#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, time, json
from pathlib import Path
import cv2, numpy as np, torch, timm, joblib
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
DEFAULT_CLASSES = ["cardboard","glass","metal","paper","plastic","trash"]

def build_transform(img_size=224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def load_classes(path: Path | None):
    if path and path.exists():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "classes" in data: return list(data["classes"])
        if isinstance(data, list): return list(data)
    return list(DEFAULT_CLASSES)

def center_roi(frame, frac=0.6):
    h, w = frame.shape[:2]
    s = int(min(h, w)*frac)
    cx, cy = w//2, h//2
    x1, y1 = max(0, cx - s//2), max(0, cy - s//2)
    x2, y2 = min(w, cx + s//2), min(h, cy + s//2)
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

@torch.inference_mode()
def extract_vit_features(model, tensor_bchw: torch.Tensor) -> np.ndarray:
    feats = model(tensor_bchw)
    if isinstance(feats, (list, tuple)): feats = feats[0]
    return feats.detach().cpu().numpy().astype(np.float32, copy=False)

def draw_overlay(frame, lines, box=None, color=(40,200,40)):
    y = 28
    for t in lines:
        cv2.putText(frame, t, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, t, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += 26
    if box:
        x1,y1,x2,y2 = box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 2)
    return frame

def main():
    ap = argparse.ArgumentParser("Webcam inference (ViT features + sklearn) | stabilisation + plein écran")
    ap.add_argument("--model", type=Path, default=Path("../models/deep/vit_pca256_histgb.joblib"))
    ap.add_argument("--classes", type=Path, default=Path("../labels.json"))
    ap.add_argument("--backbone", default="vit_small_patch16_224")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--device", choices=["cpu","cuda"], default=None)

    # --- “No waste” controls ---
    ap.add_argument("--conf", type=float, default=0.9, help="min proba top-1")
    ap.add_argument("--margin", type=float, default=0.25, help="min (top1 - top2)")
    ap.add_argument("--roi-frac", type=float, default=0.55, help="fraction du min(h,w) pour la zone analysée")
    ap.add_argument("--lap-th", type=float, default=120.0, help="seuil variance du Laplacien sous lequel 'No waste'")

    # --- Rejets supplémentaires ---
    ap.add_argument("--min-edges", type=float, default=0.020, help="densité minimale d'arêtes (Canny) dans le ROI")
    ap.add_argument("--hsv-sat", type=float, default=0.08, help="seuil de saturation moyenne HSV")
    ap.add_argument("--motion-th", type=float, default=5.0, help="seuil d’activité (diff inter-frame)")
    ap.add_argument("--face-suppress", action="store_true", help="ignorer si visage détecté")

    # --- Stabilisation / UX ---
    ap.add_argument("--ema", type=float, default=0.7, help="coefficient EMA des proba (0..1, haut = +lente)")
    ap.add_argument("--min-switch-ms", type=int, default=1500, help="temps min entre 2 changements de label")
    ap.add_argument("--hold-ms", type=int, default=800, help="durée min d'affichage avant MAJ")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--fullscreen", action="store_true", help="démarrer en plein écran")
    args = ap.parse_args()

    # Chargements
    class_names = load_classes(args.classes)
    clf = joblib.load(args.model)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.backbone, pretrained=True, num_classes=0).eval().to(device)
    tfm = build_transform(args.img_size)

    win = "Webcam - Garbage Classifier"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    face_cascade = None
    if args.face_suppress:
        face_cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW if hasattr(cv2,'CAP_DSHOW') else 0)
    prev_gray_full = None
    prev, fps = time.time(), 0.0

    ema_probs = None
    current_label, current_conf = "No waste", 0.0
    last_change_ms = int(time.time()*1000)
    last_update_ms = last_change_ms

    print(f"[INFO] Device={device} | Backbone={args.backbone} | Classes={class_names}")

    while True:
        ok, frame = cap.read()
        if not ok: break
        if args.flip: frame = cv2.flip(frame, 1)

        roi, box = center_roi(frame, args.roi_frac)
        show_unknown_reason = ""

        # --- GATES DE REJET AVANT INFÉRENCE ---
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv_roi  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv_roi[...,1]) / 255.0)
        lapvar = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        if lapvar < args.lap_th:
            show_unknown_reason = f"No waste (low texture: {lapvar:.0f})"

        if show_unknown_reason == "":
            edges = cv2.Canny(gray_roi, 50, 150)
            edge_density = float(np.count_nonzero(edges)) / edges.size
            if edge_density < args.min_edges:
                show_unknown_reason = f"No waste (few edges: {edge_density*100:.1f}%)"

        if show_unknown_reason == "" and mean_sat < args.hsv_sat:
            show_unknown_reason = f"No waste (low saturation: {mean_sat:.2f})"

        if show_unknown_reason == "":
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_full_blur = cv2.GaussianBlur(gray_full, (5,5), 0)
            if prev_gray_full is None:
                prev_gray_full = gray_full_blur.copy()
            diff = cv2.absdiff(prev_gray_full, gray_full_blur)
            motion_score = float(np.mean(diff))
            prev_gray_full = gray_full_blur
            if motion_score < args.motion_th:
                show_unknown_reason = f"No waste (low motion: {motion_score:.1f})"

        if show_unknown_reason == "" and face_cascade is not None:
            faces = face_cascade.detectMultiScale(gray_roi, scaleFactor=1.2, minNeighbors=5,
                                                  flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60))
            if len(faces) > 0:
                show_unknown_reason = "No waste (face detected)"

        # --- INFÉRENCE ---
        probs = None
        if show_unknown_reason == "":
            inp = tfm(roi).unsqueeze(0).to(device, non_blocking=True)
            feats = extract_vit_features(model, inp)
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(feats)[0].astype(np.float32)
            else:
                if hasattr(clf, "decision_function"):
                    s = clf.decision_function(feats)[0]
                    s = np.atleast_1d(s); s = s - np.max(s)
                    probs = (np.exp(s) / np.sum(np.exp(s))).astype(np.float32)
                else:
                    pred = int(clf.predict(feats)[0])
                    probs = np.zeros(len(class_names), dtype=np.float32); probs[pred] = 1.0
            if ema_probs is None:
                ema_probs = probs.copy()
            else:
                ema_probs = args.ema * ema_probs + (1 - args.ema) * probs

        # --- LABEL FINAL ---
        candidate_label, candidate_conf, topline = "No waste", 0.0, ""
        if probs is not None:
            order = np.argsort(ema_probs)[::-1]
            top1, top2 = order[0], order[1]
            conf1, conf2 = float(ema_probs[top1]), float(ema_probs[top2])
            if (conf1 >= args.conf) and ((conf1 - conf2) >= args.margin):
                candidate_label, candidate_conf = class_names[top1], conf1
                k = max(1, min(args.topk, len(ema_probs)))
                topline = " / ".join([f"{class_names[i]} {ema_probs[i]*100:.0f}%" for i in order[:k]])
            else:
                show_unknown_reason = "No waste (low confidence/margin)"

        now_ms = int(time.time()*1000)
        if candidate_label != current_label and (now_ms - last_change_ms) >= args.min_switch_ms:
            current_label, current_conf = candidate_label, candidate_conf
            last_change_ms = now_ms
        elif candidate_label == current_label and (now_ms - last_update_ms) >= args.hold_ms:
            current_conf = candidate_conf
            last_update_ms = now_ms

        # --- FPS + affichage ---
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - prev))
        prev = now

        lines = [f"Top-1: {current_label}  ({current_conf*100:.1f}%)", f"FPS: {fps:.1f}"]
        if topline and current_label != "No waste":
            lines.insert(1, topline)
        elif show_unknown_reason:
            lines.insert(1, show_unknown_reason)

        draw_overlay(frame, lines, box=box)
        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')): break
        elif key == ord('f'):
            fs = cv2.getWindowProperty(win, cv2.WND_PROP_FULLSCREEN)
            target = cv2.WINDOW_NORMAL if int(fs) == 1 else cv2.WINDOW_FULLSCREEN
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, target)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
