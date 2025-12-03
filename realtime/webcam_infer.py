#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, time, json
from pathlib import Path
import cv2, numpy as np, torch, timm, joblib
from torchvision import transforms

# ImageNet normalization constants used for ViT
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Default class list in case
DEFAULT_CLASSES = ["cardboard","glass","metal","paper","plastic","trash"]

def build_transform(img_size=224):
    """
    Construct the preprocessing pipeline for the ViT:
    - convert OpenCV BGR array into PIL image
    - resize
    - convert to tensor
    - normalize using ImageNet statistics
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def load_classes(path: Path | None):
    """
    Load class names from a JSON file
    Accepts:
      - {"classes":[...]}
      - ["list", "of", "labels"]
    Defaults to DEFAULT_CLASSES if file missing.
    """
    if path and path.exists():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "classes" in data: return list(data["classes"])
        if isinstance(data, list): return list(data)
    return list(DEFAULT_CLASSES)

def center_roi(frame, frac=0.6):
    """
    Extract a centered square Region Of Interest (ROI) from the frame.
    frac: fraction of the smaller dimension used as ROI size.
    Returns the cropped ROI and its bounding box coordinates.
    """
    h, w = frame.shape[:2]
    s = int(min(h, w)*frac)
    cx, cy = w//2, h//2
    x1, y1 = max(0, cx - s//2), max(0, cy - s//2)
    x2, y2 = min(w, cx + s//2), min(h, cy + s//2)
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

@torch.inference_mode()
def extract_vit_features(model, tensor_bchw: torch.Tensor) -> np.ndarray:
    """
    Forward pass through ViT to obtain features.
    Supports models returning a tuple/list by selecting the first output.
    Returns NumPy float32 array.
    """
    feats = model(tensor_bchw)
    if isinstance(feats, (list, tuple)): feats = feats[0]
    return feats.detach().cpu().numpy().astype(np.float32, copy=False)

def draw_overlay(frame, lines, box=None, color=(40,200,40)):
    """
    Draw text lines + bounding box on the webcam frame.
    Text uses a double-stroke layer for visibility.
    """
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
    """
    Main real-time webcam loop:
      - capture frame
      - extract ROI
      - optional Laplacian gate ("No waste" mode)
      - extract ViT features
      - run sklearn classifier
      - compute probabilities + top-k
      - apply confidence / margin rules
      - overlay results + FPS
    """
    ap = argparse.ArgumentParser("Webcam inference (ViT features + sklearn) with 'No waste' mode")
    
    # Model and label paths
    ap.add_argument("--model", type=Path, default=Path("../models/deep/vit_pca256_histgb.joblib"))
    ap.add_argument("--classes", type=Path, default=Path("../labels.json"))
    ap.add_argument("--backbone", default="vit_small_patch16_224")
    ap.add_argument("--img-size", type=int, default=224)
    
    # Webcam / video settings
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--flip", action="store_true")
    
    # Device choice
    ap.add_argument("--device", choices=["cpu","cuda"], default=None)

    # --- “No waste” controls ---
    ap.add_argument("--conf", type=float, default=0.70, help="min proba top-1")
    ap.add_argument("--margin", type=float, default=0.10, help="min (top1 - top2)")
    ap.add_argument("--roi-frac", type=float, default=0.60, help="fraction du min(h,w) pour la zone analysée")
    ap.add_argument("--gate", choices=["none","lapvar"], default="lapvar",
                    help="filtre d'activation de scène")
    ap.add_argument("--lap-th", type=float, default=60.0,
                    help="seuil variance du Laplacien (texture) sous lequel on dit 'No waste'")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    # Load class list and sklearn classifier
    class_names = load_classes(args.classes)
    clf = joblib.load(args.model)
    
    # Select device (CPU unless CUDA available)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load ViT backbone (features only → num_classes=0)
    model = timm.create_model(args.backbone, pretrained=True, num_classes=0).eval().to(device)
    
    # Preprocessing pipeline
    tfm = build_transform(args.img_size)

    # Initialize webcam
    cap = cv2.VideoCapture(args.cam, cv2.CAP_AVFOUNDATION)
    prev, fps = time.time(), 0.0

    print(f"[INFO] Device={device} | Backbone={args.backbone} | Classes={class_names}")

    while True:
        ok, frame = cap.read()
        if not ok: break
        if args.flip: frame = cv2.flip(frame, 1)

        # --- Extract centered ROI ---
        roi, box = center_roi(frame, args.roi_frac)

        # --- Laplacian variance gate: detects empty/untextured background ---
        show_unknown_reason = ""
        if args.gate == "lapvar":
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            lapvar = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lapvar < args.lap_th:
                show_unknown_reason = f"No waste (low texture: {lapvar:.0f})"

        # --- Classification if gate is satisfied ---
        label, conf = "No waste", 0.0
        topline = ""
        if show_unknown_reason == "":
            # Preprocess input for ViT
            inp = tfm(roi).unsqueeze(0).to(device, non_blocking=True)
            
            # Extract ViT features
            feats = extract_vit_features(model, inp)
            
            # Compute probabilities using sklearn classifier
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(feats)[0]
            else:
                # Handle classifiers without predict_proba()
                if hasattr(clf, "decision_function"):
                    s = clf.decision_function(feats)[0]
                    s = np.atleast_1d(s)
                    s = s - np.max(s)
                    probs = np.exp(s) / np.sum(np.exp(s))
                else:
                    # Fallback: pure predicted class
                    pred = int(clf.predict(feats)[0]); probs = np.zeros(len(class_names)); probs[pred]=1.0

            # Sort probabilities
            order = np.argsort(probs)[::-1]
            top1, top2 = order[0], (order[1] if len(order)>1 else order[0])
            conf1, conf2 = float(probs[top1]), float(probs[top2])

            # --- Apply “No waste” rule (confidence + margin) ---
            if (conf1 >= args.conf) and ((conf1 - conf2) >= args.margin):
                label, conf = class_names[top1], conf1
                k = max(1, min(args.topk, len(probs)))
                topline = " / ".join([f"{class_names[i]} {probs[i]*100:.0f}%" for i in order[:k]])
            else:
                show_unknown_reason = "No waste (low confidence/margin)"

        # --- FPS + overlay ---
        now = time.time()
        fps = 0.9*fps + 0.1*(1.0/max(1e-6, now - prev)); prev = now

        # Build overlay text
        lines = [f"Top-1: {label}  ({conf*100:.1f}%)", f"FPS: {fps:.1f}"]
        if topline: lines.insert(1, topline)
        if show_unknown_reason and not topline:
            lines.insert(1, show_unknown_reason)

        # Draw overlay and show frame
        draw_overlay(frame, lines, box=box)
        cv2.imshow("Webcam - Garbage Classifier", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
