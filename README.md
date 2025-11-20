<h1 align="center">  TrashNet.IA — Real-Time Waste Classification System </h1>
<p align="center">
  <img src="pics/logo.png" alt="TrashNet.IA Logo" width="280">
</p>

A complete end-to-end computer vision pipeline designed to detect and classify waste in real time using a webcam.  
This project combines traditional image processing, deep learning feature extraction, dimensionality reduction, and lightweight machine-learning classifiers to achieve high-performance inference on consumer hardware.
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg">
  <img src="https://img.shields.io/badge/OpenCV-Live%20Inference-green.svg">
  <img src="https://img.shields.io/badge/Model-ViT%20%2B%20PCA%20%2B%20HGB-orange.svg">
  <img src="https://img.shields.io/badge/Accuracy-~0.88-lightgrey.svg">
  <img src="https://img.shields.io/badge/License-MIT-black.svg">
</p>
---

## *Overview*

TrashNet.IA is built on the Garbage Classification dataset, containing six categories of waste:  
*cardboard, glass, metal, paper, plastic, trash*.  
The images exhibit significant variability in background, lighting, and viewpoint, allowing the resulting model to generalize well to real-world usage.

The project includes all stages of an ML workflow:

* preprocessing and foreground segmentation  
* handcrafted feature extraction  
* deep feature extraction (ResNet / Vision Transformer)  
* dimensionality reduction (PCA)  
* model training (scikit-learn)  
* real-time inference with OpenCV  

The deployed classifier uses a *Vision Transformer embedding → PCA (256 components) → Histogram Gradient Boosting* model.

---

## *Project Structure*

### *1. Handcrafted Features (01_handcrafted_features.html)*  
This module explores classical computer vision descriptors such as color histograms, HOG, LBP, and edge-based features.  
It establishes a baseline and highlights the limitations of handcrafted approaches when dealing with high intra-class variance.

### *2. Deep Features with ResNet and ViT (02_deep_features_resnet_vit.html)*  
High-level image embeddings are extracted using pretrained convolutional networks and the Vision Transformer.  
A PCA projection reduces the embedding dimension before training several shallow classifiers.  
The final selected model achieves strong performance with a compact memory footprint, making it suitable for real-time deployment.

### *3. Segmentation, Detection and Batch Classification (03_detection_segmentation_classif.html)*  
Foreground segmentation isolates the waste object before classification, improving robustness.  
Batch evaluation scripts, diagnostics, and visual overlays are also included.
## *System Architecture*

mermaid
flowchart TD

A[Webcam Stream] --> B[Frame Acquisition<br>OpenCV]
B --> C[ROI Extraction<br>Foreground Segmentation]
C --> D[Transform to 224×224<br>Normalize]
D --> E[Vision Transformer<br>Feature Extraction]
E --> F[PCA 256-D Projection]
F --> G[HistGradientBoosting Classifier]
G --> H[Temporal Smoothing<br>EMA + Voting]
H --> I[Prediction Overlay<br>FPS / Confidence]
I --> J[Real-Time Display]

subgraph Dataset Expansion
    C --> S[Save ROI<br>Hotkey 1–6]
    S --> T[Build New Dataset]
end

---

## **Real-Time Inference: webcam_infer.py**

The live inference script provides:

* dynamic window resizing  
* optional fullscreen mode  
* foreground ROI extraction  
* temporal smoothing (EMA + voting queue)  
* uncertainty rejection based on entropy and confidence margins  
* optional face suppression  
* real-time overlays with class predictions and probabilities  

It supports dataset expansion by allowing on-the-fly saving of labeled samples directly from webcam input.

---
### **About the .joblib Model**

The final model stored as a .joblib file represents the core of the TrashNet.IA classification system. It condenses the entire training pipeline—feature extraction, dimensionality reduction, and machine-learning decisions—into a single deployable artifact. We created this file to ensure fast loading, reproducibility, and portability across machines without requiring heavy deep-learning frameworks at inference time.  

The model contains a PCA-compressed embedding space derived from Vision Transformer features, followed by a Histogram Gradient Boosting classifier trained on the six waste categories. By separating feature extraction from classification and saving the lightweight classifier in .joblib format, we achieve near-instant predictions, minimal memory usage, and easy integration in real-time applications such as webcam detection. This design allows us to benefit from the representational power of deep networks while keeping inference efficient and accessible, even on systems without GPU acceleration.

## *Model Performance*

Using *ViT + PCA + HistGradientBoosting*, the system achieves:

* *Accuracy:* ~0.88  
* *Macro F1-score:* ~0.83  
* High recall on common classes such as paper, metal, and cardboard  
* Improved robustness thanks to segmentation and uncertainty gates  

These results are consistent across both notebook evaluations and real-time inference.

---

## *Installation*

bash
git clone https://github.com/<your-username>/trashnet.ia.git
cd trashnet.ia
pip install -r requirements.txt


Models and labels should be placed in:


/models/deep/
labels.json


To launch real-time inference:

bash
python webcam_infer.py --model models/deep/vit_pca256_histgb.joblib


---

---
