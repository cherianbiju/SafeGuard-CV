# SafeGuard-CV 🦺
### Real-Time PPE Helmet Detection System for Construction Site Safety

SafeGuard-CV is a real-time computer vision system that detects missing PPE (helmets) on construction sites using a fine-tuned YOLOv8s model, optimized for edge deployment via ONNX FP16 quantization.

---

## Problem Statement
Construction sites are high-risk environments where workers often skip wearing safety helmets. SafeGuard-CV automatically detects PPE violations in real-time using a camera feed, enabling instant safety alerts.

---

## Project Structure
```
SafeGuard-CV/
├── safeguard-cv(finetuning).ipynb   # Model training notebook (Kaggle)
├── safeguard-cv(validation).ipynb   # Model validation notebook (Kaggle)
├── live_inference.py                # Real-time inference script
└── README.md
```

---

## Dataset
- **Source:** [Hardhat Detection by Michael](https://universe.roboflow.com/michael-8jeqe/hardhat-detection-iukt9) — Roboflow Universe
- **Images:** 19,800
- **Classes:** Helmet, No-Helmet, Person
- **Split:** Train / Validation / Test

---

## ⚡ Phase 1: Training (FP32 Baseline)
- **Model:** YOLOv8s 
- **Epochs:** 25
- **Batch Size:** 16

### Baseline Validation Results (FP32)
| Class | mAP50 | mAP50-95 |
|---|---|---|
| All | 0.572 | 0.289 |
| Helmet | 0.665 | 0.334 |
| No-Helmet | 0.582 | 0.266 |
| Person | 0.471 | 0.267 |

## ⚡ Phase 2: Edge Conversion & Quantization
- **Format:** ONNX
- **Quantization:** FP16
- **Command:**
```bash
yolo export model=best.pt format=onnx half=True
```

## 📊 Performance Benchmark Table

| Metric | FP32 (Base) | FP16 ONNX (Edge) | Difference |
|---|---|---|---|
| **Model Size** | 21.5 MB | 21.4 MB | -0.1 MB |
| **mAP50** | 0.572 | 0.570 | -0.002 |
| **mAP50-95** | 0.289 | 0.289 | **0.000 ** |
| **Accuracy Drop** | - | **0.00%** | No loss |
| **Avg FPS (RTX 3050)** | 44.5 FPS | 36.7 FPS | **+12 FPS ** |

## ⚡ Phase 3: Live Inference Script
The `live_inference.py` script loads the base model and edge model, and runs real-time inference on a video with the following overlays:
```
- Bounding boxes with class labels and confidence scores
- Live Inference FPS (excluding rendering time)
- Preprocessing latency (ms)
- Postprocessing / NMS latency (ms)
```
---
## Model Weights

| Model | Format | Size | Link |
|---|---|---|---|
| best_fp32.pt | PyTorch FP32 | 21.5 MB | [Download](https://drive.google.com/file/d/1cmFggU_9lFcmii4nhkKHsiE3FwDOmCSy/view?usp=drive_link) |
| best_fp16.onnx | ONNX FP16 | 21.4 MB | [Download](https://drive.google.com/file/d/1zAjvvnFMbBKyos1MyuiMo-AIbLDy95pX/view?usp=drive_link) |

## Demo Video
🔗 [Watch Demo Video](https://drive.google.com/file/d/1y47EmafdXdO-dMjd7QP5Lav-sB0ElR3t/view?usp=drive_link)

---

## Tech Stack
- Python 3.11.9
- YOLOv8s (Ultralytics)
- ONNX Runtime
- OpenCV
- Roboflow
- Kaggle (Tesla T4 GPU)

---


### How to Run
```bash
# Install dependencies
pip install ultralytics opencv-python numpy

# Run with edge model 
python live_inference.py
```
---
