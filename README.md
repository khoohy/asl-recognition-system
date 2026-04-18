# Real-Time ASL Recognition System (WLASL300)

## Overview

This project is a real-time American Sign Language (ASL) recognition system that uses webcam input to classify 300 isolated ASL signs. It processes hand, pose, and facial landmarks in real time and outputs predictions as on-screen text and speech.

The system is designed for consumer hardware and runs fully on a single GPU-enabled laptop.

---

## Key Features

- Real-time webcam-based ASL recognition
- 300-class vocabulary (WLASL300 dataset)
- Multi-modal feature extraction (hands, pose, face)
- Temporal deep learning model (BiLSTM + attention)
- Prediction smoothing and stabilization for live use
- Text + speech output (TTS support)

---

## System Pipeline


Webcam Input
↓
MediaPipe Landmark Extraction
↓
180D Feature Construction
↓
30-frame Temporal Buffer
↓
BiLSTM + Attention Model
↓
Top-K Prediction
↓
UI Display + Text-to-Speech


---

## 180D Feature Representation

Each frame is encoded into a structured 180-dimensional vector:

### 1. Hands (0–125)
- 126 features (2 hands × 21 landmarks × 3D)
- Wrist-centered normalization
- Scale normalization to remove camera distance bias

### 2. Pose (126–146)
- 21 features from upper-body joints
- Includes shoulders, elbows, wrists, and nose
- Normalized using shoulder midpoint reference

### 3. Face (147–179)
- 33 features from key facial landmarks
- Eye-centered normalization
- Helps disambiguate visually similar signs (e.g., MOTHER vs FATHER)

---

## Model Architecture

### Temporal Model
- 2-layer Bidirectional LSTM
- Hidden size: 512
- Captures forward and backward motion context

### Attention Mechanism
- Soft attention over time steps
- Focuses on informative frames
- Reduces noise from irrelevant motion

### Training Enhancements
- Targeted sampling for difficult sign pairs
- Balanced WLASL300 class distribution handling

---

## Real-Time Inference System

### Key Components

- **Inference Bridge**
  - Ensures training-inference consistency
  - Loads model + preprocessing pipeline

- **Sequence Buffer**
  - Maintains rolling 30-frame window

- **Stabilization System**
  - Confidence threshold filtering (≥ 0.65)
  - Majority vote over time window
  - Reduces flickering predictions

- **Grace Period Handling**
  - Maintains context during temporary hand loss

---

## Performance

### Validation Set
- Top-1 Accuracy: ~67%
- Top-5 Accuracy: ~89%

### Test Set (Held-out signers)
- Top-1 Accuracy: ~60%
- Top-5 Accuracy: ~86%

These results reflect real-world signer variation and unseen conditions.

---

## Design Philosophy

This system is designed around three core principles:

- **Real-time performance on consumer hardware**
- **Robustness under noisy webcam conditions**
- **Consistency between training and deployment pipelines**

---

## Deployment Optimizations

- Reduced dataset size from ~60GB → ~17GB
- Removed raw video storage after feature extraction
- Optimized MediaPipe landmark caching
- Offline TTS (Windows SAPI) for low-latency output

---
## Project Scope

This system performs isolated sign recognition only (single sign at a time).
Sentence-level ASL translation and grammar modeling are not included.

---

## Future Improvements
Improve signer-independent generalization
Add latency benchmarking for real-time evaluation
Explore pose + hand fusion improvements
ONNX / TensorRT deployment optimization

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
Run training
python scripts/train_model_300.py --device cuda
Run real-time inference
python src/main.py --model models/asl_model_300.pt --use-wlasl300

