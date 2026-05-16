# Real-Time ASL Recognition System

Real-time American Sign Language (ASL) recognition using webcam input, MediaPipe landmarks, and a temporal deep learning classifier trained on a 300-sign WLASL subset.

This repository combines:

- real-time webcam inference
- WLASL300 data preparation utilities
- training and evaluation scripts
- preprocessing shared between offline training and live deployment
- optional text-to-speech output for predicted signs

Demo video: [GitHub-hosted preview](https://github.com/user-attachments/assets/9807f81d-f893-447c-b9a1-706aef9c525b)

## Overview

The system captures live video, extracts hand landmarks with MediaPipe, converts them into normalized frame-level features, builds a short temporal sequence, and predicts the most likely ASL gloss with a BiLSTM-based model. The newer WLASL300 path also supports pose and compact face features for better disambiguation between visually similar signs.

Primary goals of the project:

- usable real-time inference on consumer hardware
- training/inference preprocessing parity
- practical robustness under noisy webcam conditions
- extensibility for dataset experiments and model ablations

## Features

- Real-time webcam-based ASL recognition
- 300-sign vocabulary based on WLASL label maps
- MediaPipe landmark extraction
- BiLSTM + temporal attention classifier
- Optional pose and face feature fusion in the WLASL300 training path
- Prediction stabilization for live inference
- Confidence gating and confusion-pair suppression
- Text-to-speech output using `pyttsx3` or `gTTS`
- Experiment runner for repeatable training comparisons

## System Pipeline

```text
Webcam / Video Frames
        |
        v
MediaPipe Landmark Extraction
        |
        v
Frame Normalization + Feature Engineering
        |
        v
Temporal Sequence Buffer
        |
        v
BiLSTM + Attention Classifier
        |
        v
Top-K Predictions + Stabilization
        |
        v
On-Screen Display + Optional Speech Output
```

## Model and Feature Representation

### Default real-time representation

The base pipeline uses hand landmarks only:

- `42 x 3` coordinates from left and right hands
- `126` features per frame after flattening
- wrist-centric normalization and scale normalization
- fixed-length temporal window, default `30` frames

### Extended WLASL300 representation

The newer training path can extend the input beyond hands:

- hands: `126` dims
- pose: selected upper-body joints
- face: compact subset of face mesh landmarks

This makes the model more useful for signs where hand shape alone is not enough.

## Current Refactor Status

- The working webcam inference path is preserved and still runs through `src/main.py`
- Production model artifacts are separated under `models/production/`
- Older experiments and historical model outputs are archived under `models/archive/`
- `backend/` and `frontend/` are placeholders for future FastAPI and React expansion

Compatibility wrappers intentionally still exist in:

- `src/modules/`
- `src/utils/preprocessing.py`
- `scripts/inference_bridge.py`

These wrappers keep legacy imports working while the implementation lives in the refactored structure.

## Repository Structure

```text
.
|-- backend/
|-- frontend/
|-- docs/
|-- models/
|   |-- archive/
|   |-- production/
|   `-- bilstm_final.pt
|-- reports/
|-- scripts/
|   |-- data/
|   |   |-- create_sample_dataset.py
|   |   |-- download_dataset.py
|   |   |-- download_from_kaggle.py
|   |   |-- extract_keypoints.py
|   |   `-- prepare_data.py
|   |-- training/
|   |   |-- train_model.py
|   |   `-- train_model_300.py
|   |-- evaluation/
|   |   |-- analyze_confusion.py
|   |   |-- inference_bridge.py
|   |   |-- test_mediapipe_hands.py
|   |   |-- test_pipeline.py
|   |   |-- test_trained_model.py
|   |   `-- verify_preprocessing_parity.py
|   |-- experiments/
|   |   `-- run_experiment_matrix.py
|   |-- inference_bridge.py          # compatibility wrapper
|   `-- __init__.py
|-- src/
|   |-- audio/
|   |   `-- text_to_speech.py
|   |-- inference/
|   |   `-- sequence_model.py
|   |-- preprocessing/
|   |   |-- keypoint_extraction.py
|   |   `-- preprocessing.py
|   |-- utils/
|   |   |-- config.py
|   |   `-- preprocessing.py         # compatibility wrapper
|   |-- video/
|   |   |-- ui.py
|   |   `-- video_capture.py
|   |-- modules/                     # compatibility wrappers
|   `-- main.py
|-- tests/
|-- data/
|-- requirements.txt
`-- README.md
```

## Requirements

### Software

- Python `3.8+`
- `pip`
- webcam for real-time inference

### Core dependencies

- `torch`
- `torchvision`
- `mediapipe`
- `opencv-python`
- `numpy`
- `scipy`
- `pyttsx3`
- `gtts`
- `onnx`
- `onnxruntime`
- `matplotlib`
- `tqdm`

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Optional: install CUDA-enabled PyTorch

If you want GPU acceleration, install the correct PyTorch build for your CUDA version from the official PyTorch index, then reinstall the remaining requirements if needed.

Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Quick Start

### Run the real-time app

```bash
python src/main.py --use-wlasl300
```

Controls:

- `q` to quit
- `s` to toggle landmark rendering

### Active production model

The default realtime WLASL300 path loads:

- `models/production/asl_wlasl300_realtime.pt`

### Run without loading a model

Useful for webcam and MediaPipe smoke testing:

```bash
python src/main.py --no-model
```

## Data Setup

The project supports a few different data sources depending on how you want to train.

### Option 1: Prepare label maps from WLASL metadata

If you already have the WLASL metadata JSON:

```bash
python scripts/data/prepare_data.py --metadata data/raw/wlasl_v0.3.json
```

This generates:

- `data/raw/label_map_300.json`
- `data/raw/label_to_index_300.json`
- `data/processed/wlasl300/selected_signs.json`

### Option 2: Download sample videos

For quick testing and exploration:

```bash
python scripts/data/download_dataset.py 10
```

### Option 3: Download processed data from Kaggle

```bash
python scripts/data/download_from_kaggle.py
```

The Kaggle helper expects the Kaggle CLI and a valid `kaggle.json` credential file in your user profile.

### Expected paths used by training

The main WLASL300 training script defaults to:

- metadata: `data/raw/wlasl_v0.3.json`
- label map: `data/raw/label_map_300.json`
- MediaPipe cache root: `data/raw/data/mp`
- Kaggle 126-dim feature root: `data/raw/kaggle/wlasl-126keypoints-2000/wlasl_keypoints_126`

## Training

The main training entry point is [`scripts/training/train_model_300.py`](scripts/training/train_model_300.py).

### Train from MediaPipe cache

```powershell
python scripts/training/train_model_300.py `
  --source mp-cache `
  --metadata data/raw/wlasl_v0.3.json `
  --label-map data/raw/label_map_300.json `
  --mp-root data/raw/data/mp `
  --device cuda `
  --epochs 50 `
  --batch-size 32 `
  --class-balanced `
  --augment `
  --use-pose `
  --use-face `
  --output-prefix models/asl_model_300_pose_face_balaug_v1
```

PowerShell single-line version:

```bash
python scripts/training/train_model_300.py --source mp-cache --metadata data/raw/wlasl_v0.3.json --label-map data/raw/label_map_300.json --mp-root data/raw/data/mp --device cuda --epochs 50 --batch-size 32 --class-balanced --augment --use-pose --use-face --output-prefix models/asl_model_300_pose_face_balaug_v1
```

### Train from Kaggle 126-dim features

```bash
python scripts/training/train_model_300.py --source kaggle-126 --kaggle-root data/raw/kaggle/wlasl-126keypoints-2000/wlasl_keypoints_126 --device cuda --epochs 50 --batch-size 32
```

### Train from JSON landmark sequences

```bash
python scripts/training/train_model_300.py --source json --landmarks-file data/raw/wlasl2000_landmarks.json --device cuda
```

### Common training options

- `--class-balanced`: weighted sampling for class imbalance
- `--augment`: enable sequence augmentation
- `--weighted-loss`: class-weighted loss
- `--focal-gamma`: focal-style loss emphasis for hard examples
- `--use-pose`: add pose features
- `--use-face`: add compact face features
- `--eval-only`: evaluate a checkpoint without retraining
- `--checkpoint`: checkpoint path for evaluation-only runs
- `--output-prefix`: output prefix for model/report/history files

## Evaluation

### Evaluate a trained checkpoint

```bash
python scripts/training/train_model_300.py --source mp-cache --eval-only --checkpoint models/production/asl_wlasl300_realtime.pt --device cuda
```

### Run a lightweight sample-video validation

```bash
python scripts/evaluation/test_trained_model.py --model models/bilstm_final.pt --samples 5
```

### Pipeline smoke test

```bash
python scripts/evaluation/test_pipeline.py
```

### Confusion analysis

```bash
python scripts/evaluation/analyze_confusion.py --checkpoint models/production/asl_wlasl300_realtime.pt
```

## Realtime Inference

The real-time app is implemented in [`src/main.py`](src/main.py) through `ASLRecognitionPipeline`.

### Standard usage

```bash
python src/main.py --use-wlasl300
```

### What the live system does

- opens the webcam
- extracts landmarks frame by frame
- preprocesses features using the same logic used in training
- maintains a rolling temporal buffer
- predicts top-K candidate signs
- stabilizes outputs using confidence, margin, motion, and vote history
- optionally speaks the stabilized sign aloud

### Current live-stability logic includes

- confidence squelching
- adaptive fallback thresholds
- per-sign overrides
- motion requirements for signs prone to idle false positives
- confusion-pair suppression
- peak-detection shortcuts for short but strong predictions

Most of these thresholds live in [`src/utils/config.py`](src/utils/config.py).

The active WLASL300 realtime bridge implementation lives in [`scripts/evaluation/inference_bridge.py`](scripts/evaluation/inference_bridge.py), while [`scripts/inference_bridge.py`](scripts/inference_bridge.py) remains as a compatibility wrapper.

## Experiments

You can launch predefined experiments with:

```bash
python scripts/experiments/run_experiment_matrix.py --matrix top1_push --device cuda
```

Dry run:

```bash
python scripts/experiments/run_experiment_matrix.py --matrix top1_push --dry-run
```

The current matrix includes variants such as:

- balanced + augmentation baseline
- pose-enhanced training
- pose + face feature fusion
- longer temporal windows
- focal-loss variants

## Configuration

Central configuration lives in [`src/utils/config.py`](src/utils/config.py).

Important settings include:

- camera resolution and FPS
- sequence length
- number of classes
- model defaults
- TTS backend
- confidence thresholds
- stabilization windows
- confusion-pair overrides

Notable defaults:

- `SEQUENCE_LENGTH = 30`
- `NUM_CLASSES = 300`
- `DEVICE` resolves to CUDA when available, otherwise CPU
- `TTS_BACKEND = "pyttsx3"`
- `ENABLE_TTS = True`

## Outputs and Artifacts

Training and experimentation produce artifacts in a few places:

- `models/production/`: active promoted model artifacts
- `models/archive/`: archived experiment checkpoints and reports
- `reports/`: evaluation outputs and analysis artifacts
- `logs/`: runtime log output

Periodic checkpoint dumps may still be produced during training workflows, but they are not part of the main active tracked structure described by this README.

`scripts/training/train_model_300.py` also writes:

- `<output-prefix>.pt`
- `<output-prefix>_best.pt`
- `<output-prefix>_history.json`
- `<output-prefix>_report.json`

## Documentation

Additional project documentation is already included:

- [`docs/SETUP.md`](docs/SETUP.md): installation and environment notes
- [`docs/API.md`](docs/API.md): API-level documentation
- [`docs/DEV.md`](docs/DEV.md): development notes
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): current refactored architecture overview
- [`docs/REFACTOR_LOG.md`](docs/REFACTOR_LOG.md): phase-by-phase refactor summary
- [`docs/FULL_SYSTEM_TECHNICAL_ANATOMY.md`](docs/FULL_SYSTEM_TECHNICAL_ANATOMY.md): deeper technical breakdown
- [`docs/PROJECT_CHANGELOG_AND_COMPONENTS.md`](docs/PROJECT_CHANGELOG_AND_COMPONENTS.md): component history and changes

## Performance Notes

The repository's earlier project notes report approximate results in this range:

- validation Top-1: about `67%`
- validation Top-5: about `89%`
- held-out test Top-1: about `60%`
- held-out test Top-5: about `86%`

Actual results depend on:

- dataset variant and cleanliness
- whether pose/face features are enabled
- signer variation
- train/val/test split strategy
- hardware and PyTorch build

## Troubleshooting

### Webcam does not open

- check whether another application is using the camera
- try a different `CAMERA_ID` in `src/utils/config.py`
- confirm OS permissions allow camera access

### CUDA is not being used

- run `python -c "import torch; print(torch.cuda.is_available())"`
- install the correct CUDA-enabled PyTorch build
- verify `nvidia-smi` works

### Low FPS during live inference

- reduce frame resolution in `Config`
- switch to CPU only if GPU drivers are unstable
- disable extra visualization work
- use a lighter checkpoint or simpler feature set

### No predictions or unstable predictions

- confirm the label map exists at `data/raw/label_map_300.json`
- make sure the checkpoint matches the expected input feature dimensions
- improve lighting and keep both hands inside frame when possible
- review confidence and stabilization thresholds in `src/utils/config.py`

### Kaggle download fails

- install Kaggle CLI
- place `kaggle.json` in the expected user directory
- verify the dataset path referenced by the script still matches your local layout

## Limitations

- focuses on isolated sign recognition, not full sentence translation
- real-world accuracy depends heavily on signer variability and camera conditions
- deployment quality depends on keeping training and inference feature formats aligned
- some helper scripts reflect older experimental paths and may need adaptation for your local data layout
