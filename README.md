
# Real-Time ASL Recognition System

This project is a webcam-based American Sign Language recognition pipeline designed for standard consumer hardware. It targets isolated-sign recognition with a 300-sign WLASL vocabulary, on-screen text output, and text-to-speech.

## Current Reality

The project currently adheres to the practical direction of your proposal:
- webcam-only input
- MediaPipe-based hand landmark extraction
- lightweight temporal model for isolated-sign recognition
- deployment on a single RTX 4050 laptop
- text plus TTS output

The project does **not** yet fully satisfy every target in the proposal:
- test accuracy is currently below the target range
- formal end-to-end latency benchmarking is still pending
- the runtime currently uses hand landmarks only, not full Holistic upper-body landmarks
- continuous signing and sentence-level translation are out of scope

## Current Best Results

Real WLASL300 run using the local MediaPipe cache:
- Validation Top-1: `55.83%`
- Validation Top-5: `81.46%`
- Held-out test Top-1: `50.61%`
- Held-out test Top-5: `77.92%`

These numbers are a credible baseline, but they are still below the project target of `65-75%` Top-1 and `85-95%` Top-5.

## What Matches the Proposal

### Objective 1: Real-time webcam pipeline
- Uses a standard webcam as the only input device
- Avoids Leap Motion, Kinect, Myo, and multi-camera rigs
- Runs on a single laptop with RTX 4050 acceleration
- Provides live UI plus TTS

### Objective 2: 300-sign recognition
- Uses WLASL metadata to select the top 300 glosses
- Trains a BiLSTM with self-attention
- Uses hand-focused landmark features with normalization and temporal resampling
- Evaluates on held-out train/val/test splits

### Objective 3: Low-latency translation output
- Produces live top-k predictions
- Displays readable on-screen text
- Supports speech output with `pyttsx3` or `gTTS`

## What Is Still In Progress

- Hitting the target accuracy range
- Formal latency measurement against the `100-150 ms` target
- More robust missing-keypoint imputation in the real WLASL300 path
- Optional pose-feature integration for better signer robustness
- Optional quantized or ONNX deployment path

## Actual Runtime Design

The current runtime path is:

```text
Webcam Frame
  -> MediaPipe Hands
  -> 126-dim hand feature vector
  -> 30-frame sliding sequence
  -> BiLSTM + self-attention
  -> Top-k predictions
  -> UI + optional TTS
```

This is intentionally lightweight and aligned with the project aim of deployability on consumer hardware.

## Dataset and Training Path

The current real training run uses:
- [data/raw/wlasl_v0.3.json](c:\Users\Khoo Han Yang\Desktop\fyp2\data\raw\wlasl_v0.3.json)
- [data/raw/data/mp](c:\Users\Khoo Han Yang\Desktop\fyp2\data\raw\data\mp)

The project no longer relies on the removed synthetic JSON landmark file.

## Key Files

- [src/main.py](c:\Users\Khoo Han Yang\Desktop\fyp2\src\main.py): webcam runtime pipeline
- [scripts/inference_bridge.py](c:\Users\Khoo Han Yang\Desktop\fyp2\scripts\inference_bridge.py): shared realtime preprocessing and prediction
- [scripts/prepare_data.py](c:\Users\Khoo Han Yang\Desktop\fyp2\scripts\prepare_data.py): WLASL300 label-map generation and shared feature engineering
- [scripts/train_model_300.py](c:\Users\Khoo Han Yang\Desktop\fyp2\scripts\train_model_300.py): training and evaluation
- [src/utils/config.py](c:\Users\Khoo Han Yang\Desktop\fyp2\src\utils\config.py): dynamic device selection and vocabulary loading

## Commands

Train on the real local MediaPipe cache:

```bash
python scripts/train_model_300.py --source mp-cache --metadata data/raw/wlasl_v0.3.json --mp-root data/raw/data/mp --epochs 20 --batch-size 64 --device cuda
```

Run the prioritized experiment matrix:

```bash
python scripts/run_experiment_matrix.py --device cuda --epochs 30 --batch-size 32
```

Evaluate on the held-out test split:

```bash
python scripts/train_model_300.py --source mp-cache --metadata data/raw/wlasl_v0.3.json --mp-root data/raw/data/mp --batch-size 64 --device cuda --eval-only --checkpoint models/asl_model_300.pt
```

Run webcam inference:

```bash
python src/main.py --model models/asl_model_300.pt --use-wlasl300
```

## Recommended Next Steps

If the goal is to align more closely with the proposal targets, the strongest next steps are:
- add formal latency logging to the webcam path
- compare hand-only versus hand-plus-pose features
- improve missing-frame handling in the WLASL300 training path
- tune the model and scheduler around the current real baseline

## Scope Reminder

This project is currently an isolated-sign recognizer, not a full ASL grammar or sentence translator. That still fits the core project direction, but it should be stated clearly in reports and presentations.

