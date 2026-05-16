# Architecture

## Current Runtime Entry Point

- [src/main.py](<C:/Users/Khoo Han Yang/Desktop/fyp2 - Copy/src/main.py:1>) is the active webcam inference entry point
- It orchestrates capture, keypoint extraction, sequence buffering, inference, UI rendering, and text-to-speech

## Runtime Modules

### `src/inference/`

- Model definitions and high-level inference pipeline
- Contains the sequence model implementation used by the legacy and realtime paths

### `src/preprocessing/`

- Keypoint extraction and preprocessing logic
- Handles MediaPipe-derived runtime features used before inference

### `src/video/`

- Webcam capture and on-screen rendering helpers
- Responsible for frame acquisition and UI drawing

### `src/audio/`

- Text-to-speech integration for spoken prediction output

### `src/utils/`

- Keeps shared configuration in `src/utils/config.py`
- `src/utils/preprocessing.py` currently exists as a compatibility wrapper to the new preprocessing package

## Realtime WLASL300 Bridge

- [scripts/evaluation/inference_bridge.py](<C:/Users/Khoo Han Yang/Desktop/fyp2 - Copy/scripts/evaluation/inference_bridge.py:1>) is the active shared bridge for WLASL300 realtime inference
- It loads the production checkpoint, rebuilds the model from checkpoint metadata, and applies shared runtime feature engineering
- [scripts/inference_bridge.py](<C:/Users/Khoo Han Yang/Desktop/fyp2 - Copy/scripts/inference_bridge.py:1>) remains as a compatibility wrapper so existing imports continue to work

## Model Artifacts

### `models/production/`

- Contains the active live checkpoint:
  - `models/production/asl_wlasl300_realtime.pt`
- Also stores the paired history and report files for the promoted production run

### `models/archive/`

- Stores older experiment checkpoints, validation snapshots, and historical reports
- Archive contents are preserved for traceability and comparison, not deleted

## Compatibility Layer

The refactor keeps wrappers in place to avoid breaking the current codebase while structure improves:

- `src/modules/`
- `src/utils/preprocessing.py`
- `scripts/inference_bridge.py`

These wrappers let legacy imports keep working while the main implementation lives in more domain-specific folders.

## High-Level Flow

1. Webcam frames enter through `src/main.py`
2. Video capture is handled by `src/video/`
3. Landmarks and feature preprocessing come from `src/preprocessing/`
4. Realtime WLASL300 inference uses `scripts/evaluation/inference_bridge.py`
5. Model classes live in `src/inference/`
6. Predictions are rendered through `src/video/ui.py`
7. Spoken output is handled by `src/audio/text_to_speech.py`

