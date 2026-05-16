# Refactor Log

## Phase 1: Repo Hygiene

- Removed generated `__pycache__` folders and `.pyc` files
- Updated `.gitignore` to exclude logs, virtual environments, raw/processed data, large archives, and checkpoint dumps
- Added scaffold folders:
  - `models/production/`
  - `models/archive/`
  - `backend/`
  - `frontend/`
  - `tests/`

## Phase 2: Model Artifact Organization

- Confirmed the active realtime checkpoint before moving anything
- Promoted the production model into:
  - `models/production/asl_wlasl300_realtime.pt`
- Moved paired report/history artifacts into `models/production/`
- Archived prior experimental checkpoints under `models/archive/`
- Kept `models/bilstm_final.pt` in place for the legacy path
- Added `models/README.md`

## Phase 3: Script Reorganization

- Reorganized `scripts/` into:
  - `scripts/data/`
  - `scripts/training/`
  - `scripts/evaluation/`
  - `scripts/experiments/`
- Moved the realtime bridge implementation to `scripts/evaluation/inference_bridge.py`
- Kept `scripts/inference_bridge.py` as a compatibility wrapper
- Updated nested script imports and path handling to remain runnable from the repo root

## Phase 4: Source Reorganization

- Reorganized `src/` into:
  - `src/inference/`
  - `src/preprocessing/`
  - `src/video/`
  - `src/audio/`
- Kept `src/utils/config.py` in place
- Left wrapper modules in `src/modules/`
- Left `src/utils/preprocessing.py` as a compatibility wrapper
- Updated `src/main.py` to import from the new domain folders without changing runtime behavior

## Refactor Principles

- Preserve the working webcam inference path
- Avoid rewriting ML logic
- Prefer compatibility wrappers over risky big-bang migration
- Separate production artifacts from historical experiments
- Make the repo easier to explain as a junior AI engineer portfolio project

