# Project Change Log And Component Reference

Project: Real-time ASL Recognition Pipeline  
Maintained as a living document for code, training, and architecture updates.  
Last updated: April 16, 2026

## How to use this document

This file is meant to be updated whenever the project changes in a meaningful way.

For each future update, add:
- what changed
- why it changed
- which files were touched
- what libraries, data sources, or tools were involved
- what effect it is expected to have on training, inference, usability, or maintainability

## Current project summary

The project is a webcam-based ASL recognition system that:
- captures live video with OpenCV
- extracts hand landmarks and optional pose plus compact face landmarks with MediaPipe
- converts landmark sequences into fixed-length model inputs
- classifies isolated signs with a BiLSTM + attention model
- shows predictions on screen and can speak them with TTS

The strongest measured WLASL300 validation direction found so far is the balanced-sampling plus augmentation path. From the saved histories in `models/`, the best validation Top-1 observed is `59.27%` and the best validation Top-5 observed is `84.48%`. The held-out test result recorded in the existing project documentation is lower, so the current gap is not just model capacity; it is also a generalization problem.

## Latest change log

### 2026-04-16: Production-only system anatomy document added

What was done:
- added `docs/FULL_SYSTEM_TECHNICAL_ANATOMY.md` as a focused technical deep-dive of the active WLASL300 production path
- documented the current `30 x 180` face-aware feature contract, including exact hand, pose, and face index ranges
- documented the shared preprocessing math, BiLSTM-plus-attention architecture, targeted class-repair strategy, realtime grace-period logic, TTS behavior, storage optimization, and current generalization gap
- explicitly corrected a few easy-to-misremember assumptions in the documentation, including that the active face-aware checkpoint is `30` frames rather than `40`, that temporal interpolation uses NumPy rather than SciPy, and that the current runtime no longer hard-blanks visible-hand paused signs based on motion alone

Why it was done:
- the project needed one accurate reference document for understanding the production-ready architecture end to end without mixing in legacy paths or outdated assumptions
- the earlier prompt was useful structurally, but some of its implementation assumptions no longer matched the live code, so the new document was written against the repository as it exists now

Files changed:
- `docs/FULL_SYSTEM_TECHNICAL_ANATOMY.md`
- `docs/PROJECT_CHANGELOG_AND_COMPONENTS.md`

Expected effect:
- faster onboarding into the active pipeline
- fewer misunderstandings about which architecture, feature width, and runtime behaviors are actually live
- a clearer bridge between the codebase, the saved checkpoints, and the engineering decisions made during debugging and retraining

### 2026-04-16: Runtime stability update for TTS, idle suppression, and temporal grace period

What was done:
- upgraded `src/modules/text_to_speech.py` to prefer native Windows SAPI speech when available, while keeping the previous backend path as fallback
- moved the offline speech path onto a more reliable worker-owned backend lifecycle so repeated speech calls do not stop after the first utterance
- added explicit realtime idle-state clearing support through `scripts/inference_bridge.py` and `src/main.py` so stale buffered predictions stop hallucinating signs such as `LATE` when hands are no longer present
- added a 10-frame temporal grace period to the WLASL300 live loop so brief hand dropouts no longer clear the sequence buffer immediately
- made the grace-period path retain the existing 180-dimensional temporal context for hands + pose + face features and resume normal inference immediately if hands reappear before the limit is exceeded
- updated the live status text to show when the system is temporarily holding context instead of resetting
- fixed a realtime crash where the grace-period path could return a raw Top-5 list instead of the standard `(sign, confidence, top_k)` tuple expected by the OpenCV UI loop
- added a low-motion gate in `scripts/inference_bridge.py` so static or near-static buffered hand sequences are treated as idle instead of flickering false signs on screen
- relaxed that low-motion gating in the next runtime pass so a sign no longer turns off just because the signer pauses mid-gesture while hands remain visible on screen
- updated the live status text so hand-visible paused states hold sign context instead of being treated as idle

Why it was done:
- the previous TTS behavior could speak once and then go silent, which made live testing unreliable
- the immediate no-hand reset was too aggressive for long or complex signs and could discard useful BiLSTM context during momentary tracking loss
- idle hallucinations while the signer was not moving needed to be suppressed without making the live model brittle to natural short dropouts
- the first grace-period implementation exposed a tuple-shape mismatch in the UI loop, and static non-sign posture was still causing rapid `Sign` flicker
- the stricter low-motion suppression also blanked legitimate long or paused signs halfway through, which felt worse than a temporary false positive for actual signing flow

Files changed:
- `src/modules/text_to_speech.py`
- `scripts/inference_bridge.py`
- `src/main.py`
- `docs/PROJECT_CHANGELOG_AND_COMPONENTS.md`

Expected effect:
- repeated TTS output should remain available throughout a session instead of dying after the first spoken word
- long signs should survive brief hand loss better because the feature buffer now persists for up to 10 consecutive missing-hand frames
- idle false positives should still clear once the grace window is exceeded, while brief tracking interruptions no longer wipe the active sequence immediately

### 2026-04-16: Face-aware checkpoint evaluated and set as live testing default

What was done:
- ran `scripts/analyze_confusion.py` on `models/asl_model_300_pose_face_balaug_v1.pt`
- saved the face-aware confusion outputs under `reports/pose_face_balaug_v1_confusion`
- checked the targeted held-out classes `MOTHER`, `FATHER`, `TALL`, and `THEORY` from the generated classification report
- updated `src/main.py` so the realtime `--use-wlasl300` path now defaults to `models/asl_model_300_pose_face_balaug_v1.pt` for live testing

Why it was done:
- the face-aware retrain needed direct validation on the exact glosses that motivated the feature upgrade
- switching the live default model makes it easier to test whether the webcam behavior improves on face-anchored signs without passing a custom checkpoint path every time

Files changed:
- `src/main.py`
- `docs/PROJECT_CHANGELOG_AND_COMPONENTS.md`

Measured effect:
- face-aware checkpoint overall held-out test metrics remained `60.03%` Top-1 and `85.80%` Top-5
- targeted held-out class results were:
- `MOTHER`: precision `1.000`, recall `1.000`, support `3`
- `FATHER`: precision `0.667`, recall `1.000`, support `2`
- `TALL`: precision `1.000`, recall `0.667`, support `3`
- `THEORY`: precision `1.000`, recall `1.000`, support `2`
- this means the new model looks strong for `MOTHER`, `FATHER`, and `THEORY` on the held-out split, while `TALL` still misses one test example

### 2026-04-16: Face-aware pipeline update, TTS fix, and vocabulary audit

What was done:
- extended the shared WLASL feature path to support a compact selected-face-landmark slice alongside hands and optional pose
- updated `scripts/train_model_300.py` so the MediaPipe cache loader can read cached `face_<video_id>.pickle` files and append normalized face features during training
- updated checkpoint metadata and the realtime bridge so saved models now carry `pose_joints` and `face_landmarks`, allowing runtime feature reconstruction without assuming the old fixed `147D` path
- updated `src/modules/keypoint_extraction.py` to extract live face mesh landmarks alongside hands and pose for webcam inference
- updated `scripts/inference_bridge.py`, `src/main.py`, and `scripts/analyze_confusion.py` so training, webcam inference, and confusion analysis all agree on pose-plus-face-aware feature layouts
- fixed the realtime TTS transition path so the same gloss can be spoken again after predictions drop back to silence instead of only speaking once
- added optional forced-gloss inclusion support to `scripts/prepare_data.py` so custom label maps can include low-frequency glosses such as `I` and `ME`
- checked the full `data/raw/wlasl_v0.3.json` metadata and confirmed that `I` and `ME` exist in WLASL, but not in the default top-300 frequency-selected vocabulary
- measured approximate metadata rank of `I` as `1779` and `ME` as `1165`, which explains why neither appears in the current `label_map_300.json`

Why it was done:
- live confusion between `MOTHER` and `FATHER` is not well addressed by hand landmarks alone because those signs depend heavily on facial anchor position such as chin versus forehead
- the local MediaPipe cache already included full face mesh pickles, so the project had a realistic path to add face-aware features without re-downloading raw videos
- retraining and deployment needed to stay aligned so a future face-aware checkpoint can be used directly in the webcam demo
- the TTS issue was hurting live usability because repeated valid recognitions could become silent after the first spoken output
- the vocabulary audit was needed because the user specifically asked whether `I` and `ME` could be added

Files changed:
- `scripts/prepare_data.py`
- `scripts/train_model_300.py`
- `scripts/inference_bridge.py`
- `scripts/analyze_confusion.py`
- `scripts/run_experiment_matrix.py`
- `src/modules/keypoint_extraction.py`
- `src/main.py`
- `src/modules/text_to_speech.py`
- `docs/PROJECT_CHANGELOG_AND_COMPONENTS.md`

Tools and libraries involved:
- MediaPipe Hands, Pose, and Face Mesh
- PyTorch
- NumPy

Expected effect:
- better separation for face-anchored sign pairs such as `MOTHER` versus `FATHER`
- a cleaner retraining path for hands + pose + face models using the existing MediaPipe cache
- correct runtime support for future face-aware checkpoints instead of assuming the old pose-only feature width
- TTS that continues working after the first spoken recognition
- a documented route to generate a custom label map that forces in `I` and `ME` if the project chooses to move beyond the strict default top-300 frequency list

### 2026-04-15: Targeted confusion follow-up and sequence-length-40 pose retrain

What was done:
- ran `scripts/analyze_confusion.py` on `models/asl_model_300_bilstm512_pose_v1.pt` and checked the specific failure patterns for `MOTHER`, `FATHER`, `TALL`, and `THEORY`
- confirmed that `MOTHER` was a weak held-out class with dominant confusion into `FATHER`, while `TALL` was less stable than `THEORY`
- manually inspected the raw MediaPipe cache statistics for those four glosses across train/val/test splits
- confirmed that `MOTHER` and `FATHER` are mostly one-handed signs in this cache and therefore rely heavily on anchor-position cues rather than two-hand interaction
- added targeted gloss oversampling support to `scripts/train_model_300.py` through `--boost-glosses` and `--boost-factor`
- launched a new pose-enhanced retrain with `sequence_length=40` as `models/asl_model_300_bilstm512_pose_seq40_targeted_v1`
- used targeted boosting for `mother,father,tall,theory` with a factor of `2.5` on top of the existing class-balanced sampler and augmentation pipeline

Why it was done:
- the live demo failures suggested that a general accuracy number was not enough; the project needed class-specific diagnosis
- the confusion report showed that `MOTHER -> FATHER` was a real held-out failure mode rather than only a webcam artifact
- the raw cache inspection showed that several of the key signs are effectively one-handed in the extracted landmarks, so stronger exposure to those anchor-sensitive examples was justified
- a `40`-frame sequence window was tested because some sign distinctions appear to depend on motion completion and longer temporal context

Files changed:
- `scripts/train_model_300.py`
- `docs/PROJECT_CHANGELOG_AND_COMPONENTS.md`

Tools and libraries involved:
- PyTorch for confusion evaluation, weighted sampling, and retraining
- NumPy for raw landmark-cache inspection and clip statistics
- existing MediaPipe cache in `data/raw/data/mp`

Measured effect:
- baseline pose model `models/asl_model_300_bilstm512_pose_v1.pt` kept the stronger overall held-out test Top-1 at `63.03%`
- the new targeted `seq40` run reached `68.10%` validation Top-1 and `88.47%` validation Top-5
- the new targeted `seq40` run reached `61.82%` test Top-1 and `85.80%` test Top-5
- targeted class behavior improved on the held-out test set:
- `MOTHER` recall improved from `0.333` to `0.667`
- `TALL` recall improved from `0.667` to `1.000`
- `FATHER` remained at `1.000` recall
- `THEORY` remained at `1.000` recall
- the trade-off is that the targeted run improved the specific weak classes and Top-5 coverage, but did not beat the original pose model on overall test Top-1
- this means the targeted oversampling idea is useful for class repair, but it should be tuned more carefully before replacing the current best general-purpose checkpoint

### 2026-04-15: Realtime pose-model deployment and demo stabilization update

What was done:
- updated the realtime WLASL300 bridge to load the finalized `models/asl_model_300_bilstm512_pose_v1.pt` checkpoint by default for the production demo path
- made `scripts/inference_bridge.py` rebuild the BiLSTM using checkpoint metadata so the runtime now matches the finalized `hidden_dim=512`, `num_layers=2`, `dropout=0.5`, and `input_dim=147` configuration
- aligned webcam preprocessing with the pose-enhanced training pipeline by extracting the same compact upper-body pose joints and normalizing them around the shoulder center before appending them to the 126 hand features
- added feature-dimension and forward-pass readiness tracking so the UI can verify that valid `147`-dimensional vectors are actually reaching the BiLSTM
- updated the realtime inference loop in `src/main.py` with a 10-frame rolling prediction buffer, a 6-vote majority requirement, and a `0.65` confidence squelch to suppress flicker and idle-time ghost outputs
- added TTS hysteresis so the speech module only fires when the stabilized prediction changes to a different valid sign
- upgraded the OpenCV overlay to show Top-3 confidence bars and a `Model Ready` status light tied to successful pose-model inference

Why it was done:
- the strongest finalized model is now the pose-enhanced `147`-dimensional BiLSTM, so the webcam demo needed to stop using the older hand-only assumptions
- training/inference parity matters more after moving to pose features because a wrong joint selection or normalization center would silently break deployment quality
- the raw per-frame predictions were too unstable for a professional demo and could cause distracting flicker or repeated TTS output while the user was idle or transitioning between signs
- the new readiness indicator makes it easier to debug the live demo and clearly show that the deployed path is using the correct production model

Files changed:
- `scripts/inference_bridge.py`
- `src/main.py`
- `src/modules/ui.py`
- `docs/PROJECT_CHANGELOG_AND_COMPONENTS.md`

Tools and libraries involved:
- PyTorch for checkpoint loading and realtime model inference
- NumPy for feature fusion and prediction-window bookkeeping
- OpenCV for the updated overlay rendering
- MediaPipe for live hand and pose landmark extraction

Expected effect:
- correct deployment of the finalized pose-enhanced `asl_model_300_bilstm512_pose_v1` model in the webcam system
- stronger training/runtime preprocessing parity for the `147`-feature inference path
- more stable and presentation-ready predictions during live use
- reduced repeated speech output and better visual confidence signaling during the demo

### 2026-04-08: Training and experiment tracking update

What was done:
- reviewed the active WLASL300 pipeline instead of relying on the older `scripts/train_model.py` baseline
- compared saved experiment histories to identify which training settings helped most
- updated `scripts/train_model_300.py` to support configurable `sequence_length`, `hidden_dim`, and `dropout`
- added optional focal loss through `--focal-gamma` to make it easier to test long-tail-focused training without rewriting the trainer again
- updated checkpoint metadata so trained models now record `sequence_length`, `hidden_dim`, `dropout`, and `focal_gamma`
- added automatic best-checkpoint test evaluation after training
- added automatic experiment report generation as `<output_prefix>_report.json`
- updated `scripts/inference_bridge.py` so realtime inference can read `sequence_length`, `hidden_dim`, and `dropout` directly from the saved checkpoint

Why it was done:
- the project already has multiple training variants, but experiment details were spread across filenames and history JSON files
- future sequence-length experiments would have been error-prone because the webcam inference path was effectively hard-coded to `30`
- adding run reports and richer checkpoint metadata makes the training process easier to reproduce and easier to explain in the report or viva
- focal loss gives a controlled way to test whether harder minority classes are dragging down Top-1

Files changed:
- `scripts/train_model_300.py`
- `scripts/inference_bridge.py`
- `docs/PROJECT_CHANGELOG_AND_COMPONENTS.md`

Tools and libraries involved:
- PyTorch for model training and loss functions
- NumPy for sequence handling
- existing WLASL metadata and MediaPipe cache already present in the repo

Expected effect:
- better experiment traceability
- safer training/inference alignment for future hyperparameter sweeps
- easier testing of long-tail optimization ideas

### 2026-04-08: Experiment matrix runner added

What was done:
- added `scripts/run_experiment_matrix.py`
- encoded the four highest-priority WLASL300 experiments into a repeatable matrix
- made the matrix runnable in sequence or previewable with `--dry-run`
- aligned the experiment names with output prefixes so resulting checkpoints and reports stay easy to compare

Why it was done:
- the next training step should be systematic, not ad hoc
- running experiments in a fixed order reduces accidental configuration drift
- it also makes it easier to discuss the improvement plan in documentation and presentations

Files changed:
- `scripts/run_experiment_matrix.py`
- `docs/PROJECT_CHANGELOG_AND_COMPONENTS.md`

Tools and libraries involved:
- Python standard library only: `argparse`, `subprocess`, `sys`, `pathlib`

Expected effect:
- faster and cleaner experiment execution
- easier comparison between baseline, pose, sequence-length, and focal-loss runs

### 2026-04-08: Confusion analysis tooling added

What was done:
- added `scripts/analyze_confusion.py`
- wired it to the real `WLASLMPDataset` test split from `wlasl_v0.3.json`
- made it load `models/asl_model_300_best.pt` by default
- made it generate a full per-class classification report
- made it identify the top confused sign pairs
- made it save a top-50 confusion-matrix heatmap to `reports/confusion_matrix.png`
- made it save an actionable weak-class summary so follow-up augmentation or pose experiments can focus on the worst signs
- kept plotting on `matplotlib` but removed the hard dependency on `scikit-learn` by implementing the core confusion metrics directly in the script

Why it was done:
- raw Top-1 and Top-5 scores do not explain where the model is failing
- confusion analysis is the fastest way to tell whether the next gains should come from pose features, better augmentation, or class-specific data cleanup
- this also makes the project easier to justify academically because it connects model behavior to concrete failure modes

Files changed:
- `scripts/analyze_confusion.py`
- `requirements.txt`
- `docs/PROJECT_CHANGELOG_AND_COMPONENTS.md`

Tools and libraries involved:
- PyTorch for checkpoint loading and inference
- matplotlib for heatmap generation

Expected effect:
- clearer diagnosis of misclassified signs
- faster identification of classes that need targeted improvement

### 2026-04-08: Data-driven optimization update

What was done:
- upgraded `scripts/analyze_confusion.py` to classify confused pairs into static-shape or temporal confusion
- added `reports/actionable_fixes.json` output for the top confused pairs
- moved stronger augmentation logic into `scripts/prepare_data.py`
- added hand-specific dropout, gaussian xy jitter, and finger-bone scaling
- upgraded `scripts/train_model_300.py` with motion-delta feature fusion and explicit configurable multi-head attention
- added OneCycleLR support for faster convergence experiments
- added `scripts/verify_preprocessing_parity.py` to verify WLASL300 runtime preprocessing parity with training

Why it was done:
- the project needs targeted, error-driven optimization rather than generic reruns
- the confusion results now feed directly into augmentation and architecture decisions
- parity validation reduces the risk of hidden preprocessing drift between training and inference

Files changed:
- `scripts/analyze_confusion.py`
- `scripts/prepare_data.py`
- `scripts/train_model_300.py`
- `scripts/inference_bridge.py`
- `scripts/verify_preprocessing_parity.py`
- `docs/PROJECT_CHANGELOG_AND_COMPONENTS.md`

Expected effect:
- stronger discriminative learning for confusing gloss pairs
- more realistic robustness to signer and webcam variation
- clearer validation of training versus deployment consistency

### 2026-04-09: High-capacity BiLSTM plateau run

What was done:
- updated the active WLASL300 training defaults to a higher-capacity BiLSTM configuration
- set `hidden_dim=512`
- set `num_layers=2`
- kept standard single-head attention with an internal hidden layer of `256`
- increased dropout to `0.5`
- used `ReduceLROnPlateau` with a starting learning rate of `1e-3`
- trained a full run as `models/asl_model_300_bilstm512_plateau_v1`

Why it was done:
- the previous baseline was close to the `60%` validation Top-1 barrier but not crossing it reliably
- a larger BiLSTM with stronger regularization was a controlled way to test whether representation capacity was the immediate bottleneck
- keeping standard attention made the comparison cleaner than changing several architectural ideas at once

Files changed:
- `scripts/train_model_300.py`
- `docs/PROJECT_CHANGELOG_AND_COMPONENTS.md`

Tools and libraries involved:
- PyTorch for BiLSTM training, attention pooling, and `ReduceLROnPlateau`
- existing MediaPipe cache and WLASL300 metadata already in the repo

Measured effect:
- validation Top-1 improved to `60.67%`
- test Top-1 improved to `57.50%`
- test Top-5 reached `83.57%`
- this configuration successfully crossed the `60%` validation barrier and improved held-out test accuracy versus the earlier rerun baseline

### 2026-04-09: Pose-enhanced high-capacity run

What was done:
- launched a pose-enhanced variant of the successful `512`-hidden-dimension, `2`-layer BiLSTM
- kept standard single-head attention with the `256`-unit attention MLP
- kept `dropout=0.5`
- kept `ReduceLROnPlateau` and `learning_rate=1e-3`
- enabled `--use-pose` so the dataset appended normalized upper-body pose features to the hand landmarks
- verified the expanded feature input dimension was handled correctly during training
- updated `scripts/analyze_confusion.py` so newer standard-attention checkpoints load correctly for post-run diagnosis

Why it was done:
- the hand-only `bilstm512_plateau_v1` run improved accuracy but still left a noticeable gap on confusing signs
- pose cues were the cleanest next test for signs that share hand shape but differ in arm placement, body anchor, or upper-body context
- the confusion script fix was required so the newer checkpoints could be compared fairly against earlier runs

Files changed:
- `scripts/analyze_confusion.py`
- `docs/PROJECT_CHANGELOG_AND_COMPONENTS.md`

Tools and libraries involved:
- PyTorch for training and checkpoint loading
- existing MediaPipe pose cache in `data/raw/data/mp`
- matplotlib-based confusion analysis already present in the repo

Measured effect:
- validation Top-1 improved to `68.00%`
- test Top-1 improved to `63.03%`
- test Top-5 improved to `84.59%`
- the pose run outperformed the hand-only `bilstm512_plateau_v1` test Top-1 by `5.53` percentage points
- several previously weak classes improved meaningfully, including `WHO`, `YES`, `DOCTOR`, `DAUGHTER`, `FATHER`, `MEDICINE`, `BIRTHDAY`, and `CORN`
- this is strong evidence that pose features helped reduce at least part of the hand-shape-driven ambiguity, even though some weak classes like `BACKPACK` and `CHECK` still need targeted cleanup

## Accuracy improvement suggestions

These suggestions are prioritized based on the codebase and the saved histories already in `models/`.

### 1. Keep balanced sampling and augmentation as the baseline

Why:
- the saved history `models/asl_model_300_poseattn_balaug_history.json` clearly outperforms the lower-learning-rate variant and edges out the weighted-loss variant on validation Top-1
- this suggests class exposure and moderate corruption-based robustness are already helping more than simply down-weighting common classes in the loss

Recommended command direction:

```bash
python scripts/train_model_300.py --source mp-cache --metadata data/raw/wlasl_v0.3.json --mp-root data/raw/data/mp --device cuda --epochs 30 --batch-size 32 --class-balanced --augment --output-prefix models/asl_model_300_balaug_v2
```

### 2. Test pose features systematically, not casually

Why:
- the code already supports `--use-pose`
- many confusing ASL pairs are not separable from hand shape alone; shoulder, elbow, and motion path cues often matter
- if pose helps, it is more likely to improve Top-1 than Top-5 because it helps resolve near-miss classes

How this project uses pose:
- pose is extracted in `src/modules/keypoint_extraction.py`
- compact pose joints are selected and normalized in `scripts/prepare_data.py`
- pose is appended after the 126 hand features inside `WLASLMPDataset`

Recommended experiment:

```bash
python scripts/train_model_300.py --source mp-cache --metadata data/raw/wlasl_v0.3.json --mp-root data/raw/data/mp --device cuda --epochs 30 --batch-size 32 --class-balanced --augment --use-pose --output-prefix models/asl_model_300_pose_balaug_v2
```

### 3. Sweep sequence length next

Why:
- the current pipeline always compresses to `30` frames
- some signs are probably being over-compressed, especially when motion path matters
- now that checkpoints store sequence length and the inference bridge reads it back, this experiment is safer to run

Recommended sweep:
- `30`
- `36`
- `40`

Expectation:
- longer windows may improve Top-5 first, then Top-1 if the extra temporal detail helps disambiguation instead of adding noise

### 4. Use focal loss as a targeted experiment, not a default assumption

Why:
- weighted loss alone did not beat the best balanced-augmentation run on validation Top-1
- focal loss is worth testing because it emphasizes hard examples differently from class weighting
- it is especially relevant if confusion is concentrated in many rare or visually similar classes

Recommended sweep:
- `--focal-gamma 1.0`
- `--focal-gamma 1.5`
- `--focal-gamma 2.0`

Suggested starting command:

```bash
python scripts/train_model_300.py --source mp-cache --metadata data/raw/wlasl_v0.3.json --mp-root data/raw/data/mp --device cuda --epochs 30 --batch-size 32 --class-balanced --augment --focal-gamma 1.5 --output-prefix models/asl_model_300_balaug_focal15
```

### 5. Add confusion analysis before changing the architecture

Why:
- the project may be losing Top-1 mostly on a relatively small subset of confusing gloss pairs
- if so, data cleanup or pose usage may help more than replacing the model

Recommended analysis:
- collect the most frequent Top-1 mistakes on the held-out test set
- identify whether those mistakes are same-handshape / different-motion signs
- identify whether the failing classes have few training examples or many noisy samples

### 6. Improve data quality before attempting a heavier model

Why:
- the training histories already show the model can fit the train split quite well
- the bigger issue is the train-to-val/test gap
- that usually means the next gains come from feature quality, split quality, missing-frame handling, and signer robustness more than raw parameter count

Specific data-side ideas for this repo:
- inspect classes with many zero-heavy frames after MediaPipe extraction
- inspect samples where one hand is missing for most of the clip
- compare per-class counts in the official WLASL train split
- test stricter sequence cleaning for zero-heavy clips before resampling

## Detailed component reference

### `src/main.py`

What it is:
- the main runtime entry point

How it is used:
- starts webcam capture
- runs MediaPipe-based keypoint extraction
- buffers sequence features
- calls the recognition model
- updates the UI
- triggers TTS when confidence is high enough

How this project uses it:
- as the end-to-end live demo path for proposal alignment and user-facing testing

### `src/modules/video_capture.py`

What it is:
- a threaded OpenCV webcam wrapper

How it is used:
- keeps frame acquisition separate from the main loop so capture does not block inference

How this project uses it:
- to maintain smoother realtime behavior on consumer hardware

### `src/modules/keypoint_extraction.py`

What it is:
- MediaPipe-based landmark extraction for both hands, a compact face mesh slice, and upper-body pose

How it is used:
- each frame is converted from BGR to RGB
- MediaPipe Hands extracts up to two hands
- MediaPipe Face Mesh can provide anchor-sensitive facial landmarks
- MediaPipe Pose extracts 33 pose landmarks
- results are returned as NumPy arrays

How this project uses it:
- hand landmarks are the main deployed feature source
- pose and compact face features can be folded into the WLASL300 training and realtime path

### `scripts/prepare_data.py`

What it is:
- the shared feature-engineering layer for WLASL300 training and realtime inference

How it is used:
- normalizes gloss labels
- normalizes hands around the wrist
- optionally normalizes selected pose joints around the shoulder center
- optionally normalizes selected face landmarks around the eye midpoint
- trims invalid sequence edges
- interpolates short gaps
- resamples all sequences to a fixed frame length

How this project uses it:
- this file is one of the most important consistency points in the repo because it defines the exact training distribution the live system tries to match

### `scripts/train_model_300.py`

What it is:
- the main WLASL300 training script

How it is used:
- loads one of several dataset backends
- creates the BiLSTM + attention model
- trains with Top-1 and Top-5 tracking
- supports class-balanced sampling, augmentation, weighted loss, warmup, optional pose features, optional face features, and optional focal loss
- saves checkpoints, history, and experiment reports

How this project uses it:
- this is the real model-development path tied to the reported WLASL300 results

### `scripts/inference_bridge.py`

What it is:
- the bridge between saved WLASL300 checkpoints and the webcam runtime

How it is used:
- loads label maps and checkpoint metadata
- rebuilds the model with the right input dimension
- applies the same preprocessing logic used in training
- returns Top-5 predictions for the live UI using the same saved hand/pose/face feature settings as training

How this project uses it:
- as the production-oriented inference layer for the stronger 300-sign path

### `src/modules/sequence_model.py`

What it is:
- the original general-purpose sequence model module

How it is used:
- contains the earlier BiLSTM and lightweight transformer abstractions

How this project uses it:
- mainly as the baseline / legacy path for the generic realtime system
- the active WLASL300 path currently uses the model defined in `scripts/train_model_300.py`

### `src/modules/ui.py`

What it is:
- an OpenCV-based visual overlay layer

How it is used:
- draws the live frame
- overlays predicted signs, confidence, top-k guesses, FPS, and system status

How this project uses it:
- to make the realtime demo understandable during testing, reporting, and presentation

### `src/modules/text_to_speech.py`

What it is:
- a background-thread TTS wrapper

How it is used:
- supports `pyttsx3` for offline speech
- optionally supports `gTTS`, though that needs internet access

How this project uses it:
- to convert recognized signs into spoken output for accessibility and demonstration

### `src/utils/config.py`

What it is:
- central project configuration

How it is used:
- stores capture settings, model defaults, directory paths, and runtime toggles

How this project uses it:
- to keep the demo path configurable without scattering constants across the codebase

### `data/raw/wlasl_v0.3.json`

What it is:
- the master WLASL metadata file

How it is used:
- provides gloss names, sample instances, and official splits

How this project uses it:
- it is the authority used to build the top-300 label map and train/val/test sample lists

### `data/raw/label_map_300.json` and `data/raw/label_to_index_300.json`

What they are:
- the forward and reverse mappings between class index and gloss

How they are used:
- training uses them for labels
- inference uses them to convert predicted indices back into readable glosses
- custom regeneration can now force-include specific low-frequency glosses if required

How this project uses them:
- as the vocabulary contract across training, evaluation, and live inference

### `data/raw/data/mp`

What it is:
- the local MediaPipe landmark cache used by the strongest training path

How it is used:
- each video id has cached left-hand, right-hand, and often pose and face pickle files

How this project uses it:
- this cache removes the need to re-run MediaPipe extraction during every training run
- it is the practical backbone of the current WLASL300 experiments

### `models/`

What it is:
- the folder containing saved checkpoints and history files

How it is used:
- each run leaves behind a `.pt` checkpoint and usually a `_history.json`

How this project uses it:
- the saved histories are currently the clearest evidence for which training ideas helped and which ones did not

## Important observations about the current system

- The active deployed representation can now be face-aware, but that path still needs retraining and validation before it becomes the strongest default model.
- `I` and `ME` exist in the full WLASL metadata but are not part of the default top-300 frequency-selected vocabulary.
- The best validation result and the reported test result are meaningfully different, so the next gains should focus on generalization.
- Balanced sampling plus augmentation already looks like the right baseline direction.
- Weighted loss alone is not enough evidence that the class-imbalance problem is solved.

## Recommended next experiment order

1. `class-balanced + augment` baseline rerun with the new experiment report output.
2. Same run plus `--use-pose`.
3. Sequence-length sweep at `30`, `36`, and `40`.
4. Best of the above plus `--focal-gamma 1.5`.
5. Confusion-matrix review before any architecture replacement.

## Ready-to-run experiment matrix

The project now includes a small runner at `scripts/run_experiment_matrix.py`.

Preview the commands first:

```bash
python scripts/run_experiment_matrix.py --dry-run
```

Run the full prioritized matrix:

```bash
python scripts/run_experiment_matrix.py --device cuda --epochs 30 --batch-size 32
```

The current matrix executes these runs in order:

1. `baseline_balaug`
   Goal:
   refresh the strongest balanced-sampling baseline with the new report output

2. `pose_balaug`
   Goal:
   test whether upper-body cues improve Top-1 by reducing near-miss classes

3. `seq36_balaug`
   Goal:
   test whether a slightly longer temporal window preserves more discriminative motion

4. `focal15_balaug`
   Goal:
   test whether focal loss improves hard-class learning beyond balanced sampling

Suggested success criteria:
- keep any run that improves validation Top-1 without hurting test Top-5 materially
- prioritize runs that also narrow the validation-to-test gap
- if `pose_balaug` wins, treat pose as a serious production candidate
- if `seq36_balaug` wins, continue with `40` next
- if `focal15_balaug` loses clearly, stop the focal-loss branch early
