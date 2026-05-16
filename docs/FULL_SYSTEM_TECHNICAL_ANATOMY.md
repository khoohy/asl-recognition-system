# Full-System Technical Anatomy

Project: Real-time ASL Recognition Pipeline  
Purpose: Internal deep-dive for understanding the active production-oriented system.  
Scope: Only the current WLASL300 webcam pipeline and the training path that supports it.

## Ground truth before reading

- The promoted production checkpoint now lives at `models/production/asl_wlasl300_realtime.pt`.
- The older filename `models/asl_model_300_pose_face_balaug_hardened_v1.pt` was migrated during the repo refactor and now resolves through the production path instead of staying at the models root.
- That checkpoint is still a `30 x 180` model.
- The hardened run has now reached `65.01%` Test Top-1 and `87.59%` Test Top-5.
- The production checkpoint is now the default live webcam model path in both `src/main.py` and `scripts/evaluation/inference_bridge.py`.
- The runtime still keeps motion as diagnostic context, but it no longer blanks a visible-hand paused sign just because motion is temporarily low.

## 1. The 180D Multi-Modal Feature Vector

- Each frame is encoded as `180` scalar features.
  - Hands: indices `0-125`
  - Pose: indices `126-146`
  - Face: indices `147-179`

- Hand block
  - Layout:
    - left hand: `21 x 3 = 63` values at `0-62`
    - right hand: `21 x 3 = 63` values at `63-125`
  - Source: MediaPipe Hands
  - Normalization:
    - each hand is centered on its wrist
    - formula: `p'_i = p_i - p_wrist`
    - each coordinate axis is then min-max scaled to `[-1, 1]`
  - Why:
    - removes camera translation
    - preserves relative finger geometry

- Pose block
  - Selected joints: `0, 11, 12, 13, 14, 15, 16`
  - Width: `7 x 3 = 21`
  - Normalization:
    - center at shoulder midpoint
    - scale by shoulder distance
    - formula:
      - `c_pose = (p_left_shoulder + p_right_shoulder) / 2`
      - `s_pose = ||p_left_shoulder_xy - p_right_shoulder_xy||`
      - `p'_i = (p_i - c_pose) / max(s_pose, epsilon)`
    - values clipped to `[-2, 2]`
  - Why:
    - preserves arm path and body anchor cues across signer position changes

- Face block
  - Selected landmarks: `10, 151, 168, 1, 2, 13, 14, 17, 152, 33, 263`
  - Width: `11 x 3 = 33`
  - Normalization:
    - center at the midpoint of landmarks `33` and `263`
    - scale by the distance between those outer-eye points
    - formula:
      - `c_face = (p_left_eye + p_right_eye) / 2`
      - `s_face = ||p_left_eye_xy - p_right_eye_xy||`
      - `p'_i = (p_i - c_face) / max(s_face, epsilon)`
    - values clipped to `[-2, 2]`
  - Why:
    - keeps facial anchor geometry stable under head movement

- Why face features matter for `MOTHER` vs `FATHER`
  - Hand shape alone is not the hard part.
  - The real distinction is hand placement against the face.
    - `MOTHER`: lower-face / chin anchor
    - `FATHER`: upper-face / forehead anchor
  - The compact face slice gives the model a signer-relative map for hand-to-face contact location.

## 2. Active Data Engineering Engine: `scripts/prepare_data.py`

- This file is the canonical preprocessing contract for the WLASL300 path.
  - gloss normalization
  - hand/pose/face normalization
  - missing-frame cleanup
  - fixed-length temporal resampling
  - augmentation for robustness

- Temporal cleanup
  - infer valid frames from nonzero feature count
  - trim invalid leading and trailing edges
  - linearly interpolate only short interior gaps
  - default short-gap fill limit: `2` frames

- Linear interpolation
  - implementation uses `numpy.interp`
  - every feature dimension is interpolated independently between valid boundary frames
  - purpose:
    - absorb brief tracker dropouts
    - avoid breaking trajectories because of 1-2 weak frames

- Dynamic resampling
  - implementation also uses `numpy.interp`
  - variable-length sequences are mapped onto a fixed temporal grid
  - current deployed face-aware checkpoint remains `30` frames
  - the codebase still supports alternative lengths for retraining experiments

- Label mapping
  - authority source: `data/raw/wlasl_v0.3.json`
  - deployed vocabulary contract: `data/raw/label_map_300.json`
  - forced inclusion support exists for low-frequency glosses such as `I` and `ME`
  - logic:
    - build the top-300 by metadata frequency
    - inject requested glosses if they fall outside the cutoff
    - displace lower-ranked defaults when necessary

- Stress-test augmentation now used in training
  - Coordinate jitter
    - stronger x/y noise than the earlier pass
    - extra pose/face channels get smaller companion jitter
  - Per-frame hand occlusion
    - `10%` of frames now have the entire `126`-dimensional hand block zeroed
    - this forces fallback to pose and face anchors
  - One-hand dropout
    - either left-hand or right-hand stream can still be zeroed across the sequence
    - this is now controlled separately from per-frame full-hand occlusion
  - Finger-bone scaling
    - finger geometry is stretched more aggressively to mimic signer anatomy variation
    - current scale range is `0.9-1.1`
  - Temporal corruption
    - some frames are zeroed
    - some frames are skipped before final resampling
    - this simulates lag, tracking flicker, and frame-rate instability

## 3. The Neural Architecture: `scripts/train_model_300.py`

- Backbone
  - 2-layer bidirectional LSTM
  - hidden size: `512`
  - dropout: `0.5`
  - input width for face-aware mode: `180`

- Why BiLSTM
  - ASL signs are temporal trajectories, not static poses.
  - A middle frame may be ambiguous until the movement destination is known.
  - Bidirectionality makes each frame representation aware of both prefix and suffix context.

- Attention pooling
  - standard learned soft attention, not transformer self-attention
  - pipeline:
    - `Linear(1024 -> 256)`
    - `Tanh`
    - `Dropout`
    - `Linear(256 -> 1)`
    - `softmax` over time
  - math:
    - `e_t = w2(tanh(W1 h_t))`
    - `a_t = softmax(e_t)`
    - `h_pool = sum_t (a_t * h_t)`
  - why:
    - upweights discriminative sub-motions
    - downweights dead air and transition frames

- Classification head
  - `Linear(1024 -> 512)`
  - `ReLU`
  - `Dropout`
  - `Linear(512 -> 300)`

- Class-balanced sampling
  - uses `WeightedRandomSampler`
  - sample weights are inverse-frequency by class
  - effect:
    - rare classes appear more often
    - common classes stop dominating the gradient stream

- Targeted boosting
  - `--boost-glosses` and `--boost-factor`
  - if a sample belongs to a target gloss, its sampler weight is multiplied again
  - this is the class-repair path for pairs like `MOTHER/FATHER` and `TALL/THEORY`

- Training-hardening protocol now in code
  - default epochs: `50`
  - default learning rate: `1e-3`
  - default scheduler: `ReduceLROnPlateau`
  - learning-rate floor: `1e-5`
  - warmup remains active for early epochs
  - batch size default remains `32`
  - DataLoader workers default remains `0`
  - CUDA AMP with gradient scaling is now supported by default unless `--no-amp` is used
  - CUDA DataLoaders now use pinned host memory automatically when training on GPU
  - Measured result of the first full hardened run:
    - validation Top-1: `72.31%`
    - validation Top-5: `89.33%`
    - test Top-1: `65.01%`
    - test Top-5: `87.59%`

## 4. Real-time Inference And Stability

- `scripts/inference_bridge.py`
  - loads checkpoint metadata
  - reconstructs the correct feature width and sequence length
  - applies the same shared preprocessing helpers used in training
  - returns Top-5 predictions for the UI stabilization layer

- `src/main.py`
  - orchestrates webcam capture, keypoint extraction, buffering, stabilization, UI, and TTS

- Sequence buffer
  - fixed-length rolling temporal buffer
  - runtime uses checkpoint metadata instead of guessing sequence length

- Temporal grace period
  - `HAND_MISSING_GRACE_FRAMES = 10`
  - logic:
    - hands visible: reset grace counter and continue
    - hands briefly missing: keep temporal context alive
    - hands missing too long: clear buffer and prediction state
  - why:
    - prevents short detector flickers from destroying BiLSTM context

- Stabilization layer
  - base confidence squelch: `0.65`
  - adaptive acceptance floor: `0.45`
  - adaptive runner-up margin: `0.12`
  - stabilization window: `10`
  - minimum vote count: `6`
  - default visible guess list: Top `5`
  - effect:
    - suppresses frame-level flicker
    - keeps UI and TTS tied to stable signs rather than noisy logits
    - allows a few known weaker-but-correct live signs to pass at lower confidence when the margin is still clear

- Peak-sign fallback
  - problem:
    - some signs hit the correct class only at the most expressive frame, then decay into a confused end pose before the standard vote window fills
  - implementation:
    - a short separate peak history stores only sign predictions that cross stricter sign-specific thresholds and a minimum rival margin
    - if the regular vote history is not yet stable, the UI can still surface a peak-stabilized sign when it appears at least `2` times inside the `5`-frame peak window
  - effect:
    - short-lived but strong sign peaks such as `ARRIVE`, `CATCH`, `HOPE`, `JACKET`, and `LAW` can remain visible instead of collapsing immediately to `...`

- Confusion-aware gating
  - some live errors are near-ties between known rival signs rather than total recognition failures
  - `src/utils/config.py` now carries explicit confusion-pair suppressors
  - if a rival sign stays too close to the winner, the pipeline holds output instead of committing too early
  - this especially matters for signs that repeatedly trade places during live testing even when the correct class is already inside Top-5

- Motion metric
  - mean absolute delta over consecutive hand features
  - originally introduced to diagnose idle hallucinations such as false `LATE`
  - current role:
    - diagnostic/status context
    - sign-specific motion gate for a small list of classes that should not fire while hands are nearly static
    - not a global hard off-switch for visible-hand pauses

- Training/runtime parity
  - strongest guarantee comes from shared use of `WLASLFeatureEngineering`
  - `scripts/verify_preprocessing_parity.py` currently validates the shared hand path exactly
  - it is still narrower than a full end-to-end 180D parity proof

## 5. Reliability, UI, And Deployment Strategy

- TTS: `src/modules/text_to_speech.py`
  - prefers native Windows SAPI when available
  - backend is initialized and used inside the worker thread
  - silence-state hysteresis prevents repeated speech loops
  - effect:
    - repeated utterances stay reliable across a session

- UI: `src/modules/ui.py` plus `src/main.py`
  - shows:
    - stabilized sign text
    - Top-5 predictions by default
    - buffering state
    - model readiness
    - held-context status during grace windows
  - effect:
    - makes the live system debuggable during real webcam use

- Storage optimization
  - repo was reduced from about `60 GB` to about `17 GB`
  - retained assets:
    - `data/raw/data/mp`
    - `data/raw/wlasl_v0.3.json`
    - `data/raw/label_map_300.json`
  - why:
    - the MP cache is the reusable substrate for retraining
    - raw video is no longer required once landmarks are cached

- Generalization gap
  - earlier face-aware checkpoint:
    - validation Top-1: `67.46%`
    - validation Top-5: `88.90%`
    - test Top-1: `60.03%`
    - test Top-5: `85.80%`
  - hardened face-aware checkpoint:
    - validation Top-1: `72.31%`
    - validation Top-5: `89.33%`
    - test Top-1: `65.01%`
    - test Top-5: `87.59%`
  - interpretation:
    - the system still has a signer-independence and domain-shift problem
    - it fits validation better than it transfers to unseen test signers
    - the hardened run improved both validation and test performance, but it did not eliminate the gap
  - mitigation direction:
    - class-balanced sampling
    - stronger augmentation
    - frame-level hand occlusion
    - one-hand dropout
    - temporal skipping and jitter
    - richer pose and face anchors
  - bottom line:
    - the main remaining problem is deployment realism, not just missing depth in the network

- Hardware-stability guardrails
  - batch size stays at `32`
  - DataLoader workers default to `0`
  - AMP is enabled by default on CUDA through `--amp`
  - `--no-amp` remains available for debugging
  - purpose:
    - reduce VRAM pressure
    - lower the chance of watchdog-style instability on laptop GPUs

## Production-only file map

- `scripts/prepare_data.py`
  - source of truth for WLASL300 feature engineering and augmentation

- `scripts/train_model_300.py`
  - source of truth for the active BiLSTM-attention training path

- `scripts/inference_bridge.py`
  - runtime adapter from checkpoint metadata to live webcam inference

- `src/main.py`
  - orchestration, stabilization, grace-period logic, and TTS triggering

- `src/modules/keypoint_extraction.py`
  - MediaPipe extraction of hands, compact face context, and upper-body pose

- `src/modules/text_to_speech.py`
  - native/offline speech backend wrapper

- `src/modules/ui.py`
  - live overlay and runtime status presentation

- `data/raw/wlasl_v0.3.json`
  - vocabulary and split authority

- `data/raw/label_map_300.json`
  - deployed class-index contract

- `data/raw/data/mp`
  - cached landmark corpus for practical retraining

## Bottom line

- The active deployed checkpoint is still a `30 x 180` temporal model.
- The hardened face-aware checkpoint is now the shared default live model path.
- The codebase now contains a harder robustness-oriented training path designed to narrow the validation-to-test gap.
- The key engineering bets are:
  - face anchors for facial-location signs
  - stronger occlusion and temporal-corruption augmentation
  - class-balanced exposure with targeted repair
  - adaptive, confusion-aware, and peak-aware live stabilization
  - grace-period buffering for live stability
  - conservative GPU guardrails for retraining on laptop hardware
