# Full-System Technical Anatomy

Project: Real-time ASL Recognition Pipeline  
Purpose: Internal deep-dive for understanding the active production-oriented system, not a viva script.  
Scope: Only the current WLASL300 webcam pipeline built around the face-aware 180D checkpoint path.

## Ground truth before reading

- The active face-aware production checkpoint is a `180`-dimensional model with a fixed `30`-frame sequence length, not `40`.
- The current live runtime keeps a motion metric for diagnostics, but it no longer hard-disables predictions just because the signer pauses with hands still visible.
- The repository still contains legacy files, but this document ignores them unless they directly support the current deployed path.

## 1. The 180D Multi-Modal Feature Vector: The input contract

- The runtime and trainer both build one frame-level vector with exactly `180` scalar values.
  - Hands occupy indices `0-125`.
  - Pose occupies indices `126-146`.
  - Face occupies indices `147-179`.

- Hand block: indices `0-125`
  - Source: `42 x 3` coordinates.
  - Layout:
    - left hand: `21 x 3 = 63` values at `0-62`
    - right hand: `21 x 3 = 63` values at `63-125`
  - Raw source comes from MediaPipe Hands.
  - Each hand is normalized independently, so left-hand translation does not corrupt right-hand geometry and vice versa.

- Pose block: indices `126-146`
  - Source joints: `0, 11, 12, 13, 14, 15, 16`
  - Count: `7 x 3 = 21` values.
  - Semantics:
    - `0`: nose
    - `11, 12`: shoulders
    - `13, 14`: elbows
    - `15, 16`: wrists
  - This compact slice gives upper-body anchor and arm-path context without paying the cost of all `33` pose landmarks.

- Face block: indices `147-179`
  - Source landmarks: `10, 151, 168, 1, 2, 13, 14, 17, 152, 33, 263`
  - Count: `11 x 3 = 33` values.
  - These points give a compact facial anchor map centered around the forehead, nose, mouth/chin region, and outer-eye anchors.
  - The last two selected landmarks, `33` and `263`, are used as the eye pair for normalization.

- Hand normalization: wrist-centric plus per-hand scaling
  - For each hand independently, the wrist landmark at index `0` is treated as the local origin.
  - Math:
    - `p'_i = p_i - p_wrist`
  - After translation, each coordinate axis is min-max scaled within that hand to `[-1, 1]`.
  - Effect:
    - removes absolute screen position
    - keeps finger shape and relative articulation
    - makes the model care more about configuration than camera placement

- Pose normalization: mid-shoulder centering plus shoulder-distance scaling
  - After selecting the `7` compact joints, the shoulder center is computed from the selected-slice indices `1` and `2`.
  - Math:
    - `c_pose = (p_left_shoulder + p_right_shoulder) / 2`
    - `s_pose = ||p_left_shoulder_xy - p_right_shoulder_xy||`
    - `p'_i = (p_i - c_pose) / max(s_pose, epsilon)`
  - Values are clipped to `[-2, 2]`.
  - Effect:
    - body anchor becomes signer-relative instead of camera-relative
    - arm path and hand-to-body placement stay meaningful across different distances from the webcam

- Face normalization: eye-midpoint centering plus eye-distance scaling
  - After selecting the `11` compact face points, the outer-eye midpoint is used as the face origin.
  - Math:
    - `c_face = (p_left_eye + p_right_eye) / 2`
    - `s_face = ||p_left_eye_xy - p_right_eye_xy||`
    - `p'_i = (p_i - c_face) / max(s_face, epsilon)`
  - Values are clipped to `[-2, 2]`.
  - If the eye distance collapses, the code falls back to a nose-centered local scale estimate.
  - Effect:
    - facial anchors remain spatially stable even when the head shifts in the frame

- Why the face block matters for `MOTHER` vs `FATHER`
  - Hand shape alone is not the hard part for that pair.
  - The real discriminant is facial anchor placement.
    - `MOTHER` is closer to chin/lower-face contact.
    - `FATHER` is closer to forehead/upper-face contact.
  - A hand-only or even hand-plus-pose system can miss this because the decisive difference is local to the face.
  - The `33` face dimensions give the model a signer-relative map of where the hand sits against the face geometry, which is the missing spatial cue for chin-vs-forehead separation.

## 2. Active Data Engineering Engine: `scripts/prepare_data.py`

- This file is the canonical feature-engineering contract for the WLASL300 path.
  - It defines gloss normalization.
  - It defines hand, pose, and face normalization.
  - It defines missing-frame cleanup.
  - It defines fixed-length temporal resampling.

- Temporal cleanup pipeline
  - Step 1: infer valid frames
    - Frames with too few nonzero values are treated as weak or missing.
    - Default threshold: at least `6` nonzero feature values.
  - Step 2: trim invalid edges
    - Leading and trailing dead regions are dropped.
    - Interior gaps are preserved for possible interpolation.
  - Step 3: interpolate short gaps
    - Short tracking dropouts are filled with linear interpolation.
    - Default maximum fill size: `2` frames.
    - Longer weak regions are left untouched instead of hallucinating a long motion path.

- Linear interpolation logic
  - The implementation uses `numpy.interp`, not SciPy.
  - For each short gap bounded by a valid left frame and a valid right frame:
    - every feature dimension is interpolated independently
    - positions are filled on the straight line between the two boundary values
  - Blueprint math for one feature:
    - if frame `t0` has value `x0` and frame `t1` has value `x1`
    - then missing frame `t` is filled by linear interpolation between `x0` and `x1`
  - Purpose:
    - absorb brief MediaPipe dropouts
    - avoid turning one missing frame into a broken trajectory

- Dynamic resampling logic
  - After cleanup, every sequence is resampled to a fixed temporal length.
  - The implementation again uses `numpy.interp` per feature dimension.
  - Old frame positions are mapped to evenly spaced new frame positions.
  - Blueprint math:
    - original positions: `linspace(0, T-1, T)`
    - target positions: `linspace(0, T-1, L)`
    - for each feature `f`, build `f_resampled(target_positions)` from `f(original_positions)`
  - Purpose:
    - convert variable-length videos into a fixed tensor for batching
    - preserve coarse motion trajectory while matching the BiLSTM input contract

- Temporal resolution standard in the current face-aware production model
  - The active face-aware checkpoint uses `30` frames.
  - Reason this matters:
    - the trainer, checkpoint metadata, inference bridge, and webcam runtime are aligned on `30`
    - there was a separate targeted `40`-frame pose experiment, but that is not the current deployed face-aware standard

- Label mapping
  - The system uses `data/raw/wlasl_v0.3.json` as the gloss authority and `label_map_300.json` as the deployed vocabulary contract.
  - The default vocabulary is frequency-selected top-300 WLASL glosses.
  - `prepare_data.py` now also supports forced inclusion of low-frequency glosses.
  - Logic:
    - normalize requested gloss names
    - build the top-`k` set from metadata frequency
    - if a requested gloss is outside the default cutoff, it is injected into the selected set and a lower-ranked gloss is displaced
  - Why this exists:
    - words such as `I` and `ME` exist in full WLASL
    - they do not appear in the default top-300 label map
    - forced inclusion allows a custom 300-class vocabulary without changing the rest of the training pipeline

## 3. The Neural Architecture: `scripts/train_model_300.py`

- Backbone: 2-layer stacked BiLSTM
  - Hidden size: `512`
  - Directionality: bidirectional
  - Dropout: `0.5`
  - Input width for the active face-aware model: `180`

- Why BiLSTM is appropriate for isolated-sign recognition
  - ASL signs are not static images; they are temporal trajectories.
  - Meaning often depends on where a motion started, where it ended, and how it evolved across time.
  - Bidirectionality matters because a middle frame can be ambiguous until the destination of the movement is known.
  - Blueprint intuition:
    - forward state captures how the sign has unfolded so far
    - backward state captures how the remaining suffix constrains interpretation
    - concatenating both makes each frame representation destination-aware

- Attention pooling
  - The active model uses standard learned soft attention, not transformer self-attention.
  - Pipeline:
    - each BiLSTM frame state goes through `Linear(1024 -> 256)`
    - then `Tanh`
    - then `Dropout`
    - then `Linear(256 -> 1)`
    - then `softmax` over the temporal dimension
  - Blueprint math:
    - `e_t = w2(tanh(W1 h_t))`
    - `a_t = softmax(e_t over time)`
    - `h_pool = sum_t (a_t * h_t)`
  - Purpose:
    - upweight discriminative sub-motions
    - downweight entry frames, exit frames, and dead air
    - avoid forcing the classifier to treat every frame as equally informative

- Classification head
  - The pooled temporal representation goes through:
    - `Linear(1024 -> 512)`
    - `ReLU`
    - `Dropout`
    - `Linear(512 -> 300)`
  - Output logits correspond exactly to the deployed 300-class label map.

- Class-balanced sampling
  - Training can assign each sample a weight inversely related to class frequency.
  - These weights drive a `WeightedRandomSampler`.
  - Effect:
    - rarer classes appear more often in minibatches
    - common classes stop dominating gradient updates just because they are more frequent

- Targeted boosting
  - The trainer also supports explicit gloss repair through `--boost-glosses` and `--boost-factor`.
  - Mechanism:
    - start from the sample weights already used for class balancing
    - for each sample whose gloss is in the target set, multiply its weight by the boost factor
  - Example blueprint:
    - if a sample weight is `w`
    - and the gloss is in `{mother, father, tall, theory}`
    - boosted weight becomes `2.5 * w`
  - Purpose:
    - force the sampler to revisit the highest-confusion classes more aggressively
    - repair known failure pairs without rewriting the model

- Why this matters for `MOTHER/FATHER` and `TALL/THEORY`
  - `MOTHER/FATHER` is mostly an anchor-location problem.
  - `TALL/THEORY` is more of a motion-path and temporal-disambiguation problem.
  - The architecture attacks both axes:
    - face/pose-aware features improve the geometry seen by the model
    - attention improves temporal focus
    - class-balanced sampling and targeted boosting increase learning pressure on the weak classes

## 4. Real-time Inference And Stability: `src/main.py` and `scripts/inference_bridge.py`

- Inference bridge responsibilities
  - Load checkpoint weights.
  - Recover checkpoint metadata such as:
    - `input_dim`
    - `sequence_length`
    - `pose_joints`
    - `face_landmarks`
  - Rebuild the exact feature extractor width expected by the trained model.
  - Apply the same hand, pose, and face preprocessing contract used in training.
  - Return Top-5 live predictions for UI stabilization.

- Sequence buffer
  - The bridge keeps a fixed-length rolling temporal buffer.
  - The target length comes from checkpoint metadata, so the runtime does not need to guess whether the model expects `30`, `36`, or `40` frames.

- Temporal grace period
  - The main webcam loop uses `HAND_MISSING_GRACE_FRAMES = 10`.
  - Logic:
    - if hands are visible, reset grace counter to `10` and continue normal appending
    - if hands disappear, decrement the counter instead of clearing immediately
    - if hands return before the counter drops below zero, keep the existing sequence context
    - if the counter is exceeded, clear the sequence buffer, cached predictions, and UI state
  - Why it exists:
    - MediaPipe hand tracking is not perfectly continuous
    - long or complex signs can include brief detector flickers
    - clearing immediately would destroy temporal continuity exactly when the BiLSTM needs it most

- Stabilization layer in the main loop
  - Live predictions are not shown directly frame by frame.
  - The top candidate must pass:
    - confidence squelch: `0.65`
    - stabilization window: `10` predictions
    - minimum majority count: `6`
  - Blueprint effect:
    - reduce flicker
    - make TTS and UI respond to stable signs rather than noisy frame-level logits

- Motion metric and the old `LATE` ghosting problem
  - The inference engine still computes a hand-motion score:
    - mean absolute delta across consecutive hand-feature vectors
    - hands only, first `126` dimensions
  - This was introduced to diagnose and suppress idle false positives such as the system hallucinating `LATE` while the user was effectively idle.
  - Important current-state note:
    - the runtime no longer uses this metric as a hard kill-switch when hands remain on screen
    - that stricter behavior was relaxed because it blanked real signs when the signer paused mid-gesture
  - Current role of the metric:
    - diagnostic/status context in the live loop
    - support for understanding whether a held sign is static versus moving

- What now turns the sign off
  - Hands still visible:
    - the system prefers to hold sign context instead of blanking immediately
  - Hands missing briefly:
    - hold context during the 10-frame grace window
  - Hands missing for too long:
    - reset to `"..."` and clear stale temporal context

- Training/runtime parity
  - The strongest parity guarantee comes from shared use of `WLASLFeatureEngineering`.
  - The inference bridge calls the same normalization helpers used during training.
  - `scripts/verify_preprocessing_parity.py` is useful, but currently narrow in scope.
    - It verifies exact parity for the shared WLASL hand-preprocessing path.
    - It does not yet serve as a full 180D face-aware parity proof across hands, pose, and face.
  - The real production safeguard today is the shared feature-engineering code path plus checkpoint metadata reconstruction.

## 5. Reliability, UI, And Deployment Strategy

- TTS architecture: `src/modules/text_to_speech.py`
  - The current offline-first path prefers native Windows SAPI when available.
  - Why that change happened:
    - repeated speech in the old setup could die after the first utterance
    - SAPI is a native Windows speech backend and is more stable for repeated low-latency calls in this environment
  - Worker-thread lifecycle:
    - the speech backend is initialized and used inside the worker thread
    - COM is initialized inside that same thread for SAPI
  - Hysteresis / silence-state logic:
    - if the displayed prediction is `"..."`, the last spoken prediction is cleared
    - if the same stabilized sign repeats, it is not spoken again immediately
    - if a new stabilized sign appears, it is enqueued for speech
  - Effect:
    - prevents repeated speech loops
    - allows the same word to be spoken again after the model genuinely returns to silence and then redetects it

- UI strategy: `src/main.py` and `src/modules/ui.py`
  - The OpenCV overlay is not just cosmetic.
  - It exposes:
    - stabilized sign text
    - Top-3 predictions
    - buffering state
    - model readiness
    - held-context messages during grace windows
  - This makes the live system debuggable without attaching a debugger during demo use.

- Storage optimization and deployment portability
  - The repository was reduced from roughly `60 GB` to about `17 GB` by removing raw videos and redundant intermediate artifacts while keeping the MediaPipe cache and metadata.
  - What was intentionally kept:
    - `data/raw/data/mp`
    - `data/raw/wlasl_v0.3.json`
    - `data/raw/label_map_300.json`
  - Engineering justification:
    - the MP cache is the reusable training substrate
    - retraining does not need the original `.mp4` files once landmarks are cached
    - smaller project size improves portability, backup practicality, and iteration speed

- Generalization gap
  - Active face-aware checkpoint:
    - validation Top-1: `67.46%`
    - validation Top-5: `88.90%`
    - test Top-1: `60.03%`
    - test Top-5: `85.80%`
  - Interpretation:
    - the system still has a real signer-independence and domain-shift problem
    - it learns the training/validation distribution better than it transfers to unseen test signers
  - Current mitigation direction:
    - class-balanced sampling
    - augmentation in the data pipeline
    - hand-coordinate perturbations such as jitter/dropout/shape corruption in training
    - richer anchor cues from pose and face features
  - The system is improved and more stable than the hand-only baseline, but it is not “solved”; the remaining gap is a deployment realism problem, not just a missing layer in the network

## Production-only file map

- `scripts/prepare_data.py`
  - Source of truth for active WLASL300 feature engineering.

- `scripts/train_model_300.py`
  - Source of truth for the active BiLSTM-attention training path.

- `scripts/inference_bridge.py`
  - Runtime adapter from checkpoint metadata to live webcam inference.

- `src/main.py`
  - End-to-end orchestration, stabilization, grace-period logic, and TTS triggering.

- `src/modules/keypoint_extraction.py`
  - MediaPipe extraction of hands, compact face context, and upper-body pose.

- `src/modules/text_to_speech.py`
  - Native/offline speech backend wrapper with worker-thread execution.

- `src/modules/ui.py`
  - Live overlay and status presentation.

- `data/raw/wlasl_v0.3.json`
  - Master vocabulary and split authority.

- `data/raw/label_map_300.json`
  - Deployed class-index contract for the current 300-class model.

- `data/raw/data/mp`
  - Cached landmark corpus that makes retraining practical without raw video regeneration.

## Bottom line

- The current system is a `30 x 180` temporal model.
- Its production strength comes from parity between training and runtime feature construction, not from raw model size alone.
- The most meaningful engineering upgrades were:
  - adding face anchors for facial-location signs
  - preserving temporal context across brief tracking loss
  - stabilizing live predictions before UI/TTS emission
  - keeping only reusable landmark-cache assets for faster iteration
