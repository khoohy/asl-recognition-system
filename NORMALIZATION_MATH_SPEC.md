"""
MATHEMATICAL SPECIFICATION: WLASL300 Normalization Pipeline
============================================================

This document provides the exact mathematical formulations for the normalization
pipeline that must be identical in training and inference.
"""

import numpy as np

# PART 1: WRIST-CENTRIC SHIFT
# ============================
#
# Mathematical Definition:
# Given a hand with 21 landmarks L = [L₀, L₁, ..., L₂₀] where each Lᵢ = (xᵢ, yᵢ, zᵢ)
#
# Step 1: Extract wrist position
#   W = L₀ = (x₀, y₀, z₀)
#
# Step 2: Translate all landmarks
#   L'ᵢ = Lᵢ - W for all i ∈ [0, 20]
#
# Result: L'₀ = (0, 0, 0) and all other landmarks are relative to wrist
#
# Example:
#   Input:  L = [(0.5, 0.2, 0.9), (0.51, 0.15, 0.88), (0.6, 0.1, 0.92), ...]
#   Wrist:  W = (0.5, 0.2, 0.9)
#   Output: L' = [(0, 0, 0), (0.01, -0.05, -0.02), (0.1, -0.1, 0.02), ...]


# PART 2: DISTANCE SCALING
# =========================
#
# Mathematical Definition:
# After wrist-centric shift, normalize by hand size
#
# Step 1: Define reference distance
#   Reference point: Middle finger MCP (metacarpophalangeal joint) at index 9
#   After shift: M' = L'₉ = (x'₉, y'₉, z'₉)
#
# Step 2: Calculate distance
#   d = ||M'_xy|| = √((x'₉)² + (y'₉)²)
#
#   Note: Use only x, y coordinates (ignore z which is confidence)
#   Note: Only scale x, y; keep z (confidence) unchanged
#
# Step 3: Scale coordinates (avoid division by zero)
#   If d > ε (where ε = 1e-5):
#       L''ᵢ_xy = L'ᵢ_xy / d  for all i ∈ [0, 20]
#       L''ᵢ_z = L'ᵢ_z  (z unchanged)
#   Else:
#       L''ᵢ = L'ᵢ  (no scaling if distance too small)
#
# Result: Hand is size-invariant (distance between wrist and MCP = 1.0)
#
# Example:
#   Input after step 1: L' = [(0, 0, 0), (0.01, -0.05, -0.02), ..., (0.1, -0.1, 0.02), ...]
#   Middle MCP (idx 9):  M' = (0.08, -0.08, 0.01)
#   Distance:           d = √(0.08² + 0.08²) = √(0.0128) ≈ 0.113
#   Output after scale: L'' = [..., (0.01/0.113, -0.05/0.113, -0.02), ..., (0.1/0.113, -0.1/0.113, 0.02), ...]
#                       L''_xy ≈ [(0.088, -0.442, -0.02), ..., (0.885, -0.885, 0.02), ...]


# PART 3: COMBINED NORMALIZATION FUNCTION
# =========================================
#
# Input: landmarks_frame ∈ ℝ^(42×3)
#   - First 21 rows: left hand
#   - Last 21 rows: right hand
#
# Output: normalized_frame ∈ ℝ^(42×3)
#   - Wrist-shifted and distance-scaled
#   - Wrist is at origin (0, 0, 0)
#   - Hand size normalized
#
# Algorithm:
#   for hand_idx in [0, 1]:
#       start_idx ← hand_idx × 21
#       end_idx ← start_idx + 21
#       
#       hand ← landmarks_frame[start_idx:end_idx]  # Shape: (21, 3)
#       
#       # Step 1: Wrist-centric shift
#       wrist ← hand[0]
#       hand ← hand - wrist  # Broadcasting: (21, 3) - (1, 3)
#       
#       # Step 2: Distance scaling
#       mcp ← hand[9]  # Middle finger MCP
#       distance ← √(mcp[0]² + mcp[1]²)  # Use x, y only
#       
#       if distance > 1e-5:
#           hand[:, 0] ← hand[:, 0] / distance  # Scale x
#           hand[:, 1] ← hand[:, 1] / distance  # Scale y
#           # hand[:, 2] unchanged (z = confidence)
#       
#       landmarks_frame[start_idx:end_idx] ← hand
#   
#   return landmarks_frame


# PART 4: TEMPORAL RESAMPLING
# ============================
#
# Mathematical Definition:
# Resample variable-length sequence to fixed 30 frames using linear interpolation
#
# Input: sequence ∈ ℝ^(N×42×3) where N is variable (N ≠ 30)
# Output: resampled ∈ ℝ^(30×42×3)
#
# Algorithm:
#   1. Create old indices: t_old = linspace(0, N-1, N)
#   2. Create new indices: t_new = linspace(0, N-1, 30)
#   3. For each feature dimension f ∈ [0, 126]:
#       y_old = sequence[:, f]  # All N values for feature f
#       y_new = interp1d(t_old, y_old, kind='linear')(t_new)
#       resampled[:, f] = y_new
#
# Result: Smooth temporal sequence with exactly 30 frames
#
# Note: Important for model input which expects fixed (30, 126) tensor


# PART 5: FEATURE VECTOR FLATTENING
# ==================================
#
# Input: landmarks_frame ∈ ℝ^(42×3)
#   Shape: [x₀, y₀, z₀, x₁, y₁, z₁, ..., x₄₁, y₄₁, z₄₁]
#           └─ left hand (21 pts) ─┘  └─ right hand (21 pts) ─┘
#
# Output: flattened ∈ ℝ^126
#   Shape: [x₀, y₀, z₀, x₁, y₁, z₁, ..., x₄₁, y₄₁, z₄₁]
#
# Operation:
#   flattened = landmarks_frame.reshape(-1)  # ℝ^(42×3) → ℝ^126


# PART 6: COMPLETE PIPELINE
# ==========================
#
# Input: Raw keypoints from MediaPipe (variable frame count)
#
# Process:
#   1. Per frame: normalize_landmarks(frame) → (42, 3)
#       ├── Wrist-centric shift (both hands)
#       └── Distance scaling (both hands)
#
#   2. Per sequence: resample_sequence(sequence, 30) → (30, 42, 3)
#       └── Linear interpolation to 30 frames
#
#   3. Per frame: flatten(frame) → (126,)
#       └── Reshape (42, 3) → (126,)
#
# Output: Model-ready tensor (30, 126)
#
# This entire pipeline MUST be identical in:
#   - Training (train_model_300.py)
#   - Inference (inference_bridge.py)
# 
# FAILURE to maintain parity causes distribution shift and 0% accuracy


# PART 7: PSEUDOCODE IMPLEMENTATION
# ==================================

def normalize_landmarks_pseudocode(landmarks_frame):
    """
    Pseudocode for exact normalization algorithm
    """
    # Validate input
    if landmarks_frame is None or landmarks_frame.shape != (42, 3):
        return np.zeros((42, 3), dtype=np.float32)
    
    normalized = landmarks_frame.copy().astype(np.float32)
    
    # Process each hand
    for hand_idx, start_idx in enumerate([0, 21]):
        hand_landmarks = normalized[start_idx:start_idx+21]  # (21, 3)
        
        # Wrist-centric shift
        wrist = hand_landmarks[0].copy()  # Index 0 is wrist
        hand_landmarks -= wrist
        
        # Distance scaling
        mcp_pos = hand_landmarks[9]  # Index 9 is middle finger MCP
        distance = np.linalg.norm(mcp_pos[:2])  # Euclidean norm of x, y
        
        # Avoid division by zero
        if distance > 1e-5:
            hand_landmarks[:, 0] /= distance  # Scale x
            hand_landmarks[:, 1] /= distance  # Scale y
            # hand_landmarks[:, 2] unchanged (z is confidence)
        
        normalized[start_idx:start_idx+21] = hand_landmarks
    
    return normalized


# PART 8: INVARIANCE PROPERTIES
# ==============================
#
# After normalization, the features have these invariances:
#
# 1. Translation Invariance
#    - Shifting entire hand doesn't change features
#    - Because we translate to wrist at origin
#
# 2. Scale Invariance
#    - Hand close or far from camera doesn't change features
#    - Because we normalize by hand size (wrist-to-MCP distance)
#
# 3. Temporal Invariance
#    - Resampling to 30 frames makes sequences comparable
#    - Prevents bias toward fast or slow signers
#
# These invariances help the model generalize from training videos
# to real-time webcam input where:
#   - User position changes (translation)
#   - User distance varies (scale)
#   - Signing speed varies (temporal)


# PART 9: COORDINATE SYSTEMS
# ===========================
#
# MediaPipe Output Coordinates:
#   - x, y: Normalized to frame (0 to 1)
#   - z: Depth coordinate (0 to 1, where 0 = far, 1 = near)
#   - Often z is actually confidence, varies by MediaPipe version
#
# After Normalization:
#   - x, y: Scaled by hand size (typically -0.5 to 0.5 range)
#   - z: Unchanged (preserved for confidence information)
#
# Important: MediaPipe uses right-hand coordinate system
#   - +x: right
#   - +y: down (image coordinates)
#   - +z: toward camera (for depth) or arbitrary (for confidence)


# PART 10: NUMERICAL STABILITY CONSIDERATIONS
# =============================================
#
# 1. Division by Zero Protection
#    if distance > 1e-5:  # Instead of if distance != 0
#    Reason: Floating point comparison safety
#
# 2. Data Type Consistency
#    Use np.float32 throughout (matches PyTorch default)
#    Reason: Prevents type mismatches in model input
#
# 3. Shape Validation
#    Always verify input is (42, 3) before normalization
#    Reason: Prevents silent errors from shape mismatches
#
# 4. NaN Handling
#    np.nan in input → returns zeros
#    Reason: Graceful degradation when detection fails


# PART 11: REFERENCE VALUES FOR TESTING
# ======================================
#
# Test Case: Single hand at origin
#   Input:  [
#     [0.5, 0.2, 0.9],  # Index 0: Wrist
#     [0.51, 0.15, 0.88],  # Index 1
#     ...
#     [0.6, 0.1, 0.92],  # Index 9: Middle MCP
#     ...
#     [0, 0, 0] × 21  # Right hand all zeros
#   ]
#
#   After wrist shift:
#   [
#     [0, 0, 0],  # Wrist at origin
#     [0.01, -0.05, -0.02],
#     ...
#     [0.1, -0.1, 0.02],  # MCP relative to wrist
#     ...
#     [-0.5, -0.2, -0.9] × 21  # Right hand shifted
#   ]
#
#   Distance = √(0.1² + 0.1²) = 0.1414
#
#   After scaling (left hand only):
#   [
#     [0, 0, 0],
#     [0.0707, -0.3536, -0.02],
#     ...
#     [0.7071, -0.7071, 0.02],
#     ...
#   ]
#
#   Note: Right hand zeros scaled by any distance = still zeros


# VERIFICATION CHECKLIST
# ======================
#
# When implementing the pipeline, verify:
#
# □ WLASLDataProcessor.normalize_landmarks() uses both shifts
# □ Wrist index = 0, Middle MCP index = 9 (MediaPipe standard)
# □ Only x, y scaled by distance (z unchanged)
# □ Same function called in BOTH train_model_300.py AND inference_bridge.py
# □ Input validation checks for (42, 3) shape
# □ Output is float32 type
# □ Temporal resampling uses linear interpolation
# □ Flattening produces 126 features per frame
# □ Model expects (batch, 30, 126) input shape


print(__doc__)
