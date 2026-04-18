"""
Preprocessing Module
Data preprocessing and normalization for ASL keypoints.
"""

import numpy as np
from typing import Optional, Tuple
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


class KeypointPreprocessor:
    """
    Handles keypoint normalization, filtering, and augmentation.
    """
    
    @staticmethod
    def normalize_keypoints(keypoints: np.ndarray, reference_point: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalize keypoints by centering on reference point (e.g., wrist or neck).
        
        Args:
            keypoints: Array of shape (N, 3) with (x, y, z)
            reference_point: Reference point for centering, defaults to mean of keypoints
        
        Returns:
            Normalized keypoints
        """
        if keypoints is None:
            return None
        
        keypoints = keypoints.copy()
        
        # Handle 2D keypoints
        if keypoints.shape[1] == 2:
            keypoints = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])
        
        if reference_point is None:
            reference_point = np.mean(keypoints[:, :2], axis=0)
        
        # Center keypoints
        keypoints[:, :2] -= reference_point
        
        return keypoints
    
    @staticmethod
    def scale_keypoints(keypoints: np.ndarray, scale_factor: Optional[float] = None) -> np.ndarray:
        """
        Scale keypoints to fixed range [0, 1].
        
        Args:
            keypoints: Array of shape (N, 3)
            scale_factor: Custom scale factor, defaults to max distance
        
        Returns:
            Scaled keypoints
        """
        if keypoints is None:
            return None
        
        keypoints = keypoints.copy()
        
        if scale_factor is None:
            # Scale to max distance
            distances = np.linalg.norm(keypoints[:, :2], axis=1)
            scale_factor = np.max(distances) + 1e-5
        
        keypoints[:, :2] /= scale_factor
        
        return keypoints
    
    @staticmethod
    def temporal_smoothing(sequence: np.ndarray, window_length: int = 5, polyorder: int = 2) -> np.ndarray:
        """
        Apply Savitzky-Golay filter to smooth temporal sequences.
        
        Args:
            sequence: Array of shape (seq_len, feature_dim)
            window_length: Filter window length (must be odd)
            polyorder: Polynomial order
        
        Returns:
            Smoothed sequence
        """
        if sequence is None or len(sequence) < window_length:
            return sequence
        
        # Ensure odd window length
        if window_length % 2 == 0:
            window_length += 1
        
        sequence = sequence.copy()
        
        # Apply filter to each dimension
        for dim in range(sequence.shape[1]):
            try:
                sequence[:, dim] = savgol_filter(sequence[:, dim], window_length, polyorder)
            except:
                # If filtering fails, skip this dimension
                pass
        
        return sequence
    
    @staticmethod
    def handle_missing_keypoints(sequence: np.ndarray, confidence_threshold: float = 0.3) -> np.ndarray:
        """
        Interpolate missing keypoints (marked by low confidence).
        
        Args:
            sequence: Array of shape (seq_len, num_keypoints, 3)
            confidence_threshold: Confidence threshold for missing keypoints
        
        Returns:
            Sequence with interpolated keypoints
        """
        if sequence is None:
            return None
        
        sequence = sequence.copy()
        
        # Mark low-confidence keypoints as missing
        if sequence.shape[2] >= 3:  # Has confidence scores
            missing_mask = sequence[:, :, 2] < confidence_threshold
        else:
            missing_mask = np.zeros((sequence.shape[0], sequence.shape[1]), dtype=bool)
        
        # Interpolate for each keypoint dimension
        for kpt_idx in range(sequence.shape[1]):
            for dim in range(2):  # x, y only
                valid_indices = np.where(~missing_mask[:, kpt_idx])[0]
                
                if len(valid_indices) < 2:
                    # Not enough valid points for interpolation
                    continue
                
                valid_values = sequence[valid_indices, kpt_idx, dim]
                
                # Create interpolation function
                f = interp1d(valid_indices, valid_values, kind='linear', fill_value='extrapolate')
                
                # Interpolate missing values
                all_indices = np.arange(sequence.shape[0])
                sequence[all_indices, kpt_idx, dim] = f(all_indices)
        
        return sequence
    
    @staticmethod
    def pad_or_truncate_sequence(sequence: np.ndarray, target_length: int, pad_value: float = 0.0) -> np.ndarray:
        """
        Pad or truncate sequence to fixed length.
        
        Args:
            sequence: Array of shape (seq_len, ...)
            target_length: Target sequence length
            pad_value: Value to use for padding
        
        Returns:
            Sequence of shape (target_length, ...)
        """
        if sequence is None:
            return None
        
        if len(sequence) == target_length:
            return sequence
        elif len(sequence) > target_length:
            # Truncate - take middle portion for better temporal coverage
            start = (len(sequence) - target_length) // 2
            return sequence[start:start + target_length]
        else:
            # Pad
            pad_width = [(0, target_length - len(sequence))] + [(0, 0)] * (sequence.ndim - 1)
            return np.pad(sequence, pad_width, constant_values=pad_value)
    
    @staticmethod
    def data_augmentation_temporal_scale(sequence: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        Augment by scaling temporal dimension (speed variation).
        
        Args:
            sequence: Array of shape (seq_len, feature_dim)
            scale_range: (min_scale, max_scale)
        
        Returns:
            Augmented sequence
        """
        import random
        
        if sequence is None:
            return None
        
        scale = random.uniform(scale_range[0], scale_range[1])
        new_length = int(len(sequence) * scale)
        
        if new_length < 2:
            return sequence
        
        # Resample using linear interpolation
        old_indices = np.linspace(0, len(sequence) - 1, len(sequence))
        new_indices = np.linspace(0, len(sequence) - 1, new_length)
        
        augmented = np.zeros((new_length, sequence.shape[1]))
        for dim in range(sequence.shape[1]):
            f = interp1d(old_indices, sequence[:, dim], kind='linear')
            augmented[:, dim] = f(new_indices)
        
        return augmented
