"""
Realtime inference bridge for the WLASL300 model.
"""

import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from scripts.data.prepare_data import WLASLFeatureEngineering
from scripts.training.train_model_300 import BiLSTMSignClassifier

DEFAULT_REALTIME_MODEL_PATH = "models/production/asl_wlasl300_realtime.pt"
LEGACY_REALTIME_MODEL_NAMES = {
    "asl_model_300_pose_face_balaug_hardened_v1.pt",
    "asl_wlasl300_realtime.pt",
}


def resolve_realtime_model_path(model_path: str) -> str:
    checkpoint_name = Path(model_path).name.lower()
    if checkpoint_name in LEGACY_REALTIME_MODEL_NAMES:
        return DEFAULT_REALTIME_MODEL_PATH
    return model_path


class InferenceBridge:
    """Load the WLASL300 model and expose shared preprocessing + Top-5 prediction."""

    def __init__(
        self,
        model_path: str = DEFAULT_REALTIME_MODEL_PATH,
        label_map_path: str = "data/raw/label_map_300.json",
        device: str = "cuda",
        sequence_length: int | None = None,
    ):
        model_path = resolve_realtime_model_path(model_path)
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.last_feature_dim = 0
        self.last_feature_vector_valid = False
        self.last_sequence_shape_valid = False
        self.last_model_forward_ok = False
        self.last_error: Optional[str] = None

        with Path(label_map_path).open("r", encoding="utf-8") as handle:
            raw_label_map = json.load(handle)
        self.index_to_gloss = {int(idx): gloss for idx, gloss in raw_label_map.items()}

        checkpoint = torch.load(model_path, map_location=self.device)
        checkpoint_sequence_length = checkpoint.get("sequence_length", 30) if isinstance(checkpoint, dict) else 30
        self.sequence_length = sequence_length or checkpoint_sequence_length
        self.input_dim = checkpoint.get("input_dim", 126) if isinstance(checkpoint, dict) else 126
        self.pose_joints = self._resolve_pose_joints(checkpoint)
        self.face_landmarks = self._resolve_face_landmarks(checkpoint)
        self.pose_feature_dim = len(self.pose_joints) * 3
        self.face_feature_dim = len(self.face_landmarks) * 3
        self.extra_feature_dim = max(0, self.input_dim - 126)
        resolved_extra_dim = self.pose_feature_dim + self.face_feature_dim
        if resolved_extra_dim != self.extra_feature_dim:
            self.face_landmarks = tuple()
            self.face_feature_dim = 0
            self.pose_feature_dim = self.extra_feature_dim
        hidden_dim = checkpoint.get("hidden_dim", 256) if isinstance(checkpoint, dict) else 256
        dropout = checkpoint.get("dropout", 0.3) if isinstance(checkpoint, dict) else 0.3
        num_heads = checkpoint.get("num_heads", 8) if isinstance(checkpoint, dict) else 8
        num_layers = checkpoint.get("num_layers", 2) if isinstance(checkpoint, dict) else 2
        self.model = BiLSTMSignClassifier(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_classes=len(self.index_to_gloss),
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
        )
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        self.checkpoint_metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
        self.checkpoint_path = str(model_path)

    def _resolve_pose_joints(self, checkpoint: object) -> Tuple[int, ...]:
        saved_pose_joints = checkpoint.get("pose_joints") if isinstance(checkpoint, dict) else None
        if saved_pose_joints:
            return tuple(int(joint_idx) for joint_idx in saved_pose_joints)

        default_pose_joints = tuple(WLASLFeatureEngineering.DEFAULT_POSE_JOINTS)
        extra_dim = max(0, self.input_dim - 126)
        if extra_dim <= 0:
            return tuple()
        if len(default_pose_joints) * 3 == extra_dim:
            return default_pose_joints

        pose_joint_count = max(0, extra_dim // 3)
        return default_pose_joints[:pose_joint_count]

    def _resolve_face_landmarks(self, checkpoint: object) -> Tuple[int, ...]:
        saved_face_landmarks = checkpoint.get("face_landmarks") if isinstance(checkpoint, dict) else None
        if saved_face_landmarks:
            return tuple(int(landmark_idx) for landmark_idx in saved_face_landmarks)
        return tuple()

    def _build_feature_vector(self, keypoints_dict: Dict) -> np.ndarray:
        left_hand = np.asarray(keypoints_dict.get("left_hand", np.zeros((21, 3))), dtype=np.float32)
        right_hand = np.asarray(keypoints_dict.get("right_hand", np.zeros((21, 3))), dtype=np.float32)
        hand_frame = np.vstack([left_hand, right_hand]).astype(np.float32)
        hand_features = WLASLFeatureEngineering.normalize_landmarks(hand_frame).reshape(-1).astype(np.float32)
        feature_parts = [hand_features]

        if self.pose_feature_dim > 0:
            pose_frame = WLASLFeatureEngineering.extract_pose_frame(
                keypoints_dict.get("pose", np.zeros((33, 3), dtype=np.float32)),
                selected_joints=self.pose_joints,
            )
            pose_features = WLASLFeatureEngineering.normalize_pose_landmarks(
                pose_frame,
                shoulder_pair=(1, 2),
            ).reshape(-1).astype(np.float32)
            feature_parts.append(pose_features)

        if self.face_feature_dim > 0:
            face_frame = WLASLFeatureEngineering.extract_face_frame(
                keypoints_dict.get("face", np.zeros((478, 3), dtype=np.float32)),
                selected_landmarks=self.face_landmarks,
            )
            face_features = WLASLFeatureEngineering.normalize_face_landmarks(face_frame).reshape(-1).astype(np.float32)
            feature_parts.append(face_features)

        return np.concatenate(feature_parts).astype(np.float32)

    def preprocess_webcam_frame(self, keypoints_dict: Dict) -> np.ndarray:
        """Apply the exact same feature engineering used during training."""
        features = self._build_feature_vector(keypoints_dict)
        self.last_feature_dim = int(features.shape[0])
        self.last_feature_vector_valid = self.last_feature_dim == self.input_dim
        if not self.last_feature_vector_valid:
            self.last_model_forward_ok = False
            self.last_error = f"Feature dimension mismatch: expected {self.input_dim}, got {self.last_feature_dim}"
            raise ValueError(self.last_error)
        self.last_error = None
        return features

    def predict_top5(self, keypoint_sequence: np.ndarray) -> List[Tuple[str, float]]:
        """Return Top-5 human-readable predictions."""
        if keypoint_sequence.shape != (self.sequence_length, self.input_dim):
            self.last_sequence_shape_valid = False
            self.last_model_forward_ok = False
            self.last_error = (
                f"Expected sequence shape ({self.sequence_length}, {self.input_dim}), got {keypoint_sequence.shape}"
            )
            raise ValueError(
                self.last_error
            )
        self.last_sequence_shape_valid = True

        with torch.no_grad():
            batch = torch.from_numpy(keypoint_sequence).float().unsqueeze(0).to(self.device)
            logits = self.model(batch)
            probabilities = torch.softmax(logits, dim=1)[0]
            top_probs, top_indices = torch.topk(probabilities, k=min(5, probabilities.shape[0]))
        self.last_model_forward_ok = True
        self.last_error = None

        results: List[Tuple[str, float]] = []
        for probability, index in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
            gloss = self.index_to_gloss.get(int(index), f"unknown_{index}")
            results.append((WLASLFeatureEngineering.format_gloss(gloss), float(probability)))
        return results


class SequenceBuffer:
    """Fixed-size sliding window over per-frame feature vectors."""

    def __init__(self, sequence_length: int = 30, feature_dim: int = 126):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.buffer: deque[np.ndarray] = deque(maxlen=sequence_length)

    def add(self, features: np.ndarray) -> Optional[np.ndarray]:
        vector = np.asarray(features, dtype=np.float32).reshape(-1)
        if vector.shape != (self.feature_dim,):
            raise ValueError(f"Expected feature vector shape ({self.feature_dim},), got {vector.shape}")
        self.buffer.append(vector)
        if len(self.buffer) < self.sequence_length:
            return None
        return np.asarray(self.buffer, dtype=np.float32)

    def clear(self) -> None:
        self.buffer.clear()

    def status(self) -> Tuple[int, int]:
        return len(self.buffer), self.sequence_length


class RealtimeInferenceEngine:
    """End-to-end webcam inference state for WLASL300 mode."""

    def __init__(
        self,
        model_path: str = DEFAULT_REALTIME_MODEL_PATH,
        label_map_path: str = "data/raw/label_map_300.json",
        device: str = "cuda",
        sequence_length: int | None = None,
        prediction_cooldown: float = 0.5,
        min_motion_delta: float = 0.012,
    ):
        self.bridge = InferenceBridge(
            model_path=model_path,
            label_map_path=label_map_path,
            device=device,
            sequence_length=sequence_length,
        )
        self.buffer = SequenceBuffer(
            sequence_length=self.bridge.sequence_length,
            feature_dim=self.bridge.input_dim,
        )
        self.prediction_cooldown = prediction_cooldown
        self.min_motion_delta = float(min_motion_delta)
        self.last_prediction: Optional[List[Tuple[str, float]]] = None
        self.last_prediction_time = 0.0
        self.last_motion_delta = 0.0

    def reset(self) -> None:
        """Clear buffered features and cached predictions when input is idle."""
        self.buffer.clear()
        self.last_prediction = None
        self.last_prediction_time = 0.0
        self.last_motion_delta = 0.0

    def retain_context(self) -> Optional[List[Tuple[str, float]]]:
        """Hold the current temporal context during brief hand dropouts."""
        return self.last_prediction

    def _compute_motion_delta(self, sequence: np.ndarray) -> float:
        """Estimate recent motion using hand features only to suppress idle false positives."""
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 2 or len(seq) < 2:
            return 0.0
        hand_dims = min(126, seq.shape[1])
        deltas = np.diff(seq[:, :hand_dims], axis=0)
        return float(np.mean(np.abs(deltas)))

    def process_frame(self, keypoints_dict: Dict) -> Optional[List[Tuple[str, float]]]:
        features = self.bridge.preprocess_webcam_frame(keypoints_dict)
        sequence = self.buffer.add(features)
        if sequence is None:
            self.last_motion_delta = 0.0
            return None

        self.last_motion_delta = self._compute_motion_delta(sequence)

        now = time.time()
        if now - self.last_prediction_time < self.prediction_cooldown:
            return self.last_prediction

        self.last_prediction = self.bridge.predict_top5(sequence)
        self.last_prediction_time = now
        return self.last_prediction

    def get_status(self) -> Dict:
        current, target = self.buffer.status()
        return {
            "buffer_frames": current,
            "buffer_target": target,
            "is_ready": current == target,
            "feature_dim": self.bridge.last_feature_dim,
            "expected_input_dim": self.bridge.input_dim,
            "uses_pose": self.bridge.pose_feature_dim > 0,
            "uses_face": self.bridge.face_feature_dim > 0,
            "motion_delta": self.last_motion_delta,
            "min_motion_delta": self.min_motion_delta,
            "model_ready": bool(
                self.bridge.last_feature_vector_valid
                and self.bridge.last_sequence_shape_valid
                and self.bridge.last_model_forward_ok
            ),
            "last_error": self.bridge.last_error,
            "last_prediction": self.last_prediction,
            "checkpoint_epoch": self.bridge.checkpoint_epoch,
            "checkpoint_metrics": self.bridge.checkpoint_metrics,
            "checkpoint_path": self.bridge.checkpoint_path,
        }
