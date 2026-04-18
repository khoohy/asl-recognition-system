"""
Data preparation helpers for WLASL300 landmark training.

This module does two jobs:
1. Select the top 300 glosses from the master WLASL metadata.
2. Provide the exact frame/sequence preprocessing used by both training
   and real-time inference.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


class WLASLFeatureEngineering:
    """Shared preprocessing for training and inference."""

    WRIST_IDX = 0
    DEFAULT_POSE_JOINTS = (0, 11, 12, 13, 14, 15, 16)
    DEFAULT_FACE_LANDMARKS = (10, 151, 168, 1, 2, 13, 14, 17, 152, 33, 263)
    HAND_DIM = 126
    HAND_JOINTS = 21
    FINGER_GROUPS = (
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    )

    @staticmethod
    def normalize_gloss(gloss: str) -> str:
        return str(gloss).strip().lower()

    @staticmethod
    def format_gloss(gloss: str) -> str:
        return WLASLFeatureEngineering.normalize_gloss(gloss).replace("_", " ").upper()

    @staticmethod
    def _scale_hand_minmax(hand: np.ndarray) -> np.ndarray:
        if hand.shape != (21, 3):
            return np.zeros((21, 3), dtype=np.float32)
        if not np.any(hand):
            return hand.astype(np.float32)

        scaled = hand.astype(np.float32).copy()
        mins = scaled.min(axis=0)
        maxs = scaled.max(axis=0)
        spans = maxs - mins

        for coord_idx in range(3):
            if spans[coord_idx] > 1e-6:
                coord = (scaled[:, coord_idx] - mins[coord_idx]) / spans[coord_idx]
                scaled[:, coord_idx] = coord * 2.0 - 1.0
            else:
                scaled[:, coord_idx] = 0.0

        return scaled

    @staticmethod
    def normalize_landmarks(landmarks_frame: np.ndarray) -> np.ndarray:
        """Apply wrist-centric normalization and min-max scaling."""
        if landmarks_frame is None:
            return np.zeros((42, 3), dtype=np.float32)

        frame = np.asarray(landmarks_frame, dtype=np.float32)
        if frame.shape != (42, 3):
            return np.zeros((42, 3), dtype=np.float32)

        normalized_hands: List[np.ndarray] = []
        for start_idx in (0, 21):
            hand = frame[start_idx:start_idx + 21].copy()
            if np.any(hand):
                wrist = hand[WLASLFeatureEngineering.WRIST_IDX].copy()
                hand -= wrist
                hand = WLASLFeatureEngineering._scale_hand_minmax(hand)
            else:
                hand = np.zeros((21, 3), dtype=np.float32)
            normalized_hands.append(hand)

        return np.vstack(normalized_hands).astype(np.float32)

    @staticmethod
    def extract_pose_frame(frame_data: Sequence, selected_joints: Sequence[int] | None = None) -> np.ndarray:
        """Extract a compact upper-body pose slice from a 33x3 pose frame."""
        joints = tuple(selected_joints or WLASLFeatureEngineering.DEFAULT_POSE_JOINTS)
        frame = np.asarray(frame_data, dtype=np.float32)
        if frame.shape == (33, 3):
            pose = frame
        elif frame.ndim == 1 and frame.size == 99:
            pose = frame.reshape(33, 3)
        elif frame.ndim == 2 and frame.shape[1] >= 3 and frame.shape[0] >= 33:
            pose = frame[:33, :3]
        else:
            return np.zeros((len(joints), 3), dtype=np.float32)
        return pose[list(joints)].astype(np.float32)

    @staticmethod
    def extract_face_frame(frame_data: Sequence, selected_landmarks: Sequence[int] | None = None) -> np.ndarray:
        """Extract a compact face slice from a MediaPipe face mesh frame."""
        landmarks = tuple(selected_landmarks or WLASLFeatureEngineering.DEFAULT_FACE_LANDMARKS)
        frame = np.asarray(frame_data, dtype=np.float32)
        landmark_count = len(landmarks)

        if frame.ndim == 1 and frame.size % 3 == 0:
            frame = frame.reshape(-1, 3)

        if frame.ndim != 2 or frame.shape[1] < 3:
            return np.zeros((landmark_count, 3), dtype=np.float32)

        face = frame[:, :3]
        if face.shape[0] <= max(landmarks):
            return np.zeros((landmark_count, 3), dtype=np.float32)
        return face[list(landmarks)].astype(np.float32)

    @staticmethod
    def normalize_pose_landmarks(
        pose_frame: np.ndarray,
        shoulder_pair: tuple[int, int] = (1, 2),
    ) -> np.ndarray:
        """Normalize selected upper-body pose joints around the shoulder center."""
        frame = np.asarray(pose_frame, dtype=np.float32)
        if frame.ndim != 2 or frame.shape[1] != 3:
            return np.zeros_like(frame, dtype=np.float32)
        if not np.any(frame):
            return np.zeros_like(frame, dtype=np.float32)

        normalized = frame.copy()
        left_idx, right_idx = shoulder_pair
        if frame.shape[0] > max(left_idx, right_idx):
            center = (frame[left_idx] + frame[right_idx]) / 2.0
            scale = np.linalg.norm(frame[left_idx, :2] - frame[right_idx, :2])
        else:
            center = frame.mean(axis=0)
            scale = np.max(np.linalg.norm(frame[:, :2] - center[:2], axis=1))

        normalized -= center
        scale = float(scale) if scale > 1e-6 else 1.0
        normalized /= scale
        normalized = np.clip(normalized, -2.0, 2.0)
        return normalized.astype(np.float32)

    @staticmethod
    def normalize_face_landmarks(
        face_frame: np.ndarray,
        eye_pair: tuple[int, int] = (-2, -1),
    ) -> np.ndarray:
        """Normalize selected face landmarks around the eye midpoint."""
        frame = np.asarray(face_frame, dtype=np.float32)
        if frame.ndim != 2 or frame.shape[1] != 3:
            return np.zeros_like(frame, dtype=np.float32)
        if not np.any(frame):
            return np.zeros_like(frame, dtype=np.float32)

        normalized = frame.copy()
        left_idx, right_idx = eye_pair
        left_eye = frame[left_idx]
        right_eye = frame[right_idx]
        center = (left_eye + right_eye) / 2.0
        scale = np.linalg.norm(left_eye[:2] - right_eye[:2])
        if scale <= 1e-6:
            nose_idx = min(3, len(frame) - 1)
            center = frame[nose_idx]
            scale = np.max(np.linalg.norm(frame[:, :2] - center[:2], axis=1))

        normalized -= center
        scale = float(scale) if scale > 1e-6 else 1.0
        normalized /= scale
        normalized = np.clip(normalized, -2.0, 2.0)
        return normalized.astype(np.float32)

    @staticmethod
    def extract_42x3_frame(frame_data: Sequence) -> np.ndarray:
        """
        Convert one dataset frame into a (42, 3) left+right hand array.

        Supported inputs:
        - dict with `left` and `right`
        - list/array with 56 landmarks (14 pose + 21 right + 21 left)
        - list/array with 42 landmarks
        - flat 126-value vector
        """
        if isinstance(frame_data, dict):
            left = np.asarray(frame_data.get("left", np.zeros((21, 3))), dtype=np.float32)
            right = np.asarray(frame_data.get("right", np.zeros((21, 3))), dtype=np.float32)
            left = left[:, :3] if left.ndim == 2 and left.shape[0] == 21 else np.zeros((21, 3), dtype=np.float32)
            right = right[:, :3] if right.ndim == 2 and right.shape[0] == 21 else np.zeros((21, 3), dtype=np.float32)
            return np.vstack([left, right]).astype(np.float32)

        arr = np.asarray(frame_data, dtype=np.float32)
        if arr.ndim == 1 and arr.size == 126:
            return arr.reshape(42, 3).astype(np.float32)

        if arr.ndim == 2 and arr.shape[1] >= 3:
            arr = arr[:, :3]
            if arr.shape[0] == 56:
                right = arr[14:35]
                left = arr[35:56]
                return np.vstack([left, right]).astype(np.float32)
            if arr.shape[0] == 42:
                return arr.astype(np.float32)

        return np.zeros((42, 3), dtype=np.float32)

    @staticmethod
    def preprocess_frame(frame_data: Sequence) -> np.ndarray:
        landmarks = WLASLFeatureEngineering.extract_42x3_frame(frame_data)
        normalized = WLASLFeatureEngineering.normalize_landmarks(landmarks)
        return normalized.reshape(-1).astype(np.float32)

    @staticmethod
    def infer_valid_frames(sequence: np.ndarray, min_nonzero: int = 6) -> np.ndarray:
        """Mark frames with enough signal to trust for interpolation and resampling."""
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 2:
            raise ValueError(f"Expected sequence shape (frames, features), got {seq.shape}")
        if len(seq) == 0:
            return np.zeros((0,), dtype=bool)
        nonzero_counts = np.count_nonzero(np.abs(seq) > 1e-6, axis=1)
        return nonzero_counts >= min_nonzero

    @staticmethod
    def trim_invalid_edges(sequence: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
        """Drop leading and trailing invalid frames while keeping interior gaps."""
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 2:
            raise ValueError(f"Expected sequence shape (frames, features), got {seq.shape}")
        if len(seq) == 0:
            return seq

        mask = valid_mask if valid_mask is not None else WLASLFeatureEngineering.infer_valid_frames(seq)
        valid_indices = np.flatnonzero(mask)
        if len(valid_indices) == 0:
            return seq[:1].copy()
        return seq[valid_indices[0]:valid_indices[-1] + 1].copy()

    @staticmethod
    def interpolate_missing_frames(
        sequence: np.ndarray,
        valid_mask: np.ndarray | None = None,
        max_gap: int | None = 2,
    ) -> np.ndarray:
        """Linearly interpolate short zero-heavy gaps while preserving longer weak segments."""
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 2:
            raise ValueError(f"Expected sequence shape (frames, features), got {seq.shape}")
        if len(seq) == 0:
            return seq

        mask = valid_mask if valid_mask is not None else WLASLFeatureEngineering.infer_valid_frames(seq)
        valid_indices = np.flatnonzero(mask)
        if len(valid_indices) == 0:
            return np.zeros_like(seq, dtype=np.float32)
        if len(valid_indices) == 1:
            return np.repeat(seq[valid_indices], len(seq), axis=0).astype(np.float32)

        interpolated = seq.copy()

        for left_idx, right_idx in zip(valid_indices[:-1], valid_indices[1:]):
            gap = int(right_idx - left_idx - 1)
            if gap <= 0:
                continue
            if max_gap is not None and gap > max_gap:
                continue

            span_positions = np.arange(left_idx, right_idx + 1, dtype=np.float32)
            for feature_idx in range(seq.shape[1]):
                interpolated[left_idx:right_idx + 1, feature_idx] = np.interp(
                    span_positions,
                    np.array([left_idx, right_idx], dtype=np.float32),
                    np.array([seq[left_idx, feature_idx], seq[right_idx, feature_idx]], dtype=np.float32),
                )
        return interpolated.astype(np.float32)

    @staticmethod
    def clean_sequence(sequence: np.ndarray, min_nonzero: int = 6, max_gap: int | None = 2) -> np.ndarray:
        """Trim empty edges and only interpolate short interior gaps before resampling."""
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 2:
            raise ValueError(f"Expected sequence shape (frames, features), got {seq.shape}")
        if len(seq) == 0:
            return seq

        valid_mask = WLASLFeatureEngineering.infer_valid_frames(seq, min_nonzero=min_nonzero)
        trimmed = WLASLFeatureEngineering.trim_invalid_edges(seq, valid_mask=valid_mask)
        trimmed_mask = WLASLFeatureEngineering.infer_valid_frames(trimmed, min_nonzero=min_nonzero)
        cleaned = WLASLFeatureEngineering.interpolate_missing_frames(trimmed, valid_mask=trimmed_mask, max_gap=max_gap)
        return cleaned.astype(np.float32)

    @staticmethod
    def resample_sequence(sequence: np.ndarray, target_length: int = 30) -> np.ndarray:
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 2:
            raise ValueError(f"Expected sequence shape (frames, features), got {seq.shape}")

        if len(seq) == 0:
            return np.zeros((target_length, 126), dtype=np.float32)
        if len(seq) == target_length:
            return seq.astype(np.float32)

        old_positions = np.linspace(0, len(seq) - 1, len(seq), dtype=np.float32)
        new_positions = np.linspace(0, len(seq) - 1, target_length, dtype=np.float32)
        resampled = np.empty((target_length, seq.shape[1]), dtype=np.float32)

        for feature_idx in range(seq.shape[1]):
            resampled[:, feature_idx] = np.interp(new_positions, old_positions, seq[:, feature_idx])

        return resampled

    @staticmethod
    def preprocess_sequence(frames: Sequence, target_length: int = 30) -> np.ndarray:
        processed_frames = [WLASLFeatureEngineering.preprocess_frame(frame) for frame in frames]
        if not processed_frames:
            return np.zeros((target_length, 126), dtype=np.float32)
        stacked = np.asarray(processed_frames, dtype=np.float32)
        stacked = WLASLFeatureEngineering.clean_sequence(stacked)
        return WLASLFeatureEngineering.resample_sequence(stacked, target_length=target_length)

    @staticmethod
    def _reshape_hands_from_features(frame_features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame = np.asarray(frame_features, dtype=np.float32)
        hand = frame[: WLASLFeatureEngineering.HAND_DIM].reshape(42, 3).copy()
        extra = frame[WLASLFeatureEngineering.HAND_DIM :].copy()
        left = hand[:21].copy()
        right = hand[21:].copy()
        return left, right, extra

    @staticmethod
    def _merge_hands_to_features(left: np.ndarray, right: np.ndarray, extra: np.ndarray) -> np.ndarray:
        hand = np.vstack([left, right]).astype(np.float32).reshape(-1)
        if extra.size == 0:
            return hand.astype(np.float32)
        return np.concatenate([hand, extra.astype(np.float32)], axis=0).astype(np.float32)

    @staticmethod
    def _apply_finger_bone_scaling(hand: np.ndarray, scale_range: tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
        hand = np.asarray(hand, dtype=np.float32).copy()
        if hand.shape != (21, 3) or not np.any(hand):
            return hand

        wrist = hand[WLASLFeatureEngineering.WRIST_IDX].copy()
        for finger in WLASLFeatureEngineering.FINGER_GROUPS:
            scale = np.random.uniform(scale_range[0], scale_range[1])
            for joint_idx in finger:
                offset = hand[joint_idx] - wrist
                hand[joint_idx] = wrist + offset * scale
        return hand.astype(np.float32)

    @staticmethod
    def augment_feature_sequence(
        sequence: np.ndarray,
        jitter_std: float = 0.01,
        hand_dropout_prob: float = 0.2,
        finger_scale_prob: float = 0.35,
        temporal_dropout_prob: float = 0.6,
        temporal_scale_prob: float = 0.5,
    ) -> np.ndarray:
        """Apply advanced hand-aware augmentation to a normalized feature sequence."""
        seq = np.asarray(sequence, dtype=np.float32).copy()
        if seq.ndim != 2 or len(seq) == 0:
            return seq

        for frame_idx in range(len(seq)):
            left, right, extra = WLASLFeatureEngineering._reshape_hands_from_features(seq[frame_idx])

            # Gaussian sensor noise on x/y only.
            if jitter_std > 0:
                for hand in (left, right):
                    if np.any(hand):
                        hand[:, :2] += np.random.normal(0.0, jitter_std, size=(WLASLFeatureEngineering.HAND_JOINTS, 2)).astype(np.float32)
                if extra.size > 0:
                    extra_view = extra.reshape(-1, 3)
                    extra_view[:, :2] += np.random.normal(0.0, jitter_std * 0.5, size=(extra_view.shape[0], 2)).astype(np.float32)
                    extra = extra_view.reshape(-1)

            # Finger-size variation to mimic different hand geometries.
            if np.random.random() < finger_scale_prob:
                left = WLASLFeatureEngineering._apply_finger_bone_scaling(left)
                right = WLASLFeatureEngineering._apply_finger_bone_scaling(right)

            seq[frame_idx] = WLASLFeatureEngineering._merge_hands_to_features(left, right, extra)

        # Hand-specific dropout to force robustness for one-handed variants.
        if seq.shape[1] >= WLASLFeatureEngineering.HAND_DIM and np.random.random() < hand_dropout_prob:
            if np.random.random() < 0.5:
                seq[:, :63] = 0.0
            else:
                seq[:, 63:126] = 0.0

        # Temporal frame dropout for tracking instability.
        if len(seq) > 10 and np.random.random() < temporal_dropout_prob:
            drop_count = max(1, int(len(seq) * np.random.uniform(0.03, 0.08)))
            drop_indices = np.random.choice(len(seq), size=drop_count, replace=False)
            seq[drop_indices] = 0.0

        if len(seq) > 5 and np.random.random() < temporal_scale_prob:
            seq = WLASLFeatureEngineering.resample_sequence(
                seq,
                target_length=max(8, int(len(seq) * np.random.uniform(0.9, 1.1))),
            )

        return seq.astype(np.float32)


class WLASLDataProcessor:
    """Metadata-driven selection and label-map generation for WLASL300."""

    def __init__(self, metadata_path: str = "data/raw/wlasl_v0.3.json"):
        self.metadata_path = Path(metadata_path)
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        with self.metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

    def get_top_signs(
        self,
        top_k: int = 300,
        include_glosses: Sequence[str] | None = None,
    ) -> List[Tuple[str, int]]:
        ranked: List[Tuple[str, int]] = []
        for entry in self.metadata:
            gloss = WLASLFeatureEngineering.normalize_gloss(entry.get("gloss", ""))
            if not gloss:
                continue
            ranked.append((gloss, len(entry.get("instances", []))))

        ranked.sort(key=lambda item: (-item[1], item[0]))
        if not include_glosses:
            return ranked[:top_k]

        required = [
            WLASLFeatureEngineering.normalize_gloss(gloss)
            for gloss in include_glosses
            if WLASLFeatureEngineering.normalize_gloss(gloss)
        ]
        required_set = set(required)
        selected = list(ranked[:top_k])
        selected_glosses = {gloss for gloss, _ in selected}

        if required_set.issubset(selected_glosses):
            return selected

        ranked_lookup = {gloss: count for gloss, count in ranked}
        extras = [(gloss, ranked_lookup[gloss]) for gloss in required if gloss in ranked_lookup and gloss not in selected_glosses]
        if not extras:
            return selected

        removable = [item for item in reversed(selected) if item[0] not in required_set]
        for extra in extras:
            if removable:
                drop_item = removable.pop(0)
                selected.remove(drop_item)
            selected.append(extra)

        selected_set = {gloss for gloss, _ in selected}
        return [item for item in ranked if item[0] in selected_set][:top_k]

    def create_label_mapping(
        self,
        top_k: int = 300,
        include_glosses: Sequence[str] | None = None,
    ) -> Tuple[Dict[str, str], Dict[str, int]]:
        top_signs = self.get_top_signs(top_k=top_k, include_glosses=include_glosses)
        index_to_gloss = {str(idx): gloss for idx, (gloss, _) in enumerate(top_signs)}
        gloss_to_index = {gloss: int(idx) for idx, gloss in index_to_gloss.items()}
        return index_to_gloss, gloss_to_index

    def save_label_maps(
        self,
        label_map_path: str = "data/raw/label_map_300.json",
        reverse_map_path: str = "data/raw/label_to_index_300.json",
        selected_signs_path: str = "data/processed/wlasl300/selected_signs.json",
        top_k: int = 300,
        include_glosses: Sequence[str] | None = None,
    ) -> Tuple[Dict[str, str], Dict[str, int]]:
        index_to_gloss, gloss_to_index = self.create_label_mapping(top_k=top_k, include_glosses=include_glosses)
        top_signs = self.get_top_signs(top_k=top_k, include_glosses=include_glosses)

        Path(label_map_path).parent.mkdir(parents=True, exist_ok=True)
        Path(reverse_map_path).parent.mkdir(parents=True, exist_ok=True)
        Path(selected_signs_path).parent.mkdir(parents=True, exist_ok=True)

        with Path(label_map_path).open("w", encoding="utf-8") as handle:
            json.dump(index_to_gloss, handle, indent=2)

        with Path(reverse_map_path).open("w", encoding="utf-8") as handle:
            json.dump(gloss_to_index, handle, indent=2)

        with Path(selected_signs_path).open("w", encoding="utf-8") as handle:
            json.dump(
                [{"index": idx, "gloss": gloss, "instances": count} for idx, (gloss, count) in enumerate(top_signs)],
                handle,
                indent=2,
            )

        return index_to_gloss, gloss_to_index

    def prepare_training_data(
        self,
        output_dir: str = "data/processed/wlasl300",
        top_k: int = 300,
        include_glosses: Sequence[str] | None = None,
        label_map_path: str = "data/raw/label_map_300.json",
        reverse_map_path: str = "data/raw/label_to_index_300.json",
    ) -> None:
        print("Preparing WLASL300 metadata...")
        index_to_gloss, _ = self.save_label_maps(
            label_map_path=label_map_path,
            reverse_map_path=reverse_map_path,
            selected_signs_path=str(Path(output_dir) / "selected_signs.json"),
            top_k=top_k,
            include_glosses=include_glosses,
        )

        summary = {
            "num_classes": len(index_to_gloss),
            "label_map_path": label_map_path,
            "label_to_index_path": reverse_map_path,
            "feature_dim": 126,
            "sequence_length": 30,
            "forced_glosses": [WLASLFeatureEngineering.normalize_gloss(gloss) for gloss in include_glosses or []],
        }
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "sign_data.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        print(f"Saved {len(index_to_gloss)} labels to {label_map_path}")
        print(f"Saved metadata summary to {output_path / 'sign_data.json'}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare WLASL label maps and metadata summary.")
    parser.add_argument("--metadata", default="data/raw/wlasl_v0.3.json")
    parser.add_argument("--output-dir", default="data/processed/wlasl300")
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--label-map-path", default="data/raw/label_map_300.json")
    parser.add_argument("--reverse-map-path", default="data/raw/label_to_index_300.json")
    parser.add_argument(
        "--include-glosses",
        default="",
        help="Comma-separated glosses to force into the label map even if they are outside the default top-k.",
    )
    args = parser.parse_args()

    try:
        forced = [item.strip() for item in args.include_glosses.split(",") if item.strip()]
        WLASLDataProcessor(metadata_path=args.metadata).prepare_training_data(
            output_dir=args.output_dir,
            top_k=args.top_k,
            include_glosses=forced,
            label_map_path=args.label_map_path,
            reverse_map_path=args.reverse_map_path,
        )
    except Exception as exc:
        print(f"Error during data preparation: {exc}")
        raise


if __name__ == "__main__":
    main()
