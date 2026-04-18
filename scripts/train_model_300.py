"""
Train a BiLSTM ASL classifier on pre-extracted WLASL landmarks.
"""

import argparse
import json
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prepare_data import WLASLDataProcessor, WLASLFeatureEngineering


class BiLSTMSignClassifier(nn.Module):
    """High-capacity BiLSTM with standard single-head attention for WLASL300."""

    def __init__(
        self,
        input_dim: int = 126,
        hidden_dim: int = 512,
        num_classes: int = 300,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_heads: int = 1,
        use_motion_delta: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_motion_delta = use_motion_delta

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        frame_weights = torch.softmax(self.attention_pool(lstm_out), dim=1)
        pooled = torch.sum(lstm_out * frame_weights, dim=1)
        return self.classifier(pooled)


class FocalCrossEntropyLoss(nn.Module):
    """Cross-entropy with optional focal reweighting for harder long-tail examples."""

    def __init__(
        self,
        gamma: float = 0.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = max(0.0, float(gamma))
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits,
            labels,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if self.gamma <= 0.0:
            return ce.mean()

        probs = torch.softmax(logits, dim=1)
        target_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        focal_factor = torch.pow(1.0 - target_probs, self.gamma)
        return (focal_factor * ce).mean()


def load_index_to_gloss(label_map_path: str) -> Dict[int, str]:
    """Load an index-to-gloss JSON map."""
    with Path(label_map_path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(idx): WLASLFeatureEngineering.normalize_gloss(gloss) for idx, gloss in raw.items()}


def parse_gloss_list(raw_value: str) -> List[str]:
    """Parse a comma-separated gloss list into normalized gloss names."""
    if not raw_value:
        return []
    return [
        WLASLFeatureEngineering.normalize_gloss(gloss)
        for gloss in raw_value.split(",")
        if WLASLFeatureEngineering.normalize_gloss(gloss)
    ]


class WLASL300Dataset(Dataset):
    """Dataset backed by a pre-extracted landmarks JSON file."""

    def __init__(
        self,
        samples: Sequence[Dict],
        index_to_gloss: Dict[int, str],
        sequence_length: int = 30,
        augment: bool = False,
    ):
        self.samples = list(samples)
        self.index_to_gloss = index_to_gloss
        self.gloss_to_index = {gloss: idx for idx, gloss in index_to_gloss.items()}
        self.sequence_length = sequence_length
        self.augment = augment
        self.input_dim = 126

    @staticmethod
    def augment_sequence(sequence: np.ndarray) -> np.ndarray:
        return WLASLFeatureEngineering.augment_feature_sequence(sequence)

    def get_sample_weights(self) -> List[float]:
        class_counts: Dict[int, int] = defaultdict(int)
        for sample in self.samples:
            class_counts[self.gloss_to_index[sample["gloss"]]] += 1
        return [1.0 / class_counts[self.gloss_to_index[sample["gloss"]]] for sample in self.samples]

    def get_class_weights(self) -> torch.Tensor:
        class_counts: Dict[int, int] = defaultdict(int)
        for sample in self.samples:
            class_counts[self.gloss_to_index[sample["gloss"]]] += 1
        weights = np.ones((len(self.index_to_gloss),), dtype=np.float32)
        for class_idx, count in class_counts.items():
            weights[class_idx] = 1.0 / max(count, 1)
        weights /= weights.mean()
        return torch.tensor(weights, dtype=torch.float32)

    @staticmethod
    def load_samples(
        landmarks_file: str,
        index_to_gloss: Dict[int, str],
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load and stratify the JSON landmark dataset into train/val/test splits."""
        with Path(landmarks_file).open("r", encoding="utf-8") as handle:
            raw_samples = json.load(handle)

        allowed_glosses = set(index_to_gloss.values())
        by_class: Dict[str, List[Dict]] = defaultdict(list)

        for sample in raw_samples:
            gloss = WLASLFeatureEngineering.normalize_gloss(sample.get("gloss", ""))
            if gloss not in allowed_glosses:
                continue

            sequence = sample.get("sequence")
            if not sequence:
                continue

            by_class[gloss].append(
                {
                    "gloss": gloss,
                    "sequence": sequence,
                    "instance_idx": int(sample.get("instance_idx", 0)),
                    "fps": sample.get("fps"),
                }
            )

        rng = random.Random(seed)
        train_samples: List[Dict] = []
        val_samples: List[Dict] = []
        test_samples: List[Dict] = []

        for class_samples in by_class.values():
            class_samples = sorted(class_samples, key=lambda item: item["instance_idx"])
            rng.shuffle(class_samples)
            total = len(class_samples)

            train_end = max(1, int(total * 0.7))
            val_end = max(train_end + 1, int(total * 0.85)) if total >= 3 else min(total, train_end + 1)

            train_split = class_samples[:train_end]
            val_split = class_samples[train_end:val_end]
            test_split = class_samples[val_end:]

            if not val_split and len(train_split) > 1:
                val_split = [train_split.pop()]
            if not test_split and len(train_split) > 1:
                test_split = [train_split.pop()]

            train_samples.extend(train_split)
            val_samples.extend(val_split)
            test_samples.extend(test_split)

        return train_samples, val_samples, test_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        label = self.gloss_to_index[sample["gloss"]]
        processed = WLASLFeatureEngineering.preprocess_sequence(
            sample["sequence"],
            target_length=self.sequence_length,
        )
        if self.augment:
            processed = self.augment_sequence(processed)
            processed = WLASLFeatureEngineering.resample_sequence(processed, target_length=self.sequence_length)
        return torch.from_numpy(processed), torch.tensor(label, dtype=torch.long)


class WLASLMPDataset(Dataset):
    """Dataset backed by the local MediaPipe pickle cache."""

    def __init__(
        self,
        samples: Sequence[Dict],
        index_to_gloss: Dict[int, str],
        mp_root: str,
        sequence_length: int = 30,
        use_pose: bool = False,
        pose_joints: Sequence[int] | None = None,
        use_face: bool = False,
        face_landmarks: Sequence[int] | None = None,
        augment: bool = False,
    ):
        self.samples = list(samples)
        self.index_to_gloss = index_to_gloss
        self.gloss_to_index = {gloss: idx for idx, gloss in index_to_gloss.items()}
        self.sequence_length = sequence_length
        self.mp_root = Path(mp_root)
        self.use_pose = use_pose
        self.pose_joints = tuple(pose_joints or WLASLFeatureEngineering.DEFAULT_POSE_JOINTS)
        self.use_face = use_face
        self.face_landmarks = tuple(face_landmarks or WLASLFeatureEngineering.DEFAULT_FACE_LANDMARKS)
        self.input_dim = 126
        if self.use_pose:
            self.input_dim += len(self.pose_joints) * 3
        if self.use_face:
            self.input_dim += len(self.face_landmarks) * 3
        self.augment = augment

    def get_sample_weights(self) -> List[float]:
        class_counts: Dict[int, int] = defaultdict(int)
        for sample in self.samples:
            class_counts[self.gloss_to_index[sample["gloss"]]] += 1
        return [1.0 / class_counts[self.gloss_to_index[sample["gloss"]]] for sample in self.samples]

    def get_class_weights(self) -> torch.Tensor:
        class_counts: Dict[int, int] = defaultdict(int)
        for sample in self.samples:
            class_counts[self.gloss_to_index[sample["gloss"]]] += 1
        weights = np.ones((len(self.index_to_gloss),), dtype=np.float32)
        for class_idx, count in class_counts.items():
            weights[class_idx] = 1.0 / max(count, 1)
        weights /= weights.mean()
        return torch.tensor(weights, dtype=torch.float32)

    @staticmethod
    def load_samples(
        metadata_path: str,
        index_to_gloss: Dict[int, str],
        mp_root: str,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Build official train/val/test splits from metadata and local pickle coverage."""
        with Path(metadata_path).open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        allowed_glosses = set(index_to_gloss.values())
        mp_root_path = Path(mp_root)
        available_ids = {entry.name for entry in mp_root_path.iterdir() if entry.is_dir()}

        train_samples: List[Dict] = []
        val_samples: List[Dict] = []
        test_samples: List[Dict] = []

        for item in metadata:
            gloss = WLASLFeatureEngineering.normalize_gloss(item.get("gloss", ""))
            if gloss not in allowed_glosses:
                continue

            for instance in item.get("instances", []):
                video_id = str(instance.get("video_id", ""))
                if video_id not in available_ids:
                    continue

                sample = {
                    "gloss": gloss,
                    "video_id": video_id,
                    "split": instance.get("split", "train"),
                }

                split = sample["split"]
                if split == "train":
                    train_samples.append(sample)
                elif split == "val":
                    val_samples.append(sample)
                elif split == "test":
                    test_samples.append(sample)

        return train_samples, val_samples, test_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        video_id = sample["video_id"]
        gloss = sample["gloss"]
        sample_dir = self.mp_root / video_id

        with (sample_dir / f"lh_{video_id}.pickle").open("rb") as handle:
            left_hand = np.asarray(pickle.load(handle), dtype=np.float32)
        with (sample_dir / f"rh_{video_id}.pickle").open("rb") as handle:
            right_hand = np.asarray(pickle.load(handle), dtype=np.float32)
        pose_path = sample_dir / f"pose_{video_id}.pickle"
        face_path = sample_dir / f"face_{video_id}.pickle"
        pose = None
        face = None
        if self.use_pose and pose_path.exists():
            with pose_path.open("rb") as handle:
                pose = np.asarray(pickle.load(handle), dtype=np.float32)
        if self.use_face and face_path.exists():
            with face_path.open("rb") as handle:
                face = np.asarray(pickle.load(handle), dtype=np.float32)

        if left_hand.ndim != 2 or left_hand.shape[1] != 63:
            left_hand = np.zeros((max(len(right_hand), 1), 63), dtype=np.float32)
        if right_hand.ndim != 2 or right_hand.shape[1] != 63:
            right_hand = np.zeros((max(len(left_hand), 1), 63), dtype=np.float32)

        frame_count = max(len(left_hand), len(right_hand))
        if self.use_pose:
            pose_frame_count = len(pose) if pose is not None and pose.ndim == 2 and pose.shape[1] == 99 else 0
            frame_count = max(frame_count, pose_frame_count, 1)
        if self.use_face:
            face_frame_count = len(face) if face is not None and face.ndim == 2 and face.shape[1] % 3 == 0 else 0
            frame_count = max(frame_count, face_frame_count, 1)
        if len(left_hand) != frame_count:
            padded = np.zeros((frame_count, 63), dtype=np.float32)
            padded[: len(left_hand)] = left_hand
            left_hand = padded
        if len(right_hand) != frame_count:
            padded = np.zeros((frame_count, 63), dtype=np.float32)
            padded[: len(right_hand)] = right_hand
            right_hand = padded
        if self.use_pose:
            if pose is None or pose.ndim != 2 or pose.shape[1] != 99:
                pose = np.zeros((frame_count, 99), dtype=np.float32)
            elif len(pose) != frame_count:
                padded = np.zeros((frame_count, 99), dtype=np.float32)
                padded[: len(pose)] = pose
                pose = padded
        if self.use_face:
            expected_face_width = max(self.face_landmarks, default=0) + 1
            if face is None or face.ndim != 2 or face.shape[1] % 3 != 0 or (face.shape[1] // 3) < expected_face_width:
                face = np.zeros((frame_count, expected_face_width * 3), dtype=np.float32)
            elif len(face) != frame_count:
                padded = np.zeros((frame_count, face.shape[1]), dtype=np.float32)
                padded[: len(face)] = face
                face = padded

        frames = []
        for frame_idx in range(frame_count):
            left = left_hand[frame_idx].reshape(21, 3)
            right = right_hand[frame_idx].reshape(21, 3)
            combined = np.vstack([left, right]).astype(np.float32)
            hand_features = WLASLFeatureEngineering.normalize_landmarks(combined).reshape(-1)
            feature_parts = [hand_features]
            if self.use_pose:
                pose_frame = WLASLFeatureEngineering.extract_pose_frame(
                    pose[frame_idx],
                    selected_joints=self.pose_joints,
                )
                pose_features = WLASLFeatureEngineering.normalize_pose_landmarks(pose_frame).reshape(-1)
                feature_parts.append(pose_features)
            if self.use_face:
                face_frame = WLASLFeatureEngineering.extract_face_frame(
                    face[frame_idx],
                    selected_landmarks=self.face_landmarks,
                )
                face_features = WLASLFeatureEngineering.normalize_face_landmarks(face_frame).reshape(-1)
                feature_parts.append(face_features)
            frame_features = np.concatenate(feature_parts).astype(np.float32)
            frames.append(frame_features)

        feature_sequence = WLASLFeatureEngineering.clean_sequence(np.asarray(frames, dtype=np.float32))
        processed = WLASLFeatureEngineering.resample_sequence(feature_sequence, target_length=self.sequence_length)
        if self.augment:
            processed = WLASL300Dataset.augment_sequence(processed)
            processed = WLASLFeatureEngineering.resample_sequence(processed, target_length=self.sequence_length)
        label = self.gloss_to_index[gloss]
        return torch.from_numpy(processed), torch.tensor(label, dtype=torch.long)


class WLASLKaggle126Dataset(Dataset):
    """Dataset backed by Kaggle fixed-length 126-dim NPY features."""

    def __init__(
        self,
        samples: Sequence[Dict],
        index_to_gloss: Dict[int, str],
        kaggle_root: str,
        sequence_length: int = 30,
        augment: bool = False,
    ):
        self.samples = list(samples)
        self.index_to_gloss = index_to_gloss
        self.gloss_to_index = {gloss: idx for idx, gloss in index_to_gloss.items()}
        self.sequence_length = sequence_length
        self.kaggle_root = Path(kaggle_root)
        self.augment = augment
        self.input_dim = 126

    def get_sample_weights(self) -> List[float]:
        class_counts: Dict[int, int] = defaultdict(int)
        for sample in self.samples:
            class_counts[self.gloss_to_index[sample["gloss"]]] += 1
        return [1.0 / class_counts[self.gloss_to_index[sample["gloss"]]] for sample in self.samples]

    def get_class_weights(self) -> torch.Tensor:
        class_counts: Dict[int, int] = defaultdict(int)
        for sample in self.samples:
            class_counts[self.gloss_to_index[sample["gloss"]]] += 1
        weights = np.ones((len(self.index_to_gloss),), dtype=np.float32)
        for class_idx, count in class_counts.items():
            weights[class_idx] = 1.0 / max(count, 1)
        weights /= weights.mean()
        return torch.tensor(weights, dtype=torch.float32)

    @staticmethod
    def load_samples(
        metadata_path: str,
        index_to_gloss: Dict[int, str],
        kaggle_root: str,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Build official train/val/test splits using Kaggle NPY coverage by video_id."""
        with Path(metadata_path).open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        allowed_glosses = set(index_to_gloss.values())
        kaggle_root_path = Path(kaggle_root)
        keypoint_dir = kaggle_root_path / "keypoints"
        available_ids = {path.stem for path in keypoint_dir.glob("*.npy")}

        train_samples: List[Dict] = []
        val_samples: List[Dict] = []
        test_samples: List[Dict] = []

        for item in metadata:
            gloss = WLASLFeatureEngineering.normalize_gloss(item.get("gloss", ""))
            if gloss not in allowed_glosses:
                continue

            for instance in item.get("instances", []):
                video_id = str(instance.get("video_id", ""))
                if video_id not in available_ids:
                    continue

                sample = {
                    "gloss": gloss,
                    "video_id": video_id,
                    "split": instance.get("split", "train"),
                    "cache_file": str(keypoint_dir / f"{video_id}.npy"),
                }

                split = sample["split"]
                if split == "train":
                    train_samples.append(sample)
                elif split == "val":
                    val_samples.append(sample)
                elif split == "test":
                    test_samples.append(sample)

        return train_samples, val_samples, test_samples

    @staticmethod
    def trim_zero_padding(sequence: np.ndarray) -> np.ndarray:
        """Remove all-zero padded frames while preserving real zero-safe data."""
        if sequence.ndim != 2:
            raise ValueError(f"Expected shape (frames, features), got {sequence.shape}")

        nonzero_mask = np.any(np.abs(sequence) > 1e-6, axis=1)
        if not np.any(nonzero_mask):
            return np.zeros((1, sequence.shape[1]), dtype=np.float32)
        return sequence[nonzero_mask].astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        gloss = sample["gloss"]
        sequence = np.load(sample["cache_file"]).astype(np.float32)
        sequence = self.trim_zero_padding(sequence)
        sequence = WLASLFeatureEngineering.clean_sequence(sequence)
        if self.augment:
            sequence = WLASL300Dataset.augment_sequence(sequence)
        processed = WLASLFeatureEngineering.resample_sequence(sequence, target_length=self.sequence_length)
        label = self.gloss_to_index[gloss]
        return torch.from_numpy(processed), torch.tensor(label, dtype=torch.long)


class ModelTrainer:
    """Training loop with Top-1/Top-5 metrics and best-model saving."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        output_prefix: str = "models/asl_model_300",
        class_weights: torch.Tensor | None = None,
        warmup_epochs: int = 0,
        sequence_length: int = 30,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        focal_gamma: float = 0.0,
        scheduler_name: str = "plateau",
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = FocalCrossEntropyLoss(
            gamma=focal_gamma,
            label_smoothing=0.1,
            weight=class_weights.to(device) if class_weights is not None else None,
        )
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = None
        self.history: List[Dict] = []
        self.best_top1 = -1.0
        self.best_epoch = 0
        self.early_stop_patience = 10
        self.no_improve_epochs = 0
        self.output_prefix = Path(output_prefix)
        self.base_learning_rate = learning_rate
        self.warmup_epochs = max(0, warmup_epochs)
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.focal_gamma = focal_gamma
        self.scheduler_name = scheduler_name

    @staticmethod
    def compute_topk(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
        topk = torch.topk(logits, k=min(k, logits.shape[1]), dim=1).indices
        correct = (topk == labels.unsqueeze(1)).any(dim=1).float().mean()
        return float(correct.item() * 100.0)

    def run_epoch(self, loader: DataLoader, training: bool) -> Tuple[float, float, float]:
        self.model.train(mode=training)
        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        total_batches = 0

        progress = tqdm(loader, desc="Train" if training else "Eval")
        for features, labels in progress:
            features = features.to(self.device)
            labels = labels.to(self.device)

            if training:
                self.optimizer.zero_grad()

            logits = self.model(features)
            loss = self.criterion(logits, labels)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if self.scheduler_name == "onecycle" and self.scheduler is not None:
                    self.scheduler.step()

            batch_top1 = self.compute_topk(logits, labels, k=1)
            batch_top5 = self.compute_topk(logits, labels, k=5)

            total_loss += float(loss.item())
            total_top1 += batch_top1
            total_top5 += batch_top5
            total_batches += 1

            progress.set_postfix(loss=f"{loss.item():.4f}", top1=f"{batch_top1:.2f}", top5=f"{batch_top5:.2f}")

        if total_batches == 0:
            raise RuntimeError("No batches were produced. Check the dataset and splits.")

        return total_loss / total_batches, total_top1 / total_batches, total_top5 / total_batches

    def save_model(self, path: str, epoch: int, index_to_gloss: Dict[int, str], metrics: Dict) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "metrics": metrics,
                "index_to_gloss": {str(idx): gloss for idx, gloss in index_to_gloss.items()},
                "input_dim": getattr(self.model, "input_dim", 126),
                "sequence_length": self.sequence_length,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "focal_gamma": self.focal_gamma,
                "num_heads": getattr(self.model, "num_heads", 1),
                "use_motion_delta": getattr(self.model, "use_motion_delta", False),
                "num_classes": len(index_to_gloss),
                "pose_joints": list(getattr(self.model, "pose_joints", ())),
                "face_landmarks": list(getattr(self.model, "face_landmarks", ())),
                "use_face": bool(getattr(self.model, "face_landmarks", ())),
            },
            path,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        index_to_gloss: Dict[int, str],
        epochs: int,
    ) -> None:
        Path("models").mkdir(parents=True, exist_ok=True)
        Path("checkpoints").mkdir(parents=True, exist_ok=True)
        started = time.time()
        if self.scheduler_name == "onecycle":
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.base_learning_rate,
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.2,
                anneal_strategy="cos",
                div_factor=10.0,
                final_div_factor=100.0,
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=3)

        for epoch in range(1, epochs + 1):
            if self.scheduler_name != "onecycle" and self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                warmup_lr = self.base_learning_rate * (epoch / self.warmup_epochs)
                for group in self.optimizer.param_groups:
                    group["lr"] = warmup_lr

            print(f"\nEpoch {epoch}/{epochs}")
            train_loss, train_top1, train_top5 = self.run_epoch(train_loader, training=True)
            val_loss, val_top1, val_top5 = self.run_epoch(val_loader, training=False)

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_top1": train_top1,
                "train_top5": train_top5,
                "val_loss": val_loss,
                "val_top1": val_top1,
                "val_top5": val_top5,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(epoch_metrics)

            print(
                f"train loss={train_loss:.4f} top1={train_top1:.2f}% top5={train_top5:.2f}% | "
                f"val loss={val_loss:.4f} top1={val_top1:.2f}% top5={val_top5:.2f}%"
            )

            if self.scheduler_name != "onecycle" and epoch > self.warmup_epochs:
                self.scheduler.step(val_top1)

            if val_top1 > self.best_top1:
                self.best_top1 = val_top1
                self.best_epoch = epoch
                self.no_improve_epochs = 0
                self.save_model(str(self.output_prefix.with_suffix(".pt")), epoch, index_to_gloss, epoch_metrics)
                self.save_model(
                    str(self.output_prefix.parent / f"{self.output_prefix.name}_best.pt"),
                    epoch,
                    index_to_gloss,
                    epoch_metrics,
                )
                print(f"Saved new best model to {self.output_prefix.with_suffix('.pt')}")
            else:
                self.no_improve_epochs += 1
                print(f"No validation improvement for {self.no_improve_epochs}/{self.early_stop_patience} epochs")

            if epoch % 5 == 0:
                self.save_model(f"checkpoints/asl_model_300_epoch_{epoch:03d}.pt", epoch, index_to_gloss, epoch_metrics)

            if self.no_improve_epochs >= self.early_stop_patience:
                print("Early stopping triggered.")
                break

        history_path = self.output_prefix.parent / f"{self.output_prefix.name}_history.json"
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)

        elapsed = time.time() - started
        print(f"\nTraining complete in {elapsed / 60.0:.1f} minutes")
        print(f"Best validation Top-1: {self.best_top1:.2f}% at epoch {self.best_epoch}")


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate a trained model on a loader and return Top-1/Top-5 metrics."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_batches = 0

    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Test"):
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += float(loss.item())
            total_top1 += ModelTrainer.compute_topk(logits, labels, k=1)
            total_top5 += ModelTrainer.compute_topk(logits, labels, k=5)
            total_batches += 1

    if total_batches == 0:
        raise RuntimeError("No test batches were produced.")

    return {
        "loss": total_loss / total_batches,
        "top1": total_top1 / total_batches,
        "top5": total_top5 / total_batches,
    }


def save_experiment_report(
    output_prefix: str,
    args: argparse.Namespace,
    trainer: ModelTrainer,
    train_samples: Sequence[Dict],
    val_samples: Sequence[Dict],
    test_samples: Sequence[Dict],
    test_metrics: Dict[str, float] | None = None,
) -> Path:
    """Persist a compact run report so experiment details stay traceable."""
    report_path = Path(f"{output_prefix}_report.json")
    best_metrics = max(trainer.history, key=lambda item: item["val_top1"]) if trainer.history else None
    all_samples = list(train_samples) + list(val_samples) + list(test_samples)
    all_glosses = {sample["gloss"] for sample in all_samples if "gloss" in sample}
    report = {
        "best_checkpoint": f"{output_prefix}.pt",
        "dataset": {
            "source": args.source,
            "metadata": args.metadata,
            "label_map": args.label_map,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
        },
        "model": {
            "input_dim": getattr(trainer.model, "input_dim", 126),
            "hidden_dim": trainer.hidden_dim,
            "dropout": trainer.dropout,
            "num_heads": getattr(trainer.model, "num_heads", 8),
            "sequence_length": trainer.sequence_length,
            "num_classes": len(all_glosses),
            "use_pose": bool(args.use_pose),
            "use_face": bool(args.use_face),
            "pose_joints": list(getattr(trainer.model, "pose_joints", ())),
            "face_landmarks": list(getattr(trainer.model, "face_landmarks", ())),
        },
        "training": {
            "epochs_requested": args.epochs,
            "epochs_completed": len(trainer.history),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_epochs": args.warmup_epochs,
            "scheduler": args.scheduler,
            "augment": bool(args.augment),
            "class_balanced": bool(args.class_balanced),
            "weighted_loss": bool(args.weighted_loss),
            "focal_gamma": args.focal_gamma,
            "boost_glosses": parse_gloss_list(args.boost_glosses),
            "boost_factor": args.boost_factor,
            "seed": args.seed,
            "device": args.device,
        },
        "best_validation": best_metrics,
        "test_metrics": test_metrics,
        "history_file": str(Path(output_prefix).parent / f"{Path(output_prefix).name}_history.json"),
    }
    if args.source == "mp-cache":
        report["dataset"]["mp_root"] = args.mp_root
    if args.source == "kaggle-126":
        report["dataset"]["kaggle_root"] = args.kaggle_root
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the WLASL300 BiLSTM model")
    parser.add_argument("--source", choices=["json", "mp-cache", "kaggle-126"], default="mp-cache")
    parser.add_argument("--landmarks-file", default="data/raw/wlasl2000_landmarks.json")
    parser.add_argument("--label-map", default="data/raw/label_map_300.json")
    parser.add_argument("--metadata", default="data/raw/wlasl_v0.3.json")
    parser.add_argument("--mp-root", default="data/raw/data/mp")
    parser.add_argument("--kaggle-root", default="data/raw/kaggle/wlasl-126keypoints-2000/wlasl_keypoints_126")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default="models/asl_model_300.pt")
    parser.add_argument("--output-prefix", default="models/asl_model_300")
    parser.add_argument("--use-pose", action="store_true")
    parser.add_argument("--use-face", action="store_true")
    parser.add_argument("--class-balanced", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--weighted-loss", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--scheduler", choices=["onecycle", "plateau"], default="plateau")
    parser.add_argument(
        "--boost-glosses",
        default="",
        help="Comma-separated glosses to oversample more heavily during class-balanced training.",
    )
    parser.add_argument(
        "--boost-factor",
        type=float,
        default=1.0,
        help="Extra multiplier for sampler weights of glosses listed in --boost-glosses.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not Path(args.label_map).exists():
        print("Label map missing. Regenerating from WLASL metadata...")
        WLASLDataProcessor().prepare_training_data()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    index_to_gloss = load_index_to_gloss(args.label_map)

    if args.source == "mp-cache":
        train_samples, val_samples, test_samples = WLASLMPDataset.load_samples(
            args.metadata,
            index_to_gloss,
            args.mp_root,
        )
        train_dataset = WLASLMPDataset(
            train_samples,
            index_to_gloss,
            mp_root=args.mp_root,
            sequence_length=args.sequence_length,
            use_pose=args.use_pose,
            use_face=args.use_face,
            augment=args.augment,
        )
        val_dataset = WLASLMPDataset(
            val_samples,
            index_to_gloss,
            mp_root=args.mp_root,
            sequence_length=args.sequence_length,
            use_pose=args.use_pose,
            use_face=args.use_face,
        )
        test_dataset = WLASLMPDataset(
            test_samples,
            index_to_gloss,
            mp_root=args.mp_root,
            sequence_length=args.sequence_length,
            use_pose=args.use_pose,
            use_face=args.use_face,
        )
    elif args.source == "kaggle-126":
        train_samples, val_samples, test_samples = WLASLKaggle126Dataset.load_samples(
            args.metadata,
            index_to_gloss,
            args.kaggle_root,
        )
        train_dataset = WLASLKaggle126Dataset(
            train_samples,
            index_to_gloss,
            kaggle_root=args.kaggle_root,
            sequence_length=args.sequence_length,
            augment=args.augment,
        )
        val_dataset = WLASLKaggle126Dataset(
            val_samples,
            index_to_gloss,
            kaggle_root=args.kaggle_root,
            sequence_length=args.sequence_length,
        )
        test_dataset = WLASLKaggle126Dataset(
            test_samples,
            index_to_gloss,
            kaggle_root=args.kaggle_root,
            sequence_length=args.sequence_length,
        )
    else:
        train_samples, val_samples, test_samples = WLASL300Dataset.load_samples(
            args.landmarks_file,
            index_to_gloss,
            seed=args.seed,
        )
        train_dataset = WLASL300Dataset(
            train_samples,
            index_to_gloss,
            sequence_length=args.sequence_length,
            augment=args.augment,
        )
        val_dataset = WLASL300Dataset(val_samples, index_to_gloss, sequence_length=args.sequence_length)
        test_dataset = WLASL300Dataset(test_samples, index_to_gloss, sequence_length=args.sequence_length)

    print(
        f"Loaded samples from {args.source}: "
        f"train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
    )

    train_sampler = None
    train_shuffle = True
    if args.class_balanced and hasattr(train_dataset, "get_sample_weights"):
        sample_weights = torch.as_tensor(train_dataset.get_sample_weights(), dtype=torch.double)
        boosted_glosses = set(parse_gloss_list(args.boost_glosses))
        if boosted_glosses and args.boost_factor > 1.0 and hasattr(train_dataset, "samples"):
            boosted_count = 0
            for sample_idx, sample in enumerate(getattr(train_dataset, "samples", [])):
                gloss = WLASLFeatureEngineering.normalize_gloss(sample.get("gloss", ""))
                if gloss in boosted_glosses:
                    sample_weights[sample_idx] *= float(args.boost_factor)
                    boosted_count += 1
            print(
                f"Applied targeted sampler boost to {boosted_count} training samples "
                f"across {len(boosted_glosses)} glosses: {sorted(boosted_glosses)} "
                f"(factor={args.boost_factor:.2f})"
            )
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_shuffle = False

    class_weights = None
    if args.weighted_loss and hasattr(train_dataset, "get_class_weights"):
        class_weights = train_dataset.get_class_weights()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=0,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    input_dim = getattr(train_dataset, "input_dim", 126)
    model = BiLSTMSignClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(index_to_gloss),
        dropout=args.dropout,
        num_heads=args.num_heads,
    )
    model.pose_joints = tuple(getattr(train_dataset, "pose_joints", ()))
    model.face_landmarks = tuple(getattr(train_dataset, "face_landmarks", ()))
    print(f"Training on {device} with {sum(p.numel() for p in model.parameters()):,} parameters")

    if args.eval_only:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        checkpoint_input_dim = checkpoint.get("input_dim", input_dim) if isinstance(checkpoint, dict) else input_dim
        checkpoint_hidden_dim = checkpoint.get("hidden_dim", args.hidden_dim) if isinstance(checkpoint, dict) else args.hidden_dim
        checkpoint_dropout = checkpoint.get("dropout", args.dropout) if isinstance(checkpoint, dict) else args.dropout
        checkpoint_num_heads = checkpoint.get("num_heads", args.num_heads) if isinstance(checkpoint, dict) else args.num_heads
        if (
            checkpoint_input_dim != input_dim
            or checkpoint_hidden_dim != args.hidden_dim
            or checkpoint_dropout != args.dropout
            or checkpoint_num_heads != args.num_heads
        ):
            model = BiLSTMSignClassifier(
                input_dim=checkpoint_input_dim,
                hidden_dim=checkpoint_hidden_dim,
                num_classes=len(index_to_gloss),
                dropout=checkpoint_dropout,
                num_heads=checkpoint_num_heads,
            )
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        metrics = evaluate_model(model, test_loader, device)
        print(
            f"Test metrics from {args.checkpoint}: "
            f"loss={metrics['loss']:.4f} top1={metrics['top1']:.2f}% top5={metrics['top5']:.2f}%"
        )
        return

    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        output_prefix=args.output_prefix,
        class_weights=class_weights,
        warmup_epochs=args.warmup_epochs,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        focal_gamma=args.focal_gamma,
        scheduler_name=args.scheduler,
    )
    trainer.train(train_loader, val_loader, index_to_gloss=index_to_gloss, epochs=args.epochs)

    best_checkpoint = Path(f"{args.output_prefix}.pt")
    test_metrics = None
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        test_metrics = evaluate_model(model, test_loader, device)
        print(
            f"Best-checkpoint test metrics: "
            f"loss={test_metrics['loss']:.4f} top1={test_metrics['top1']:.2f}% top5={test_metrics['top5']:.2f}%"
        )

    report_path = save_experiment_report(
        output_prefix=args.output_prefix,
        args=args,
        trainer=trainer,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        test_metrics=test_metrics,
    )
    print(f"Saved experiment report to {report_path}")


if __name__ == "__main__":
    main()
