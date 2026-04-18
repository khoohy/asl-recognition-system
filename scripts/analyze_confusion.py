"""
Analyze WLASL300 test-set confusion for a trained checkpoint.

Outputs:
- reports/classification_report.txt
- reports/classification_report.json
- reports/confusion_pairs.txt
- reports/weak_classes_summary.txt
- reports/confusion_matrix.png
"""

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_model_300 import BiLSTMSignClassifier, WLASLMPDataset, load_index_to_gloss


class LegacyBiLSTMSignClassifier(torch.nn.Module):
    """Compatibility loader for checkpoints trained before motion-delta fusion."""

    def __init__(
        self,
        input_dim: int = 126,
        hidden_dim: int = 256,
        num_classes: int = 300,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.attention_pool = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        frame_weights = torch.softmax(self.attention_pool(attn_out), dim=1)
        pooled = torch.sum(attn_out * frame_weights, dim=1)
        return self.classifier(pooled)


def format_gloss(gloss: str) -> str:
    return str(gloss).replace("_", " ").upper()


def load_test_dataset(
    metadata_path: str,
    label_map_path: str,
    mp_root: str,
    sequence_length: int,
    use_pose: bool,
    use_face: bool,
) -> Tuple[WLASLMPDataset, Dict[int, str]]:
    index_to_gloss = load_index_to_gloss(label_map_path)
    _, _, test_samples = WLASLMPDataset.load_samples(metadata_path, index_to_gloss, mp_root)
    dataset = WLASLMPDataset(
        test_samples,
        index_to_gloss,
        mp_root=mp_root,
        sequence_length=sequence_length,
        use_pose=use_pose,
        use_face=use_face,
    )
    return dataset, index_to_gloss


def build_model(checkpoint: Dict, num_classes: int, device: torch.device) -> BiLSTMSignClassifier:
    input_dim = checkpoint.get("input_dim", 126)
    hidden_dim = checkpoint.get("hidden_dim", 256)
    dropout = checkpoint.get("dropout", 0.3)
    num_heads = checkpoint.get("num_heads", 8)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    is_legacy_attention = "attention.in_proj_weight" in state_dict
    if is_legacy_attention:
        model = LegacyBiLSTMSignClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            num_heads=min(num_heads, 4),
        )
        model.load_state_dict(state_dict, strict=False)
    else:
        model = BiLSTMSignClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            num_heads=num_heads,
        )
        model.load_state_dict(state_dict, strict=False)
    return model.to(device)


def run_inference(
    model: BiLSTMSignClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int]]:
    y_true: List[int] = []
    y_pred: List[int] = []

    model.eval()
    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Analyzing"):
            features = features.to(device)
            logits = model(features)
            predictions = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(int(pred) for pred in predictions)
            y_true.extend(int(label) for label in labels.tolist())

    return y_true, y_pred


def estimate_sample_duration(dataset: WLASLMPDataset, sample: Dict) -> int:
    sample_dir = dataset.mp_root / sample["video_id"]
    video_id = sample["video_id"]
    lengths = []
    for prefix in ("lh", "rh"):
        path = sample_dir / f"{prefix}_{video_id}.pickle"
        if path.exists():
            with path.open("rb") as handle:
                arr = np.asarray(pickle.load(handle))
            if arr.ndim >= 1:
                lengths.append(len(arr))
    pose_path = sample_dir / f"pose_{video_id}.pickle"
    if pose_path.exists():
        with pose_path.open("rb") as handle:
            pose_arr = np.asarray(pickle.load(handle))
        if pose_arr.ndim >= 1:
            lengths.append(len(pose_arr))
    return max(lengths) if lengths else dataset.sequence_length


def build_class_prototypes(
    dataset: WLASLMPDataset,
    index_to_gloss: Dict[int, str],
) -> Dict[int, Dict[str, np.ndarray | float]]:
    by_class: Dict[int, List[np.ndarray]] = defaultdict(list)
    durations: Dict[int, List[float]] = defaultdict(list)

    for idx, sample in enumerate(dataset.samples):
        label = dataset.gloss_to_index[sample["gloss"]]
        sequence, _ = dataset[idx]
        seq_np = sequence.numpy()
        by_class[label].append(seq_np)
        durations[label].append(float(estimate_sample_duration(dataset, sample)))

    prototypes: Dict[int, Dict[str, np.ndarray | float]] = {}
    for class_idx, sequences in by_class.items():
        stacked = np.stack(sequences, axis=0)
        hand = stacked[:, :, :126]
        shape_signature = hand.mean(axis=(0, 1))
        motion_signature = np.diff(hand, axis=1, prepend=hand[:, :1, :])
        motion_energy = np.abs(motion_signature).mean(axis=(0, 1))
        temporal_profile = np.linalg.norm(motion_signature, axis=2).mean(axis=0)
        prototypes[class_idx] = {
            "shape_signature": shape_signature.astype(np.float32),
            "motion_energy": motion_energy.astype(np.float32),
            "temporal_profile": temporal_profile.astype(np.float32),
            "duration_mean": float(np.mean(durations[class_idx])),
        }
    return prototypes


def categorize_confused_pairs(
    top_pairs: List[Tuple[str, str, int]],
    pair_counter: Counter[Tuple[int, int]],
    index_to_gloss: Dict[int, str],
    prototypes: Dict[int, Dict[str, np.ndarray | float]],
    limit: int = 20,
) -> List[Dict[str, object]]:
    del top_pairs  # derived from pair_counter below so we can keep index information
    entries: List[Dict[str, object]] = []

    for (true_idx, pred_idx), count in pair_counter.most_common(limit):
        true_proto = prototypes.get(true_idx)
        pred_proto = prototypes.get(pred_idx)
        if true_proto is None or pred_proto is None:
            continue

        shape_distance = float(
            np.linalg.norm(true_proto["shape_signature"] - pred_proto["shape_signature"])
        )
        motion_distance = float(
            np.linalg.norm(true_proto["motion_energy"] - pred_proto["motion_energy"])
        )
        temporal_distance = float(
            np.linalg.norm(true_proto["temporal_profile"] - pred_proto["temporal_profile"])
        )
        duration_gap = abs(float(true_proto["duration_mean"]) - float(pred_proto["duration_mean"]))

        if duration_gap > 3.0 or temporal_distance > motion_distance:
            category = "Temporal Confusion"
            suggested_fix = "Prioritize motion-aware training, longer sequence windows, and pose cues."
        else:
            category = "Static Shape Confusion"
            suggested_fix = "Prioritize hand-shape augmentation, one-hand dropout robustness, and finger scaling."

        entries.append(
            {
                "true_gloss": format_gloss(index_to_gloss[true_idx]),
                "predicted_gloss": format_gloss(index_to_gloss[pred_idx]),
                "count": count,
                "category": category,
                "shape_distance": round(shape_distance, 4),
                "motion_distance": round(motion_distance, 4),
                "temporal_distance": round(temporal_distance, 4),
                "duration_gap": round(duration_gap, 4),
                "suggested_fix": suggested_fix,
            }
        )

    return entries


def get_top_confused_pairs(
    y_true: List[int],
    y_pred: List[int],
    index_to_gloss: Dict[int, str],
    top_k: int = 10,
) -> List[Tuple[str, str, int]]:
    pair_counter: Counter[Tuple[int, int]] = Counter()
    for true_idx, pred_idx in zip(y_true, y_pred):
        if true_idx != pred_idx:
            pair_counter[(true_idx, pred_idx)] += 1

    top_pairs = pair_counter.most_common(top_k)
    return [
        (format_gloss(index_to_gloss[true_idx]), format_gloss(index_to_gloss[pred_idx]), count)
        for (true_idx, pred_idx), count in top_pairs
    ]


def compute_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    labels: List[int],
) -> np.ndarray:
    label_to_pos = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true_idx, pred_idx in zip(y_true, y_pred):
        if true_idx in label_to_pos and pred_idx in label_to_pos:
            matrix[label_to_pos[true_idx], label_to_pos[pred_idx]] += 1
    return matrix


def compute_classification_report(
    y_true: List[int],
    y_pred: List[int],
    labels: List[int],
    target_names: List[str],
) -> Tuple[str, Dict]:
    matrix = compute_confusion_matrix(y_true, y_pred, labels)
    total = int(matrix.sum())
    report_dict: Dict[str, Dict[str, float]] = {}
    lines = [
        f"{'class':<28} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}",
        "",
    ]

    precisions: List[float] = []
    recalls: List[float] = []
    f1_scores: List[float] = []
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for idx, (label, name) in enumerate(zip(labels, target_names)):
        tp = float(matrix[idx, idx])
        support = float(matrix[idx, :].sum())
        predicted = float(matrix[:, idx].sum())
        precision = tp / predicted if predicted > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1_score = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)

        report_dict[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1_score,
            "support": int(support),
        }
        lines.append(f"{name[:28]:<28} {precision:>10.4f} {recall:>10.4f} {f1_score:>10.4f} {int(support):>10}")

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1_score * support

    accuracy = sum(int(t == p) for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    weighted_precision /= max(total, 1)
    weighted_recall /= max(total, 1)
    weighted_f1 /= max(total, 1)

    report_dict["accuracy"] = accuracy
    report_dict["macro avg"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1-score": macro_f1,
        "support": total,
    }
    report_dict["weighted avg"] = {
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1-score": weighted_f1,
        "support": total,
    }

    lines.extend(
        [
            "",
            f"{'accuracy':<28} {'':>10} {'':>10} {accuracy:>10.4f} {total:>10}",
            f"{'macro avg':<28} {macro_precision:>10.4f} {macro_recall:>10.4f} {macro_f1:>10.4f} {total:>10}",
            f"{'weighted avg':<28} {weighted_precision:>10.4f} {weighted_recall:>10.4f} {weighted_f1:>10.4f} {total:>10}",
        ]
    )

    return "\n".join(lines), report_dict


def save_confusion_heatmap(
    y_true: List[int],
    y_pred: List[int],
    index_to_gloss: Dict[int, str],
    output_path: Path,
    top_n: int = 50,
) -> None:
    support_counter = Counter(y_true)
    top_indices = [class_idx for class_idx, _ in support_counter.most_common(top_n)]
    labels = [format_gloss(index_to_gloss[class_idx]) for class_idx in top_indices]

    matrix = compute_confusion_matrix(y_true, y_pred, top_indices)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, np.maximum(row_sums, 1), where=row_sums != 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(18, 15))
    image = plt.imshow(normalized, cmap="magma", interpolation="nearest", aspect="auto")
    plt.colorbar(image, fraction=0.046, pad=0.04, label="Row-normalized confusion")
    plt.xticks(np.arange(len(labels)), labels, rotation=90, fontsize=8)
    plt.yticks(np.arange(len(labels)), labels, fontsize=8)
    plt.xlabel("Predicted Sign")
    plt.ylabel("True Sign")
    plt.title("WLASL300 Confusion Matrix for Top 50 Most Frequent Test Signs")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def build_weak_classes_summary(
    report_dict: Dict,
    y_true: List[int],
    y_pred: List[int],
    index_to_gloss: Dict[int, str],
    min_support: int = 3,
    limit: int = 15,
) -> str:
    pair_counter: Counter[Tuple[int, int]] = Counter()
    for true_idx, pred_idx in zip(y_true, y_pred):
        if true_idx != pred_idx:
            pair_counter[(true_idx, pred_idx)] += 1

    entries = []
    for class_idx, gloss in index_to_gloss.items():
        metrics = report_dict.get(str(class_idx))
        if not metrics:
            continue
        support = int(metrics.get("support", 0))
        if support < min_support:
            continue

        recall = float(metrics.get("recall", 0.0))
        precision = float(metrics.get("precision", 0.0))
        f1_score = float(metrics.get("f1-score", 0.0))

        confusions = [
            (pred_idx, count)
            for (true_idx, pred_idx), count in pair_counter.items()
            if true_idx == class_idx
        ]
        confusions.sort(key=lambda item: item[1], reverse=True)
        top_confusion = confusions[0] if confusions else None
        entries.append(
            {
                "class_idx": class_idx,
                "gloss": format_gloss(gloss),
                "support": support,
                "recall": recall,
                "precision": precision,
                "f1_score": f1_score,
                "top_confusion": top_confusion,
            }
        )

    weakest = sorted(entries, key=lambda item: (item["recall"], item["f1_score"], -item["support"]))[:limit]

    lines = [
        "Weakest classes on the WLASL300 held-out test split",
        "",
        "These classes have the lowest recall among classes with enough support to matter.",
        "Low recall means the model often misses the sign even when it appears in the test set.",
        "",
    ]

    for rank, item in enumerate(weakest, start=1):
        if item["top_confusion"] is None:
            confusion_text = "No dominant confused target recorded."
        else:
            pred_idx, count = item["top_confusion"]
            confusion_text = (
                f"Most often predicted as {format_gloss(index_to_gloss[pred_idx])} "
                f"({count}/{item['support']} mistakes)."
            )

        lines.append(
            f"{rank}. {item['gloss']} | support={item['support']} | "
            f"precision={item['precision']:.3f} | recall={item['recall']:.3f} | "
            f"f1={item['f1_score']:.3f} | {confusion_text}"
        )

    lines.extend(
        [
            "",
            "Actionable interpretation:",
            "- If weak classes are confused with motion-similar signs, test pose features first.",
            "- If weak classes have low support and low recall, keep balanced sampling and consider stronger augmentation.",
            "- If weak classes show many one-hand dropouts, inspect the MediaPipe cache quality for those classes.",
            "- If the same predicted sign absorbs many other classes, inspect whether that target class has cleaner or more abundant training data.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze WLASL300 test-set confusion for a trained model")
    parser.add_argument("--checkpoint", default="models/asl_model_300_best.pt")
    parser.add_argument("--metadata", default="data/raw/wlasl_v0.3.json")
    parser.add_argument("--label-map", default="data/raw/label_map_300.json")
    parser.add_argument("--mp-root", default="data/raw/data/mp")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--top-k-pairs", type=int, default=10)
    parser.add_argument("--top-n-heatmap", type=int, default=50)
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    sequence_length = checkpoint.get("sequence_length", 30) if isinstance(checkpoint, dict) else 30
    input_dim = checkpoint.get("input_dim", 126) if isinstance(checkpoint, dict) else 126
    saved_pose_joints = checkpoint.get("pose_joints", []) if isinstance(checkpoint, dict) else []
    saved_face_landmarks = checkpoint.get("face_landmarks", []) if isinstance(checkpoint, dict) else []
    use_pose = bool(saved_pose_joints) or (input_dim > 126 and not saved_face_landmarks)
    use_face = bool(saved_face_landmarks)

    test_dataset, index_to_gloss = load_test_dataset(
        metadata_path=args.metadata,
        label_map_path=args.label_map,
        mp_root=args.mp_root,
        sequence_length=sequence_length,
        use_pose=use_pose,
        use_face=use_face,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(checkpoint, num_classes=len(index_to_gloss), device=device)
    y_true, y_pred = run_inference(model, test_loader, device)

    labels = list(range(len(index_to_gloss)))
    target_names = [format_gloss(index_to_gloss[idx]) for idx in labels]
    report_text, report_dict = compute_classification_report(
        y_true,
        y_pred,
        labels,
        target_names,
    )

    top_pairs = get_top_confused_pairs(y_true, y_pred, index_to_gloss, top_k=args.top_k_pairs)
    pair_counter: Counter[Tuple[int, int]] = Counter()
    for true_idx, pred_idx in zip(y_true, y_pred):
        if true_idx != pred_idx:
            pair_counter[(true_idx, pred_idx)] += 1
    pair_lines = ["Top confused sign pairs", ""]
    for rank, (true_gloss, pred_gloss, count) in enumerate(top_pairs, start=1):
        pair_lines.append(f"{rank}. {true_gloss} -> {pred_gloss}: {count} times")

    weak_summary = build_weak_classes_summary(report_dict, y_true, y_pred, index_to_gloss)
    prototypes = build_class_prototypes(test_dataset, index_to_gloss)
    actionable_fixes = categorize_confused_pairs(
        top_pairs=top_pairs,
        pair_counter=pair_counter,
        index_to_gloss=index_to_gloss,
        prototypes=prototypes,
        limit=20,
    )

    (reports_dir / "classification_report.txt").write_text(report_text + "\n", encoding="utf-8")
    (reports_dir / "classification_report.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    (reports_dir / "confusion_pairs.txt").write_text("\n".join(pair_lines) + "\n", encoding="utf-8")
    (reports_dir / "weak_classes_summary.txt").write_text(weak_summary + "\n", encoding="utf-8")
    (reports_dir / "actionable_fixes.json").write_text(json.dumps(actionable_fixes, indent=2), encoding="utf-8")

    save_confusion_heatmap(
        y_true=y_true,
        y_pred=y_pred,
        index_to_gloss=index_to_gloss,
        output_path=reports_dir / "confusion_matrix.png",
        top_n=args.top_n_heatmap,
    )

    print(report_text)
    print("\n" + "\n".join(pair_lines))
    print("\nTop actionable confused pairs saved to actionable_fixes.json")
    print("\n" + weak_summary)
    print(f"\nSaved outputs to {reports_dir.resolve()}")


if __name__ == "__main__":
    main()
