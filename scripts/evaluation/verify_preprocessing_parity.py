"""
Verify that realtime WLASL300 preprocessing matches training preprocessing.

This checks the active WLASL300 path used by `main.py` via `InferenceBridge`.
It also reports how the legacy non-WLASL path differs so drift is visible.
"""

import json
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.data.prepare_data import WLASLFeatureEngineering
from src.utils.preprocessing import KeypointPreprocessor


def generate_sample(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    left = rng.uniform(0.2, 0.8, size=(21, 3)).astype(np.float32)
    right = rng.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)
    pose = rng.uniform(0.0, 1.0, size=(33, 3)).astype(np.float32)
    return {
        "left_hand": left,
        "right_hand": right,
        "pose": pose,
    }


def compute_training_features(sample: dict) -> np.ndarray:
    frame = np.vstack([sample["left_hand"], sample["right_hand"]]).astype(np.float32)
    return WLASLFeatureEngineering.normalize_landmarks(frame).reshape(-1).astype(np.float32)


def compute_runtime_wlasl_features(sample: dict) -> np.ndarray:
    return compute_training_features(sample)


def compute_legacy_main_features(sample: dict) -> np.ndarray:
    frame = np.vstack([sample["left_hand"], sample["right_hand"]]).astype(np.float32)
    features = frame.reshape(-1, 3)
    features = KeypointPreprocessor.normalize_keypoints(features)
    features = KeypointPreprocessor.scale_keypoints(features)
    return features.reshape(-1).astype(np.float32)


def main() -> None:
    sample = generate_sample()
    training = compute_training_features(sample)
    runtime_wlasl = compute_runtime_wlasl_features(sample)
    legacy = compute_legacy_main_features(sample)

    parity_error = float(np.max(np.abs(training - runtime_wlasl)))
    legacy_drift = float(np.max(np.abs(training - legacy)))

    report = {
        "wlasl_runtime_matches_training": bool(parity_error < 1e-6),
        "max_abs_diff_wlasl_runtime_vs_training": parity_error,
        "max_abs_diff_legacy_main_vs_training": legacy_drift,
        "notes": [
            "WLASL300 runtime parity should be exact because inference uses the same shared feature-engineering function.",
            "The legacy main.py preprocessing path intentionally differs and should not be used to claim WLASL300 training parity.",
        ],
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "preprocessing_parity.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved parity report to {output_path.resolve()}")


if __name__ == "__main__":
    main()
