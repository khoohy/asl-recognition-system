"""
Run a prioritized WLASL300 experiment matrix.

This helper keeps the next set of model runs consistent and documented.
"""

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = {
    "baseline_balaug": {
        "description": "Balanced sampling + augmentation baseline refresh",
        "extra_args": [
            "--class-balanced",
            "--augment",
            "--output-prefix",
            "models/asl_model_300_balaug_v2",
        ],
    },
    "pose_balaug": {
        "description": "Balanced sampling + augmentation + pose features",
        "extra_args": [
            "--class-balanced",
            "--augment",
            "--use-pose",
            "--output-prefix",
            "models/asl_model_300_pose_balaug_v2",
        ],
    },
    "pose_face_balaug": {
        "description": "Balanced sampling + augmentation + pose + compact face features",
        "extra_args": [
            "--class-balanced",
            "--augment",
            "--use-pose",
            "--use-face",
            "--output-prefix",
            "models/asl_model_300_pose_face_balaug_v1",
        ],
    },
    "seq36_balaug": {
        "description": "Balanced sampling + augmentation + longer 36-frame window",
        "extra_args": [
            "--class-balanced",
            "--augment",
            "--sequence-length",
            "36",
            "--output-prefix",
            "models/asl_model_300_seq36_balaug_v2",
        ],
    },
    "focal15_balaug": {
        "description": "Balanced sampling + augmentation + focal loss gamma 1.5",
        "extra_args": [
            "--class-balanced",
            "--augment",
            "--focal-gamma",
            "1.5",
            "--output-prefix",
            "models/asl_model_300_balaug_focal15_v2",
        ],
    },
}

MATRICES = {
    "top1_push": [
        "baseline_balaug",
        "pose_balaug",
        "pose_face_balaug",
        "seq36_balaug",
        "focal15_balaug",
    ],
}


def build_base_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/training/train_model_300.py",
        "--source",
        args.source,
        "--metadata",
        args.metadata,
        "--label-map",
        args.label_map,
        "--mp-root",
        args.mp_root,
        "--kaggle-root",
        args.kaggle_root,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--seed",
        str(args.seed),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a predefined WLASL300 experiment matrix")
    parser.add_argument("--matrix", choices=sorted(MATRICES.keys()), default="top1_push")
    parser.add_argument("--source", choices=["json", "mp-cache", "kaggle-126"], default="mp-cache")
    parser.add_argument("--metadata", default="data/raw/wlasl_v0.3.json")
    parser.add_argument("--label-map", default="data/raw/label_map_300.json")
    parser.add_argument("--mp-root", default="data/raw/data/mp")
    parser.add_argument("--kaggle-root", default="data/raw/kaggle/wlasl-126keypoints-2000/wlasl_keypoints_126")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_command = build_base_command(args)
    experiment_ids = MATRICES[args.matrix]

    Path("models").mkdir(parents=True, exist_ok=True)

    print(f"Running matrix: {args.matrix}")
    for index, experiment_id in enumerate(experiment_ids, start=1):
        experiment = EXPERIMENTS[experiment_id]
        command = base_command + experiment["extra_args"]
        print(f"\n[{index}/{len(experiment_ids)}] {experiment_id}")
        print(experiment["description"])
        print(" ".join(command))

        if args.dry_run:
            continue

        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise SystemExit(
                f"Experiment '{experiment_id}' failed with exit code {completed.returncode}. "
                "Fix that run before continuing so results stay comparable."
            )

    if args.dry_run:
        print("\nDry run complete.")
    else:
        print("\nExperiment matrix complete.")


if __name__ == "__main__":
    main()
