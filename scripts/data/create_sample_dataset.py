"""
Create sample WLASL300 dataset structure for testing.
Generates mock video frames for testing the pipeline without full dataset download.
"""

import json
import os
import numpy as np
import cv2
from pathlib import Path
from typing import List


def create_sample_dataset(output_dir: str = "data/raw", num_signs: int = 10, frames_per_sign: int = 30):
    """
    Create sample dataset with mock video frames.
    
    Args:
        output_dir: Base directory for dataset
        num_signs: Number of sign samples to create
        frames_per_sign: Frames per video
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ASL sample vocabulary (first 10 signs from WLASL300)
    signs = [
        "ABOUT", "ABOVE", "ACCEPT", "ACCESS", "ACCORDING",
        "ACROSS", "ACT", "ACTION", "ACTIVE", "ACTIVITY"
    ][:num_signs]
    
    print(f"Creating sample dataset with {num_signs} signs...")
    print(f"Frames per sign: {frames_per_sign}\n")
    
    # Create directory structure and sample videos
    for sign_idx, sign_name in enumerate(signs, 1):
        sign_dir = output_dir / sign_name
        sign_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 1 sample video per sign
        video_file = sign_dir / f"{sign_name}_video_000.mp4"
        
        if not video_file.exists():
            print(f"[{sign_idx}/{num_signs}] Creating {sign_name} sample video...")
            create_sample_video(str(video_file), frames_per_sign)
        else:
            print(f"[{sign_idx}/{num_signs}] {sign_name} video already exists")
    
    # Create metadata JSON
    create_metadata_json(output_dir, signs)
    
    print(f"\n✓ Sample dataset created at {output_dir}")
    print(f"  Total signs: {len(signs)}")
    print(f"  Total videos: {len(signs)}")
    print(f"  Frames per video: {frames_per_sign}")


def create_sample_video(output_path: str, num_frames: int = 30, fps: int = 30, width: int = 640, height: int = 480):
    """
    Create a sample video file with motion patterns.
    
    Args:
        output_path: Path to save video
        num_frames: Number of frames to generate
        fps: Frames per second
        width: Video width
        height: Video height
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_idx in range(num_frames):
        # Create frame with simulated hand motion
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Draw some animated circles to simulate hand position
        center_x = int(width * (0.3 + 0.4 * np.sin(frame_idx * np.pi / num_frames)))
        center_y = int(height * (0.4 + 0.2 * np.cos(frame_idx * np.pi / num_frames)))
        
        cv2.circle(frame, (center_x, center_y), 50, (0, 255, 0), -1)
        cv2.circle(frame, (center_x - 30, center_y), 20, (255, 0, 0), -1)
        cv2.circle(frame, (center_x + 30, center_y), 20, (255, 0, 0), -1)
        
        # Add frame counter
        cv2.putText(frame, f"Frame {frame_idx+1}/{num_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        out.write(frame)
    
    out.release()


def create_metadata_json(dataset_dir: Path, signs: List[str]):
    """Create WLASL300 metadata JSON file."""
    metadata = []
    
    for sign_idx, sign_name in enumerate(signs):
        sign_entry = {
            "gloss": sign_name,
            "instances": [
                {
                    "video_id": f"{sign_name}_video_000",
                    "signer_id": 1,
                    "url": "",
                    "fps": 30,
                    "duration": 1.0
                }
            ]
        }
        metadata.append(sign_entry)
    
    metadata_file = dataset_dir / "wlasl_v0.3.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to {metadata_file}")


if __name__ == "__main__":
    import sys
    
    num_signs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    print("\n" + "="*60)
    print("WLASL300 Sample Dataset Creator")
    print("="*60 + "\n")
    
    create_sample_dataset(num_signs=num_signs, frames_per_sign=num_frames)
    
    print("\n" + "="*60)
    print("Sample Dataset Ready!")
    print("="*60 + "\n")
