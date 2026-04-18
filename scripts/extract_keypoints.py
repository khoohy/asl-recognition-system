"""
Demo: Process video dataset to extract keypoints
Demonstrates how to use the system with the sample dataset.
"""

import sys
import json
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules.keypoint_extraction import KeypointExtractor
from src.utils.config import Config
from src.utils.preprocessing import KeypointPreprocessor


def process_video_dataset(dataset_dir: str = "data/raw", output_dir: str = "data/processed", limit_signs: int = 5):
    """
    Process WLASL300 dataset to extract keypoints.
    
    Args:
        dataset_dir: Input dataset directory
        output_dir: Output processed keypoints directory
        limit_signs: Limit number of signs to process (for testing)
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not dataset_dir.exists():
        print(f"✗ Dataset not found at {dataset_dir}")
        return
    
    print(f"\n{'='*60}")
    print("Processing WLASL300 Dataset for Keypoint Extraction")
    print(f"{'='*60}\n")
    
    # Load metadata
    metadata_file = dataset_dir / "wlasl_v0.3.json"
    if not metadata_file.exists():
        print(f"✗ Metadata file not found: {metadata_file}")
        return
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Total signs in dataset: {len(metadata)}")
    print(f"Processing: {min(limit_signs, len(metadata))} signs\n")
    
    # Initialize extractor
    extractor = KeypointExtractor()
    
    processed_count = 0
    failed_count = 0
    
    # Process each sign
    for sign_idx, gloss_entry in enumerate(metadata[:limit_signs]):
        gloss_name = gloss_entry['gloss']
        sign_dir = dataset_dir / gloss_name
        
        print(f"[{sign_idx+1}/{min(limit_signs, len(metadata))}] Processing: {gloss_name}")
        
        if not sign_dir.exists():
            print(f"  ✗ Sign directory not found")
            failed_count += 1
            continue
        
        # Find video file
        video_files = list(sign_dir.glob("*.*"))
        if not video_files:
            print(f"  ✗ No video files found")
            failed_count += 1
            continue
        
        video_file = video_files[0]
        print(f"  Processing: {video_file.name}")
        
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                print(f"  ✗ Cannot open video file")
                failed_count += 1
                continue
            
            # Extract keypoints from all frames
            keypoint_sequence = []
            frame_count = 0
            valid_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Extract keypoints
                keypoints = extractor.extract_keypoints(frame)
                
                if keypoints is not None:
                    hand_landmarks = extractor.get_hand_landmarks(keypoints)
                    if hand_landmarks is not None:
                        # Preprocess
                        features = hand_landmarks.flatten()
                        features = KeypointPreprocessor.normalize_keypoints(features.reshape(-1, 3)).flatten()
                        features = KeypointPreprocessor.scale_keypoints(features.reshape(-1, 3)).flatten()
                        
                        keypoint_sequence.append(features)
                        valid_frames += 1
            
            cap.release()
            
            if len(keypoint_sequence) == 0:
                print(f"  ✗ No valid keypoints extracted ({frame_count} frames, 0 valid)")
                failed_count += 1
                continue
            
            # Pad/truncate to fixed length
            sequence = np.array(keypoint_sequence)
            sequence = KeypointPreprocessor.pad_or_truncate_sequence(sequence, Config.SEQUENCE_LENGTH)
            
            # Save processed keypoints
            output_sign_dir = output_dir / gloss_name
            output_sign_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_sign_dir / f"{video_file.stem}_keypoints.npy"
            np.save(str(output_file), sequence)
            
            print(f"  ✓ Extracted keypoints: {valid_frames}/{frame_count} frames")
            print(f"    Saved: {output_file.name}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ✗ Error processing video: {e}")
            failed_count += 1
    
    extractor.release()
    
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"  Successfully processed: {processed_count}/{limit_signs}")
    print(f"  Failed: {failed_count}/{limit_signs}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Verify output
    output_files = list(output_dir.glob("*/*.npy"))
    print(f"Total keypoint files generated: {len(output_files)}")
    
    if output_files:
        # Show sample
        sample_file = output_files[0]
        sample_data = np.load(sample_file)
        print(f"\nSample keypoint data: {sample_file.parent.name}/{sample_file.name}")
        print(f"  Shape: {sample_data.shape}")
        print(f"  Data type: {sample_data.dtype}")
        print(f"  Range: [{sample_data.min():.3f}, {sample_data.max():.3f}]")


def main():
    """Run demo."""
    import sys
    
    print("\n" + "="*70)
    print(" "*25 + "ASL KEYPOINT EXTRACTION DEMO")
    print("="*70)
    
    # Check if dataset exists
    dataset_dir = Path("data/raw")
    if not dataset_dir.exists():
        print(f"\n✗ Dataset not found. Create sample dataset first:")
        print(f"  python scripts/create_sample_dataset.py")
        return 1
    
    # Process dataset
    try:
        process_video_dataset(limit_signs=5)
        print("\n✓ Demo completed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
