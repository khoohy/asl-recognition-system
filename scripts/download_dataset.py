"""
WLASL300 Dataset Downloader
Downloads video samples from WLASL300 dataset.
"""

import os
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# WLASL300 dataset info
WLASL_JSON_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/WLASL_v0.3.json"
WLASL_VIDEO_BASE_URL = "https://github.com/dxli94/WLASL/raw/master/data/"


def download_wlasl_metadata(output_dir: str = "data/raw") -> Optional[dict]:
    """
    Download WLASL dataset metadata (JSON file).
    
    Args:
        output_dir: Directory to save metadata
    
    Returns:
        Loaded metadata dictionary or None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "wlasl_v0.3.json"
    
    if output_file.exists():
        print(f"Metadata already exists: {output_file}")
        with open(output_file, 'r') as f:
            return json.load(f)
    
    print("Downloading WLASL300 metadata...")
    try:
        urllib.request.urlretrieve(WLASL_JSON_URL, output_file)
        print(f"✓ Metadata saved to {output_file}")
        
        with open(output_file, 'r') as f:
            return json.load(f)
    
    except Exception as e:
        print(f"✗ Failed to download metadata: {e}")
        return None


def download_sample_videos(num_samples: int = 10, output_dir: str = "data/raw") -> int:
    """
    Download sample videos from WLASL300.
    
    Args:
        num_samples: Number of sample videos to download
        output_dir: Directory to save videos
    
    Returns:
        Number of successfully downloaded videos
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download metadata first
    metadata = download_wlasl_metadata(output_dir)
    if metadata is None:
        print("Could not load metadata. Aborting download.")
        return 0
    
    print(f"\nDownloading {num_samples} sample videos...")
    print("Note: This may take a while depending on internet speed.\n")
    
    downloaded = 0
    failed = 0
    
    for idx, gloss in enumerate(metadata[:num_samples]):
        gloss_name = gloss.get('gloss', f'sign_{idx}')
        
        # Create subdirectory for each sign
        sign_dir = output_dir / gloss_name
        sign_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to download one video for this sign
        if 'instances' in gloss and len(gloss['instances']) > 0:
            video_id = gloss['instances'][0]['video_id']
            
            # Try different video formats
            for fmt in ['mp4', 'avi', 'mov']:
                video_url = f"{WLASL_VIDEO_BASE_URL}videos/{gloss_name}/{video_id}.{fmt}"
                video_file = sign_dir / f"{video_id}.{fmt}"
                
                if video_file.exists():
                    print(f"[{idx+1}/{num_samples}] ✓ {gloss_name}/{video_id}.{fmt} (cached)")
                    downloaded += 1
                    break
                
                try:
                    print(f"[{idx+1}/{num_samples}] Downloading {gloss_name}/{video_id}.{fmt}...")
                    urllib.request.urlretrieve(video_url, video_file)
                    print(f"                ✓ Successfully saved")
                    downloaded += 1
                    break
                
                except urllib.error.HTTPError as e:
                    if fmt == 'mov':
                        print(f"                ✗ Not found in any format")
                        failed += 1
                except Exception as e:
                    if fmt == 'mov':
                        print(f"                ✗ Error: {e}")
                        failed += 1
    
    print(f"\n{'='*50}")
    print(f"Download Summary:")
    print(f"  Successfully downloaded: {downloaded}")
    print(f"  Failed: {failed}")
    print(f"  Total: {num_samples}")
    print(f"{'='*50}\n")
    
    return downloaded


def verify_dataset(data_dir: str = "data/raw") -> dict:
    """
    Verify downloaded dataset.
    
    Args:
        data_dir: Directory containing dataset
    
    Returns:
        Dictionary with verification stats
    """
    data_dir = Path(data_dir)
    
    stats = {
        'metadata_exists': (data_dir / 'wlasl_v0.3.json').exists(),
        'num_sign_folders': 0,
        'total_videos': 0,
        'video_formats': set()
    }
    
    if not data_dir.exists():
        return stats
    
    for item in data_dir.iterdir():
        if item.is_dir():
            stats['num_sign_folders'] += 1
            for video_file in item.glob('*.*'):
                stats['total_videos'] += 1
                stats['video_formats'].add(video_file.suffix.lower())
    
    stats['video_formats'] = list(stats['video_formats'])
    
    return stats


if __name__ == "__main__":
    import sys
    
    # Allow custom number of samples
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    
    print("\n" + "="*60)
    print("WLASL300 Dataset Downloader")
    print("="*60 + "\n")
    
    # Download metadata
    print("Step 1: Download Metadata")
    print("-" * 60)
    metadata = download_wlasl_metadata()
    
    if metadata:
        print(f"Total signs in WLASL300: {len(metadata)}\n")
    
    # Download sample videos
    print("Step 2: Download Sample Videos")
    print("-" * 60)
    downloaded = download_sample_videos(num_samples)
    
    # Verify
    print("Step 3: Verify Dataset")
    print("-" * 60)
    stats = verify_dataset()
    print(f"Metadata file: {'✓' if stats['metadata_exists'] else '✗'}")
    print(f"Sign folders: {stats['num_sign_folders']}")
    print(f"Total videos: {stats['total_videos']}")
    print(f"Video formats: {', '.join(stats['video_formats']) if stats['video_formats'] else 'None'}")
    
    print("\n" + "="*60)
    print("Dataset Ready!")
    print("="*60 + "\n")
