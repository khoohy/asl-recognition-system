"""Download WLASL300 dataset from Kaggle."""

import subprocess
import json
from pathlib import Path
import os

def download_from_kaggle():
    """Download WLASL dataset from Kaggle."""
    
    print("\n" + "="*70)
    print("DOWNLOAD WLASL300 FROM KAGGLE")
    print("="*70 + "\n")
    
    # Step 1: Check if Kaggle CLI is installed
    print("[STEP 1] Checking Kaggle CLI\n")
    
    try:
        result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
        print(f"  [OK] Kaggle CLI found: {result.stdout.strip()}\n")
    except FileNotFoundError:
        print("  [ERROR] Kaggle CLI not installed\n")
        print("  Install with: pip install kaggle\n")
        return False
    
    # Step 2: Check credentials
    print("[STEP 2] Checking Kaggle credentials\n")
    
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_config.exists():
        print("  [ERROR] Kaggle credentials not found\n")
        print("  To set up Kaggle API:")
        print("    1. Go to: https://www.kaggle.com/settings/account")
        print("    2. Click 'Create New API Token'")
        print("    3. Save kaggle.json to: ~/.kaggle/kaggle.json")
        print("    4. Run: chmod 600 ~/.kaggle/kaggle.json  (Linux/Mac)\n")
        print("  On Windows, kaggle.json should be at:")
        print(f"    {kaggle_config}\n")
        return False
    else:
        print(f"  [OK] Kaggle credentials found at {kaggle_config}\n")
    
    # Step 3: Download dataset
    print("[STEP 3] Downloading WLASL dataset from Kaggle\n")
    
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = "risangdevs/wlasl-processed"
    
    print(f"  Dataset: {dataset_name}")
    print(f"  Destination: {data_dir.absolute()}")
    print(f"  Size: ~50-100 GB")
    print(f"  Time: 4-8 hours (varies by connection)\n")
    
    print("  Starting download...")
    print("  (This will take a while. Please keep your connection stable.)\n")
    
    try:
        cmd = [
            "kaggle", "datasets", "download", 
            "-d", dataset_name,
            "-p", str(data_dir),
            "--unzip"
        ]
        
        print(f"  Command: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n  [OK] Download complete!\n")
            return True
        else:
            print("\n  [ERROR] Download failed\n")
            return False
            
    except Exception as e:
        print(f"\n  [ERROR] {e}\n")
        return False

def verify_download():
    """Verify the downloaded dataset."""
    
    print("[STEP 4] Verifying dataset\n")
    
    data_dir = Path("data/raw")
    
    # Count sign directories
    sign_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.isupper()]
    
    print(f"  Sign folders found: {len(sign_dirs)}")
    
    if len(sign_dirs) >= 300:
        print(f"  [OK] WLASL300 complete dataset downloaded!\n")
        
        sample_signs = sorted([d.name for d in sign_dirs])[:10]
        print(f"  Sample signs: {', '.join(sample_signs)}\n")
        
        # Check metadata
        meta_files = list(data_dir.glob("wlasl_v*.json"))
        if meta_files:
            with open(meta_files[0]) as f:
                meta = json.load(f)
            print(f"  Metadata: {len(meta)} signs\n")
        
        return True
    else:
        print(f"  [WARNING] Only {len(sign_dirs)} folders found (expected 300)\n")
        return False

def main():
    """Main download pipeline."""
    
    # Download
    success = download_from_kaggle()
    
    if success:
        # Verify
        complete = verify_download()
        
        if complete:
            print("[READY FOR TRAINING]\n")
            print("  Train with:")
            print("  $ python scripts/training/train_model.py \\")
            print("      --epochs 30 \\")
            print("      --batch-size 8 \\")
            print("      --model bilstm \\")
            print("      --dataset wlasl300\n")
            print("  Training targets:")
            print("    - Top-1 Accuracy: 65-75%")
            print("    - Top-5 Accuracy: 85-95%")
            print("    - Latency: <100-150ms per frame\n")
        else:
            print("[NEXT STEPS]\n")
            print("  1. Check data/raw/ folder")
            print("  2. Verify all 300 sign folders are present")
            print("  3. Run training once download is complete\n")
    else:
        print("[TROUBLESHOOTING]\n")
        print("  If download fails:")
        print("    1. Check internet connection")
        print("    2. Verify Kaggle API credentials")
        print("    3. Try again: python scripts/data/download_from_kaggle.py\n")

if __name__ == "__main__":
    main()
