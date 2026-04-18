"""
WLASL300 Download from Kaggle - Setup Guide

This script helps you download the complete WLASL300 dataset from Kaggle.
"""

def setup_kaggle():
    """Setup and guide for Kaggle download."""
    
    print("\n" + "="*70)
    print("KAGGLE SETUP GUIDE - WLASL300 DOWNLOAD")
    print("="*70 + "\n")
    
    print("[STEP 1] Install Kaggle CLI (Already Done)\n")
    print("  Command: pip install kaggle")
    print("  Status: ✓ Installed\n")
    
    print("[STEP 2] Get Kaggle API Token\n")
    print("  1. Go to: https://www.kaggle.com/settings/account")
    print("  2. Scroll down to 'API' section")
    print("  3. Click 'Create New API Token'")
    print("     (This downloads kaggle.json)")
    print("  4. Save the file to your home directory:\n")
    
    import os
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_file = os.path.join(kaggle_dir, "kaggle.json")
    
    print(f"     Windows: {os.path.expanduser('~')}\\.kaggle\\kaggle.json")
    print(f"     Linux/Mac: {kaggle_dir}/kaggle.json\n")
    
    print("[STEP 3] Run Download\n")
    print("  Command: python -c \"")
    print("import subprocess")
    print("subprocess.run(['kaggle', 'datasets', 'download', '-d', 'risangdevs/wlasl-processed', '-p', 'data/raw', '--unzip'])")
    print("  \"\n")
    
    print("[STEP 4] Or use this simpler method:\n")
    print("  kaggle datasets download -d risangdevs/wlasl-processed -p data/raw --unzip\n")
    
    print("[ALTERNATIVE] If Kaggle is slow:\n")
    print("  1. Visit: https://www.kaggle.com/datasets/risangdevs/wlasl-processed")
    print("  2. Click 'Download'")
    print("  3. Extract to: data/raw/\n")
    
    print("[EXPECTED DOWNLOAD SIZE]\n")
    print("  - Dataset size: 50-100 GB")
    print("  - Estimated time: 4-8 hours")
    print("  - Network: Requires stable internet connection\n")
    
    print("[VERIFICATION]\n")
    print("  After download, verify with:")
    print("  $ ls data/raw/ | grep -E '^[A-Z]+$' | wc -l")
    print("  (should show ~300)\n")
    
    print("[TRAIN MODEL]\n")
    print("  Once verified, train with:")
    print("  $ python scripts/train_model.py \\")
    print("      --epochs 30 \\")
    print("      --batch-size 8 \\")
    print("      --model bilstm \\")
    print("      --dataset wlasl300\n")
    
    print("="*70)
    print("\n[IMPORTANT] Create kaggle.json file first, then run:")
    print("  kaggle datasets download -d risangdevs/wlasl-processed -p data/raw --unzip\n")

if __name__ == "__main__":
    setup_kaggle()
