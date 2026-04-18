# Installation & Setup Guide

## System Requirements

### Minimum
- **OS**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+
- **CPU**: Intel i5 / AMD Ryzen 5 (4+ cores)
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: NVIDIA GTX 1650 / RTX 3050 (optional but recommended)
- **Python**: 3.8+

### Recommended
- **OS**: Windows 11 / Ubuntu 22.04
- **CPU**: Intel i7-12700 / AMD Ryzen 7 5800X
- **RAM**: 16GB
- **GPU**: NVIDIA RTX 3050 / RTX 4050 (8GB VRAM)

---

## Step-by-Step Installation

### 1. Install Python

Download Python 3.10+ from [python.org](https://www.python.org/downloads/)

Verify installation:
```bash
python --version
```

---

### 2. Clone Repository

```bash
cd c:\Users\Khoo Han Yang\Desktop\fyp2
```

---

### 3. Create Virtual Environment

#### Option A: Using venv (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Option B: Using Conda
```bash
conda create -n asl-recognition python=3.10
conda activate asl-recognition
```

---

### 4. Install Dependencies

#### Basic Installation
```bash
pip install -r requirements.txt
```

#### GPU Support (NVIDIA CUDA 11.8)
```bash
# PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Other requirements
pip install -r requirements.txt
```

#### GPU Support (NVIDIA CUDA 12.1)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

#### CPU-Only (Slower but works everywhere)
```bash
pip install -r requirements.txt
# Install CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

### 5. Verify Installation

```bash
# Check Python packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import mediapipe; print(f'MediaPipe: {mediapipe.__version__}')"

# Check GPU availability
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Name: {torch.cuda.get_device_name(0)}')"
```

---

## Troubleshooting Installation

### Issue: PyTorch CUDA version mismatch

**Error:**
```
NVIDIA GPU not detected, falling back to CPU
```

**Solution:**
1. Install CUDA Toolkit matching your PyTorch version
2. Verify NVIDIA drivers are installed:
   ```bash
   nvidia-smi
   ```
3. Reinstall PyTorch with correct CUDA version:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
   ```

---

### Issue: MediaPipe not working on Windows

**Error:**
```
ImportError: cannot import name 'mediapipe'
```

**Solution:**
```bash
pip install --upgrade mediapipe
# If still fails, try with constraints:
pip install mediapipe==0.10.0 --no-deps
pip install attrs dataclasses-json flatbuffers
```

---

### Issue: Webcam not detected

**Error:**
```
Error: Cannot open camera 0
```

**Solutions:**
1. Check if camera is plugged in and enabled
2. Try different camera ID:
   ```python
   # In config.py or main.py
   Config.CAMERA_ID = 1  # Try 1, 2, etc.
   ```
3. Restart computer (sometimes helps with camera driver issues)
4. Update camera drivers

---

### Issue: Out of Memory (OOM) error

**Error:**
```
CUDA out of memory. Tried to allocate X.00 GiB
```

**Solutions:**
1. Reduce batch size:
   ```python
   Config.BATCH_SIZE = 16  # Reduce from 32
   ```
2. Reduce sequence length:
   ```python
   Config.SEQUENCE_LENGTH = 15  # Reduce from 30
   ```
3. Use CPU instead:
   ```python
   Config.DEVICE = "cpu"
   ```
4. Enable model quantization:
   ```python
   Config.ENABLE_QUANTIZATION = True
   Config.QUANTIZATION_TYPE = "fp16"
   ```

---

### Issue: Very slow inference (low FPS)

**Solutions:**
1. Reduce frame resolution:
   ```python
   Config.VIDEO_FRAME_WIDTH = 480
   Config.VIDEO_FRAME_HEIGHT = 360
   ```
2. Use BiLSTM instead of Transformer:
   ```python
   Config.MODEL_TYPE = "bilstm"
   ```
3. Enable quantization:
   ```python
   Config.ENABLE_QUANTIZATION = True
   ```
4. Use CPU-GPU profiling:
   ```bash
   python -m torch.utils._python_dispatch --help
   ```

---

## Configuration for Different Hardware

### High-End (RTX 4090)
```python
# src/utils/config.py
Model.DEVICE = "cuda"
Config.SEQUENCE_LENGTH = 60
Config.MODEL_TYPE = "transformer"
Config.VIDEO_FRAME_WIDTH = 1280
Config.VIDEO_FRAME_HEIGHT = 720
```

### Mid-Range (RTX 3050/4050)
```python
Config.DEVICE = "cuda"
Config.SEQUENCE_LENGTH = 30
Config.MODEL_TYPE = "bilstm"
Config.VIDEO_FRAME_WIDTH = 640
Config.VIDEO_FRAME_HEIGHT = 480
Config.ENABLE_QUANTIZATION = True
```

### Budget (GTX 1660/CPU)
```python
Config.DEVICE = "cuda"  # or "cpu"
Config.SEQUENCE_LENGTH = 15
Config.MODEL_TYPE = "bilstm"
Config.VIDEO_FRAME_WIDTH = 480
Config.VIDEO_FRAME_HEIGHT = 360
Config.ENABLE_QUANTIZATION = True
Config.QUANTIZATION_TYPE = "int8"
```

### CPU-Only
```python
Config.DEVICE = "cpu"
Config.SEQUENCE_LENGTH = 20
Config.MODEL_TYPE = "bilstm"
Config.VIDEO_FRAME_WIDTH = 480
Config.VIDEO_FRAME_HEIGHT = 360
Config.BATCH_SIZE = 1
```

---

## Environment Variables

### Set GPU device ID (useful for multi-GPU systems)
```bash
# Windows
set CUDA_VISIBLE_DEVICES=0

# Linux/macOS
export CUDA_VISIBLE_DEVICES=0
```

### Set number of CPU threads
```bash
# Windows
set OMP_NUM_THREADS=4

# Linux/macOS
export OMP_NUM_THREADS=4
```

---

## Testing Installation

Run the test script to verify everything works:

```bash
# Basic functionality test
python -c "from src.main import ASLRecognitionPipeline; print('Installation successful!')"

# Full system test (requires webcam)
python src/main.py
# Press 'q' to quit after 5 seconds
```

---

## Uninstallation

### Remove virtual environment
```bash
# Windows
rmdir /s venv

# Linux/macOS
rm -rf venv
```

### Remove conda environment
```bash
conda remove --name asl-recognition --all
```

---

## Next Steps

After successful installation:

1. **Download WLASL300 dataset** (for training):
   ```
   https://github.com/dxli94/WLASL
   ```

2. **Start using the system**:
   ```bash
   python src/main.py
   ```

3. **Check the README** for usage instructions

4. **Review API documentation** in `docs/API.md`

---

## Support

- **GitHub Issues**: Report bugs or ask questions
- **Documentation**: See `docs/` folder
- **Examples**: Check `scripts/` folder for usage examples

---

**Last Updated**: April 2026
