# Development Guide

## Project Structure Deep Dive

### Directory Layout

```
fyp2/
├── src/                           # Source code
│   ├── modules/                   # Core pipeline components
│   │   ├── __init__.py
│   │   ├── video_capture.py       # Real-time video input
│   │   ├── keypoint_extraction.py # MediaPipe integration
│   │   ├── sequence_model.py      # DL models (BiLSTM, Transformer)
│   │   ├── text_to_speech.py      # TTS output module
│   │   └── ui.py                  # Visualization
│   ├── utils/                     # Utilities
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration management
│   │   └── preprocessing.py       # Data preprocessing
│   └── main.py                    # Pipeline orchestration
│
├── data/                          # Data storage
│   ├── raw/                       # WLASL300 videos (to download)
│   └── processed/                 # Extracted keypoints
│
├── checkpoints/                   # Trained model weights
├── docs/                          # Documentation
│   ├── API.md                     # API reference
│   ├── SETUP.md                   # Installation guide
│   └── DEV.md                     # This file
│
├── scripts/                       # (To be created)
│   ├── train_model.py            # Model training
│   ├── extract_keypoints.py      # Preprocess dataset
│   ├── evaluate_model.py         # Model evaluation
│   └── download_dataset.py       # WLASL300 downloader
│
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
└── README.md                      # Project overview
```

---

## Module Design Patterns

### 1. VideoCapture - Threading Pattern

Uses background thread for continuous frame capture without blocking inference:

```python
class VideoCapture:
    def __init__(self):
        self.thread = None
        self.is_running = False
        self.frame_buffer = deque(maxlen=2)
    
    def start(self):
        # Start background thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def _capture_loop(self):
        # Runs in background, continuously fills buffer
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_buffer.append(frame)
    
    def get_frame(self):
        # Main thread fetches latest frame without waiting
        if self.frame_buffer:
            return self.frame_buffer[-1]
        return None
```

**Benefits:**
- Non-blocking frame access
- Smooth 30 FPS capture
- Main thread remains responsive for inference

---

### 2. KeypointExtractor - Stateful Extraction

MediaPipe maintains internal state for smooth tracking:

```python
class KeypointExtractor:
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,        # Stateful (faster)
            model_complexity=1,             # Balanced
            smooth_landmarks=True           # Temporal smoothing
        )
    
    def extract_keypoints(self, frame):
        # Maintains state across frames
        results = self.holistic.process(frame)
        return self._results_to_dict(results)
```

**Why stateful?**
- Faster frame-to-frame processing
- Better temporal consistency
- Lower computational overhead

---

### 3. Preprocessing - Modular Pipeline

Composable preprocessing steps:

```python
# Each step is independent and reusable
sequence = KeypointPreprocessor.normalize_keypoints(sequence)
sequence = KeypointPreprocessor.scale_keypoints(sequence)
sequence = KeypointPreprocessor.temporal_smoothing(sequence)
sequence = KeypointPreprocessor.handle_missing_keypoints(sequence)
sequence = KeypointPreprocessor.pad_or_truncate_sequence(sequence, 30)
```

**Advantages:**
- Easy to test individual steps
- Can be reordered or skipped as needed
- Fits different preprocessing pipelines

---

### 4. Model Inference - Pipeline Abstraction

High-level inference interface hiding model details:

```python
class SignClassificationPipeline:
    def __init__(self, model_type: str = "bilstm"):
        # Can swap models without changing interface
        if model_type == "bilstm":
            self.model = BiLSTMSignClassifier(...)
        elif model_type == "transformer":
            self.model = LightweightTransformerClassifier(...)
    
    def predict(self, sequence):
        # Unified interface
        with torch.no_grad():
            logits = self.model(sequence)
            return torch.max(probabilities)
```

**Benefits:**
- Easy model switching
- Consistent inference API
- Device abstraction (CUDA/CPU)

---

### 5. Real-time Pipeline - Orchestration

Main pipeline coordinates all modules with timing:

```python
class ASLRecognitionPipeline:
    def run(self):
        frame_buffer = deque(maxlen=30)
        
        while True:
            # Non-blocking capture
            frame = self.video_capture.get_frame()
            if frame is None:
                continue
            
            # Fast keypoint extraction
            keypoints = self.keypoint_extractor.extract_keypoints(frame)
            
            # Preprocess
            features = self.preprocess_keypoints(keypoints)
            frame_buffer.append(features)
            
            # Predict when buffer full
            if len(frame_buffer) == 30:
                prediction = self.model_pipeline.predict(frame_buffer)
                self.tts_engine.speak(prediction)
                frame_buffer.clear()
            
            # Render UI
            display = self.ui.draw_prediction(frame, prediction)
            cv2.imshow("ASL", display)
```

---

## Data Flow Diagram

```
┌─────────────────┐
│   Webcam Input  │
└────────┬────────┘
         │ (30 FPS, raw BGR frames)
         ↓
┌─────────────────────────────┐
│ VideoCapture (Background)   │
│ - Threading                 │
│ - Frame buffering           │
└────────┬────────────────────┘
         │ (latest frame)
         ↓
┌─────────────────────────────┐
│ KeypointExtractor (MediaPipe)
│ - 42 hand landmarks         │
│ - Normalized [0, 1]         │
│ - Confidence scores         │
└────────┬────────────────────┘
         │ (keypoint dict)
         ↓
┌─────────────────────────────┐
│ Preprocessing               │
│ - Normalization             │
│ - Smoothing                 │
│ - Imputation                │
└────────┬────────────────────┘
         │ (feature vector)
         ↓
┌─────────────────────────────┐
│ Keypoint Buffer (30 frames) │
│ - Temporal sequence         │
│ - Shape: (30, 126)          │
└────────┬────────────────────┘
         │ (full sequence)
         ↓
┌──────────────────────────────┐
│ Model Inference              │
│ - BiLSTM / Transformer       │
│ - Forward pass               │
│ - Softmax probabilities      │
└────────┬─────────────────────┘
         │ (class_id, confidence)
         ↓
┌──────────────────────────────┐
│ Post-Processing              │
│ - Sign vocabulary lookup     │
│ - Top-K filtering            │
│ - Confidence threshold       │
└────────┬─────────────────────┘
         │ (sign name, confidence)
         ↓
    ┌────┴────┐
    │          │
    ↓          ↓
┌───────┐  ┌──────────┐
│  TTS  │  │ UI Render│
│ Audio │  │ Display  │
└───────┘  └──────────┘
```

---

## Latency Optimization Strategies

### 1. Batch Processing (Not Applicable Here)
- Single-sign recognition: 1 inference per 30 frames
- Trade-off: Latency vs throughput

### 2. Model Quantization
```python
# FP32 (standard) → FP16 (half precision)
model = model.half()  # 2x speed, minimal accuracy loss

# INT8 (aggressive) - requires calibration
# 4x speed, might impact accuracy
```

### 3. Operator Fusion
```python
# PyTorch graph optimization
model = torch.jit.script(model)
model = torch.jit.optimize_for_inference(model)
```

### 4. Async Processing
```python
# Non-blocking TTS
self.tts_engine.enqueue_speech(sign_name)
# Main thread continues inference
```

### 5. Multi-threading
```python
# Video capture in background thread
# TTS in separate thread
# Main thread handles inference and UI
```

---

## Adding New Features

### Example 1: Add Fingerspelling Support

```python
# Step 1: Create new module
src/modules/fingerspelling_recognizer.py

class FingerSpellingRecognizer:
    def __init__(self):
        # Detect individual finger configurations
        self.hand_classifier = FingerConfigClassifier()
    
    def recognize_letter(self, hand_landmarks):
        # Single frame prediction
        finger_config = self.hand_classifier.predict(hand_landmarks)
        return self._finger_config_to_letter(finger_config)

# Step 2: Integrate into pipeline
class ASLRecognitionPipeline:
    def __init__(self):
        ...
        self.fingerspelling = FingerSpellingRecognizer()
    
    def run(self):
        # Detect mode (sign vs fingerspelling)
        if self._detect_spelling_mode(keypoints):
            letter = self.fingerspelling.recognize_letter(hand_keypoints)
            self.tts_engine.speak(letter)
```

---

### Example 2: Add Continuous Signing Support

```python
# Use CTC loss for alignment
class CTCASLTranslator(nn.Module):
    def __init__(self):
        self.encoder = nn.LSTM(126, 256, bidirectional=True)
        self.ctc_loss = nn.CTCLoss()
    
    def forward(self, sequences, targets, input_lengths, target_lengths):
        # Input: continuous video frames
        # Output: aligned sign sequences
        encoded = self.encoder(sequences)
        loss = self.ctc_loss(encoded, targets, input_lengths, target_lengths)
        return loss
```

---

## Testing Strategy

### Unit Tests
```python
# tests/test_preprocessing.py
def test_keypoint_normalization():
    kpts = np.random.randn(42, 3)
    normalized = KeypointPreprocessor.normalize_keypoints(kpts)
    assert np.allclose(normalized.mean(), 0, atol=0.1)
```

### Integration Tests
```python
# tests/test_pipeline.py
def test_end_to_end():
    pipeline = ASLRecognitionPipeline()
    frame = cv2.imread("test_image.png")
    prediction = pipeline.model_pipeline.predict(frame)
    assert prediction[1] > 0  # Confidence > 0
```

### Performance Tests
```python
# tests/test_latency.py
def test_inference_latency():
    pipeline = ASLRecognitionPipeline()
    sequence = np.random.randn(30, 126)
    
    start = time.perf_counter()
    _, _ = pipeline.model_pipeline.predict(sequence)
    latency = (time.perf_counter() - start) * 1000
    
    assert latency < 150  # Must be < 150ms
```

---

## Performance Profiling

### Profile inference latency
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
prediction = model.predict(sequence)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

### GPU memory usage
```python
import torch

# Before
print(torch.cuda.memory_allocated())

# Your code
prediction = model.predict(sequence)

# After
print(torch.cuda.memory_allocated())
print(torch.cuda.max_memory_allocated())
```

---

## Code Style Guidelines

### Naming Conventions
```python
# Classes: PascalCase
class VideoCapture:
    pass

# Functions/Methods: snake_case
def extract_keypoints(frame):
    pass

# Constants: UPPER_SNAKE_CASE
SEQUENCE_LENGTH = 30

# Private: leading underscore
def _private_method(self):
    pass
```

### Documentation
```python
def extract_keypoints(self, frame: np.ndarray) -> Optional[dict]:
    """
    Extract keypoints from a frame.
    
    Args:
        frame: Input video frame (BGR format from OpenCV)
    
    Returns:
        Dictionary containing extracted keypoints or None if extraction fails
    
    Raises:
        ValueError: If frame is empty
    """
    pass
```

---

## Common Gotchas

### 1. MediaPipe RGB vs BGR
```python
# WRONG - MediaPipe expects RGB
results = holistic.process(frame)  # OpenCV gives BGR!

# CORRECT
frame_rgb = frame[:, :, ::-1]
results = holistic.process(frame_rgb)
```

### 2. PyTorch Tensor Device
```python
# WRONG - Mixing CPU and GPU tensors
logits = model(gpu_tensor) + cpu_tensor  # Error!

# CORRECT
logits = model(gpu_tensor) + cpu_tensor.to(device)
```

### 3. Keypoint Normalization Order
```python
# CORRECT order matters!
keypoints = KeypointPreprocessor.normalize_keypoints(kpts)
keypoints = KeypointPreprocessor.scale_keypoints(keypoints)
keypoints = KeypointPreprocessor.temporal_smoothing(keypoints)
```

---

## Contributing Guidelines

1. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes** with clear commits

3. **Add tests** for new functionality

4. **Update documentation** in docstrings and README

5. **Submit pull request** with description

---

**Last Updated**: April 2026
