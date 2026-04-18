# API Reference

## Core Classes

### VideoCapture

Real-time video acquisition from webcam.

```python
from src.modules.video_capture import VideoCapture

# Initialize
capture = VideoCapture(
    camera_id=0,
    frame_width=640,
    frame_height=480,
    fps=30
)

# Start in background thread
capture.start()

# Get latest frame
frame_data = capture.get_frame()
if frame_data:
    frame, frame_id = frame_data
    # Process frame...

# Stop capture
capture.stop()
```

**Methods:**
- `start()` → bool: Start video capture thread
- `get_frame()` → Optional[Tuple]: Get latest frame and frame ID
- `stop()`: Stop capture and cleanup
- `get_fps()` → float: Get capture stream FPS

---

### KeypointExtractor

Extract hand and body landmarks using MediaPipe.

```python
from src.modules.keypoint_extraction import KeypointExtractor

extractor = KeypointExtractor(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Extract keypoints from frame
keypoints = extractor.extract_keypoints(frame)

if keypoints:
    hand_landmarks = extractor.get_hand_landmarks(keypoints)
    pose_landmarks = extractor.get_upper_body_pose(keypoints)

extractor.release()
```

**Methods:**
- `extract_keypoints(frame)` → Optional[dict]: Extract all landmarks
- `get_hand_landmarks(keypoints)` → Optional[np.ndarray]: Get hand keypoints (42, 3)
- `get_upper_body_pose(keypoints)` → Optional[np.ndarray]: Get pose keypoints
- `release()`: Cleanup MediaPipe resources

**Keypoint Dictionary Structure:**
```python
{
    'left_hand': np.ndarray of shape (21, 3),    # x, y, z
    'right_hand': np.ndarray of shape (21, 3),
    'pose': np.ndarray of shape (33, 3),
    'face': np.ndarray of shape (468, 3),
    'timestamp': None
}
```

---

### KeypointPreprocessor

Preprocessing utilities for keypoint sequences.

```python
from src.utils.preprocessing import KeypointPreprocessor

# Normalize keypoints
normalized = KeypointPreprocessor.normalize_keypoints(keypoints)

# Scale to [0, 1]
scaled = KeypointPreprocessor.scale_keypoints(normalized)

# Smooth temporal noise
smoothed = KeypointPreprocessor.temporal_smoothing(
    sequence,
    window_length=5,
    polyorder=2
)

# Handle missing keypoints
interpolated = KeypointPreprocessor.handle_missing_keypoints(
    sequence,
    confidence_threshold=0.3
)

# Pad/truncate to fixed length
padded = KeypointPreprocessor.pad_or_truncate_sequence(
    sequence,
    target_length=30,
    pad_value=0.0
)

# Data augmentation
augmented = KeypointPreprocessor.data_augmentation_temporal_scale(
    sequence,
    scale_range=(0.9, 1.1)
)
```

**Static Methods:**
- `normalize_keypoints()`: Center keypoints
- `scale_keypoints()`: Scale to fixed range
- `temporal_smoothing()`: Apply Savitzky-Golay filter
- `handle_missing_keypoints()`: Interpolate low-confidence keypoints
- `pad_or_truncate_sequence()`: Fixed-length sequences
- `data_augmentation_temporal_scale()`: Speed variation augmentation

---

### BiLSTMSignClassifier

Bidirectional LSTM with attention for sign classification.

```python
from src.modules.sequence_model import BiLSTMSignClassifier
import torch

model = BiLSTMSignClassifier(
    input_dim=126,      # 42 landmarks * 3
    hidden_dim=256,
    num_classes=300,
    num_layers=2,
    dropout=0.3
)

# Forward pass
input_tensor = torch.randn(batch_size, seq_len, 126)
logits, attention_weights = model(input_tensor)

# Get predictions
probs = torch.softmax(logits, dim=1)
confidence, predicted_class = torch.max(probs, dim=1)
```

**Architecture:**
- Input projection
- Bidirectional LSTM (2 layers)
- Multi-head attention (4 heads)
- Classification head (2 FC layers)

---

### LightweightTransformerClassifier

Lightweight Transformer for temporal modeling.

```python
from src.modules.sequence_model import LightweightTransformerClassifier

model = LightweightTransformerClassifier(
    input_dim=126,
    hidden_dim=256,
    num_classes=300,
    num_heads=4,
    num_layers=2,
    dropout=0.3
)

logits = model(input_tensor)  # (batch, num_classes)
```

**Architecture:**
- Input projection
- Transformer encoder (2 layers, 4 heads)
- Classification head

---

### SignClassificationPipeline

High-level interface for model inference.

```python
from src.modules.sequence_model import SignClassificationPipeline
import numpy as np

pipeline = SignClassificationPipeline(
    model_type="bilstm",  # or "transformer"
    num_classes=300,
    device="cuda"
)

# Inference
keypoint_sequence = np.random.randn(30, 126).astype(np.float32)
predicted_id, confidence = pipeline.predict(keypoint_sequence)

# Get sign name
from src.utils.config import Config
sign_name = Config.get_sign_name(predicted_id)
print(f"{sign_name}: {confidence:.2%}")
```

**Methods:**
- `predict(keypoint_sequence)` → Tuple[int, float]: Predict class and confidence

---

### TextToSpeechEngine

Text-to-speech output with async queue.

```python
from src.modules.text_to_speech import TextToSpeechEngine

tts = TextToSpeechEngine(use_gpt=False)  # pyttsx3 (offline)
tts.start()

# Immediate speech
tts.speak("HELLO")

# Queued speech (non-blocking)
tts.enqueue_speech("GOODBYE")

tts.stop()
```

**Methods:**
- `start()`: Start TTS worker thread
- `speak(text)`: Immediate speech output
- `enqueue_speech(text)`: Queue for async speech
- `stop()`: Stop and cleanup

---

### RealtimeUI

Real-time visualization utilities.

```python
from src.modules.ui import RealtimeUI
import cv2

ui = RealtimeUI(frame_width=640, frame_height=480)

# Draw keypoints
display_frame = ui.draw_keypoints(frame, keypoints, keypoint_type="hand")

# Draw prediction
display_frame = ui.draw_prediction(
    display_frame,
    prediction_text="HELLO",
    confidence=0.95,
    top_k=[("HELLO", 0.95), ("HI", 0.03), ("HEY", 0.02)]
)

# Draw FPS
display_frame = ui.draw_fps(display_frame, fps=28.5)

# Show frame
cv2.imshow("ASL Recognition", display_frame)
cv2.waitKey(1)
```

**Methods:**
- `draw_keypoints()`: Overlay detected landmarks
- `draw_prediction()`: Show sign, confidence, top-k
- `draw_fps()`: FPS counter
- `draw_status()`: Status message

---

### ASLRecognitionPipeline

Complete end-to-end pipeline.

```python
from src.main import ASLRecognitionPipeline

# Initialize
pipeline = ASLRecognitionPipeline(
    model_checkpoint="path/to/model.pth"  # Optional
)

# Run continuous recognition
pipeline.run(headless=False)
```

**Methods:**
- `run(headless=False)`: Start real-time pipeline
- `cleanup()`: Release all resources

---

## Configuration

```python
from src.utils.config import Config

# Video settings
Config.CAMERA_ID = 0
Config.VIDEO_FRAME_WIDTH = 640
Config.VIDEO_FRAME_HEIGHT = 480
Config.VIDEO_FPS = 30

# Model settings
Config.MODEL_TYPE = "bilstm"
Config.SEQUENCE_LENGTH = 30
Config.NUM_CLASSES = 300

# Inference
Config.DEVICE = "cuda"
Config.MAX_LATENCY_MS = 150
Config.ENABLE_QUANTIZATION = True

# Get sign information
sign_name = Config.get_sign_name(sign_id=0)      # "ABOUT"
sign_id = Config.get_sign_id("HELLO")            # Returns ID or -1

# Export config as dict
config_dict = Config.to_dict()
```

---

## Data Shapes

### Keypoint Array Shapes

| Component | Shape | Description |
|-----------|-------|-------------|
| Single hand | (21, 3) | 21 landmarks (x, y, z) |
| Both hands | (42, 3) | Left + right hands |
| Upper body pose | (6, 3) | Shoulders, elbows, wrists |
| Full sequence | (seq_len, 126) | Flattened hand landmarks |

### Model Input/Output

| Stage | Shape | Notes |
|-------|-------|-------|
| Model input | (batch, seq_len, 126) | Batched sequences |
| Logits (output) | (batch, 300) | Per-class scores |
| Probabilities | (batch, 300) | Softmax output |

---

## Error Handling

```python
try:
    keypoints = extractor.extract_keypoints(frame)
    if keypoints is None:
        print("No hands detected in frame")
    
    hand_landmarks = extractor.get_hand_landmarks(keypoints)
    if hand_landmarks is None:
        print("Insufficient keypoint confidence")
    
except Exception as e:
    print(f"Extraction error: {e}")
finally:
    extractor.release()
```

---

**Last Updated**: April 2026
