"""
Configuration Module
Central configuration management for the ASL recognition system.
"""

import json
from pathlib import Path
from typing import Dict, Any


class Config:
    """
    Central configuration for ASL recognition pipeline.
    """
    
    # Project structure
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    DOCS_DIR = PROJECT_ROOT / "docs"
    LABEL_MAP_PATH = RAW_DATA_DIR / "label_map_300.json"
    
    # Video capture settings
    CAMERA_ID = 0
    VIDEO_FRAME_WIDTH = 640
    VIDEO_FRAME_HEIGHT = 480
    VIDEO_FPS = 30
    
    # MediaPipe settings
    MEDIAPIPE_DETECTION_CONFIDENCE = 0.7
    MEDIAPIPE_TRACKING_CONFIDENCE = 0.5
    
    # Keypoint extraction
    SEQUENCE_LENGTH = 30  # Number of frames per sign
    INPUT_FEATURE_DIM = 126  # 42 hand landmarks * 3 (x, y, z)
    
    # Model settings
    MODEL_TYPE = "bilstm"  # "bilstm" or "transformer"
    MODEL_HIDDEN_DIM = 256
    NUM_CLASSES = 300  # WLASL300 vocabulary size
    NUM_LAYERS = 2
    NUM_HEADS = 4  # For transformer
    DROPOUT = 0.3
    
    # Training settings
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    OPTIMIZER = "adamw"
    WEIGHT_DECAY = 0.0001
    
    # Data augmentation
    USE_DATA_AUGMENTATION = True
    TEMPORAL_SCALE_RANGE = (0.9, 1.1)
    ROTATION_RANGE = (-10, 10)  # degrees
    JITTER_RANGE = (0.95, 1.05)
    
    # Preprocessing
    NORMALIZE_KEYPOINTS = True
    SCALE_KEYPOINTS = True
    TEMPORAL_SMOOTHING = True
    SMOOTH_WINDOW_LENGTH = 5
    SMOOTH_POLY_ORDER = 2
    CONFIDENCE_THRESHOLD = 0.3
    
    # Inference settings
    DEVICE = "cpu"  # Resolved dynamically below
    INFERENCE_FPS_TARGET = 30
    MAX_LATENCY_MS = 150  # Total latency target in milliseconds
    ENABLE_QUANTIZATION = True
    QUANTIZATION_TYPE = "fp16"  # "fp16" or "int8"
    
    # Output settings
    ENABLE_TTS = True
    TTS_BACKEND = "pyttsx3"  # "pyttsx3" or "gtts"
    TTS_SPEED = 150
    DISPLAY_TOP_K = 3
    DISPLAY_CONFIDENCE_THRESHOLD = 0.5
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = PROJECT_ROOT / "logs" / "asl_system.log"
    
    # ASL Sign Vocabulary is loaded from the generated WLASL300 label map when available.
    ASL_VOCABULARY: Dict[int, str] = {}

    @classmethod
    def resolve_device(cls) -> str:
        """Prefer CUDA when a CUDA-enabled PyTorch install is available."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    @classmethod
    def load_vocabulary(cls) -> Dict[int, str]:
        """Load the real WLASL300 label map if it exists, else use an empty fallback."""
        if cls.LABEL_MAP_PATH.exists():
            with cls.LABEL_MAP_PATH.open("r", encoding="utf-8") as handle:
                raw_map = json.load(handle)
            return {int(idx): str(gloss).upper() for idx, gloss in raw_map.items()}
        return {}
    
    @classmethod
    def get_sign_name(cls, sign_id: int) -> str:
        """Get sign name from ID."""
        return cls.ASL_VOCABULARY.get(sign_id, f"UNKNOWN_{sign_id}")
    
    @classmethod
    def get_sign_id(cls, sign_name: str) -> int:
        """Get sign ID from name."""
        for sign_id, name in cls.ASL_VOCABULARY.items():
            if name == sign_name.upper():
                return sign_id
        return -1
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        for directory in [cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, 
                         cls.CHECKPOINTS_DIR, cls.DOCS_DIR, cls.LOG_FILE.parent]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {key: getattr(cls, key) for key in dir(cls) 
                if not key.startswith('_') and key.isupper()}


# Create directories on import
Config.create_directories()
Config.DEVICE = Config.resolve_device()
Config.ASL_VOCABULARY = Config.load_vocabulary()
