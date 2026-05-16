"""
Main entry point for ASL Recognition System.
Orchestrates video capture, keypoint extraction, model inference, and TTS output.
"""

import cv2
import numpy as np
import time
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Optional, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.video.video_capture import VideoCapture
from src.preprocessing.keypoint_extraction import KeypointExtractor
from src.inference.sequence_model import SignClassificationPipeline
from src.audio.text_to_speech import TextToSpeechEngine
from src.video.ui import RealtimeUI
from src.utils.config import Config
from src.preprocessing.preprocessing import KeypointPreprocessor

# Import new 300-sign inference engine (WLASL300 production)
try:
    from scripts.inference_bridge import RealtimeInferenceEngine
    INFERENCE_ENGINE_AVAILABLE = True
except ImportError:
    INFERENCE_ENGINE_AVAILABLE = False


class ASLRecognitionPipeline:
    """
    End-to-end ASL recognition pipeline combining all modules.
    Handles real-time capture, processing, inference, and output.
    """
    
    WLASL300_DEFAULT_MODEL = "models/production/asl_wlasl300_realtime.pt"
    WLASL300_LEGACY_MODEL_NAMES = {
        "asl_model_300_pose_face_balaug_hardened_v1.pt",
        "asl_wlasl300_realtime.pt",
    }
    STABILIZATION_WINDOW = Config.REALTIME_STABILIZATION_WINDOW
    STABILIZATION_MIN_COUNT = Config.REALTIME_STABILIZATION_MIN_COUNT
    CONFIDENCE_SQUELCH = Config.REALTIME_BASE_CONFIDENCE_SQUELCH
    HAND_MISSING_GRACE_FRAMES = 10

    def __init__(self, model_checkpoint: Optional[str] = None, use_wlasl300: bool = False):
        """
        Initialize the complete ASL recognition pipeline.
        
        Args:
            model_checkpoint: Path to saved model checkpoint (optional)
            use_wlasl300: If True, use the new 300-sign WLASL model with RealtimeInferenceEngine
        """
        Config.create_directories()
        
        print("Initializing ASL Recognition Pipeline...")
        
        self.use_wlasl300 = use_wlasl300 and INFERENCE_ENGINE_AVAILABLE
        self.inference_engine = None
        self.model_pipeline = None
        
        # Initialize video and keypoint extraction (always needed)
        self.video_capture = VideoCapture(
            camera_id=Config.CAMERA_ID,
            frame_width=Config.VIDEO_FRAME_WIDTH,
            frame_height=Config.VIDEO_FRAME_HEIGHT,
            fps=Config.VIDEO_FPS
        )
        
        self.keypoint_extractor = KeypointExtractor(
            confidence_threshold=Config.MEDIAPIPE_DETECTION_CONFIDENCE
        )
        
        # Initialize inference engine based on mode
        if self.use_wlasl300:
            print("[INFO] Initializing WLASL300 RealtimeInferenceEngine...")
            try:
                model_path = self._resolve_wlasl300_model_path(model_checkpoint)
                label_map_path = "data/raw/label_map_300.json"
                
                self.inference_engine = RealtimeInferenceEngine(
                    model_path=model_path,
                    label_map_path=label_map_path,
                    device=Config.DEVICE,
                    prediction_cooldown=0.0,
                )
                print(f"[OK] WLASL300 engine loaded: {model_path}")
            except Exception as e:
                print(f"[WARNING] Could not load WLASL300 engine: {e}")
                print("[INFO] Falling back to standard model pipeline")
                self.use_wlasl300 = False
        
        # Use standard model pipeline if not using WLASL300
        if not self.use_wlasl300:
            self.model_pipeline = SignClassificationPipeline(
                model_type=Config.MODEL_TYPE,
                num_classes=Config.NUM_CLASSES,
                device=Config.DEVICE
            )
            
            if model_checkpoint and model_checkpoint.lower() != "none":
                self.load_model(model_checkpoint)
        
        self.tts_engine = TextToSpeechEngine(use_gpt=(Config.TTS_BACKEND == "gtts"))
        
        self.ui = RealtimeUI(
            frame_width=Config.VIDEO_FRAME_WIDTH,
            frame_height=Config.VIDEO_FRAME_HEIGHT
        )
        
        # State management
        self.keypoint_buffer = deque(maxlen=Config.SEQUENCE_LENGTH)
        self.fps_history = deque(maxlen=30)
        self.last_prediction = None
        self.last_prediction_time = 0
        self.prediction_cooldown = 0.0 if self.use_wlasl300 else 0.5
        self.prediction_history = deque(maxlen=self.STABILIZATION_WINDOW)
        self.peak_prediction_history = deque(maxlen=Config.REALTIME_PEAK_HISTORY_WINDOW)
        self.latest_top_k: List[Tuple[str, float]] = []
        self.last_spoken_prediction: Optional[str] = None
        self.current_display_text = "..."
        self.hand_missing_grace_remaining = self.HAND_MISSING_GRACE_FRAMES
        self.last_motion_delta = 0.0
        
        print("Pipeline initialized successfully!")

    def _resolve_wlasl300_model_path(self, model_checkpoint: Optional[str]) -> str:
        if not model_checkpoint or model_checkpoint.lower() == "none":
            return self.WLASL300_DEFAULT_MODEL

        checkpoint_name = Path(model_checkpoint).name.lower()
        if checkpoint_name == "bilstm_final.pt":
            return self.WLASL300_DEFAULT_MODEL
        if checkpoint_name in self.WLASL300_LEGACY_MODEL_NAMES:
            return self.WLASL300_DEFAULT_MODEL
        return model_checkpoint

    def _update_stabilized_prediction(
        self,
        top_predictions: Optional[List[Tuple[str, float]]],
        motion_delta: float = 0.0,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        if not top_predictions:
            self.latest_top_k = []
            self.prediction_history.append(None)
            self.peak_prediction_history.append(None)
            self.current_display_text = "..."
            return ("...", 0.0, [])

        self.latest_top_k = top_predictions[: Config.DISPLAY_TOP_K]
        top_sign, top_confidence = top_predictions[0]
        candidate = top_sign if self._should_accept_candidate(top_predictions, motion_delta) else None
        self.prediction_history.append(candidate)
        peak_candidate = top_sign if self._is_peak_candidate(top_predictions, motion_delta) else None
        self.peak_prediction_history.append(peak_candidate)
        if candidate is None:
            peak_sign = self._get_peak_stabilized_sign()
            if peak_sign is not None:
                self.current_display_text = peak_sign
                return (peak_sign, top_confidence, self.latest_top_k)
            self.current_display_text = "..."
            return ("...", top_confidence, self.latest_top_k)

        counts = Counter(sign for sign in self.prediction_history if sign)
        stabilized_sign = "..."
        stabilized_count = 0
        if counts:
            stabilized_sign, stabilized_count = counts.most_common(1)[0]

        if stabilized_count < self.STABILIZATION_MIN_COUNT:
            peak_sign = self._get_peak_stabilized_sign()
            if peak_sign is not None:
                self.current_display_text = peak_sign
                return (peak_sign, top_confidence, self.latest_top_k)
            self.current_display_text = "..."
            return ("...", top_confidence, self.latest_top_k)

        self.current_display_text = stabilized_sign
        stabilized_confidence = 0.0
        for sign_name, confidence in top_predictions:
            if sign_name == stabilized_sign:
                stabilized_confidence = confidence
                break
        return (stabilized_sign, stabilized_confidence, self.latest_top_k)

    def _should_accept_candidate(
        self,
        top_predictions: List[Tuple[str, float]],
        motion_delta: float,
    ) -> bool:
        if not top_predictions:
            return False

        top_sign, top_confidence = top_predictions[0]
        runner_up_confidence = top_predictions[1][1] if len(top_predictions) > 1 else 0.0
        confidence_margin = top_confidence - runner_up_confidence

        min_confidence = Config.REALTIME_SIGN_CONFIDENCE_OVERRIDES.get(
            top_sign,
            self.CONFIDENCE_SQUELCH,
        )
        adaptive_floor = max(
            Config.REALTIME_ADAPTIVE_CONFIDENCE_FLOOR,
            min_confidence - 0.10,
        )

        required_motion = Config.REALTIME_SIGN_MOTION_REQUIREMENTS.get(top_sign)
        if required_motion is not None and motion_delta < required_motion:
            return False

        rivals = Config.REALTIME_CONFUSION_PAIRS.get(top_sign, set())
        if rivals:
            for rival_sign, rival_confidence in top_predictions[1:]:
                if rival_sign in rivals and (top_confidence - rival_confidence) < 0.08:
                    return False

        if top_confidence >= min_confidence:
            return True

        return top_confidence >= adaptive_floor and confidence_margin >= Config.REALTIME_ADAPTIVE_MARGIN

    def _is_peak_candidate(
        self,
        top_predictions: List[Tuple[str, float]],
        motion_delta: float,
    ) -> bool:
        if not top_predictions:
            return False

        top_sign, top_confidence = top_predictions[0]
        peak_threshold = Config.REALTIME_PEAK_SIGN_CONFIDENCE_OVERRIDES.get(top_sign)
        if peak_threshold is None or top_confidence < peak_threshold:
            return False

        runner_up_confidence = top_predictions[1][1] if len(top_predictions) > 1 else 0.0
        if (top_confidence - runner_up_confidence) < Config.REALTIME_PEAK_MARGIN:
            return False

        required_motion = Config.REALTIME_SIGN_MOTION_REQUIREMENTS.get(top_sign)
        if required_motion is not None and motion_delta < required_motion:
            return False

        rivals = Config.REALTIME_CONFUSION_PAIRS.get(top_sign, set())
        for rival_sign, rival_confidence in top_predictions[1:]:
            if rival_sign in rivals and (top_confidence - rival_confidence) < 0.08:
                return False

        return True

    def _get_peak_stabilized_sign(self) -> Optional[str]:
        counts = Counter(sign for sign in self.peak_prediction_history if sign)
        if not counts:
            return None

        peak_sign, peak_count = counts.most_common(1)[0]
        if peak_count >= Config.REALTIME_PEAK_MIN_COUNT:
            return peak_sign
        return None

    def _handle_tts_transition(self, prediction_text: str) -> None:
        if not Config.ENABLE_TTS:
            return
        if prediction_text == "...":
            self.last_spoken_prediction = None
            return
        if prediction_text == self.last_spoken_prediction:
            return
        self.tts_engine.enqueue_speech(prediction_text)
        self.last_spoken_prediction = prediction_text

    def _reset_realtime_prediction_state(self) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Clear cached buffers/predictions so idle frames do not hallucinate signs."""
        self.prediction_history.clear()
        self.peak_prediction_history.clear()
        self.latest_top_k = []
        self.current_display_text = "..."
        self.last_prediction = ("...", 0.0, [])
        self.last_prediction_time = 0.0
        self.hand_missing_grace_remaining = self.HAND_MISSING_GRACE_FRAMES
        self.last_motion_delta = 0.0
        if self.use_wlasl300 and self.inference_engine is not None:
            self.inference_engine.reset()
        else:
            self.keypoint_buffer.clear()
        self._handle_tts_transition("...")
        return self.last_prediction
    
    def load_model(self, checkpoint_path: str):
        """Load trained model from checkpoint."""
        try:
            import torch
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
                return False
            
            print(f"Loading model from {checkpoint_path}...")
            # Use CPU map_location for safety, device will be set when needed
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle both checkpoint dict and raw state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model_pipeline.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Assume it's a state dict directly
                self.model_pipeline.model.load_state_dict(checkpoint)
            
            print(f"[OK] Model loaded successfully from {checkpoint_path}")
            return True
        except Exception as e:
            print(f"[WARNING] Could not load model checkpoint: {e}")
            return False
    
    def preprocess_keypoints(self, keypoints: dict) -> Optional[np.ndarray]:
        """
        Preprocess extracted keypoints for model inference.
        
        Args:
            keypoints: Dictionary from KeypointExtractor
        
        Returns:
            Preprocessed keypoint array or None
        """
        if self.use_wlasl300 and self.inference_engine is not None:
            return self.inference_engine.bridge.preprocess_webcam_frame(keypoints)

        # Extract hand landmarks
        hand_landmarks = self.keypoint_extractor.get_hand_landmarks(keypoints)
        if hand_landmarks is None:
            return None
        
        # Flatten to feature vector
        features = hand_landmarks.flatten()  # Shape: (126,)
        
        # Apply preprocessing
        if Config.NORMALIZE_KEYPOINTS:
            features = KeypointPreprocessor.normalize_keypoints(features.reshape(-1, 3)).flatten()
        
        if Config.SCALE_KEYPOINTS:
            features = KeypointPreprocessor.scale_keypoints(features.reshape(-1, 3)).flatten()
        
        return features
    
    def predict_sign(self, keypoints: Optional[dict] = None) -> Optional[Tuple[str, float, List[Tuple[str, float]]]]:
        """
        Make prediction from accumulated keypoint buffer.
        Uses RealtimeInferenceEngine for WLASL300 if available, else standard pipeline.
        
        Returns:
            Tuple of (sign_name, confidence, top_k_predictions) or None
        """
        # WLASL300 path: use inference engine directly with raw keypoints
        if self.use_wlasl300 and self.inference_engine is not None:
            if keypoints is None:
                stabilized_prediction = self._update_stabilized_prediction(None, motion_delta=0.0)
                self.last_prediction = stabilized_prediction
                return stabilized_prediction
            try:
                top5_predictions = self.inference_engine.process_frame(keypoints)
                self.last_motion_delta = self.inference_engine.last_motion_delta
                stabilized_prediction = self._update_stabilized_prediction(
                    top5_predictions,
                    motion_delta=self.last_motion_delta,
                )
                self.last_prediction = stabilized_prediction
                self.last_prediction_time = time.time()
                self._handle_tts_transition(stabilized_prediction[0])
                return self.last_prediction
            except Exception as e:
                print(f"Inference engine error: {e}")
                return self.last_prediction

        # Check cooldown for the legacy path only.
        if time.time() - self.last_prediction_time < self.prediction_cooldown:
            return self.last_prediction
        
        # Standard path: Use accumulated buffer with standard model
        if len(self.keypoint_buffer) < Config.SEQUENCE_LENGTH:
            return None
        
        # Prepare sequence
        sequence = np.array(self.keypoint_buffer)
        
        # Apply temporal smoothing
        if Config.TEMPORAL_SMOOTHING:
            sequence = KeypointPreprocessor.temporal_smoothing(
                sequence,
                window_length=Config.SMOOTH_WINDOW_LENGTH,
                polyorder=Config.SMOOTH_POLY_ORDER
            )
        
        # Model inference
        try:
            predicted_id, confidence = self.model_pipeline.predict(sequence)
            
            # Get sign name
            sign_name = Config.get_sign_name(predicted_id)
            
            # Generate top-k predictions (placeholder)
            top_k = [(sign_name, confidence)]
            
            # Update prediction cache
            self.last_prediction = (sign_name, confidence, top_k)
            self.last_prediction_time = time.time()
            
            # Trigger TTS if confidence is high
            if confidence >= Config.DISPLAY_CONFIDENCE_THRESHOLD and Config.ENABLE_TTS:
                self.tts_engine.enqueue_speech(sign_name)
            
            return self.last_prediction
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def run(self, headless: bool = False):
        """
        Run the complete ASL recognition pipeline.
        
        Args:
            headless: If True, run without display (for testing)
        """
        print("\nStarting ASL Recognition Pipeline...")
        print("Press 'q' to quit, 's' to toggle showing keypoints")
        
        # Start components
        if not self.video_capture.start():
            print("Error: Could not start video capture")
            return
        
        if Config.ENABLE_TTS:
            self.tts_engine.start()
        
        show_keypoints = False
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Capture frame
                frame_data = self.video_capture.get_frame()
                if frame_data is None:
                    continue
                
                frame, frame_id = frame_data
                frame_count += 1
                
                # Extract keypoints
                keypoints = self.keypoint_extractor.extract_keypoints(frame)
                hands_present = bool(
                    keypoints is not None
                    and (
                        np.any(keypoints.get("left_hand", 0.0))
                        or np.any(keypoints.get("right_hand", 0.0))
                    )
                )

                model_ready = False
                if self.use_wlasl300 and self.inference_engine is not None:
                    if hands_present:
                        self.hand_missing_grace_remaining = self.HAND_MISSING_GRACE_FRAMES
                        prediction = self.predict_sign(keypoints=keypoints)
                    else:
                        self.hand_missing_grace_remaining -= 1
                        if self.hand_missing_grace_remaining < 0:
                            prediction = self._reset_realtime_prediction_state()
                        else:
                            prediction = self.last_prediction
                    engine_status = self.inference_engine.get_status()
                    model_ready = bool(engine_status.get("model_ready"))

                    if engine_status.get("last_error"):
                        status = engine_status["last_error"]
                        color = (0, 0, 255)
                    elif not hands_present and self.hand_missing_grace_remaining >= 0:
                        status = (
                            f"Holding context {self.hand_missing_grace_remaining + 1}/{self.HAND_MISSING_GRACE_FRAMES} "
                            f"| Buffer {engine_status['buffer_frames']}/{engine_status['buffer_target']}"
                        )
                        color = (0, 215, 255)
                    elif not hands_present:
                        status = f"Waiting for hands | Buffer {engine_status['buffer_frames']}/{engine_status['buffer_target']}"
                        color = (0, 165, 255)
                    elif not engine_status["is_ready"]:
                        status = (
                            f"Buffering {engine_status['buffer_frames']}/{engine_status['buffer_target']} "
                            f"| Feature {engine_status['feature_dim']}/{engine_status['expected_input_dim']}"
                        )
                        color = (0, 255, 255)
                    elif prediction and prediction[0] != "...":
                        status = f"Stable sign: {prediction[0]}"
                        color = (0, 255, 0)
                    elif hands_present and prediction:
                        status = f"Holding sign context | Motion {engine_status['motion_delta']:.4f}"
                        color = (0, 215, 255)
                    else:
                        status = "Listening for a stable sign..."
                        color = (0, 255, 255)
                else:
                    if keypoints is None:
                        status = "No hands detected"
                        color = (0, 0, 255)
                    else:
                        features = self.preprocess_keypoints(keypoints)
                        if features is not None:
                            self.keypoint_buffer.append(features)
                            status = f"Buffering: {len(self.keypoint_buffer)}/{Config.SEQUENCE_LENGTH}"
                            color = (0, 255, 255)
                        else:
                            status = "Keypoints invalid"
                            color = (0, 0, 255)
                    prediction = self.predict_sign(keypoints=keypoints)
                
                # Render frame
                display_frame = self.ui.draw_frame(frame)
                
                # Draw keypoints if toggled on
                if show_keypoints and keypoints is not None:
                    display_frame = self.ui.draw_keypoints(display_frame, keypoints)
                
                if prediction:
                    sign_name, confidence, top_k = prediction
                    display_frame = self.ui.draw_prediction(
                        display_frame, sign_name, confidence, top_k
                    )
                
                # Calculate and display FPS
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                self.fps_history.append(current_fps)
                avg_fps = np.mean(self.fps_history)
                
                display_frame = self.ui.draw_fps(display_frame, avg_fps)
                display_frame = self.ui.draw_status(
                    display_frame,
                    status,
                    color,
                    model_ready=model_ready,
                    expected_input_dim=engine_status["expected_input_dim"] if self.use_wlasl300 and self.inference_engine is not None else None,
                )
                
                # Display
                if not headless:
                    cv2.imshow("ASL Recognition Pipeline", display_frame)
                
                # Handle user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    show_keypoints = not show_keypoints
                    status_msg = 'ON' if show_keypoints else 'OFF'
                    print(f"Keypoints display: {status_msg}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup and release resources."""
        print("\nCleaning up...")
        self.video_capture.stop()
        if Config.ENABLE_TTS:
            self.tts_engine.stop()
        self.keypoint_extractor.release()
        cv2.destroyAllWindows()
        print("Cleanup complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ASL Recognition Pipeline')
    parser.add_argument('--model', type=str, default='models/bilstm_final.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--no-model', action='store_true',
                       help='Run without loading a model (inference disabled)')
    parser.add_argument('--use-wlasl300', action='store_true',
                       help='Use the new 300-sign WLASL model with RealtimeInferenceEngine')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = ASLRecognitionPipeline(
        model_checkpoint=None if args.no_model else args.model,
        use_wlasl300=args.use_wlasl300
    )
    pipeline.run()
