"""
User Interface Module
Real-time visualization of ASL recognition pipeline.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List


class RealtimeUI:
    """
    Handles real-time visualization for ASL recognition system.
    Displays live video, detected keypoints, predictions, and confidence.
    """
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        """
        Initialize UI.
        
        Args:
            frame_width: Display frame width
            frame_height: Display frame height
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 1
        self.font_color = (255, 255, 255)  # White
    
    def draw_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw base frame with UI elements.
        
        Args:
            frame: Input video frame
        
        Returns:
            Frame with UI elements
        """
        # Ensure frame is in BGR format
        if len(frame.shape) != 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        return frame.copy()
    
    def draw_keypoints(self, frame: np.ndarray, keypoints: dict, keypoint_type: str = "hand") -> np.ndarray:
        """
        Draw detected keypoints on the frame.
        
        Args:
            frame: Input video frame
            keypoints: Dictionary of keypoints from extraction
            keypoint_type: "hand", "pose", or "all"
        
        Returns:
            Frame with drawn keypoints
        """
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw hand keypoints
        if keypoint_type in ["hand", "all"]:
            for hand_key in ["left_hand", "right_hand"]:
                hand_kpts = keypoints.get(hand_key)
                if hand_kpts is not None:
                    color = (0, 255, 0) if hand_key == "right_hand" else (255, 0, 0)
                    self._draw_skeleton(frame, hand_kpts, w, h, color)
        
        # Draw pose keypoints
        if keypoint_type in ["pose", "all"]:
            pose_kpts = keypoints.get("pose")
            if pose_kpts is not None:
                self._draw_skeleton(frame, pose_kpts, w, h, (0, 255, 255))
        
        return frame
    
    @staticmethod
    def _draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, frame_w: int, frame_h: int, color: Tuple[int, int, int], radius: int = 4):
        """
        Draw keypoint skeleton on frame.
        
        Args:
            frame: Frame to draw on
            keypoints: Keypoints array (N, 3) with normalized coordinates
            frame_w: Frame width
            frame_h: Frame height
            color: RGB color for keypoints
            radius: Point radius
        """
        if keypoints is None or len(keypoints) == 0:
            return
        
        for kpt in keypoints:
            x, y, z = kpt[0], kpt[1], kpt[2]
            
            # Convert normalized coordinates to pixel coordinates
            pixel_x = int(x * frame_w)
            pixel_y = int(y * frame_h)
            
            # Clamp to frame bounds
            if 0 <= pixel_x < frame_w and 0 <= pixel_y < frame_h:
                cv2.circle(frame, (pixel_x, pixel_y), radius, color, -1)
    
    def draw_prediction(self, frame: np.ndarray, prediction_text: str, confidence: float, top_k: List[Tuple[str, float]]) -> np.ndarray:
        """
        Draw prediction and confidence on frame.
        
        Args:
            frame: Input frame
            prediction_text: Predicted sign name
            confidence: Confidence score (0-1)
            top_k: List of (sign_name, score) tuples for top-k predictions
        
        Returns:
            Frame with prediction info
        """
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw semi-transparent background panel for predictions
        panel_height = 140
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, panel_height + 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Draw main prediction
        main_text = f"Sign: {prediction_text}"
        conf_text = f"Confidence: {confidence:.2%}"

        main_color = (0, 255, 0) if prediction_text != "..." else (180, 180, 180)
        cv2.putText(frame, main_text, (20, 40), self.font, self.font_scale + 0.2, main_color, 2)
        cv2.putText(frame, conf_text, (20, 70), self.font, self.font_scale, self.font_color, self.font_thickness)
        
        # Draw top-3 confidence bars.
        if top_k:
            topk_y = 102
            cv2.putText(frame, "Top 3 guesses", (20, topk_y), self.font, 0.5, (220, 220, 220), 1)

            bar_left = 20
            bar_right = min(w - 30, 250)
            bar_width = max(80, bar_right - bar_left)
            for idx, (sign, score) in enumerate(top_k[:3]):
                row_y = topk_y + 18 + (idx * 34)
                text = f"{idx + 1}. {sign}"
                cv2.putText(frame, text, (bar_left, row_y), self.font, 0.45, (210, 210, 210), 1)

                bar_y = row_y + 8
                cv2.rectangle(frame, (bar_left, bar_y), (bar_left + bar_width, bar_y + 12), (70, 70, 70), -1)
                filled_width = int(bar_width * float(np.clip(score, 0.0, 1.0)))
                bar_color = (0, 200, 0) if idx == 0 else (0, 165, 255)
                cv2.rectangle(frame, (bar_left, bar_y), (bar_left + filled_width, bar_y + 12), bar_color, -1)
                cv2.putText(
                    frame,
                    f"{score:.0%}",
                    (bar_left + bar_width + 10, bar_y + 10),
                    self.font,
                    0.4,
                    (220, 220, 220),
                    1,
                )

        return frame
    
    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Draw FPS counter on frame.
        
        Args:
            frame: Input frame
            fps: Current FPS value
        
        Returns:
            Frame with FPS display
        """
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (w - 150, 30), self.font, self.font_scale, (0, 255, 0), 2)
        
        return frame
    
    def draw_status(
        self,
        frame: np.ndarray,
        status: str,
        status_color: Tuple[int, int, int] = (0, 255, 0),
        model_ready: bool = False,
        expected_input_dim: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw status message on frame.
        
        Args:
            frame: Input frame
            status: Status text
            status_color: Color for status text
        
        Returns:
            Frame with status
        """
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        cv2.putText(frame, status, (10, h - 20), self.font, 0.6, status_color, 2)

        light_center = (w - 155, h - 25)
        light_color = (0, 200, 0) if model_ready else (0, 0, 255)
        cv2.circle(frame, light_center, 8, light_color, -1)
        label = "Model Ready"
        if expected_input_dim is not None:
            label = f"{label} ({expected_input_dim}D)"
        cv2.putText(frame, label, (w - 135, h - 20), self.font, 0.5, self.font_color, 1)
        
        return frame
