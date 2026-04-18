"""
Video Capture Module
Handles real-time video acquisition from webcam for ASL recognition.
"""

import cv2
import threading
from collections import deque
from typing import Optional, Callable


class VideoCapture:
    """
    Real-time video capture from webcam with threading support.
    Ensures smooth frame acquisition without blocking the main inference loop.
    """
    
    def __init__(self, camera_id: int = 0, frame_width: int = 640, frame_height: int = 480, fps: int = 30):
        """
        Initialize video capture.
        
        Args:
            camera_id: Webcam device ID (default: 0 for primary camera)
            frame_width: Width of captured frames
            frame_height: Height of captured frames
            fps: Target frames per second
        """
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        
        self.cap = None
        self.frame_buffer = deque(maxlen=2)  # Keep last 2 frames
        self.is_running = False
        self.thread = None
        self.frame_count = 0
        
    def start(self) -> bool:
        """Start video capture in a background thread."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # Set video properties for optimal real-time performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            if not self.cap.isOpened():
                print(f"Error: Cannot open camera {self.camera_id}")
                return False
            
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            print(f"Video capture started (Camera {self.camera_id})")
            return True
            
        except Exception as e:
            print(f"Error initializing video capture: {e}")
            return False
    
    def _capture_loop(self):
        """Background thread loop for continuous frame capture."""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_buffer.append(frame)
                self.frame_count += 1
            else:
                print("Failed to read frame from camera")
                break
    
    def get_frame(self) -> Optional[tuple]:
        """
        Get the latest captured frame.
        
        Returns:
            Tuple of (frame, frame_id) or None if no frame available
        """
        if self.frame_buffer:
            return self.frame_buffer[-1], self.frame_count
        return None
    
    def stop(self):
        """Stop video capture and cleanup resources."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        print("Video capture stopped")
    
    def get_fps(self) -> float:
        """Get actual FPS of the capture stream."""
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0.0
