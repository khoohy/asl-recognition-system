"""Hand, face, and upper-body keypoint extraction module using MediaPipe."""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Dict, List, Optional


class KeypointExtractor:
    """Extract hand, face, and pose keypoints from video frames using MediaPipe."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the keypoint extractor with MediaPipe hands.
        
        Args:
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize MediaPipe hands, face mesh, and pose.
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold,
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold,
        )
    
    def extract_keypoints(self, frame: np.ndarray) -> Dict:
        """
        Extract hand landmarks from a video frame using MediaPipe.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with 'left_hand', 'right_hand', 'face', and 'pose' keypoints
        """
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        hand_results = self.hands.process(frame_rgb)
        face_results = self.face_mesh.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)
        
        left_hand = np.zeros((21, 3), dtype=np.float32)
        right_hand = np.zeros((21, 3), dtype=np.float32)
        face = np.zeros((478, 3), dtype=np.float32)
        pose = np.zeros((33, 3), dtype=np.float32)

        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                # Convert landmarks to numpy array
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                ], dtype=np.float32)
                
                if handedness.classification[0].label == 'Right':
                    right_hand = landmarks
                else:
                    left_hand = landmarks

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            face_points = np.array(
                [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark],
                dtype=np.float32,
            )
            face[: len(face_points)] = face_points

        if pose_results.pose_landmarks:
            pose = np.array(
                [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark],
                dtype=np.float32,
            )

        return {
            'left_hand': left_hand,
            'right_hand': right_hand,
            'face': face,
            'pose': pose,
            'frame_shape': (h, w)
        }
    
    def extract_hand_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract hand keypoints and flatten them.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Flattened keypoints array (shape: 126,) - 21 landmarks * 3 coordinates * 2 hands
        """
        keypoints = self.extract_keypoints(frame)
        
        # Create array for both hands (21 landmarks * 3 coordinates each)
        # Order: left hand (63) + right hand (63) = 126 total
        left_hand_kps = keypoints['left_hand']
        right_hand_kps = keypoints['right_hand']
        
        # Flatten and concatenate
        flattened = np.concatenate([left_hand_kps.flatten(), right_hand_kps.flatten()])
        return flattened.astype(np.float32)
    
    def visualize_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw hand landmarks on frame for visualization.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Frame with landmarks drawn
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process to get landmarks for drawing
        hand_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2),
            )
        
        return frame
    
    def get_hand_landmarks(self, keypoints: Dict) -> Optional[np.ndarray]:
        """
        Extract concatenated hand landmarks (both hands).
        
        Args:
            keypoints: Dictionary from extract_keypoints()
        
        Returns:
            Concatenated hand landmarks array of shape (42, 3) or None
        """
        left_hand = keypoints.get('left_hand')
        right_hand = keypoints.get('right_hand')
        
        if left_hand is None or right_hand is None:
            return None
        
        # Concatenate left and right hands
        combined = np.concatenate([left_hand, right_hand], axis=0)
        return combined
    
    def release(self):
        """Release resources."""
        if hasattr(self, 'hands') and self.hands is not None:
            self.hands.close()
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            self.face_mesh.close()
        if hasattr(self, 'pose') and self.pose is not None:
            self.pose.close()
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        try:
            if hasattr(self, 'hands') and self.hands is not None:
                self.hands.close()
            if hasattr(self, 'face_mesh') and self.face_mesh is not None:
                self.face_mesh.close()
            if hasattr(self, 'pose') and self.pose is not None:
                self.pose.close()
        except:
            pass  # Ignore errors during cleanup
