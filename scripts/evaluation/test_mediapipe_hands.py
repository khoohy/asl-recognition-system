"""Test MediaPipe hand detection with webcam."""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.modules.keypoint_extraction import KeypointExtractor


def main():
    """Test hand detection using MediaPipe."""
    print("\n" + "="*60)
    print("MediaPipe Hand Detection Test")
    print("="*60 + "\n")
    print("Press 'q' to exit\n")
    
    # Initialize detector
    try:
        extractor = KeypointExtractor(confidence_threshold=0.5)
        print("✓ Hand detector initialized (MediaPipe)\n")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("✓ Webcam opened\n")
    
    frame_count = 0
    hands_detected_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Mirror for selfie view
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Extract keypoints
        try:
            keypoints_dict = extractor.extract_keypoints(frame)
            left_hand = keypoints_dict['left_hand']
            right_hand = keypoints_dict['right_hand']
            
            # Check if hands detected (not all zeros)
            left_detected = not np.all(left_hand == 0)
            right_detected = not np.all(right_hand == 0)
            num_hands = int(left_detected) + int(right_detected)
            
            if num_hands > 0:
                hands_detected_frames += 1
        except Exception as e:
            print(f"❌ Error in extraction: {e}")
            num_hands = 0
        
        # Visualize landmarks on frame
        display_frame = extractor.visualize_landmarks(frame)
        
        # Add info text
        info_text = f"Hands detected: {num_hands} | Frame: {frame_count}"
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0) if num_hands > 0 else (0, 0, 255), 2)
        
        detection_rate = 100 * hands_detected_frames / max(frame_count, 1)
        cv2.putText(display_frame, f"Detection rate: {detection_rate:.1f}%", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)
        
        # Show frame
        cv2.imshow("MediaPipe Hand Detection", display_frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n✓ Exiting...")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    print(f"\nResults:")
    print(f"  Total frames: {frame_count}")
    print(f"  Frames with hands: {hands_detected_frames}")
    print(f"  Detection rate: {100*hands_detected_frames/max(frame_count, 1):.1f}%")
    print("✓ Test complete!\n")


if __name__ == "__main__":
    main()
