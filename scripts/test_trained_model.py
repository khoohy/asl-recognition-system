"""
Test the trained model on sample videos without real-time webcam.
Useful for quick validation before connecting hardware.
"""

import sys
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules.sequence_model import BiLSTMSignClassifier
from src.modules.keypoint_extraction import KeypointExtractor
from src.utils.config import Config
from src.utils.preprocessing import KeypointPreprocessor


def test_trained_model(model_checkpoint='models/bilstm_final.pt', num_samples=5):
    """
    Test trained model on sample dataset videos.
    
    Args:
        model_checkpoint: Path to trained model
        num_samples: Number of videos to test
    """
    
    print("\n" + "="*60)
    print("Testing Trained Model")
    print("="*60 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load metadata
    metadata_file = Path('data/raw/wlasl_v0.3.json')
    if not metadata_file.exists():
        print(f"❌ Metadata file not found: {metadata_file}")
        return
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Found {len(metadata)} ASL signs\n")
    
    # Load model
    print(f"Loading model from {model_checkpoint}...")
    model = BiLSTMSignClassifier(
        input_dim=126,
        hidden_dim=256,
        num_classes=Config.NUM_CLASSES,
        num_layers=2,
        dropout=0.3
    )
    
    checkpoint = torch.load(model_checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully\n")
    
    # Initialize components
    keypoint_extractor = KeypointExtractor()
    preprocessor = KeypointPreprocessor()
    
    # Test on sample videos
    print("="*60)
    print("Testing on Sample Videos")
    print("="*60 + "\n")
    
    results = []
    
    for sign_idx, sign_info in enumerate(metadata[:num_samples]):
        gloss = sign_info.get('gloss', f'SIGN_{sign_idx}')
        
        for instance in sign_info.get('instances', []):
            video_id = instance.get('video_id', '')
            
            # Try different possible paths
            possible_paths = [
                Path('data/raw') / f"{video_id}.mp4",
                Path('data/raw') / gloss / f"{video_id}.mp4",
            ]
            
            video_path = None
            for p in possible_paths:
                if p.exists():
                    video_path = p
                    break
            
            if video_path is None:
                print(f"[{sign_idx+1}/{num_samples}] ⚠️  {gloss}: Video not found")
                continue
            
            print(f"[{sign_idx+1}/{num_samples}] Testing {gloss} ({video_path.name})...")
            
            # Extract keypoints from video
            keypoints_sequence = []
            cap = cv2.VideoCapture(str(video_path))
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                hand_keypoints = keypoint_extractor.extract_hand_keypoints(frame)
                if hand_keypoints is not None:
                    keypoints_sequence.append(hand_keypoints)
                else:
                    keypoints_sequence.append(np.zeros(126))
                
                frame_count += 1
            
            cap.release()
            
            if not keypoints_sequence:
                print(f"   ⚠️  No frames extracted")
                continue
            
            # Preprocess
            keypoints_array = np.array(keypoints_sequence)
            keypoints_array = preprocessor.normalize_keypoints(keypoints_array)
            keypoints_array = preprocessor.scale_keypoints(keypoints_array, scale_factor=1.0)
            keypoints_array = preprocessor.pad_or_truncate_sequence(
                keypoints_array,
                target_length=Config.SEQUENCE_LENGTH
            )
            
            # Run inference
            with torch.no_grad():
                keypoints_tensor = torch.FloatTensor(keypoints_array).unsqueeze(0).to(device)
                outputs = model(keypoints_tensor)
                
                # Handle tuple output
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Get predictions
                probs = torch.softmax(logits, dim=1)
                confidence, predicted_class = torch.max(probs, 1)
                
                predicted_class = predicted_class.item()
                confidence = confidence.item()
                
                # Map to sign
                if predicted_class < len(metadata):
                    predicted_sign = metadata[predicted_class].get('gloss', f'SIGN_{predicted_class}')
                else:
                    predicted_sign = f'UNKNOWN_{predicted_class}'
                
                match = "✅ CORRECT" if predicted_sign == gloss else f"❌ WRONG (predicted {predicted_sign})"
                print(f"   {match} | Confidence: {confidence*100:.1f}%")
                
                results.append({
                    'true_label': gloss,
                    'predicted_label': predicted_sign,
                    'confidence': confidence,
                    'correct': predicted_sign == gloss
                })
            
            break  # Only test first instance per sign
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    if results:
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        accuracy = 100 * correct / total
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        print(f"Avg Confidence: {avg_confidence*100:.1f}%")
        print(f"\nDetailed Results:")
        for r in results:
            print(f"  {r['true_label']:15} → {r['predicted_label']:15} ({r['confidence']*100:5.1f}%)")
    else:
        print("No results to summarize")
    
    print("\n" + "="*60)
    print("✅ Test Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained ASL model')
    parser.add_argument('--model', type=str, default='models/bilstm_final.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of videos to test')
    
    args = parser.parse_args()
    test_trained_model(model_checkpoint=args.model, num_samples=args.samples)
