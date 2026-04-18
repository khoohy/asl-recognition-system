"""
Training script for ASL Sign Recognition Model.
Loads dataset, trains BiLSTM/Transformer model, and saves checkpoints.
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules.keypoint_extraction import KeypointExtractor
from src.modules.sequence_model import BiLSTMSignClassifier, LightweightTransformerClassifier
from src.utils.config import Config
from src.utils.preprocessing import KeypointPreprocessor


class ASLDataset(Dataset):
    """Dataset for ASL sign videos with extracted keypoints."""
    
    def __init__(self, video_dir, metadata_file, keypoint_extractor, preprocessor):
        """
        Initialize ASL dataset.
        
        Args:
            video_dir: Directory containing video files
            metadata_file: JSON file with sign labels and metadata
            keypoint_extractor: KeypointExtractor instance
            preprocessor: KeypointPreprocessor instance
        """
        self.video_dir = Path(video_dir)
        self.keypoint_extractor = keypoint_extractor
        self.preprocessor = preprocessor
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.videos = []
        self.labels = []
        
        # Build dataset from metadata
        # Metadata is a list of signs, each with "gloss" and "instances"
        if isinstance(metadata, list):
            for sign_id, sign_info in enumerate(metadata):
                gloss = sign_info.get('gloss', f'SIGN_{sign_id}')
                for instance in sign_info.get('instances', []):
                    video_id = instance.get('video_id', '')
                    # Try different possible paths for the video
                    possible_paths = [
                        self.video_dir / f"{video_id}.mp4",
                        self.video_dir / gloss / f"{video_id}.mp4",
                    ]
                    for video_path in possible_paths:
                        if video_path.exists():
                            self.videos.append(str(video_path))
                            self.labels.append(sign_id)
                            break
        
        print(f"Dataset loaded: {len(self.videos)} videos with {len(set(self.labels))} unique signs")
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        """Extract keypoints from video and return as tensor."""
        video_path = self.videos[idx]
        label = self.labels[idx]
        
        # Extract keypoints from video
        keypoints = self._extract_keypoints_from_video(video_path)
        
        # Preprocess keypoints
        keypoints = self.preprocessor.normalize_keypoints(keypoints)
        keypoints = self.preprocessor.scale_keypoints(keypoints, scale_factor=1.0)
        keypoints = self.preprocessor.pad_or_truncate_sequence(
            keypoints, 
            target_length=Config.SEQUENCE_LENGTH
        )
        
        # Convert to tensor
        keypoints_tensor = torch.FloatTensor(keypoints)
        label_tensor = torch.LongTensor([label])
        
        return keypoints_tensor, label_tensor
    
    def _extract_keypoints_from_video(self, video_path, max_frames=None):
        """
        Extract keypoints from video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to extract (None = all)
            
        Returns:
            numpy array of shape (num_frames, 126) for hand landmarks
        """
        keypoints_list = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_count >= max_frames:
                    break
                
                # Extract hand keypoints
                hand_keypoints = self.keypoint_extractor.extract_hand_keypoints(frame)
                
                if hand_keypoints is not None:
                    keypoints_list.append(hand_keypoints)
                else:
                    # If no hands detected, use zeros
                    keypoints_list.append(np.zeros(126))
                
                frame_count += 1
            
            cap.release()
            
        except Exception as e:
            print(f"Error extracting keypoints from {video_path}: {e}")
            return np.zeros((1, 126))
        
        if not keypoints_list:
            return np.zeros((1, 126))
        
        return np.array(keypoints_list)


class ModelTrainer:
    """Trainer for ASL recognition models."""
    
    def __init__(self, model, device='cpu', learning_rate=0.001):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to use ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_model_path = None
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (keypoints, labels) in enumerate(pbar):
            keypoints = keypoints.to(self.device)
            labels = labels.squeeze().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(keypoints)
            # Handle both tuple (logits, attention) and single tensor returns
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            for keypoints, labels in pbar:
                keypoints = keypoints.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                outputs = self.model(keypoints)
                # Handle both tuple (logits, attention) and single tensor returns
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                num_batches += 1
                
                accuracy = 100 * correct / total
                pbar.set_postfix({'loss': loss.item(), 'acc': f'{accuracy:.2f}%'})
        
        avg_loss = total_loss / num_batches
        avg_accuracy = 100 * correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_accuracy)
        
        return avg_loss, avg_accuracy
    
    def save_checkpoint(self, path, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }
        torch.save(checkpoint, path)
        
        if is_best:
            self.best_model_path = path
            print(f"✓ Saved best model to {path}")
        else:
            print(f"  Saved checkpoint to {path}")


def train_model(
    num_epochs=10,
    batch_size=8,
    learning_rate=0.001,
    model_type='bilstm',
    dataset_dir='data/raw',
    metadata_file='data/raw/wlasl_v0.3.json',
    checkpoint_dir='models'
):
    """
    Train ASL recognition model.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        model_type: 'bilstm' or 'transformer'
        dataset_dir: Directory containing video files
        metadata_file: JSON file with metadata
        checkpoint_dir: Directory to save checkpoints
    """
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset exists
    if not Path(metadata_file).exists():
        print(f"❌ Metadata file not found: {metadata_file}")
        print("Please run: python scripts/create_sample_dataset.py 10 30")
        return
    
    print("\n" + "="*60)
    print("ASL Sign Recognition Model Training")
    print("="*60)
    
    # Initialize components
    print("\nInitializing components...")
    keypoint_extractor = KeypointExtractor()
    preprocessor = KeypointPreprocessor()
    
    # Create dataset
    print("Loading dataset...")
    dataset = ASLDataset(
        video_dir=dataset_dir,
        metadata_file=metadata_file,
        keypoint_extractor=keypoint_extractor,
        preprocessor=preprocessor
    )
    
    if len(dataset) == 0:
        print("❌ No videos found in dataset!")
        return
    
    # Split dataset (80/20 train/val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    print(f"\nCreating {model_type.upper()} model...")
    if model_type.lower() == 'bilstm':
        model = BiLSTMSignClassifier(
            input_dim=126,
            hidden_dim=256,
            num_classes=Config.NUM_CLASSES,
            num_layers=2,
            dropout=0.3
        )
    else:
        model = LightweightTransformerClassifier(
            input_dim=126,
            hidden_dim=256,
            num_classes=Config.NUM_CLASSES,
            num_heads=4,
            num_layers=2,
            dropout=0.3
        )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = ModelTrainer(model, device=device, learning_rate=learning_rate)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_acc = trainer.validate(val_loader)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        trainer.scheduler.step(val_loss)
        
        # Save checkpoint
        checkpoint_file = checkpoint_path / f"{model_type}_epoch_{epoch:03d}.pt"
        is_best = val_loss < trainer.best_val_loss
        trainer.save_checkpoint(str(checkpoint_file), epoch, is_best=is_best)
        
        if is_best:
            trainer.best_val_loss = val_loss
    
    # Final summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Best model saved to: {trainer.best_model_path}")
    print(f"Final validation accuracy: {trainer.val_accuracies[-1]:.2f}%")
    
    # Save final model
    final_model_path = checkpoint_path / f"{model_type}_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Save training history
    history = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'val_accuracies': trainer.val_accuracies
    }
    history_path = checkpoint_path / f"{model_type}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    return model, trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ASL Recognition Model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model', type=str, default='bilstm', 
                       choices=['bilstm', 'transformer'], help='Model type')
    parser.add_argument('--dataset-dir', type=str, default='data/raw', 
                       help='Dataset directory')
    parser.add_argument('--metadata', type=str, default='data/raw/wlasl_v0.3.json',
                       help='Metadata file path')
    
    args = parser.parse_args()
    
    train_model(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_type=args.model,
        dataset_dir=args.dataset_dir,
        metadata_file=args.metadata
    )
