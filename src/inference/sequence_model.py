"""
Sequence Model Module
BiLSTM and Lightweight Transformer models for temporal ASL sign recognition.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


class BiLSTMSignClassifier(nn.Module):
    """
    Bidirectional LSTM with attention mechanism for ASL sign classification.
    Optimized for real-time inference on consumer hardware.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2, dropout: float = 0.3):
        """
        Initialize BiLSTM classifier.
        
        Args:
            input_dim: Input feature dimension (e.g., 126 for hand+pose keypoints * 3)
            hidden_dim: LSTM hidden dimension
            num_classes: Number of ASL sign classes (300 for WLASL300)
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # *2 for bidirectional
            num_heads=4,
            batch_first=True,
            dropout=dropout
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Tuple of (logits, attention_weights)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        
        # Apply attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last time step
        last_output = attn_out[:, -1, :]  # (batch, hidden_dim*2)
        
        # Classification
        logits = self.fc(last_output)  # (batch, num_classes)
        
        return logits, attn_weights


class LightweightTransformerClassifier(nn.Module):
    """
    Lightweight Temporal Transformer for ASL sign classification.
    Reduced complexity with fewer heads and layers for real-time performance.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.3):
        """
        Initialize Lightweight Transformer classifier.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_classes: Number of ASL sign classes
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # Transformer forward
        transformer_out = self.transformer(x)  # (batch, seq_len, hidden_dim)
        
        # Take the last time step
        last_output = transformer_out[:, -1, :]  # (batch, hidden_dim)
        
        # Classification
        logits = self.fc(last_output)  # (batch, num_classes)
        
        return logits


class SignClassificationPipeline:
    """
    High-level interface for sign classification with model loading and inference.
    """
    
    def __init__(self, model_type: str = "bilstm", num_classes: int = 300, device: str = "cuda"):
        """
        Initialize classification pipeline.
        
        Args:
            model_type: "bilstm" or "transformer"
            num_classes: Number of sign classes
            device: "cuda" or "cpu"
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.model_type = model_type
        
        # Create model
        input_dim = 126  # 42 hand landmarks * 3 (x, y, z)
        hidden_dim = 256
        
        if model_type == "bilstm":
            self.model = BiLSTMSignClassifier(input_dim, hidden_dim, num_classes)
        elif model_type == "transformer":
            self.model = LightweightTransformerClassifier(input_dim, hidden_dim, num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def predict(self, keypoint_sequence: np.ndarray) -> Tuple[int, float]:
        """
        Make prediction on a keypoint sequence.
        
        Args:
            keypoint_sequence: Array of shape (seq_len, input_dim)
        
        Returns:
            Tuple of (predicted_class_id, confidence_score)
        """
        with torch.no_grad():
            # Prepare input
            x = torch.from_numpy(keypoint_sequence).float().unsqueeze(0)  # (1, seq_len, input_dim)
            x = x.to(self.device)
            
            # Forward pass
            if self.model_type == "bilstm":
                logits, _ = self.model(x)
            else:
                logits = self.model(x)
            
            # Get prediction
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
            
            return predicted_class.item(), confidence.item()
