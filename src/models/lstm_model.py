"""
LSTM Model for Predictive Maintenance
Multi-variate time series prediction with attention mechanism.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for LSTM model."""
    input_size: int = 20  # Number of sensor features
    hidden_size: int = 128
    num_layers: int = 3
    sequence_length: int = 168  # 7 days * 24 hours
    output_size: int = 4  # Multi-horizon: 1h, 4h, 24h, 7d
    dropout: float = 0.2
    attention_heads: int = 8
    bidirectional: bool = False
    use_attention: bool = True
    
class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for LSTM outputs."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by number of heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # Generate Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        output = self.output_linear(attended)
        
        return output, attention_weights.mean(dim=1)  # Average attention across heads

class PredictiveMaintenanceLSTM(nn.Module):
    """
    Multi-variate LSTM with attention for predictive maintenance.
    Predicts failure probability at multiple time horizons.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input projection layer
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        # Attention mechanism
        if config.use_attention:
            self.attention = MultiHeadAttention(
                config.hidden_size, 
                config.attention_heads, 
                config.dropout
            )
        else:
            self.attention = None
        
        # Classification heads for different time horizons
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 4, config.output_size),
            nn.Sigmoid()
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.output_size),
            nn.Softplus()  # Ensure positive uncertainty values
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Dictionary containing predictions and uncertainty estimates
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention if enabled
        if self.attention is not None:
            attended_out, attention_weights = self.attention(lstm_out)
            # Use the last attended output
            final_representation = attended_out[:, -1, :]
        else:
            # Use the last LSTM output
            final_representation = lstm_out[:, -1, :]
            attention_weights = None
        
        # Classification predictions
        predictions = self.classifier(final_representation)
        
        # Uncertainty estimates
        uncertainty = self.uncertainty_head(final_representation)
        
        return {
            'predictions': predictions,
            'uncertainty': uncertainty,
            'attention_weights': attention_weights,
            'hidden_states': lstm_out
        }
    
    def predict_with_confidence(self, x: torch.Tensor, num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        Monte Carlo dropout for uncertainty estimation.
        
        Args:
            x: Input tensor
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        self.train()  # Enable dropout for uncertainty estimation
        
        predictions = []
        uncertainties = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                output = self.forward(x)
                predictions.append(output['predictions'])
                uncertainties.append(output['uncertainty'])
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (num_samples, batch_size, output_size)
        uncertainties = torch.stack(uncertainties, dim=0)
        
        # Calculate statistics
        mean_predictions = predictions.mean(dim=0)
        std_predictions = predictions.std(dim=0)
        
        # Confidence intervals (95%)
        confidence_intervals = {
            'lower': torch.clamp(mean_predictions - 1.96 * std_predictions, 0, 1),
            'upper': torch.clamp(mean_predictions + 1.96 * std_predictions, 0, 1)
        }
        
        self.eval()  # Disable dropout
        
        return {
            'predictions': mean_predictions,
            'uncertainty': uncertainties.mean(dim=0),
            'confidence_intervals': confidence_intervals,
            'prediction_std': std_predictions
        }

class EnsembleLSTM(nn.Module):
    """Ensemble of LSTM models for improved prediction accuracy."""
    
    def __init__(self, config: ModelConfig, num_models: int = 5):
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([
            PredictiveMaintenanceLSTM(config) for _ in range(num_models)
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble."""
        predictions = []
        uncertainties = []
        
        for model in self.models:
            output = model(x)
            predictions.append(output['predictions'])
            uncertainties.append(output['uncertainty'])
        
        # Average predictions
        ensemble_predictions = torch.stack(predictions).mean(dim=0)
        ensemble_uncertainty = torch.stack(uncertainties).mean(dim=0)
        
        return {
            'predictions': ensemble_predictions,
            'uncertainty': ensemble_uncertainty,
            'individual_predictions': predictions
        }

def load_model(model_path: str, config: ModelConfig) -> PredictiveMaintenanceLSTM:
    """Load a trained model from file."""
    model = PredictiveMaintenanceLSTM(config)
    
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path, map_location='cuda')
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded from {model_path}")
    return model

def save_model(model: PredictiveMaintenanceLSTM, model_path: str, 
               optimizer_state: Optional[Dict] = None, 
               epoch: int = 0, 
               metrics: Optional[Dict] = None) -> None:
    """Save a trained model to file."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'epoch': epoch
    }
    
    if optimizer_state is not None:
        checkpoint['optimizer_state_dict'] = optimizer_state
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, model_path)
    logger.info(f"Model saved to {model_path}")

def get_model_summary(model: PredictiveMaintenanceLSTM) -> str:
    """Get a summary of the model architecture."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
    Model Summary:
    - Total parameters: {total_params:,}
    - Trainable parameters: {trainable_params:,}
    - Input size: {model.config.input_size}
    - Hidden size: {model.config.hidden_size}
    - LSTM layers: {model.config.num_layers}
    - Sequence length: {model.config.sequence_length}
    - Output size: {model.config.output_size}
    - Attention heads: {model.config.attention_heads}
    - Bidirectional: {model.config.bidirectional}
    - Dropout: {model.config.dropout}
    """
    
    return summary
