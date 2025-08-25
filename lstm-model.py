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
            bidirectional=False
        )
        
        # Attention mechanism
        self.attention = MultiHeadAttention(
            config.hidden_size, 
            config.attention_heads, 
            config.dropout
        )
        
        # Classification heads for different time horizons
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 4, config.output_size)  # Multi-horizon outputs
        )
        
        # Regression head for time-to-failure estimation
        self.ttf_regressor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
            nn.ReLU()  # Ensure positive TTF
        )
        
        # Uncertainty estimation (output log variance)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, config.output_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Dictionary containing:
                - failure_probs: Failure probabilities for each horizon
                - ttf_estimate: Time-to-failure estimate
                - uncertainty: Prediction uncertainty (log variance)
                - attention_weights: Attention weights for interpretability
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x_proj = self.input_projection(x)  # (batch, seq_len, hidden_size)
        
        # LSTM forward pass
        lstm_output, (hidden, cell) = self.lstm(x_proj)  # (batch, seq_len, hidden_size)
        
        # Apply attention
        attended_output, attention_weights = self.attention(lstm_output)
        
        # Use the last attended output for predictions
        final_representation = attended_output[:, -1, :]  # (batch, hidden_size)
        
        # Multi-horizon failure probability prediction
        failure_probs = torch.sigmoid(self.classifier(final_representation))
        
        # Time-to-failure regression
        ttf_estimate = self.ttf_regressor(final_representation)
        
        # Uncertainty estimation (log variance)
        log_variance = self.uncertainty_head(final_representation)
        
        return {
            'failure_probs': failure_probs,
            'ttf_estimate': ttf_estimate,
            'uncertainty': log_variance,
            'attention_weights': attention_weights,
            'hidden_representation': final_representation
        }
    
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty quantification using Monte Carlo dropout.
        
        Args:
            x: Input tensor
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Mean predictions and uncertainty estimates
        """
        self.train()  # Enable dropout for MC sampling
        
        predictions = []
        ttf_predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                output = self.forward(x)
                predictions.append(output['failure_probs'])
                ttf_predictions.append(output['ttf_estimate'])
        
        self.eval()  # Return to eval mode
        
        # Calculate statistics
        predictions = torch.stack(predictions)  # (num_samples, batch, output_size)
        ttf_predictions = torch.stack(ttf_predictions)  # (num_samples, batch, 1)
        
        mean_failure_probs = predictions.mean(dim=0)
        std_failure_probs = predictions.std(dim=0)
        
        mean_ttf = ttf_predictions.mean(dim=0)
        std_ttf = ttf_predictions.std(dim=0)
        
        # Calculate confidence intervals (assuming normal distribution)
        failure_prob_lower = mean_failure_probs - 1.96 * std_failure_probs
        failure_prob_upper = mean_failure_probs + 1.96 * std_failure_probs
        
        ttf_lower = mean_ttf - 1.96 * std_ttf
        ttf_upper = mean_ttf + 1.96 * std_ttf
        
        return {
            'mean_failure_probs': mean_failure_probs,
            'failure_prob_confidence_interval': torch.stack([failure_prob_lower, failure_prob_upper], dim=-1),
            'mean_ttf': mean_ttf,
            'ttf_confidence_interval': torch.stack([ttf_lower, ttf_upper], dim=-1),
            'epistemic_uncertainty': std_failure_probs,
            'ttf_uncertainty': std_ttf
        }

class EnsembleLSTM(nn.Module):
    """
    Ensemble of LSTM models for improved robustness and uncertainty estimation.
    """
    
    def __init__(self, config: ModelConfig, num_models: int = 5):
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([
            PredictiveMaintenanceLSTM(config) for _ in range(num_models)
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble."""
        outputs = [model(x) for model in self.models]
        
        # Aggregate predictions
        failure_probs = torch.stack([out['failure_probs'] for out in outputs])
        ttf_estimates = torch.stack([out['ttf_estimate'] for out in outputs])
        
        mean_failure_probs = failure_probs.mean(dim=0)
        std_failure_probs = failure_probs.std(dim=0)
        
        mean_ttf = ttf_estimates.mean(dim=0)
        std_ttf = ttf_estimates.std(dim=0)
        
        # Average attention weights
        attention_weights = torch.stack([out['attention_weights'] for out in outputs]).mean(dim=0)
        
        return {
            'failure_probs': mean_failure_probs,
            'ttf_estimate': mean_ttf,
            'uncertainty': std_failure_probs,
            'ttf_uncertainty': std_ttf,
            'attention_weights': attention_weights
        }

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in failure prediction.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    """
    Combined loss function for multi-task learning.
    """
    
    def __init__(self, 
                 classification_weight: float = 1.0,
                 regression_weight: float = 0.5,
                 uncertainty_weight: float = 0.1):
        super().__init__()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.uncertainty_weight = uncertainty_weight
        
        self.focal_loss = FocalLoss(alpha=2.0, gamma=2.0)
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of losses
        """
        # Classification loss (multi-horizon failure prediction)
        classification_loss = self.focal_loss(
            predictions['failure_probs'], 
            targets['failure_labels']
        )
        
        # Regression loss (time-to-failure)
        # Only calculate for samples with known TTF
        ttf_mask = targets['ttf_labels'] > 0
        if ttf_mask.sum() > 0:
            regression_loss = self.mse_loss(
                predictions['ttf_estimate'][ttf_mask],
                targets['ttf_labels'][ttf_mask]
            )
        else:
            regression_loss = torch.tensor(0.0, device=predictions['ttf_estimate'].device)
        
        # Uncertainty regularization (encourage calibrated predictions)
        uncertainty_loss = torch.mean(predictions['uncertainty'])
        
        # Combined loss
        total_loss = (
            self.classification_weight * classification_loss +
            self.regression_weight * regression_loss +
            self.uncertainty_weight * uncertainty_loss
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'regression_loss': regression_loss,
            'uncertainty_loss': uncertainty_loss
        }

def create_model(config: ModelConfig, ensemble: bool = False, num_ensemble_models: int = 5) -> nn.Module:
    """
    Factory function to create LSTM models.
    
    Args:
        config: Model configuration
        ensemble: Whether to create ensemble model
        num_ensemble_models: Number of models in ensemble
        
    Returns:
        Initialized model
    """
    if ensemble:
        return EnsembleLSTM(config, num_ensemble_models)
    else:
        return PredictiveMaintenanceLSTM(config)

# Example model configurations
DEFAULT_CONFIG = ModelConfig(
    input_size=20,
    hidden_size=128,
    num_layers=3,
    sequence_length=168,  # 7 days
    output_size=4,  # 1h, 4h, 24h, 7d horizons
    dropout=0.2,
    attention_heads=8
)

LARGE_CONFIG = ModelConfig(
    input_size=20,
    hidden_size=256,
    num_layers=4,
    sequence_length=336,  # 14 days
    output_size=4,
    dropout=0.3,
    attention_heads=16
)

COMPACT_CONFIG = ModelConfig(
    input_size=20,
    hidden_size=64,
    num_layers=2,
    sequence_length=72,  # 3 days
    output_size=4,
    dropout=0.1,
    attention_heads=4
)