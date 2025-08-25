"""
Training script for Predictive Maintenance LSTM model.
Includes MLflow tracking, hyperparameter optimization, and model evaluation.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.lstm_model import PredictiveMaintenanceLSTM, ModelConfig, save_model
from data.feature_engineering import FeatureEngineer, FeatureConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class MultiHorizonLoss(nn.Module):
    """Custom loss function for multi-horizon prediction."""
    
    def __init__(self, horizon_weights: Optional[list] = None):
        super().__init__()
        if horizon_weights is None:
            # Weight shorter horizons more heavily
            horizon_weights = [1.0, 0.8, 0.6, 0.4]  # 1h, 4h, 24h, 7d
        self.horizon_weights = torch.tensor(horizon_weights)
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted multi-horizon loss."""
        if self.horizon_weights.device != predictions.device:
            self.horizon_weights = self.horizon_weights.to(predictions.device)
        
        # Compute focal loss for each horizon
        horizon_losses = []
        for i in range(predictions.shape[1]):
            horizon_loss = self.focal_loss(predictions[:, i], targets[:, i])
            horizon_losses.append(horizon_loss)
        
        # Weight and sum the losses
        weighted_loss = sum(w * loss for w, loss in zip(self.horizon_weights, horizon_losses))
        return weighted_loss

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience

def load_data(data_dir: Path) -> Dict[str, np.ndarray]:
    """Load processed training data."""
    logger.info(f"Loading data from {data_dir}")
    
    features = np.load(data_dir / "features.npy")
    targets = np.load(data_dir / "targets.npy")
    
    # Load equipment IDs
    with open(data_dir / "equipment_ids.txt", 'r') as f:
        equipment_ids = [line.strip() for line in f.readlines()]
    
    logger.info(f"Loaded {len(features)} sequences with {features.shape[1]} time steps and {features.shape[2]} features")
    
    return {
        'features': features,
        'targets': targets,
        'equipment_ids': equipment_ids
    }

def create_data_loaders(features: np.ndarray, targets: np.ndarray, 
                       batch_size: int = 32, test_size: float = 0.2,
                       val_size: float = 0.2, random_state: int = 42) -> Dict[str, DataLoader]:
    """Create train, validation, and test data loaders."""
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, targets, test_size=test_size, random_state=random_state, stratify=None
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=None
    )
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    logger.info(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_data': (X_train, y_train),
        'val_data': (X_val, y_val),
        'test_data': (X_test, y_test)
    }

def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate model performance."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets in data_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs = model(batch_features)
            predictions = outputs['predictions']
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics for each horizon
    metrics = {}
    horizons = ['1h', '4h', '24h', '7d']
    
    for i, horizon in enumerate(horizons):
        pred_horizon = all_predictions[:, i]
        target_horizon = all_targets[:, i]
        
        # ROC AUC
        if len(np.unique(target_horizon)) > 1:
            roc_auc = roc_auc_score(target_horizon, pred_horizon)
            metrics[f'roc_auc_{horizon}'] = roc_auc
        
        # PR AUC
        precision, recall, _ = precision_recall_curve(target_horizon, pred_horizon)
        pr_auc = auc(recall, precision)
        metrics[f'pr_auc_{horizon}'] = pr_auc
        
        # Binary accuracy
        binary_pred = (pred_horizon > 0.5).astype(int)
        accuracy = (binary_pred == target_horizon).mean()
        metrics[f'accuracy_{horizon}'] = accuracy
    
    # Average metrics
    metrics['roc_auc_mean'] = np.mean([metrics[f'roc_auc_{h}'] for h in horizons if f'roc_auc_{h}' in metrics])
    metrics['pr_auc_mean'] = np.mean([metrics[f'pr_auc_{h}'] for h in horizons])
    metrics['accuracy_mean'] = np.mean([metrics[f'accuracy_{h}'] for h in horizons])
    
    return metrics

def train_epoch(model: nn.Module, data_loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(data_loader, desc="Training")
    for batch_features, batch_targets in progress_bar:
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_features)
        predictions = outputs['predictions']
        
        # Compute loss
        loss = criterion(predictions, batch_targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(data_loader)

def validate_epoch(model: nn.Module, data_loader: DataLoader, criterion: nn.Module,
                  device: torch.device) -> float:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_features, batch_targets in data_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs = model(batch_features)
            predictions = outputs['predictions']
            
            loss = criterion(predictions, batch_targets)
            total_loss += loss.item()
    
    return total_loss / len(data_loader)

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                test_loader: DataLoader, config: Dict[str, Any], 
                experiment_name: str = "predictive_maintenance") -> Dict[str, Any]:
    """Train the model with MLflow tracking."""
    
    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config)
        
        # Set up training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        model = model.to(device)
        
        # Loss function and optimizer
        criterion = MultiHorizonLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=config['patience'])
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{config['epochs']}")
            
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss = validate_epoch(model, val_loader, criterion, device)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                mlflow.log_metric('best_val_loss', best_val_loss)
            
            # Early stopping
            if early_stopping(val_loss):
                logger.info("Early stopping triggered")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Evaluate on test set
        test_metrics = evaluate_model(model, test_loader, device)
        mlflow.log_metrics(test_metrics)
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
        
        # Save model locally
        model_path = Path("models") / f"lstm_model_v{mlflow.active_run().info.run_id}.pth"
        model_path.parent.mkdir(exist_ok=True)
        save_model(model, str(model_path), optimizer.state_dict(), config['epochs'], test_metrics)
        
        logger.info(f"Training completed. Model saved to {model_path}")
        logger.info(f"Test metrics: {test_metrics}")
        
        return {
            'model': model,
            'test_metrics': test_metrics,
            'model_path': str(model_path),
            'run_id': mlflow.active_run().info.run_id
        }

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Predictive Maintenance LSTM Model")
    parser.add_argument("--config-file", type=str, required=True, help="Configuration file path")
    parser.add_argument("--data-dir", type=str, required=True, help="Processed data directory")
    parser.add_argument("--experiment-name", type=str, default="predictive_maintenance", help="MLflow experiment name")
    parser.add_argument("--epochs", type=int, help="Number of training epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, help="Batch size (overrides config)")
    parser.add_argument("--learning-rate", type=float, help="Learning rate (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    
    # Load data
    data_dir = Path(args.data_dir)
    data = load_data(data_dir)
    
    # Create data loaders
    loaders = create_data_loaders(
        data['features'], 
        data['targets'],
        batch_size=config['batch_size'],
        test_size=0.2,
        val_size=0.2
    )
    
    # Create model
    model_config = ModelConfig(
        input_size=data['features'].shape[2],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        sequence_length=data['features'].shape[1],
        output_size=data['targets'].shape[1],
        dropout=config['dropout'],
        attention_heads=config['attention_heads'],
        bidirectional=config.get('bidirectional', False),
        use_attention=config.get('use_attention', True)
    )
    
    model = PredictiveMaintenanceLSTM(model_config)
    
    # Train model
    results = train_model(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        test_loader=loaders['test'],
        config=config,
        experiment_name=args.experiment_name
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {results['model_path']}")
    logger.info(f"MLflow run ID: {results['run_id']}")
    logger.info(f"Test metrics: {results['test_metrics']}")

if __name__ == "__main__":
    main()
