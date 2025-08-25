"""
Feature Engineering for Predictive Maintenance
Handles data preprocessing, feature extraction, and data validation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import argparse
import yaml
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    sequence_length: int = 168  # 7 days * 24 hours
    prediction_horizons: List[int] = None  # [1, 4, 24, 168] hours
    feature_columns: List[str] = None
    target_columns: List[str] = None
    scaler_type: str = "robust"  # "standard" or "robust"
    
    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [1, 4, 24, 168]
        if self.feature_columns is None:
            self.feature_columns = [
                'vibration_x', 'vibration_y', 'vibration_z',
                'temperature', 'pressure', 'current', 'voltage', 'flow_rate',
                'suction_pressure', 'discharge_pressure', 'rpm', 'power_factor'
            ]
        if self.target_columns is None:
            self.target_columns = ['failure_1h', 'failure_4h', 'failure_24h', 'failure_7d']

class FeatureEngineer:
    """Handles feature engineering for predictive maintenance data."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.scaler = self._get_scaler()
        self.feature_stats = {}
        
    def _get_scaler(self):
        """Get the appropriate scaler based on configuration."""
        if self.config.scaler_type == "robust":
            return RobustScaler()
        else:
            return StandardScaler()
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate input data quality."""
        logger.info("Validating data quality...")
        
        # Check for required columns
        missing_cols = set(self.config.feature_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for null values
        null_counts = df[self.config.feature_columns].isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Found null values: {null_counts[null_counts > 0]}")
        
        # Check for infinite values
        inf_counts = np.isinf(df[self.config.feature_columns]).sum()
        if inf_counts.sum() > 0:
            logger.warning(f"Found infinite values: {inf_counts[inf_counts > 0]}")
        
        # Check data types
        for col in self.config.feature_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.error(f"Column {col} is not numeric")
                return False
        
        logger.info("Data validation completed successfully")
        return True
    
    def create_sequences(self, df: pd.DataFrame, equipment_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        logger.info(f"Creating sequences for equipment {equipment_id}")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Extract features
        features = df[self.config.feature_columns].values
        
        # Create sequences
        X, y = [], []
        
        for i in range(len(features) - self.config.sequence_length):
            # Input sequence
            sequence = features[i:i + self.config.sequence_length]
            
            # Create targets for different horizons
            targets = []
            for horizon in self.config.prediction_horizons:
                if i + self.config.sequence_length + horizon < len(features):
                    # Check if failure occurs within horizon
                    failure_occurred = self._check_failure_in_horizon(df, i + self.config.sequence_length, horizon)
                    targets.append(1.0 if failure_occurred else 0.0)
                else:
                    targets.append(0.0)  # No failure if not enough future data
            
            X.append(sequence)
            y.append(targets)
        
        return np.array(X), np.array(y)
    
    def _check_failure_in_horizon(self, df: pd.DataFrame, start_idx: int, horizon: int) -> bool:
        """Check if failure occurs within the specified horizon."""
        end_idx = min(start_idx + horizon, len(df))
        
        # Look for failure indicators in the horizon
        horizon_data = df.iloc[start_idx:end_idx]
        
        # Define failure conditions based on sensor thresholds
        failure_conditions = [
            horizon_data['vibration_x'] > 1.0,
            horizon_data['vibration_y'] > 1.0,
            horizon_data['temperature'] > 100,
            horizon_data['pressure'] < 120,
            horizon_data['current'] > 15
        ]
        
        # Check if any failure condition is met
        return any(condition.any() for condition in failure_conditions)
    
    def extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features from time series data."""
        logger.info("Extracting statistical features...")
        
        # Group by equipment and time windows
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Calculate rolling statistics
        window_sizes = [6, 12, 24]  # 6h, 12h, 24h windows
        
        for col in self.config.feature_columns:
            for window in window_sizes:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window=window, min_periods=1).mean()
                
                # Rolling std
                df[f'{col}_rolling_std_{window}h'] = df[col].rolling(window=window, min_periods=1).std()
                
                # Rolling min/max
                df[f'{col}_rolling_min_{window}h'] = df[col].rolling(window=window, min_periods=1).min()
                df[f'{col}_rolling_max_{window}h'] = df[col].rolling(window=window, min_periods=1).max()
        
        # Add cyclical features for time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def fit_scaler(self, df: pd.DataFrame) -> None:
        """Fit the scaler on training data."""
        logger.info("Fitting scaler on training data...")
        
        # Calculate feature statistics
        self.feature_stats = {
            'mean': df[self.config.feature_columns].mean(),
            'std': df[self.config.feature_columns].std(),
            'min': df[self.config.feature_columns].min(),
            'max': df[self.config.feature_columns].max()
        }
        
        # Fit scaler
        self.scaler.fit(df[self.config.feature_columns])
        logger.info("Scaler fitted successfully")
    
    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler."""
        return self.scaler.transform(df[self.config.feature_columns])
    
    def inverse_transform_features(self, features: np.ndarray) -> np.ndarray:
        """Inverse transform features back to original scale."""
        return self.scaler.inverse_transform(features)
    
    def process_equipment_data(self, df: pd.DataFrame, equipment_id: str) -> Dict[str, np.ndarray]:
        """Process data for a single equipment."""
        logger.info(f"Processing data for equipment {equipment_id}")
        
        # Validate data
        if not self.validate_data(df):
            raise ValueError(f"Data validation failed for equipment {equipment_id}")
        
        # Extract statistical features
        df = self.extract_statistical_features(df)
        
        # Create sequences
        X, y = self.create_sequences(df, equipment_id)
        
        return {
            'features': X,
            'targets': y,
            'equipment_id': equipment_id,
            'num_sequences': len(X)
        }
    
    def prepare_training_data(self, data_dir: Path) -> Dict[str, np.ndarray]:
        """Prepare training data from multiple equipment."""
        logger.info("Preparing training data...")
        
        all_features = []
        all_targets = []
        equipment_ids = []
        
        # Process each equipment file
        for file_path in data_dir.glob("*.parquet"):
            equipment_id = file_path.stem
            df = pd.read_parquet(file_path)
            
            try:
                processed_data = self.process_equipment_data(df, equipment_id)
                all_features.append(processed_data['features'])
                all_targets.append(processed_data['targets'])
                equipment_ids.extend([equipment_id] * processed_data['num_sequences'])
                
            except Exception as e:
                logger.error(f"Error processing {equipment_id}: {e}")
                continue
        
        # Combine all data
        if not all_features:
            raise ValueError("No valid data found")
        
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_targets, axis=0)
        
        logger.info(f"Prepared training data: {X.shape[0]} sequences, {X.shape[1]} time steps, {X.shape[2]} features")
        
        return {
            'features': X,
            'targets': y,
            'equipment_ids': equipment_ids
        }

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Feature Engineering for Predictive Maintenance")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory with raw data")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--config-file", type=str, help="Configuration file path")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = FeatureConfig(**config_dict)
    else:
        config = FeatureConfig()
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(config)
    
    # Process data
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare training data
    training_data = feature_engineer.prepare_training_data(input_path)
    
    # Save processed data
    np.save(output_path / "features.npy", training_data['features'])
    np.save(output_path / "targets.npy", training_data['targets'])
    
    # Save equipment IDs
    with open(output_path / "equipment_ids.txt", 'w') as f:
        for eq_id in training_data['equipment_ids']:
            f.write(f"{eq_id}\n")
    
    # Save scaler
    import joblib
    joblib.dump(feature_engineer.scaler, output_path / "scaler.pkl")
    
    logger.info(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    main()
