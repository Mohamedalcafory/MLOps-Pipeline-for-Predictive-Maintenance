"""
Data Drift Detection for Predictive Maintenance
Monitors data distribution changes and model performance degradation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    reference_window_days: int = 30
    detection_window_days: int = 7
    drift_threshold: float = 0.15
    statistical_test_alpha: float = 0.05
    min_sample_size: int = 100
    feature_columns: List[str] = None
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                'vibration_x', 'vibration_y', 'vibration_z',
                'temperature', 'pressure', 'current', 'voltage', 'flow_rate'
            ]

class DriftDetector:
    """Detects data drift in sensor data and model performance."""
    
    def __init__(self, config: DriftConfig):
        self.config = config
        self.reference_data = None
        self.reference_stats = {}
        self.drift_history = []
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)  # For dimensionality reduction
        
    def set_reference_data(self, data: pd.DataFrame) -> None:
        """Set reference data for drift detection."""
        logger.info("Setting reference data for drift detection")
        
        # Filter to recent data
        if 'timestamp' in data.columns:
            cutoff_date = data['timestamp'].max() - timedelta(days=self.config.reference_window_days)
            reference_data = data[data['timestamp'] >= cutoff_date].copy()
        else:
            reference_data = data.tail(self.config.min_sample_size * 2).copy()
        
        # Calculate reference statistics
        self.reference_data = reference_data
        self.reference_stats = self._calculate_statistics(reference_data)
        
        # Fit scaler and PCA on reference data
        feature_data = reference_data[self.config.feature_columns].dropna()
        if len(feature_data) > 0:
            self.scaler.fit(feature_data)
            self.pca.fit(self.scaler.transform(feature_data))
        
        logger.info(f"Reference data set with {len(reference_data)} samples")
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate statistical measures for drift detection."""
        stats_dict = {}
        
        for col in self.config.feature_columns:
            if col in data.columns:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    stats_dict[col] = {
                        'mean': col_data.mean(),
                        'std': col_data.std(),
                        'median': col_data.median(),
                        'q25': col_data.quantile(0.25),
                        'q75': col_data.quantile(0.75),
                        'skewness': col_data.skew(),
                        'kurtosis': col_data.kurtosis(),
                        'min': col_data.min(),
                        'max': col_data.max()
                    }
        
        return stats_dict
    
    def calculate_drift(self, current_data: pd.DataFrame) -> float:
        """Calculate overall drift score for current data."""
        if self.reference_data is None:
            logger.warning("No reference data set. Returning 0 drift score.")
            return 0.0
        
        if len(current_data) < self.config.min_sample_size:
            logger.warning(f"Insufficient current data: {len(current_data)} samples")
            return 0.0
        
        # Calculate drift scores for different methods
        drift_scores = []
        
        # 1. Statistical drift (KS test)
        ks_scores = self._calculate_ks_drift(current_data)
        if ks_scores:
            drift_scores.extend(ks_scores)
        
        # 2. Distribution drift (Jensen-Shannon divergence)
        js_scores = self._calculate_js_drift(current_data)
        if js_scores:
            drift_scores.extend(js_scores)
        
        # 3. PCA drift
        pca_score = self._calculate_pca_drift(current_data)
        if pca_score is not None:
            drift_scores.append(pca_score)
        
        # 4. Statistical moment drift
        moment_scores = self._calculate_moment_drift(current_data)
        if moment_scores:
            drift_scores.extend(moment_scores)
        
        if not drift_scores:
            logger.warning("No drift scores calculated")
            return 0.0
        
        # Calculate overall drift score (average of all methods)
        overall_drift = np.mean(drift_scores)
        
        # Log drift detection
        drift_info = {
            'timestamp': datetime.utcnow(),
            'overall_drift': overall_drift,
            'method_scores': {
                'ks_drift': np.mean(ks_scores) if ks_scores else None,
                'js_drift': np.mean(js_scores) if js_scores else None,
                'pca_drift': pca_score,
                'moment_drift': np.mean(moment_scores) if moment_scores else None
            },
            'sample_size': len(current_data)
        }
        
        self.drift_history.append(drift_info)
        
        # Keep only recent history
        if len(self.drift_history) > 100:
            self.drift_history = self.drift_history[-100:]
        
        logger.info(f"Drift score calculated: {overall_drift:.4f}")
        return overall_drift
    
    def _calculate_ks_drift(self, current_data: pd.DataFrame) -> List[float]:
        """Calculate Kolmogorov-Smirnov test drift scores."""
        ks_scores = []
        
        for col in self.config.feature_columns:
            if col in current_data.columns and col in self.reference_stats:
                ref_data = self.reference_data[col].dropna()
                cur_data = current_data[col].dropna()
                
                if len(ref_data) > 0 and len(cur_data) > 0:
                    try:
                        # Perform KS test
                        ks_statistic, p_value = stats.ks_2samp(ref_data, cur_data)
                        
                        # Convert to drift score (0-1, where 1 is high drift)
                        drift_score = 1 - p_value if p_value < self.config.statistical_test_alpha else 0
                        ks_scores.append(drift_score)
                        
                    except Exception as e:
                        logger.warning(f"KS test failed for {col}: {e}")
        
        return ks_scores
    
    def _calculate_js_drift(self, current_data: pd.DataFrame) -> List[float]:
        """Calculate Jensen-Shannon divergence drift scores."""
        js_scores = []
        
        for col in self.config.feature_columns:
            if col in current_data.columns and col in self.reference_stats:
                ref_data = self.reference_data[col].dropna()
                cur_data = current_data[col].dropna()
                
                if len(ref_data) > 0 and len(cur_data) > 0:
                    try:
                        # Create histograms for comparison
                        bins = np.linspace(
                            min(ref_data.min(), cur_data.min()),
                            max(ref_data.max(), cur_data.max()),
                            50
                        )
                        
                        ref_hist, _ = np.histogram(ref_data, bins=bins, density=True)
                        cur_hist, _ = np.histogram(cur_data, bins=bins, density=True)
                        
                        # Calculate Jensen-Shannon divergence
                        js_divergence = jensenshannon(ref_hist, cur_hist)
                        
                        # Normalize to 0-1 range (JS divergence is 0-1)
                        js_scores.append(js_divergence)
                        
                    except Exception as e:
                        logger.warning(f"JS divergence calculation failed for {col}: {e}")
        
        return js_scores
    
    def _calculate_pca_drift(self, current_data: pd.DataFrame) -> Optional[float]:
        """Calculate PCA-based drift score."""
        try:
            # Prepare feature data
            feature_data = current_data[self.config.feature_columns].dropna()
            
            if len(feature_data) == 0:
                return None
            
            # Transform data
            scaled_data = self.scaler.transform(feature_data)
            pca_data = self.pca.transform(scaled_data)
            
            # Calculate distance from reference distribution
            ref_pca = self.pca.transform(self.scaler.transform(
                self.reference_data[self.config.feature_columns].dropna()
            ))
            
            # Calculate Mahalanobis distance
            ref_mean = np.mean(ref_pca, axis=0)
            ref_cov = np.cov(ref_pca.T)
            
            if np.linalg.det(ref_cov) > 1e-10:  # Check if covariance is invertible
                inv_cov = np.linalg.inv(ref_cov)
                distances = []
                
                for point in pca_data:
                    diff = point - ref_mean
                    distance = np.sqrt(diff.T @ inv_cov @ diff)
                    distances.append(distance)
                
                # Convert to drift score
                avg_distance = np.mean(distances)
                pca_score = min(1.0, avg_distance / 10.0)  # Normalize
                return pca_score
            
        except Exception as e:
            logger.warning(f"PCA drift calculation failed: {e}")
        
        return None
    
    def _calculate_moment_drift(self, current_data: pd.DataFrame) -> List[float]:
        """Calculate statistical moment drift scores."""
        moment_scores = []
        
        for col in self.config.feature_columns:
            if col in current_data.columns and col in self.reference_stats:
                cur_data = current_data[col].dropna()
                
                if len(cur_data) > 0:
                    try:
                        # Calculate current statistics
                        cur_mean = cur_data.mean()
                        cur_std = cur_data.std()
                        cur_skew = cur_data.skew()
                        cur_kurt = cur_data.kurtosis()
                        
                        # Get reference statistics
                        ref_stats = self.reference_stats[col]
                        
                        # Calculate relative differences
                        mean_diff = abs(cur_mean - ref_stats['mean']) / (abs(ref_stats['mean']) + 1e-8)
                        std_diff = abs(cur_std - ref_stats['std']) / (abs(ref_stats['std']) + 1e-8)
                        skew_diff = abs(cur_skew - ref_stats['skewness']) / (abs(ref_stats['skewness']) + 1e-8)
                        kurt_diff = abs(cur_kurt - ref_stats['kurtosis']) / (abs(ref_stats['kurtosis']) + 1e-8)
                        
                        # Average moment drift
                        moment_score = np.mean([mean_diff, std_diff, skew_diff, kurt_diff])
                        moment_scores.append(min(1.0, moment_score))
                        
                    except Exception as e:
                        logger.warning(f"Moment drift calculation failed for {col}: {e}")
        
        return moment_scores
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict:
        """Detect if significant drift has occurred."""
        drift_score = self.calculate_drift(current_data)
        
        drift_detected = drift_score > self.config.drift_threshold
        
        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'threshold': self.config.drift_threshold,
            'timestamp': datetime.utcnow(),
            'sample_size': len(current_data)
        }
    
    def get_drift_trend(self, window_days: int = 7) -> Dict:
        """Get drift trend over time."""
        if not self.drift_history:
            return {'trend': 'insufficient_data', 'drift_scores': []}
        
        # Filter recent history
        cutoff_time = datetime.utcnow() - timedelta(days=window_days)
        recent_history = [
            entry for entry in self.drift_history 
            if entry['timestamp'] >= cutoff_time
        ]
        
        if len(recent_history) < 2:
            return {'trend': 'insufficient_data', 'drift_scores': []}
        
        # Calculate trend
        drift_scores = [entry['overall_drift'] for entry in recent_history]
        
        # Simple linear trend
        x = np.arange(len(drift_scores))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, drift_scores)
        
        if slope > 0.01:
            trend = 'increasing'
        elif slope < -0.01:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': slope,
            'r_squared': r_value ** 2,
            'drift_scores': drift_scores,
            'mean_drift': np.mean(drift_scores),
            'std_drift': np.std(drift_scores)
        }
    
    def get_feature_drift_analysis(self, current_data: pd.DataFrame) -> Dict:
        """Get detailed drift analysis by feature."""
        if self.reference_data is None:
            return {}
        
        feature_analysis = {}
        
        for col in self.config.feature_columns:
            if col in current_data.columns and col in self.reference_stats:
                ref_data = self.reference_data[col].dropna()
                cur_data = current_data[col].dropna()
                
                if len(ref_data) > 0 and len(cur_data) > 0:
                    try:
                        # KS test
                        ks_stat, ks_p = stats.ks_2samp(ref_data, cur_data)
                        
                        # Distribution comparison
                        bins = np.linspace(
                            min(ref_data.min(), cur_data.min()),
                            max(ref_data.max(), cur_data.max()),
                            30
                        )
                        ref_hist, _ = np.histogram(ref_data, bins=bins, density=True)
                        cur_hist, _ = np.histogram(cur_data, bins=bins, density=True)
                        js_div = jensenshannon(ref_hist, cur_hist)
                        
                        # Statistical moments
                        ref_stats = self.reference_stats[col]
                        cur_mean = cur_data.mean()
                        cur_std = cur_data.std()
                        cur_skew = cur_data.skew()
                        cur_kurt = cur_data.kurtosis()
                        
                        feature_analysis[col] = {
                            'ks_statistic': ks_stat,
                            'ks_p_value': ks_p,
                            'js_divergence': js_div,
                            'mean_change': (cur_mean - ref_stats['mean']) / (abs(ref_stats['mean']) + 1e-8),
                            'std_change': (cur_std - ref_stats['std']) / (abs(ref_stats['std']) + 1e-8),
                            'skewness_change': cur_skew - ref_stats['skewness'],
                            'kurtosis_change': cur_kurt - ref_stats['kurtosis'],
                            'reference_stats': ref_stats,
                            'current_stats': {
                                'mean': cur_mean,
                                'std': cur_std,
                                'skewness': cur_skew,
                                'kurtosis': cur_kurt
                            }
                        }
                        
                    except Exception as e:
                        logger.warning(f"Feature drift analysis failed for {col}: {e}")
        
        return feature_analysis
    
    def reset_reference_data(self) -> None:
        """Reset reference data (useful for model retraining)."""
        logger.info("Resetting reference data")
        self.reference_data = None
        self.reference_stats = {}
        self.drift_history = []
    
    def save_drift_state(self, filepath: str) -> None:
        """Save drift detection state to file."""
        import pickle
        
        state = {
            'reference_data': self.reference_data,
            'reference_stats': self.reference_stats,
            'drift_history': self.drift_history,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Drift state saved to {filepath}")
    
    def load_drift_state(self, filepath: str) -> None:
        """Load drift detection state from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.reference_data = state['reference_data']
        self.reference_stats = state['reference_stats']
        self.drift_history = state['drift_history']
        self.config = state['config']
        
        logger.info(f"Drift state loaded from {filepath}")
