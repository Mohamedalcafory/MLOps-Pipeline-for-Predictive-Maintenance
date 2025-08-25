#!/usr/bin/env python3
"""
Synthetic Sensor Data Generator for Predictive Maintenance
Generates realistic equipment sensor data with failure patterns.
"""
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EquipmentType(Enum):
    PUMP = "pump"
    MOTOR = "motor"
    COMPRESSOR = "compressor"
    TURBINE = "turbine"

class FailureMode(Enum):
    BEARING_WEAR = "bearing_wear"
    SEAL_FAILURE = "seal_failure"
    IMPELLER_DAMAGE = "impeller_damage"
    WINDING_OVERHEATING = "winding_overheating"
    ROTOR_IMBALANCE = "rotor_imbalance"
    VALVE_LEAKAGE = "valve_leakage"
    PISTON_WEAR = "piston_wear"
    COOLING_FAILURE = "cooling_failure"

@dataclass
class SensorConfig:
    """Configuration for sensor data generation."""
    name: str
    normal_range: Tuple[float, float]
    failure_range: Tuple[float, float]
    unit: str
    noise_std: float = 0.02

class EquipmentSimulator:
    """Simulates equipment sensor data with realistic failure patterns."""
    
    def __init__(self, equipment_id: str, equipment_type: EquipmentType):
        self.equipment_id = equipment_id
        self.equipment_type = equipment_type
        self.current_health = 1.0  # 1.0 = perfect health, 0.0 = failed
        self.failure_mode = None
        self.failure_start_time = None
        
        # Define sensor configurations for different equipment types
        self.sensor_configs = self._get_sensor_configs()
        
        # Failure patterns
        self.failure_modes = self._get_failure_modes()
    
    def _get_sensor_configs(self) -> Dict[str, SensorConfig]:
        """Get sensor configurations based on equipment type."""
        base_configs = {
            'vibration_x': SensorConfig('vibration_x', (0.1, 0.3), (0.8, 1.5), 'mm/s', 0.01),
            'vibration_y': SensorConfig('vibration_y', (0.1, 0.3), (0.8, 1.5), 'mm/s', 0.01),
            'vibration_z': SensorConfig('vibration_z', (0.1, 0.3), (0.8, 1.5), 'mm/s', 0.01),
            'temperature': SensorConfig('temperature', (70, 85), (95, 120), '°C', 0.5),
            'pressure': SensorConfig('pressure', (145, 155), (120, 140), 'psi', 0.3),
            'current': SensorConfig('current', (8.5, 9.5), (12, 16), 'A', 0.05),
            'voltage': SensorConfig('voltage', (380, 420), (350, 380), 'V', 1.0),
            'flow_rate': SensorConfig('flow_rate', (95, 105), (60, 90), 'L/min', 0.2),
        }
        
        # Equipment-specific sensors
        if self.equipment_type == EquipmentType.PUMP:
            base_configs.update({
                'suction_pressure': SensorConfig('suction_pressure', (10, 15), (5, 10), 'psi', 0.1),
                'discharge_pressure': SensorConfig('discharge_pressure', (150, 160), (120, 140), 'psi', 0.2),
            })
        elif self.equipment_type == EquipmentType.MOTOR:
            base_configs.update({
                'rpm': SensorConfig('rpm', (1750, 1800), (1600, 1750), 'rpm', 2.0),
                'power_factor': SensorConfig('power_factor', (0.85, 0.95), (0.6, 0.8), '', 0.01),
            })
        elif self.equipment_type == EquipmentType.COMPRESSOR:
            base_configs.update({
                'inlet_pressure': SensorConfig('inlet_pressure', (14.5, 15.5), (12, 14), 'psi', 0.1),
                'outlet_pressure': SensorConfig('outlet_pressure', (120, 130), (90, 110), 'psi', 0.3),
            })
        
        return base_configs
    
    def _get_failure_modes(self) -> List[FailureMode]:
        """Get possible failure modes for equipment type."""
        failure_map = {
            EquipmentType.PUMP: [FailureMode.BEARING_WEAR, FailureMode.SEAL_FAILURE, FailureMode.IMPELLER_DAMAGE],
            EquipmentType.MOTOR: [FailureMode.BEARING_WEAR, FailureMode.WINDING_OVERHEATING, FailureMode.ROTOR_IMBALANCE],
            EquipmentType.COMPRESSOR: [FailureMode.VALVE_LEAKAGE, FailureMode.PISTON_WEAR, FailureMode.COOLING_FAILURE],
            EquipmentType.TURBINE: [FailureMode.BEARING_WEAR, FailureMode.ROTOR_IMBALANCE, FailureMode.COOLING_FAILURE],
        }
        return failure_map.get(self.equipment_type, [FailureMode.BEARING_WEAR])
    
    def _update_health(self, hours_since_start: float):
        """Update equipment health based on time and failure progression."""
        if self.failure_mode is None:
            # Random chance of failure initiation (very low probability)
            if random.random() < 0.0001:  # 0.01% chance per hour
                self.failure_mode = random.choice(self.failure_modes)
                self.failure_start_time = hours_since_start
                logger.info(f"Failure initiated for {self.equipment_id}: {self.failure_mode.value}")
        
        if self.failure_mode is not None:
            # Gradual health degradation
            time_since_failure = hours_since_start - self.failure_start_time
            
            # Different failure progression rates
            if self.failure_mode == FailureMode.BEARING_WEAR:
                # Slow degradation over 200-500 hours
                degradation_rate = 1 / (300 + random.uniform(-100, 200))
            elif self.failure_mode == FailureMode.SEAL_FAILURE:
                # Medium degradation over 100-200 hours
                degradation_rate = 1 / (150 + random.uniform(-50, 50))
            else:
                # Fast degradation over 50-100 hours
                degradation_rate = 1 / (75 + random.uniform(-25, 25))
            
            self.current_health = max(0.0, 1.0 - (time_since_failure * degradation_rate))
    
    def generate_sensor_reading(self, timestamp: datetime, hours_since_start: float) -> Dict:
        """Generate a single sensor reading."""
        self._update_health(hours_since_start)
        
        reading = {
            'equipment_id': self.equipment_id,
            'equipment_type': self.equipment_type.value,
            'timestamp': timestamp.isoformat(),
            'health_score': self.current_health,
            'failure_mode': self.failure_mode.value if self.failure_mode else None,
            'time_to_failure_hours': self._calculate_time_to_failure(),
        }
        
        # Generate sensor values
        for sensor_name, config in self.sensor_configs.items():
            value = self._generate_sensor_value(config)
            reading[sensor_name] = round(value, 3)
        
        # Add derived features
        reading.update(self._calculate_derived_features(reading))
        
        return reading
    
    def _generate_sensor_value(self, config: SensorConfig) -> float:
        """Generate a sensor value based on health and failure mode."""
        # Base value interpolation between normal and failure ranges
        normal_min, normal_max = config.normal_range
        failure_min, failure_max = config.failure_range
        
        # Interpolate based on health score
        health_factor = self.current_health
        
        if health_factor > 0.8:
            # Healthy range
            base_value = random.uniform(normal_min, normal_max)
        elif health_factor > 0.5:
            # Degrading - mix of normal and failure values
            weight = (health_factor - 0.5) / 0.3
            normal_val = random.uniform(normal_min, normal_max)
            failure_val = random.uniform(failure_min, failure_max)
            base_value = weight * normal_val + (1 - weight) * failure_val
        else:
            # Failing range
            base_value = random.uniform(failure_min, failure_max)
        
        # Add noise
        noise = random.gauss(0, config.noise_std * base_value)
        
        # Failure mode specific adjustments
        if self.failure_mode:
            base_value = self._apply_failure_mode_effects(config.name, base_value)
        
        return base_value + noise
    
    def _apply_failure_mode_effects(self, sensor_name: str, base_value: float) -> float:
        """Apply failure mode specific effects to sensor readings."""
        if self.failure_mode == FailureMode.BEARING_WEAR:
            if 'vibration' in sensor_name:
                # Increase vibration with bearing wear
                return base_value * (1.2 + (1 - self.current_health) * 0.8)
            elif sensor_name == 'temperature':
                # Increase temperature with bearing wear
                return base_value + (1 - self.current_health) * 20
        
        elif self.failure_mode == FailureMode.WINDING_OVERHEATING:
            if sensor_name == 'temperature':
                # Significant temperature increase
                return base_value + (1 - self.current_health) * 40
            elif sensor_name == 'current':
                # Current fluctuations
                return base_value * (1 + (1 - self.current_health) * 0.3)
        
        elif self.failure_mode == FailureMode.SEAL_FAILURE:
            if sensor_name == 'pressure':
                # Pressure drop with seal failure
                return base_value * (0.8 - (1 - self.current_health) * 0.3)
            elif sensor_name == 'flow_rate':
                # Flow rate reduction
                return base_value * (0.9 - (1 - self.current_health) * 0.4)
        
        return base_value
    
    def _calculate_time_to_failure(self) -> Optional[float]:
        """Calculate estimated time to failure in hours."""
        if self.failure_mode is None:
            return None
        
        if self.current_health <= 0.1:
            return 0.0
        elif self.current_health <= 0.3:
            return random.uniform(1, 24)  # 1-24 hours
        elif self.current_health <= 0.5:
            return random.uniform(24, 72)  # 1-3 days
        elif self.current_health <= 0.7:
            return random.uniform(72, 168)  # 3-7 days
        else:
            return random.uniform(168, 720)  # 1-4 weeks
    
    def _calculate_derived_features(self, reading: Dict) -> Dict:
        """Calculate derived features from raw sensor data."""
        derived = {}
        
        # Vibration magnitude
        vib_x = reading.get('vibration_x', 0)
        vib_y = reading.get('vibration_y', 0)
        vib_z = reading.get('vibration_z', 0)
        derived['vibration_magnitude'] = round(np.sqrt(vib_x**2 + vib_y**2 + vib_z**2), 3)
        
        # Temperature-Current ratio (thermal efficiency indicator)
        temp = reading.get('temperature', 75)
        current = reading.get('current', 9)
        derived['temp_current_ratio'] = round(temp / current, 3)
        
        # Pressure differential (for pumps/compressors)
        if 'suction_pressure' in reading and 'discharge_pressure' in reading:
            derived['pressure_differential'] = round(
                reading['discharge_pressure'] - reading['suction_pressure'], 3
            )
        elif 'inlet_pressure' in reading and 'outlet_pressure' in reading:
            derived['pressure_differential'] = round(
                reading['outlet_pressure'] - reading['inlet_pressure'], 3
            )
        
        # Power estimation
        voltage = reading.get('voltage', 400)
        current = reading.get('current', 9)
        power_factor = reading.get('power_factor', 0.9)
        derived['estimated_power_kw'] = round(voltage * current * power_factor / 1000, 3)
        
        return derived

class DatasetGenerator:
    """Generates complete datasets for predictive maintenance."""
    
    def __init__(self, 
                 num_equipment: int = 50,
                 duration_days: int = 365,
                 sampling_interval_minutes: int = 60):
        self.num_equipment = num_equipment
        self.duration_days = duration_days
        self.sampling_interval = timedelta(minutes=sampling_interval_minutes)
        self.equipment_simulators = []
        
        # Create equipment simulators
        self._create_equipment_simulators()
    
    def _create_equipment_simulators(self):
        """Create equipment simulators for different types."""
        equipment_types = list(EquipmentType)
        
        for i in range(self.num_equipment):
            equipment_type = random.choice(equipment_types)
            equipment_id = f"{equipment_type.value}_{i:03d}"
            
            simulator = EquipmentSimulator(equipment_id, equipment_type)
            self.equipment_simulators.append(simulator)
        
        logger.info(f"Created {len(self.equipment_simulators)} equipment simulators")
    
    def generate_dataset(self, output_path: Path):
        """Generate complete dataset and save to file."""
        logger.info(f"Generating dataset for {self.duration_days} days...")
        
        start_time = datetime(2023, 1, 1)
        end_time = start_time + timedelta(days=self.duration_days)
        
        all_readings = []
        current_time = start_time
        
        while current_time < end_time:
            hours_since_start = (current_time - start_time).total_seconds() / 3600
            
            # Generate readings for all equipment
            for simulator in self.equipment_simulators:
                reading = simulator.generate_sensor_reading(current_time, hours_since_start)
                all_readings.append(reading)
            
            current_time += self.sampling_interval
            
            # Progress logging
            if len(all_readings) % 10000 == 0:
                logger.info(f"Generated {len(all_readings)} readings...")
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_readings)
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet for efficiency
        parquet_path = output_path.with_suffix('.parquet')
        df.to_parquet(parquet_path, index=False)
        
        # Also save as CSV for easy inspection
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Dataset saved to {parquet_path} and {csv_path}")
        logger.info(f"Total readings: {len(all_readings)}")
        logger.info(f"Equipment count: {df['equipment_id'].nunique()}")
        logger.info(f"Failure events: {df['failure_mode'].notna().sum()}")
        
        # Generate summary statistics
        self._generate_summary_stats(df, output_path.parent)
        
        return df
    
    def _generate_summary_stats(self, df: pd.DataFrame, output_dir: Path):
        """Generate and save summary statistics."""
        stats = {
            'dataset_info': {
                'total_readings': len(df),
                'equipment_count': df['equipment_id'].nunique(),
                'time_range': {
                    'start': df['timestamp'].min(),
                    'end': df['timestamp'].max()
                },
                'failure_events': df['failure_mode'].notna().sum(),
                'equipment_types': df['equipment_type'].value_counts().to_dict()
            },
            'sensor_statistics': {},
            'failure_statistics': df.groupby('failure_mode')['equipment_id'].nunique().to_dict()
        }
        
        # Sensor statistics
        sensor_columns = [col for col in df.columns if col not in [
            'equipment_id', 'equipment_type', 'timestamp', 'health_score', 
            'failure_mode', 'time_to_failure_hours'
        ]]
        
        for sensor in sensor_columns:
            stats['sensor_statistics'][sensor] = {
                'mean': float(df[sensor].mean()),
                'std': float(df[sensor].std()),
                'min': float(df[sensor].min()),
                'max': float(df[sensor].max())
            }
        
        # Save statistics
        stats_path = output_dir / 'dataset_summary.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Summary statistics saved to {stats_path}")

def main():
    """Main function for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic predictive maintenance dataset")
    parser.add_argument('--num-equipment', type=int, default=50, 
                       help='Number of equipment to simulate')
    parser.add_argument('--duration-days', type=int, default=365,
                       help='Duration of simulation in days')
    parser.add_argument('--sampling-interval-minutes', type=int, default=60,
                       help='Sampling interval in minutes')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                       help='Output directory for generated data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    generator = DatasetGenerator(
        num_equipment=args.num_equipment,
        duration_days=args.duration_days,
        sampling_interval_minutes=args.sampling_interval_minutes
    )
    
    output_path = output_dir / f'sensor_data_{args.num_equipment}equipment_{args.duration_days}days'
    dataset = generator.generate_dataset(output_path)
    
    logger.info("Dataset generation completed successfully!")
    
    # Display sample data
    print("\nSample of generated data:")
    print(dataset.head())
    print(f"\nDataset shape: {dataset.shape}")
    print(f"Columns: {list(dataset.columns)}")

if __name__ == "__main__":
    main()