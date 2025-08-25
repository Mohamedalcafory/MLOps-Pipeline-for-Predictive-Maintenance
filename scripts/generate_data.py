#!/usr/bin/env python3
"""
Synthetic Sensor Data Generator
Generates realistic equipment sensor data for predictive maintenance testing.
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

# Import the data generator from the main module
from data_generator import EquipmentSimulator, EquipmentType, FailureMode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_equipment_data(
    equipment_id: str,
    equipment_type: EquipmentType,
    start_date: datetime,
    duration_days: int,
    failure_probability: float = 0.1
) -> pd.DataFrame:
    """Generate sensor data for a single equipment."""
    logger.info(f"Generating data for {equipment_id} ({equipment_type.value})")
    
    # Create equipment simulator
    simulator = EquipmentSimulator(equipment_id, equipment_type)
    
    # Generate data
    data = simulator.generate_sensor_data(
        start_date=start_date,
        duration_days=duration_days,
        failure_probability=failure_probability
    )
    
    return data

def generate_multiple_equipment(
    num_equipment: int,
    equipment_types: List[EquipmentType],
    start_date: datetime,
    duration_days: int,
    output_dir: Path,
    failure_probability: float = 0.1
) -> None:
    """Generate data for multiple equipment."""
    logger.info(f"Generating data for {num_equipment} equipment over {duration_days} days")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data for each equipment
    for i in range(num_equipment):
        # Select equipment type
        equipment_type = random.choice(equipment_types)
        equipment_id = f"{equipment_type.value}_{i+1:03d}"
        
        # Generate data
        data = generate_equipment_data(
            equipment_id=equipment_id,
            equipment_type=equipment_type,
            start_date=start_date,
            duration_days=duration_days,
            failure_probability=failure_probability
        )
        
        # Save to file
        output_file = output_dir / f"{equipment_id}.parquet"
        data.to_parquet(output_file, index=False)
        
        logger.info(f"Saved data for {equipment_id} to {output_file}")
    
    # Create metadata file
    metadata = {
        'num_equipment': num_equipment,
        'equipment_types': [et.value for et in equipment_types],
        'start_date': start_date.isoformat(),
        'duration_days': duration_days,
        'failure_probability': failure_probability,
        'generated_at': datetime.utcnow().isoformat(),
        'total_files': num_equipment
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Generated {num_equipment} equipment data files")
    logger.info(f"Metadata saved to {metadata_file}")

def create_sample_request_data(
    equipment_id: str,
    equipment_type: EquipmentType,
    output_file: Path
) -> None:
    """Create a sample request file for API testing."""
    logger.info(f"Creating sample request for {equipment_id}")
    
    # Generate recent data (last 7 days)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    
    # Generate data
    data = generate_equipment_data(
        equipment_id=equipment_id,
        equipment_type=equipment_type,
        start_date=start_date,
        duration_days=7,
        failure_probability=0.05  # Low failure probability for sample
    )
    
    # Convert to API request format
    sensor_readings = []
    for _, row in data.iterrows():
        reading = {
            "timestamp": row['timestamp'].isoformat(),
            "vibration_x": float(row['vibration_x']),
            "vibration_y": float(row['vibration_y']),
            "vibration_z": float(row['vibration_z']),
            "temperature": float(row['temperature']),
            "pressure": float(row['pressure']),
            "current": float(row['current']),
            "voltage": float(row['voltage']),
            "flow_rate": float(row['flow_rate'])
        }
        
        # Add equipment-specific sensors
        if equipment_type == EquipmentType.PUMP:
            reading.update({
                "suction_pressure": float(row['suction_pressure']),
                "discharge_pressure": float(row['discharge_pressure'])
            })
        elif equipment_type == EquipmentType.MOTOR:
            reading.update({
                "rpm": float(row['rpm']),
                "power_factor": float(row['power_factor'])
            })
        
        sensor_readings.append(reading)
    
    # Create request payload
    request_payload = {
        "equipment_id": equipment_id,
        "equipment_type": equipment_type.value,
        "sensor_data": sensor_readings
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(request_payload, f, indent=2)
    
    logger.info(f"Sample request saved to {output_file}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate synthetic sensor data")
    parser.add_argument("--num-equipment", type=int, default=50, 
                       help="Number of equipment to generate data for")
    parser.add_argument("--duration-days", type=int, default=365,
                       help="Duration of data generation in days")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                       help="Output directory for generated data")
    parser.add_argument("--equipment-types", nargs="+", 
                       choices=['pump', 'motor', 'compressor', 'turbine'],
                       default=['pump', 'motor', 'compressor', 'turbine'],
                       help="Types of equipment to generate")
    parser.add_argument("--failure-probability", type=float, default=0.1,
                       help="Probability of equipment failure during the period")
    parser.add_argument("--start-date", type=str, default=None,
                       help="Start date for data generation (YYYY-MM-DD)")
    parser.add_argument("--sample-request", action="store_true",
                       help="Generate sample API request file")
    
    args = parser.parse_args()
    
    # Parse start date
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_date = datetime.utcnow() - timedelta(days=args.duration_days)
    
    # Convert equipment types
    equipment_types = [EquipmentType(et) for et in args.equipment_types]
    
    # Generate data
    output_dir = Path(args.output_dir)
    generate_multiple_equipment(
        num_equipment=args.num_equipment,
        equipment_types=equipment_types,
        start_date=start_date,
        duration_days=args.duration_days,
        output_dir=output_dir,
        failure_probability=args.failure_probability
    )
    
    # Generate sample request if requested
    if args.sample_request:
        sample_equipment_id = f"{equipment_types[0].value}_001"
        sample_request_file = output_dir / "sample_request.json"
        create_sample_request_data(
            equipment_id=sample_equipment_id,
            equipment_type=equipment_types[0],
            output_file=sample_request_file
        )
    
    logger.info("Data generation completed successfully!")

if __name__ == "__main__":
    main()
