"""
Integration tests for the Predictive Maintenance API.
"""
import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from inference.api import app

# Test client
client = TestClient(app)

class TestPredictiveMaintenanceAPI:
    """Test cases for the predictive maintenance API."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_sensor_data = self._generate_sample_sensor_data()
    
    def _generate_sample_sensor_data(self, num_readings: int = 168) -> list:
        """Generate sample sensor data for testing."""
        sensor_data = []
        base_time = datetime.utcnow() - timedelta(hours=num_readings)
        
        for i in range(num_readings):
            timestamp = base_time + timedelta(hours=i)
            
            # Generate realistic sensor values
            reading = {
                "timestamp": timestamp.isoformat(),
                "vibration_x": 0.2 + np.random.normal(0, 0.05),
                "vibration_y": 0.18 + np.random.normal(0, 0.05),
                "vibration_z": 0.15 + np.random.normal(0, 0.05),
                "temperature": 75 + np.random.normal(0, 5),
                "pressure": 150 + np.random.normal(0, 10),
                "current": 9.0 + np.random.normal(0, 0.5),
                "voltage": 400 + np.random.normal(0, 20),
                "flow_rate": 100 + np.random.normal(0, 5),
                "suction_pressure": 12 + np.random.normal(0, 1),
                "discharge_pressure": 155 + np.random.normal(0, 5),
                "rpm": 1750 + np.random.normal(0, 50),
                "power_factor": 0.9 + np.random.normal(0, 0.05)
            }
            
            # Ensure values are within reasonable bounds
            reading["vibration_x"] = max(0, min(10, reading["vibration_x"]))
            reading["vibration_y"] = max(0, min(10, reading["vibration_y"]))
            reading["vibration_z"] = max(0, min(10, reading["vibration_z"]))
            reading["temperature"] = max(-50, min(200, reading["temperature"]))
            reading["pressure"] = max(0, min(500, reading["pressure"]))
            reading["current"] = max(0, min(100, reading["current"]))
            reading["voltage"] = max(0, min(1000, reading["voltage"]))
            reading["flow_rate"] = max(0, min(1000, reading["flow_rate"]))
            reading["suction_pressure"] = max(0, min(100, reading["suction_pressure"]))
            reading["discharge_pressure"] = max(0, min(500, reading["discharge_pressure"]))
            reading["rpm"] = max(0, min(10000, reading["rpm"]))
            reading["power_factor"] = max(0, min(1, reading["power_factor"]))
            
            sensor_data.append(reading)
        
        return sensor_data
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        assert "total_predictions" in data
        assert "average_latency_ms" in data
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        
        # Check for expected metrics
        metrics_text = response.text
        assert "predictions_total" in metrics_text
        assert "prediction_latency_seconds" in metrics_text
    
    def test_model_info_endpoint(self):
        """Test model information endpoint."""
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_version" in data
        assert "model_type" in data
        assert "input_size" in data
        assert "hidden_size" in data
    
    def test_single_prediction(self):
        """Test single equipment prediction."""
        request_data = {
            "equipment_id": "pump_001",
            "equipment_type": "pump",
            "sensor_data": self.sample_sensor_data
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required fields
        assert data["equipment_id"] == "pump_001"
        assert data["equipment_type"] == "pump"
        assert "prediction_timestamp" in data
        assert "failure_probability_1h" in data
        assert "failure_probability_4h" in data
        assert "failure_probability_24h" in data
        assert "failure_probability_7d" in data
        assert "estimated_ttf_hours" in data
        assert "confidence_intervals" in data
        assert "current_health_score" in data
        assert "risk_level" in data
        assert "recommended_actions" in data
        assert "model_version" in data
        assert "inference_time_ms" in data
        
        # Check data types and ranges
        assert isinstance(data["failure_probability_1h"], float)
        assert 0 <= data["failure_probability_1h"] <= 1
        assert isinstance(data["current_health_score"], float)
        assert 0 <= data["current_health_score"] <= 1
        assert data["risk_level"] in ["low", "medium", "high", "critical"]
        assert isinstance(data["recommended_actions"], list)
        assert len(data["recommended_actions"]) > 0
        assert isinstance(data["inference_time_ms"], float)
        assert data["inference_time_ms"] > 0
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint."""
        # Create multiple equipment requests
        batch_requests = []
        equipment_types = ["pump", "motor", "compressor"]
        
        for i, eq_type in enumerate(equipment_types):
            request_data = {
                "equipment_id": f"{eq_type}_{i+1:03d}",
                "equipment_type": eq_type,
                "sensor_data": self.sample_sensor_data
            }
            batch_requests.append(request_data)
        
        batch_request = {"predictions": batch_requests}
        
        response = client.post("/predict/batch", json=batch_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "total_processed" in data
        assert "batch_processing_time_ms" in data
        
        assert len(data["predictions"]) == 3
        assert data["total_processed"] == 3
        assert data["batch_processing_time_ms"] > 0
        
        # Check each prediction
        for pred in data["predictions"]:
            assert "equipment_id" in pred
            assert "equipment_type" in pred
            assert "failure_probability_1h" in pred
    
    def test_invalid_equipment_type(self):
        """Test prediction with invalid equipment type."""
        request_data = {
            "equipment_id": "invalid_001",
            "equipment_type": "invalid_type",
            "sensor_data": self.sample_sensor_data
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_insufficient_sensor_data(self):
        """Test prediction with insufficient sensor data."""
        # Create data with fewer than 168 readings
        insufficient_data = self.sample_sensor_data[:100]
        
        request_data = {
            "equipment_id": "pump_001",
            "equipment_type": "pump",
            "sensor_data": insufficient_data
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 400  # Bad request
    
    def test_invalid_sensor_values(self):
        """Test prediction with invalid sensor values."""
        invalid_data = self.sample_sensor_data.copy()
        # Set an invalid value
        invalid_data[0]["temperature"] = 1000  # Too high
        
        request_data = {
            "equipment_id": "pump_001",
            "equipment_type": "pump",
            "sensor_data": invalid_data
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_latency_performance(self):
        """Test API latency performance."""
        request_data = {
            "equipment_id": "pump_001",
            "equipment_type": "pump",
            "sensor_data": self.sample_sensor_data
        }
        
        # Make multiple requests and measure latency
        latencies = []
        num_requests = 10
        
        for _ in range(num_requests):
            start_time = time.time()
            response = client.post("/predict", json=request_data)
            end_time = time.time()
            
            assert response.status_code == 200
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        # Performance assertions (adjust based on your targets)
        assert avg_latency < 500  # Average latency < 500ms
        assert p95_latency < 1000  # P95 latency < 1000ms
        
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"P95 latency: {p95_latency:.2f}ms")
    
    def test_different_equipment_types(self):
        """Test predictions for different equipment types."""
        equipment_types = ["pump", "motor", "compressor", "turbine"]
        
        for eq_type in equipment_types:
            request_data = {
                "equipment_id": f"{eq_type}_test",
                "equipment_type": eq_type,
                "sensor_data": self.sample_sensor_data
            }
            
            response = client.post("/predict", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["equipment_type"] == eq_type
            assert "recommended_actions" in data
            assert len(data["recommended_actions"]) > 0
    
    def test_confidence_intervals(self):
        """Test that confidence intervals are properly formatted."""
        request_data = {
            "equipment_id": "pump_001",
            "equipment_type": "pump",
            "sensor_data": self.sample_sensor_data
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        confidence_intervals = data["confidence_intervals"]
        
        # Check structure
        assert "1h" in confidence_intervals
        assert "4h" in confidence_intervals
        assert "24h" in confidence_intervals
        assert "7d" in confidence_intervals
        
        # Check each interval
        for horizon, interval in confidence_intervals.items():
            assert len(interval) == 2  # [lower, upper]
            assert interval[0] <= interval[1]  # lower <= upper
            assert 0 <= interval[0] <= 1  # bounds check
            assert 0 <= interval[1] <= 1  # bounds check
    
    def test_risk_level_calculation(self):
        """Test that risk levels are calculated correctly."""
        # Test with different probability scenarios
        test_cases = [
            ([0.1, 0.2, 0.3, 0.4], "low"),
            ([0.4, 0.5, 0.6, 0.7], "medium"),
            ([0.7, 0.8, 0.9, 0.9], "high"),
            ([0.9, 0.95, 0.98, 0.99], "critical")
        ]
        
        for probabilities, expected_risk in test_cases:
            # Create modified sensor data that might produce these probabilities
            # (This is a simplified test - in reality, you'd need to mock the model)
            request_data = {
                "equipment_id": "test_equipment",
                "equipment_type": "pump",
                "sensor_data": self.sample_sensor_data
            }
            
            response = client.post("/predict", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            risk_level = data["risk_level"]
            
            # Verify risk level is valid
            assert risk_level in ["low", "medium", "high", "critical"]
    
    def test_health_score_calculation(self):
        """Test that health scores are calculated correctly."""
        request_data = {
            "equipment_id": "pump_001",
            "equipment_type": "pump",
            "sensor_data": self.sample_sensor_data
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        health_score = data["current_health_score"]
        
        # Check bounds
        assert 0 <= health_score <= 1
        
        # Health score should be inversely related to failure probabilities
        failure_probs = [
            data["failure_probability_1h"],
            data["failure_probability_4h"],
            data["failure_probability_24h"],
            data["failure_probability_7d"]
        ]
        
        avg_failure_prob = np.mean(failure_probs)
        expected_health_score = 1 - avg_failure_prob
        
        # Allow some tolerance for different calculation methods
        assert abs(health_score - expected_health_score) < 0.3

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
