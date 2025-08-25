"""
FastAPI Inference Service for Predictive Maintenance
Real-time prediction API with <200ms latency and monitoring.
"""
import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import structlog
import mlflow
import mlflow.pytorch
from sqlalchemy import create_engine
from redis import Redis

from models.lstm_model import PredictiveMaintenanceLSTM, ModelConfig
from data.feature_engineering import FeatureEngineer
from monitoring.drift_detection import DriftDetector

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'predictions_total', 
    'Total number of predictions made',
    ['equipment_type', 'prediction_type', 'status']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    ['equipment_type', 'prediction_type'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
)

MODEL_INFO = Info(
    'model_info',
    'Information about the loaded model'
)

DRIFT_SCORE = Gauge(
    'data_drift_score',
    'Current data drift score',
    ['equipment_type']
)

ERROR_COUNTER = Counter(
    'prediction_errors_total',
    'Total number of prediction errors',
    ['error_type', 'equipment_type']
)

# Pydantic models
class SensorReading(BaseModel):
    """Single sensor reading."""
    timestamp: datetime
    vibration_x: float = Field(..., ge=0, le=10)
    vibration_y: float = Field(..., ge=0, le=10)
    vibration_z: float = Field(..., ge=0, le=10)
    temperature: float = Field(..., ge=-50, le=200)
    pressure: float = Field(..., ge=0, le=500)
    current: float = Field(..., ge=0, le=100)
    voltage: float = Field(..., ge=0, le=1000)
    flow_rate: float = Field(..., ge=0, le=1000)
    
    # Optional equipment-specific sensors
    suction_pressure: Optional[float] = Field(None, ge=0, le=100)
    discharge_pressure: Optional[float] = Field(None, ge=0, le=500)
    rpm: Optional[float] = Field(None, ge=0, le=10000)
    power_factor: Optional[float] = Field(None, ge=0, le=1)

class PredictionRequest(BaseModel):
    """Request for real-time prediction."""
    equipment_id: str
    equipment_type: str = Field(..., regex="^(pump|motor|compressor|turbine)$")
    sensor_data: List[SensorReading] = Field(..., min_items=24, max_items=336)  # 1-14 days
    
    @validator('sensor_data')
    def validate_chronological_order(cls, v):
        """Ensure sensor data is in chronological order."""
        timestamps = [reading.timestamp for reading in v]
        if timestamps != sorted(timestamps):
            raise ValueError("Sensor data must be in chronological order")
        return v

class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    requests: List[PredictionRequest] = Field(..., max_items=100)

class PredictionResponse(BaseModel):
    """Response with prediction results."""
    equipment_id: str
    equipment_type: str
    prediction_timestamp: datetime
    
    # Multi-horizon failure probabilities
    failure_probability_1h: float = Field(..., ge=0, le=1)
    failure_probability_4h: float = Field(..., ge=0, le=1)
    failure_probability_24h: float = Field(..., ge=0, le=1)
    failure_probability_7d: float = Field(..., ge=0, le=1)
    
    # Time to failure estimate
    estimated_ttf_hours: Optional[float] = Field(None, ge=0)
    
    # Confidence intervals
    confidence_intervals: Dict[str, List[float]] = Field(...)  # [lower, upper] for each horizon
    
    # Health and recommendations
    current_health_score: float = Field(..., ge=0, le=1)
    risk_level: str = Field(..., regex="^(low|medium|high|critical)$")
    recommended_actions: List[str]
    
    # Model metadata
    model_version: str
    inference_time_ms: float
    data_drift_score: float

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    model_loaded: bool
    model_version: Optional[str]
    uptime_seconds: float

class ModelRegistry:
    """Manages model loading and versioning."""
    
    def __init__(self):
        self.model = None
        self.model_version = None
        self.feature_engineer = None
        self.drift_detector = None
        self.loaded_at = None
        
    async def load_latest_model(self):
        """Load the latest production model from MLflow."""
        try:
            logger.info("Loading latest production model from MLflow...")
            
            # Load model from MLflow registry
            model_uri = "models:/predictive_maintenance_lstm/Production"
            self.model = mlflow.pytorch.load_model(model_uri)
            self.model.eval()
            
            # Get model version info
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(
                "predictive_maintenance_lstm", 
                stages=["Production"]
            )[0]
            self.model_version = latest_version.version
            
            # Initialize feature engineer
            self.feature_engineer = FeatureEngineer()
            
            # Initialize drift detector
            self.drift_detector = DriftDetector()
            
            self.loaded_at = datetime.utcnow()
            
            # Update Prometheus info
            MODEL_INFO.info({
                'version': self.model_version,
                'loaded_at': self.loaded_at.isoformat(),
                'model_type': 'LSTM'
            })
            
            logger.info("Model loaded successfully", version=self.model_version)
            
        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

# Global model registry
model_registry = ModelRegistry()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Predictive Maintenance API...")
    await model_registry.load_latest_model()
    yield
    
    # Shutdown
    logger.info("Shutting down Predictive Maintenance API...")

# Create FastAPI app
app = FastAPI(
    title="Predictive Maintenance API",
    description="Real-time equipment failure prediction with LSTM models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup time for uptime calculation
startup_time = time.time()

def get_risk_level(failure_probs: torch.Tensor) -> str:
    """Determine risk level based on failure probabilities."""
    max_prob = torch.max(failure_probs).item()
    
    if max_prob >= 0.8:
        return "critical"
    elif max_prob >= 0.6:
        return "high"
    elif max_prob >= 0.3:
        return "medium"
    else:
        return "low"

def get_recommendations(risk_level: str, equipment_type: str, failure_probs: torch.Tensor) -> List[str]:
    """Generate maintenance recommendations based on risk level."""
    recommendations = []
    
    if risk_level == "critical":
        recommendations.extend([
            "IMMEDIATE SHUTDOWN RECOMMENDED",
            "Schedule emergency maintenance",
            "Inspect for visible damage or unusual sounds"
        ])
    elif risk_level == "high":
        recommendations.extend([
            "Schedule maintenance within 24 hours",
            "Increase monitoring frequency",
            "Prepare replacement parts"
        ])
    elif risk_level == "medium":
        recommendations.extend([
            "Schedule maintenance within 1 week",
            "Monitor vibration levels closely",
            "Check lubrication levels"
        ])
    else:
        recommendations.extend([
            "Continue normal operation",
            "Maintain regular monitoring schedule"
        ])
    
    # Equipment-specific recommendations
    if equipment_type == "pump":
        if failure_probs[0] > 0.5:  # 1h probability high
            recommendations.append("Check for cavitation or suction issues")
    elif equipment_type == "motor":
        if failure_probs[1] > 0.4:  # 4h probability high
            recommendations.append("Check electrical connections and windings")
    
    return recommendations

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        model_loaded=model_registry.is_loaded(),
        model_version=model_registry.model_version,
        uptime_seconds=time.time() - startup_time
    )

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_failure(request: PredictionRequest):
    """
    Predict equipment failure probability in real-time.
    Target: <200ms P95 latency
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    logger.info("Received prediction request", 
               equipment_id=request.equipment_id,
               equipment_type=request.equipment_type,
               request_id=request_id)
    
    if not model_registry.is_loaded():
        ERROR_COUNTER.labels(error_type='model_not_loaded', equipment_type=request.equipment_type).inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert sensor data to DataFrame
        sensor_data = pd.DataFrame([reading.dict() for reading in request.sensor_data])
        
        # Feature engineering
        features = model_registry.feature_engineer.transform(sensor_data, request.equipment_type)
        
        # Convert to tensor and add batch dimension
        feature_tensor = torch.FloatTensor(features.values).unsqueeze(0)  # (1, seq_len, features)
        
        # Model inference
        with torch.no_grad():
            predictions = model_registry.model(feature_tensor)
            
            # Extract predictions with uncertainty
            failure_probs = predictions['failure_probs'].squeeze(0)  # Remove batch dimension
            ttf_estimate = predictions['ttf_estimate'].squeeze(0).item() if predictions['ttf_estimate'].numel() > 0 else None
            uncertainty = predictions['uncertainty'].squeeze(0)
            
            # Monte Carlo uncertainty estimation
            uncertainty_predictions = model_registry.model.predict_with_uncertainty(feature_tensor, num_samples=50)
            confidence_intervals = {
                '1h': uncertainty_predictions['failure_prob_confidence_interval'][0, 0].tolist(),
                '4h': uncertainty_predictions['failure_prob_confidence_interval'][0, 1].tolist(),
                '24h': uncertainty_predictions['failure_prob_confidence_interval'][0, 2].tolist(),
                '7d': uncertainty_predictions['failure_prob_confidence_interval'][0, 3].tolist(),
            }
        
        # Data drift detection
        drift_score = model_registry.drift_detector.detect_drift(features, request.equipment_type)
        DRIFT_SCORE.labels(equipment_type=request.equipment_type).set(drift_score)
        
        # Calculate health score (inverse of max failure probability)
        health_score = 1.0 - torch.max(failure_probs).item()
        
        # Determine risk level and recommendations
        risk_level = get_risk_level(failure_probs)
        recommendations = get_recommendations(risk_level, request.equipment_type, failure_probs)
        
        # Calculate inference time
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Record metrics
        PREDICTION_COUNTER.labels(
            equipment_type=request.equipment_type,
            prediction_type='realtime',
            status='success'
        ).inc()
        
        PREDICTION_LATENCY.labels(
            equipment_type=request.equipment_type,
            prediction_type='realtime'
        ).observe(time.time() - start_time)
        
        response = PredictionResponse(
            equipment_id=request.equipment_id,
            equipment_type=request.equipment_type,
            prediction_timestamp=datetime.utcnow(),
            failure_probability_1h=failure_probs[0].item(),
            failure_probability_4h=failure_probs[1].item(),
            failure_probability_24h=failure_probs[2].item(),
            failure_probability_7d=failure_probs[3].item(),
            estimated_ttf_hours=ttf_estimate,
            confidence_intervals=confidence_intervals,
            current_health_score=health_score,
            risk_level=risk_level,
            recommended_actions=recommendations,
            model_version=model_registry.model_version,
            inference_time_ms=inference_time_ms,
            data_drift_score=drift_score
        )
        
        logger.info("Prediction completed successfully",
                   equipment_id=request.equipment_id,
                   risk_level=risk_level,
                   inference_time_ms=inference_time_ms,
                   request_id=request_id)
        
        return response
        
    except Exception as e:
        ERROR_COUNTER.labels(error_type=type(e).__name__, equipment_type=request.equipment_type).inc()
        logger.error("Prediction failed",
                    equipment_id=request.equipment_id,
                    error=str(e),
                    request_id=request_id)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction for multiple equipment.
    """
    start_time = time.time()
    
    logger.info("Received batch prediction request", batch_size=len(request.requests))
    
    if not model_registry.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    failed_requests = []
    
    for req in request.requests:
        try:
            result = await predict_failure(req)
            results.append(result)
        except Exception as e:
            failed_requests.append({
                'equipment_id': req.equipment_id,
                'error': str(e)
            })
            logger.error("Batch prediction item failed",
                        equipment_id=req.equipment_id,
                        error=str(e))
    
    batch_time = time.time() - start_time
    
    logger.info("Batch prediction completed",
               total_requests=len(request.requests),
               successful=len(results),
               failed=len(failed_requests),
               batch_time_seconds=batch_time)
    
    return {
        'predictions': results,
        'failed_requests': failed_requests,
        'summary': {
            'total_requests': len(request.requests),
            'successful': len(results),
            'failed': len(failed_requests),
            'batch_processing_time_seconds': batch_time
        }
    }

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    if not model_registry.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        'model_version': model_registry.model_version,
        'loaded_at': model_registry.loaded_at.isoformat(),
        'model_type': 'LSTM with Attention',
        'supported_equipment_types': ['pump', 'motor', 'compressor', 'turbine'],
        'prediction_horizons': ['1h', '4h', '24h', '7d'],
        'features_required': model_registry.feature_engineer.get_feature_names(),
        'uptime_seconds': time.time() - startup_time
    }

@app.post("/model/reload")
async def reload_model():
    """Reload the latest model (admin endpoint)."""
    try:
        await model_registry.load_latest_model()
        return {
            'status': 'success',
            'model_version': model_registry.model_version,
            'reloaded_at': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("Model reload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)