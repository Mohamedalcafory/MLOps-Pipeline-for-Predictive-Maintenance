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
import os

from models.lstm_model import PredictiveMaintenanceLSTM, ModelConfig, load_model
from data.feature_engineering import FeatureEngineer, FeatureConfig
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
    """Request for equipment failure prediction."""
    equipment_id: str = Field(..., description="Unique equipment identifier")
    equipment_type: str = Field(..., description="Type of equipment (pump, motor, compressor, turbine)")
    sensor_data: List[SensorReading] = Field(..., min_items=168, max_items=1000, 
                                            description="Time series sensor data (minimum 168 readings for 7 days)")
    
    @validator('equipment_type')
    def validate_equipment_type(cls, v):
        valid_types = ['pump', 'motor', 'compressor', 'turbine']
        if v.lower() not in valid_types:
            raise ValueError(f'Equipment type must be one of {valid_types}')
        return v.lower()

class PredictionResponse(BaseModel):
    """Response containing failure predictions."""
    equipment_id: str
    equipment_type: str
    prediction_timestamp: datetime
    failure_probability_1h: float = Field(..., ge=0, le=1)
    failure_probability_4h: float = Field(..., ge=0, le=1)
    failure_probability_24h: float = Field(..., ge=0, le=1)
    failure_probability_7d: float = Field(..., ge=0, le=1)
    estimated_ttf_hours: float = Field(..., ge=0)
    confidence_intervals: Dict[str, List[float]]
    current_health_score: float = Field(..., ge=0, le=1)
    risk_level: str = Field(..., description="low, medium, high, critical")
    recommended_actions: List[str]
    model_version: str
    inference_time_ms: float
    data_drift_score: Optional[float] = None

class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    predictions: List[PredictionRequest] = Field(..., max_items=100)

class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[PredictionResponse]
    total_processed: int
    batch_processing_time_ms: float

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    model_loaded: bool
    model_version: Optional[str]
    uptime_seconds: float
    total_predictions: int
    average_latency_ms: float

# Global variables
model: Optional[PredictiveMaintenanceLSTM] = None
feature_engineer: Optional[FeatureEngineer] = None
drift_detector: Optional[DriftDetector] = None
redis_client: Optional[Redis] = None
start_time = time.time()
total_predictions = 0
total_latency = 0.0

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global model, feature_engineer, drift_detector, redis_client
    
    # Startup
    logger.info("Starting Predictive Maintenance API...")
    
    # Initialize Redis
    try:
        redis_client = Redis(host='redis', port=6379, decode_responses=True)
        redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None
    
    # Load model
    try:
        model_path = os.getenv('MODEL_REGISTRY_URI', 'models/lstm_model_latest.pth')
        config = ModelConfig()
        model = load_model(model_path, config)
        MODEL_INFO.info({
            'model_version': '1.0.0',
            'model_type': 'lstm',
            'input_size': config.input_size,
            'hidden_size': config.hidden_size
        })
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
    
    # Initialize feature engineer
    feature_config = FeatureConfig()
    feature_engineer = FeatureEngineer(feature_config)
    
    # Initialize drift detector
    drift_detector = DriftDetector()
    
    logger.info("API startup completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")
    if redis_client:
        redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="Predictive Maintenance API",
    description="Real-time equipment failure prediction using LSTM models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_risk_level(probabilities: List[float]) -> str:
    """Determine risk level based on failure probabilities."""
    max_prob = max(probabilities)
    
    if max_prob >= 0.8:
        return "critical"
    elif max_prob >= 0.6:
        return "high"
    elif max_prob >= 0.3:
        return "medium"
    else:
        return "low"

def get_recommended_actions(risk_level: str, equipment_type: str) -> List[str]:
    """Get recommended actions based on risk level and equipment type."""
    actions = {
        'critical': [
            "Immediate shutdown required",
            "Schedule emergency maintenance",
            "Notify maintenance team immediately"
        ],
        'high': [
            "Schedule maintenance within 24 hours",
            "Increase monitoring frequency",
            "Prepare replacement parts"
        ],
        'medium': [
            "Schedule maintenance within 1 week",
            "Monitor closely for changes",
            "Check lubrication levels"
        ],
        'low': [
            "Continue normal operation",
            "Schedule routine maintenance",
            "Monitor for trend changes"
        ]
    }
    
    base_actions = actions.get(risk_level, [])
    
    # Add equipment-specific actions
    if equipment_type == 'pump':
        base_actions.extend([
            "Check pump alignment",
            "Inspect impeller condition",
            "Verify seal integrity"
        ])
    elif equipment_type == 'motor':
        base_actions.extend([
            "Check motor bearings",
            "Inspect winding insulation",
            "Verify cooling system"
        ])
    
    return base_actions

def calculate_health_score(probabilities: List[float]) -> float:
    """Calculate overall health score (0-1, where 1 is perfect health)."""
    # Weight shorter horizons more heavily
    weights = [0.4, 0.3, 0.2, 0.1]  # 1h, 4h, 24h, 7d
    weighted_prob = sum(w * p for w, p in zip(weights, probabilities))
    return 1.0 - weighted_prob

def estimate_ttf(probabilities: List[float]) -> float:
    """Estimate time to failure in hours."""
    # Simple estimation based on probability trends
    if probabilities[0] > 0.8:  # High immediate risk
        return 1.0
    elif probabilities[1] > 0.6:  # High 4h risk
        return 4.0
    elif probabilities[2] > 0.4:  # Medium 24h risk
        return 24.0
    elif probabilities[3] > 0.2:  # Low 7d risk
        return 168.0
    else:
        return 720.0  # 30 days if low risk

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global total_predictions, total_latency, start_time
    
    avg_latency = total_latency / max(total_predictions, 1)
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.utcnow(),
        model_loaded=model is not None,
        model_version="1.0.0" if model is not None else None,
        uptime_seconds=time.time() - start_time,
        total_predictions=total_predictions,
        average_latency_ms=avg_latency * 1000
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
async def predict_failure(request: PredictionRequest):
    """Predict equipment failure probability."""
    global model, feature_engineer, drift_detector, total_predictions, total_latency
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Convert sensor data to DataFrame
        sensor_df = pd.DataFrame([reading.dict() for reading in request.sensor_data])
        
        # Check for data drift
        drift_score = None
        if drift_detector:
            drift_score = drift_detector.calculate_drift(sensor_df)
            DRIFT_SCORE.labels(equipment_type=request.equipment_type).set(drift_score)
        
        # Preprocess data
        if feature_engineer:
            # Validate data
            if not feature_engineer.validate_data(sensor_df):
                raise HTTPException(status_code=400, detail="Invalid sensor data")
            
            # Transform features
            features = feature_engineer.transform_features(sensor_df)
        else:
            # Fallback preprocessing
            feature_columns = ['vibration_x', 'vibration_y', 'vibration_z', 'temperature', 
                             'pressure', 'current', 'voltage', 'flow_rate']
            features = sensor_df[feature_columns].values
        
        # Ensure correct sequence length
        if len(features) < model.config.sequence_length:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient data. Need at least {model.config.sequence_length} readings"
            )
        
        # Take the last sequence_length readings
        features = features[-model.config.sequence_length:]
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
        
        # Move to GPU if available
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            model = model.cuda()
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            predictions = outputs['predictions'].cpu().numpy()[0]
            uncertainty = outputs['uncertainty'].cpu().numpy()[0]
        
        # Calculate additional metrics
        risk_level = get_risk_level(predictions.tolist())
        health_score = calculate_health_score(predictions.tolist())
        ttf_hours = estimate_ttf(predictions.tolist())
        recommended_actions = get_recommended_actions(risk_level, request.equipment_type)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        horizons = ['1h', '4h', '24h', '7d']
        for i, horizon in enumerate(horizons):
            pred = predictions[i]
            unc = uncertainty[i]
            confidence_intervals[horizon] = [
                max(0, pred - 1.96 * unc),
                min(1, pred + 1.96 * unc)
            ]
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Update metrics
        total_predictions += 1
        total_latency += inference_time
        
        PREDICTION_COUNTER.labels(
            equipment_type=request.equipment_type,
            prediction_type='single',
            status='success'
        ).inc()
        
        PREDICTION_LATENCY.labels(
            equipment_type=request.equipment_type,
            prediction_type='single'
        ).observe(inference_time)
        
        # Cache result if Redis is available
        if redis_client:
            cache_key = f"prediction:{request.equipment_id}:{int(time.time() / 300)}"  # 5-minute cache
            cache_data = {
                'predictions': predictions.tolist(),
                'risk_level': risk_level,
                'health_score': health_score,
                'timestamp': datetime.utcnow().isoformat()
            }
            redis_client.setex(cache_key, 300, str(cache_data))
        
        return PredictionResponse(
            equipment_id=request.equipment_id,
            equipment_type=request.equipment_type,
            prediction_timestamp=datetime.utcnow(),
            failure_probability_1h=float(predictions[0]),
            failure_probability_4h=float(predictions[1]),
            failure_probability_24h=float(predictions[2]),
            failure_probability_7d=float(predictions[3]),
            estimated_ttf_hours=ttf_hours,
            confidence_intervals=confidence_intervals,
            current_health_score=health_score,
            risk_level=risk_level,
            recommended_actions=recommended_actions,
            model_version="1.0.0",
            inference_time_ms=inference_time * 1000,
            data_drift_score=drift_score
        )
        
    except Exception as e:
        ERROR_COUNTER.labels(
            error_type=type(e).__name__,
            equipment_type=request.equipment_type
        ).inc()
        
        PREDICTION_COUNTER.labels(
            equipment_type=request.equipment_type,
            prediction_type='single',
            status='error'
        ).inc()
        
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_failures(request: BatchPredictionRequest):
    """Batch prediction for multiple equipment."""
    global model, total_predictions, total_latency
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.predictions) > 100:
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    start_time = time.time()
    results = []
    
    try:
        for pred_request in request.predictions:
            # Process each prediction
            result = await predict_failure(pred_request)
            results.append(result)
        
        batch_time = time.time() - start_time
        
        return BatchPredictionResponse(
            predictions=results,
            total_processed=len(results),
            batch_processing_time_ms=batch_time * 1000
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_version": "1.0.0",
        "model_type": "LSTM",
        "input_size": model.config.input_size,
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.num_layers,
        "sequence_length": model.config.sequence_length,
        "output_size": model.config.output_size,
        "attention_heads": model.config.attention_heads,
        "bidirectional": model.config.bidirectional,
        "dropout": model.config.dropout
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
