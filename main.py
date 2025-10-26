from fastapi import FastAPI, HTTPException
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
import uvicorn
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)
model_data = {}

LOCAL_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_data["pipeline"] = joblib.load('attached_assets/vmodel_pipeline_1760914232646.pkl')
        print("Model loaded successfully")
        print(f"Required features: {model_data['pipeline']['features']}")
        HOST_URL = LOCAL_REDIS_URL  # replace this according to the Lab Requirements
        redis = asyncio.from_url(HOST_URL, encoding="utf8", decode_responses=True)

        # We initialize the connection to Redis and declare that all keys in the
        # database will be prefixed with w255-cache-predict. Do not change this
        # prefix for the submission.
        FastAPICache.init(RedisBackend(redis), prefix="w255-cache-prediction")

        yield
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
    yield
    model_data.clear()

app = FastAPI(
    title="Encounter Prediction API",
    description="API for predicting encounter probability using a trained ML model",
    version="1.0.0",
    lifespan=lifespan
)

class PredictionInput(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "hdiff": 0.5,
            "hdiff_sin": 0.479,
            "hdiff_cos": 0.878,
            "hdiff_norm": 0.25,
            "spd_diff": 1.2,
            "spd_diff_sq": 1.44,
            "loitering_sub2": 0.3,
            "hour": 14.0,
            "is_weekend": 0,
            "same_mid": 1
        }
    })
    
    hdiff: float = Field(..., description="Heading difference")
    hdiff_sin: float = Field(..., description="Sine of heading difference")
    hdiff_cos: float = Field(..., description="Cosine of heading difference")
    hdiff_norm: float = Field(..., description="Normalized heading difference")
    spd_diff: float = Field(..., description="Speed difference")
    spd_diff_sq: float = Field(..., description="Squared speed difference")
    loitering_sub2: float = Field(..., description="Loitering subscore 2")
    hour: float = Field(..., description="Hour of day")
    is_weekend: int = Field(..., description="Weekend indicator (0 or 1)")
    same_mid: int = Field(..., description="Same MID indicator (0 or 1)")

class PredictionOutput(BaseModel):
    probability: float = Field(..., description="Prediction probability (0.0 to 1.0)")
    prediction: bool = Field(..., description="Binary prediction (True/False) based on 0.5 threshold")
    confidence: str = Field(..., description="Confidence level (low/medium/high)")

@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Encounter Prediction API is running",
        "status": "healthy",
        "model_loaded": "pipeline" in model_data
    }

@app.get("/health", tags=["Health"])
async def health_check():
    if "pipeline" not in model_data:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "features_required": model_data["pipeline"]['features'],
        "feature_count": len(model_data["pipeline"]['features'])
    }

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
@cache()
async def predict(input_data: PredictionInput):
    if "pipeline" not in model_data:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features_dict = input_data.model_dump()
        df = pd.DataFrame([features_dict])
        
        pipeline = model_data["pipeline"]
        feature_order = pipeline['features']
        df = df[feature_order]
        
        model = pipeline['model']
        probability = model.predict_proba(df)[0][1]
        
        prediction = bool(probability >= 0.5)
        
        if probability >= 0.75:
            confidence = "high"
        elif probability >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictionOutput(
            probability=float(probability),
            prediction=prediction,
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/model-info", tags=["Model"])
async def model_info():
    if "pipeline" not in model_data:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    pipeline = model_data["pipeline"]
    return {
        "features": pipeline['features'],
        "schema": pipeline['schema'],
        "thresholds": pipeline['thresholds'],
        "meta": pipeline.get('meta', {})
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
