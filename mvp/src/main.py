from contextlib import asynccontextmanager
from pathlib import Path
import os
import logging
import hashlib
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import uvicorn

#-------------------------------------------
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio as aioredis  # redis>=5

#Redis code --------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("encounter-api")
#----------------------------------------------

# global in-memory store
model_data = {}

# Resolve project root: .../mvp/
BASE_DIR = Path(__file__).resolve().parents[1]

# Allow MODEL_PATH override, else default to mvp/vmodel_pipeline.pkl
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "vmodel_pipeline.pkl"))


# Redis config via env with sane defaults for k8s Service "redis-service"
REDIS_URL = os.getenv("REDIS_URL", "redis://redis-service:6379/0")
#REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_PREFIX = os.getenv("CACHE_PREFIX", "w255-cache-prediction")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))  # 5 minutes by default
#-------------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at: {MODEL_PATH}\n"
                f"Tip: put the file at {BASE_DIR/'vmodel_pipeline.pkl'} "
                f"or set MODEL_PATH to an absolute path."
            )
        model_data["pipeline"] = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        # Optional: validate expected keys
        for k in ("features", "model"):
            if k not in model_data["pipeline"]:
                raise KeyError(f"Missing '{k}' in loaded pipeline object.")
        print(f"Required features: {model_data['pipeline']['features']}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    #yield  # model available during app lifetime
    #model_data.clear()  # clean up on shutdown


    # Connect Redis & init cache ---------------------------------------
    try:
        redis = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
        # Smoke test the connection
        await redis.ping()
        FastAPICache.init(RedisBackend(redis), prefix=CACHE_PREFIX)
        logger.info(f"Redis cache initialized at {REDIS_URL} with prefix '{CACHE_PREFIX}'")
    except Exception as e:
        logger.exception("Failed to initialize Redis cache")
        raise

    yield

    # Cleanup (optional)
    model_data.clear()
    try:
        backend = FastAPICache.get_backend()
        if backend and hasattr(backend, "redis"):
            await backend.redis.close()
            logger.info("Redis connection closed")
    except Exception:
        pass
#----------------------------------------------------------------------------------------------
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
