# Encounter Prediction API

## Overview
A FastAPI application that loads a pre-trained machine learning model to predict encounter probabilities. The API accepts feature data via POST requests and returns both a probability score (float 0.0-1.0) and a boolean prediction.

## Recent Changes
- **2025-10-19**: Initial project setup
  - Installed Python 3.11 with FastAPI, uvicorn, scikit-learn, pandas, numpy
  - Created FastAPI server with prediction endpoint
  - Loaded model pipeline from vmodel_pipeline_1760914232646.pkl using joblib
  - Added health check and model info endpoints
  - Configured workflow to run server on port 5000

## Project Architecture

### Model
- **Type**: CalibratedClassifierCV with LogisticRegression
- **Features**: 10 required input features for prediction
- **Loading**: Model is loaded using joblib (not pickle) during startup
- **Version**: Trained with scikit-learn 1.6.1, running on 1.7.2 (compatible with warnings)

### API Endpoints

#### 1. GET /
- Root endpoint showing API status
- Returns: `{message, status, model_loaded}`

#### 2. GET /health
- Health check endpoint
- Returns: Model status and required features list

#### 3. POST /predict
- Main prediction endpoint
- **Input**: JSON with 10 required features
- **Output**: 
  - `probability`: float (0.0 to 1.0)
  - `prediction`: boolean (true/false based on 0.5 threshold)
  - `confidence`: string (low/medium/high)

#### 4. GET /model-info
- Returns model metadata, features, schema, and thresholds

### Required Input Features
1. `hdiff`: float - Heading difference
2. `hdiff_sin`: float - Sine of heading difference
3. `hdiff_cos`: float - Cosine of heading difference
4. `hdiff_norm`: float - Normalized heading difference
5. `spd_diff`: float - Speed difference
6. `spd_diff_sq`: float - Squared speed difference
7. `loitering_sub2`: float - Loitering subscore 2
8. `hour`: float - Hour of day
9. `is_weekend`: int - Weekend indicator (0 or 1)
10. `same_mid`: int - Same MID indicator (0 or 1)

## Usage Example

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## API Documentation
Interactive API documentation is available at:
- Swagger UI: http://localhost:5000/docs
- ReDoc: http://localhost:5000/redoc

## Technical Stack
- **Framework**: FastAPI
- **Server**: Uvicorn (ASGI)
- **ML Libraries**: scikit-learn, pandas, numpy
- **Validation**: Pydantic
- **Port**: 5000 (bound to 0.0.0.0)

## Notes
- Model file must be loaded with joblib (not standard pickle)
- Server binds to 0.0.0.0:5000 for Replit compatibility
- Confidence levels: high (>0.75), medium (0.4-0.75), low (<0.4)
