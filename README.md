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

### Launch 
- uvicorn main:app --reload


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
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hdiff": 18,
    "hdiff_sin": 0.3090169944,
    "hdiff_cos": 0.9510565163,
    "hdiff_norm": 0.1,
    "hour": 18,
    "is_weekend": 1,
    "loitering_sub2": 1,
    "same_mid": 0,
    "spd_diff": 0,
    "spd_diff_sq": 0
  }'
```
```bash
## Example 2 Postive 

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hdiff": 5.6,
    "hdiff_sin": 0.09758289976,
    "hdiff_cos": 0.9952274,
    "hdiff_norm": 0.03111111111,
    "hour": 0,
    "is_weekend": 1,
    "loitering_sub2": 1,
    "same_mid": 1,
    "spd_diff": 0,
    "spd_diff_sq": 0
  }'


```bash
## Example 3 Postive 
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hdiff": 20,
    "hdiff_sin": 0.3420201433,
    "hdiff_cos": 0.9396926208,
    "hdiff_norm": 0.1111111111,
    "hour": 7,
    "is_weekend": 1,
    "loitering_sub2": 1,
    "same_mid": 1,
    "spd_diff": 0,
    "spd_diff_sq": 0
  }'

```bash
## Example 4 Postive 
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hdiff": 8.8,
    "hdiff_sin": 0.1529858363,
    "hdiff_cos": 0.9882283814,
    "hdiff_norm": 0.04888888889,
    "hour": 17,
    "is_weekend": 1,
    "loitering_sub2": 1,
    "same_mid": 1,
    "spd_diff": 0,
    "spd_diff_sq": 0
  }'
```
Test Redis 
redis-cli KEYS "cache-prediction*"

kubectl -n w255 get svc
kubectl -n w255 port-forward svc/prediction-service 8000:8000
# in another shell:
curl http://127.0.0.1:8000/health

# or SCAN 0 MATCH "cache-prediction*"


# 1. Start minikube
minikube start --driver=docker --kubernetes-version=v1.32.1

# 2. Point docker build to minikube's docker daemon (very important)
eval $(minikube -p minikube docker-env)

# 3. Build your API image locally (no push needed)
docker build -t app:latest .

# 4. Apply Kubernetes Namespace
kubectl apply -f infra/namespace.yaml
kubectl config set-context --current --namespace=w255

# 5. Deploy it in K8s (e.g., via your deployment.yaml)
kubectl apply -f infra/

# 6. Port Forward to Access the API
kubectl port-forward service/prediction-service 8000:8000
Visit the Swagger UI at:
http://localhost:8000/lab/docs

## API Documentation
Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Technical Stack
- **Framework**: FastAPI
- **Server**: Uvicorn (ASGI)
- **ML Libraries**: scikit-learn, pandas, numpy
- **Validation**: Pydantic
- **Port**: 8000 (bound to 0.0.0.0)

## Notes
- Model file must be loaded with joblib (not standard pickle)
- Server binds to 0.0.0.0:8000 
- Confidence levels: high (>0.75), medium (0.4-0.75), low (<0.4)
