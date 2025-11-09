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
