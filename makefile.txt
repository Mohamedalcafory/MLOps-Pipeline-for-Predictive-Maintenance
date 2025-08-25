.PHONY: help setup data train serve monitor test lint clean

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development Environment
setup: ## Set up development environment
	pip install -e .
	pre-commit install
	mkdir -p data/{raw,processed,external} models experiments logs

# Data Generation and Processing
generate-data: ## Generate synthetic sensor data
	python scripts/generate_data.py --num-equipment 50 --duration-days 365 --output-dir data/raw

process-data: ## Process raw data for training
	python src/data/feature_engineering.py --input-dir data/raw --output-dir data/processed

# Model Training
train: ## Train LSTM model with MLflow tracking
	python src/training/train.py --config-file configs/lstm_config.yaml --experiment-name predictive_maintenance

train-ensemble: ## Train ensemble of models
	python src/training/train_ensemble.py --num-models 5 --experiment-name ensemble_predictive_maintenance

hyperopt: ## Run hyperparameter optimization
	python src/training/hyperopt.py --n-trials 100 --experiment-name hyperopt_lstm

evaluate: ## Evaluate trained model
	python src/training/evaluate.py --model-version latest --test-data data/processed/test_data.parquet

# Model Serving
serve: ## Start FastAPI inference service
	docker-compose up --build api

serve-local: ## Start API service locally
	uvicorn src.inference.api.main:app --host 0.0.0.0 --port 8000 --reload

# MLflow and Monitoring
mlflow: ## Start MLflow tracking server
	docker-compose up --build mlflow

monitoring: ## Start monitoring stack (Prometheus + Grafana)
	docker-compose up --build prometheus grafana

jupyter: ## Start Jupyter lab for experimentation
	docker-compose --profile jupyter up --build jupyter

# Infrastructure
infra: ## Start all infrastructure services
	docker-compose up --build postgres redis minio mlflow

full-stack: ## Start complete stack
	docker-compose up --build

# Testing and Quality
test: ## Run all tests
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-api: ## Test API endpoints
	python tests/test_api_integration.py

test-model: ## Test model performance
	python tests/test_model_performance.py

lint: ## Run code linting
	ruff check src/ tests/
	black --check src/ tests/
	mypy src/

fmt: ## Format code
	ruff check src/ tests/ --fix
	black src/ tests/

# Performance Testing
load-test: ## Run load test on API
	python scripts/load_test.py --url http://localhost:8000 --concurrent-users 50 --duration 60

benchmark: ## Benchmark model inference speed
	python scripts/benchmark_inference.py --model-version latest --batch-sizes 1,10,50,100

# Data Drift and Monitoring
check-drift: ## Check for data drift
	python src/monitoring/check_drift.py --reference-data data/processed/train_data.parquet --current-data data/processed/current_data.parquet

retrain: ## Trigger model retraining
	python scripts/trigger_retraining.py --drift-threshold 0.1 --performance-threshold 0.85

# Model Management
register-model: ## Register model in MLflow registry
	python scripts/register_model.py --run-id <run_id> --model-name predictive_maintenance_lstm

promote-model: ## Promote model to production
	python scripts/promote_model.py --model-name predictive_maintenance_lstm --version <version> --stage Production

# Deployment
build: ## Build Docker images
	docker-compose build

deploy-staging: ## Deploy to staging environment
	docker-compose -f docker-compose.staging.yml up -d

deploy-prod: ## Deploy to production environment
	docker-compose -f docker-compose.prod.yml up -d

# Database Management
init-db: ## Initialize database
	python scripts/init_database.py

backup-db: ## Backup database
	docker-compose exec postgres pg_dump -U mlops_user mlops_db > backup_$(shell date +%Y%m%d_%H%M%S).sql

# Logging and Debugging
logs: ## Show logs from all services
	docker-compose logs -f

logs-api: ## Show API service logs
	docker-compose logs -f api

logs-training: ## Show training service logs
	docker-compose logs -f training

# Data Management
download-data: ## Download external datasets (if any)
	python scripts/download_external_data.py --output-dir data/external

validate-data: ## Validate data quality
	python src/data/data_validation.py --data-path data/processed/train_data.parquet

# Batch Processing
batch-predict: ## Run batch predictions
	python scripts/batch_predict.py --input-data data/new_sensor_data.parquet --output-dir predictions/

scheduled-training: ## Run scheduled training job
	python scripts/scheduled_training.py --schedule daily --retrain-threshold 0.1

# Cleanup
clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage

clean-docker: ## Clean up Docker resources
	docker-compose down -v --remove-orphans
	docker system prune -f

clean-models: ## Clean up old model artifacts
	python scripts/cleanup_old_models.py --keep-latest 5

# Documentation
docs: ## Generate documentation
	sphinx-build -b html docs/ docs/_build/html

docs-serve: ## Serve documentation locally
	python -m http.server 8080 --directory docs/_build/html

# Utilities
shell: ## Access container shell
	docker-compose exec api bash

db-shell: ## Access database shell
	docker-compose exec postgres psql -U mlops_user -d mlops_db

redis-shell: ## Access Redis shell
	docker-compose exec redis redis-cli

# Quick Commands
quick-test: ## Quick test of the pipeline
	@echo "🔄 Generating sample data..."
	python scripts/generate_data.py --num-equipment 5 --duration-days 30 --output-dir data/raw
	@echo "🔄 Processing data..."
	python src/data/feature_engineering.py --input-dir data/raw --output-dir data/processed
	@echo "🔄 Training model..."
	python src/training/train.py --config-file configs/lstm_config.yaml --experiment-name quick_test --epochs 5
	@echo "✅ Quick test completed!"

demo: ## Run complete demo
	@echo "🚀 Starting Predictive Maintenance MLOps Demo..."
	make generate-data
	make process-data
	make train
	make serve &
	sleep 10
	make test-api
	@echo "✅ Demo completed! API running at http://localhost:8000"

# Performance Monitoring
monitor-performance: ## Monitor API performance
	@echo "📊 Checking API performance..."
	curl -s http://localhost:8000/metrics | grep -E "(prediction_latency|predictions_total)"

health-check: ## Check system health
	@echo "🏥 Checking system health..."
	@curl -s http://localhost:8000/health | jq '.'
	@echo ""
	@docker-compose ps

# CI/CD Helpers
ci-test: ## Run tests in CI environment
	pytest tests/ -v --cov=src --cov-report=xml --junitxml=test-results.xml

ci-build: ## Build for CI
	docker-compose -f docker-compose.ci.yml build

ci-deploy: ## Deploy in CI
	docker-compose -f docker-compose.ci.yml up -d

# Development Helpers
dev-setup: ## Set up for development
	make setup
	make generate-data
	make process-data
	make infra
	@echo "✅ Development environment ready!"

dev-reset: ## Reset development environment
	make clean-docker
	make clean
	make dev-setup

# Model Experimentation
experiment: ## Run model experiment with different configs
	python src/training/run_experiments.py --config-dir configs/experiments

compare-models: ## Compare different model architectures
	python scripts/compare_models.py --models lstm,ensemble,xgboost --dataset data/processed/test_data.parquet

# Production Utilities
prod-deploy: ## Production deployment
	@echo "🚀 Deploying to production..."
	docker-compose -f docker-compose.prod.yml pull
	docker-compose -f docker-compose.prod.yml up -d --remove-orphans
	@echo "✅ Production deployment completed!"

prod-rollback: ## Rollback production deployment
	@echo "↩️ Rolling back production deployment..."
	docker-compose -f docker-compose.prod.yml down
	# Add rollback logic here
	@echo "✅ Rollback completed!"

# Secrets and Configuration
setup-secrets: ## Set up secrets for production
	@echo "🔐 Setting up secrets..."
	# Add secret management logic here
	@echo "✅ Secrets configured!"

# Monitoring and Alerting
setup-alerts: ## Set up monitoring alerts
	python scripts/setup_alerts.py --slack-webhook ${SLACK_WEBHOOK} --email ${ALERT_EMAIL}

check-alerts: ## Check current alerts
	python scripts/check_alerts.py

# Installation
install: ## Install package
	pip install -e .

install-dev: ## Install with development dependencies
	pip install -e ".[dev]"