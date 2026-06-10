# Developer workflow for the ASL CNN classifier.
# Assumes `uv` is installed; the venv lives at .venv (Python 3.12).

PY := .venv/bin/python

.PHONY: install sample-train train eval gradcam calibration benchmark benchmark-backends export-onnx quantize serve camera test lint format docker-build clean

# Create the venv, install dev deps, and regenerate the committed sample data.
install:
	uv venv --python 3.12
	uv pip install -r requirements-dev.txt
	$(PY) -m src.make_sample_data

# Quick end-to-end smoke train on the tiny committed sample set (CPU, 2 epochs).
sample-train:
	$(PY) -m src.train --config configs/train_custom_cnn.yaml --data_dir data/sample --num_epochs 2 --device cpu

# Full training run on the real dataset.
train:
	$(PY) -m src.train --config configs/train_custom_cnn.yaml

# Evaluate a trained checkpoint (uses sample data here for a fast check).
eval:
	$(PY) -m src.eval --config configs/train_custom_cnn.yaml --checkpoint artifacts/checkpoints/best_model.pth --data_dir data/sample

# Grad-CAM explainability overlay for a single image (uses sample data here).
gradcam:
	$(PY) -m src.gradcam --checkpoint artifacts/checkpoints/best_model.pth --source data/sample/A/0.png --device cpu

# Calibration: Expected Calibration Error + reliability diagram on the test split.
calibration:
	$(PY) -m src.calibration --checkpoint artifacts/checkpoints/best_model.pth --data_dir data/sample --device cpu

# Inference throughput/latency benchmark.
benchmark:
	$(PY) -m src.benchmark --num_frames 1000 --device cpu --test_dir data/sample

# Multi-backend latency benchmark (PyTorch FP32 / ONNX Runtime / INT8) on CPU.
benchmark-backends:
	$(PY) -m src.benchmark_backends --num_frames 200 --source data/sample

# Export the model to ONNX (dynamic batch axis, fixed 3x128x128).
export-onnx:
	$(PY) -m src.export_onnx --output artifacts/model.onnx --device cpu

# Dynamic INT8 quantization + on-disk FP32 vs INT8 size report.
quantize:
	$(PY) -m src.quantize --output artifacts/quantization.json --device cpu

# Run the FastAPI inference service (GET /health, POST /predict).
serve:
	$(PY) -m uvicorn src.serve:app --host 0.0.0.0 --port 8000

# Live webcam inference.
camera:
	$(PY) -m src.infer_camera

# Run the test suite with coverage.
test:
	$(PY) -m pytest -q --cov=src --cov-report=term-missing

# Lint + format checks (non-mutating).
lint:
	$(PY) -m ruff check src tests && $(PY) -m black --check src tests

# Auto-format and apply lint fixes.
format:
	$(PY) -m black src tests && $(PY) -m ruff check --fix src tests

# Build the CPU Docker image.
docker-build:
	docker build -t asl-cnn-classifier .

# Remove generated artifacts (keeping .gitkeep) and bytecode caches.
clean:
	find artifacts -mindepth 1 -not -name '.gitkeep' -delete
	find . -type d -name __pycache__ -exec rm -rf {} +
