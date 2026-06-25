# Developer workflow for the ASL CNN classifier.
# Assumes `uv` is installed; the venv lives at .venv (Python 3.12).

PY := .venv/bin/python

.PHONY: install download-real download-crossval download-diverse download-hemg sample-train train train-real eval eval-real eval-realworld precrop precrop-clean train-cropped train-cropped-midaug train-cropped-clean eval-realworld-cropped eval-realworld-cropped-midaug eval-realworld-cropped-clean check-overlap train-diverse eval-realworld-diverse train-diverse-midaug eval-realworld-diverse-midaug gradcam calibration benchmark benchmark-backends export-onnx quantize serve camera test lint format mypy typecheck docker-build docker-run docker-test deploy-hf deploy-hf-dryrun clean

# Target Hugging Face Space, e.g. `export HF_SPACE=you/asl-cnn-classifier`.
HF_SPACE ?=

# Create the venv, install dev deps, and regenerate the committed sample data.
install:
	uv venv --python 3.12
	uv pip install -r requirements-dev.txt
	$(PY) -m src.make_sample_data

# Download the real 26-class ASL dataset from the public HF Hub (no credentials)
# into data/asl_real/<CLASS>/<i>.png. Add --max_per_class N for a fast subset.
download-real:
	$(PY) -m src.download_hf_data --out_dir data/asl_real

# Download a SECOND, more diverse 26-class A–Z ASL dataset (EitanG98/asl_letters:
# different signers + real photo backgrounds) into data/asl_crossval/, capped at
# 30/class, for the honest cross-dataset generalization measurement.
download-crossval:
	$(PY) -m src.download_hf_data --dataset asl_letters --out_dir data/asl_crossval --max_per_class 30

# Download a genuinely DIVERSE 2nd TRAINING source (aliciiavs: multiple signers,
# real varied backgrounds, A–Y) into data/asl_diverse/ — the antidote to
# Marxulia's single-signer overfit. Run `make check-overlap` after this.
download-diverse:
	$(PY) -m src.download_hf_data --dataset diverse --out_dir data/asl_diverse

# Fallback diversity source (Hemg: balanced, includes J/Z, dark cropped-hand bg).
download-hemg:
	$(PY) -m src.download_hf_data --dataset hemg --out_dir data/asl_hemg

# Quick end-to-end smoke train on the tiny committed sample set (CPU, 2 epochs).
sample-train:
	$(PY) -m src.train --config configs/train_custom_cnn.yaml --data_dir data/sample --num_epochs 2 --device cpu

# Full training run on the real dataset.
train:
	$(PY) -m src.train --config configs/train_custom_cnn.yaml

# Train the MobileNetV2 transfer model on the real ASL dataset (run
# `make download-real` first). ~35 min on Apple-Silicon MPS; reaches ~98% val.
train-real:
	$(PY) -m src.train --config configs/train_real_mobilenet.yaml

# Robustness retrain with aggressive domain augmentation → SEPARATE checkpoint
# (artifacts/checkpoints_robust/). Does NOT clobber the verified baseline.
train-robust:
	$(PY) -m src.train --config configs/train_robust_mobilenet.yaml

# Evaluate the robustness checkpoint on the held-out real test split.
eval-robust:
	$(PY) -m src.eval --checkpoint artifacts/checkpoints_robust/best_model.pth --data_dir data/asl_real

# Evaluate a trained checkpoint (uses sample data here for a fast check).
eval:
	$(PY) -m src.eval --config configs/train_custom_cnn.yaml --checkpoint artifacts/checkpoints/best_model.pth --data_dir data/sample

# Evaluate the real-data checkpoint on the held-out real test split.
eval-real:
	$(PY) -m src.eval --checkpoint artifacts/checkpoints/best_model.pth --data_dir data/asl_real

# HONEST cross-dataset generalization: run the deployed checkpoint on a DIFFERENT
# dataset (run `make download-crossval` first). Hand-crop ON; writes
# artifacts/realworld_eval.json. This is NOT the 96.8% same-dataset benchmark.
eval-realworld:
	$(PY) -m src.eval_realworld --checkpoint artifacts/checkpoints/best_model.pth --data_dir data/asl_crossval

# --- Crop-consistent retrain (close the train/serve hand-crop mismatch) -------
# Pre-crop the real training set to the MediaPipe hand region (same geometry as
# the browser, CROP_MARGIN=0.35) into data/asl_real_cropped/. Run `make
# download-real` first. Writes data/asl_real_cropped/_precrop_report.json.
precrop:
	$(PY) scripts/precrop_dataset.py --in_dir data/asl_real --out_dir data/asl_real_cropped

# Same as precrop, but DROP the ~15% non-photographic sketch line drawings that
# pollute crop-consistent training (they all fail hand detection). Writes
# data/asl_real_cropped_clean/.
precrop-clean:
	$(PY) scripts/precrop_dataset.py --in_dir data/asl_real --out_dir data/asl_real_cropped_clean --drop-sketches

# Experiment A: crop-consistent fine-tune, standard aug → SEPARATE checkpoint
# (artifacts/checkpoints_cropped/). Does NOT clobber the deployed baseline.
train-cropped:
	$(PY) -m src.train --config configs/train_real_mobilenet_cropped.yaml

# Experiment B: crop-consistent fine-tune with the medium aug regime.
train-cropped-midaug:
	$(PY) -m src.train --config configs/train_real_mobilenet_cropped_midaug.yaml

# Experiment A-clean: crop-consistent fine-tune on the sketch-filtered clean set.
train-cropped-clean:
	$(PY) -m src.train --config configs/train_real_mobilenet_cropped_clean.yaml

# Gate Experiment A on the cross-dataset set; writes a DISTINCT artifact so the
# deployed baseline's realworld_eval.json is preserved for comparison.
eval-realworld-cropped:
	$(PY) -m src.eval_realworld --checkpoint artifacts/checkpoints_cropped/best_model.pth --data_dir data/asl_crossval --output artifacts/realworld_eval_cropped.json

# Gate Experiment B on the cross-dataset set (distinct artifact).
eval-realworld-cropped-midaug:
	$(PY) -m src.eval_realworld --checkpoint artifacts/checkpoints_cropped_midaug/best_model.pth --data_dir data/asl_crossval --output artifacts/realworld_eval_cropped_midaug.json

# Gate Experiment A-clean on the cross-dataset set (distinct artifact).
eval-realworld-cropped-clean:
	$(PY) -m src.eval_realworld --checkpoint artifacts/checkpoints_cropped_clean/best_model.pth --data_dir data/asl_crossval --output artifacts/realworld_eval_cropped_clean.json

# --- Diverse multi-source training (the data-domain lever) --------------------
# GUARD: scan the diverse training set for near-duplicates of the held-out eval
# set. MUST report ~0 contamination before training, or the gate is poisoned.
check-overlap:
	$(PY) scripts/check_eval_overlap.py --train_dir data/asl_diverse --eval_dir data/asl_crossval

# Experiment D1: train on the UNION of Marxulia + aliciiavs → SEPARATE checkpoint
# (artifacts/checkpoints_diverse/). Run download-real, download-diverse, and
# check-overlap first.
train-diverse:
	$(PY) -m src.train --config configs/train_real_mobilenet_diverse.yaml

# Gate D1 on the cross-dataset set (distinct artifact).
eval-realworld-diverse:
	$(PY) -m src.eval_realworld --checkpoint artifacts/checkpoints_diverse/best_model.pth --data_dir data/asl_crossval --output artifacts/realworld_eval_diverse.json

# Experiment D2: diverse merge + medium augmentation (push past D1).
train-diverse-midaug:
	$(PY) -m src.train --config configs/train_real_mobilenet_diverse_midaug.yaml

eval-realworld-diverse-midaug:
	$(PY) -m src.eval_realworld --checkpoint artifacts/checkpoints_diverse_midaug/best_model.pth --data_dir data/asl_crossval --output artifacts/realworld_eval_diverse_midaug.json

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

# Export the ONNX model to the web showcase's committed static asset path and
# regenerate the cross-language parity fixtures from it. Run after retraining so
# the live site and the parity gate both track the current checkpoint.
export-onnx-web:
	$(PY) -m src.export_onnx --output web/public/model/model.onnx --device cpu
	$(PY) -m src.gen_parity_fixtures --onnx web/public/model/model.onnx

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

# Static type-check gate (scoped to src via pyproject.toml).
mypy:
	$(PY) -m mypy src

# Alias for the type-check gate.
typecheck: mypy

# Build the CPU Docker image.
docker-build:
	docker build -t asl-cnn-classifier .

# Run a headless single-image inference inside the container (proves the image works).
docker-run:
	docker run --rm asl-cnn-classifier \
		python -m src.infer_camera --source data/sample/A/0.png --device cpu

# Build the image, then run in-container inference end-to-end.
docker-test: docker-build docker-run

# Deploy the Gradio demo to a Hugging Face Space (idempotent — safe to re-run).
# Requires HF_TOKEN (write scope) in the environment and HF_SPACE=<user>/<space>.
# Get a token at https://huggingface.co/settings/tokens. See docs/DEPLOY.md.
deploy-hf:
	$(PY) scripts/deploy_hf.py --space-id $(HF_SPACE) --sdk gradio

# Preview what WOULD be uploaded and the resolved Space URL — no token, no network.
deploy-hf-dryrun:
	$(PY) scripts/deploy_hf.py --space-id $(HF_SPACE) --sdk gradio --dry-run

# Remove generated artifacts (keeping .gitkeep) and bytecode caches.
clean:
	find artifacts -mindepth 1 -not -name '.gitkeep' -delete
	find . -type d -name __pycache__ -exec rm -rf {} +
