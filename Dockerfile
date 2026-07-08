# CPU image for training/eval/benchmark on machines without a GPU.
# Uses the PyTorch CPU wheel index to avoid pulling huge CUDA wheels.
FROM python:3.12-slim

WORKDIR /app

# opencv-python-headless still dynamically links libglib at import time; install
# the minimal runtime lib so `import cv2` works on the slim base. (No GUI/GL
# libraries are needed because we use the headless OpenCV wheel.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install torch/torchvision from the CPU wheel index first (avoids pulling huge
# CUDA wheels), then install the project from pyproject.toml which resolves the
# remaining dependencies from PyPI. Copy only the build metadata first for
# Docker layer caching — dependencies change far less often than source code.
COPY pyproject.toml README.md ./
COPY src/__init__.py src/__init__.py
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2,<3.0" "torchvision>=0.17,<1.0" \
    && pip install --no-cache-dir .

# Copy the project source (includes the committed data/sample fixture, so the
# image can run a headless inference end-to-end without external data).
COPY . .

# No network ports are served; this is a batch/CLI image. (No EXPOSE.)
# Default to a headless single-image inference on the committed sample fixture,
# which proves the full preprocess -> model -> prediction path runs in-container.
# (If no trained checkpoint is baked in, infer_camera falls back to a randomly
# initialized model with a warning, so this stays runnable out of the box.)
CMD ["python", "-m", "src.infer_camera", \
     "--source", "data/sample/A/0.png", "--device", "cpu"]

# ---------------------------------------------------------------------------
# GPU variant (commented): swap the base image for a CUDA-enabled PyTorch image
# and drop the CPU index install above, since torch ships preinstalled there.
#
#   FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
#   WORKDIR /app
#   COPY pyproject.toml src/__init__.py ./
#   RUN pip install --no-cache-dir .
#   COPY . .
#   CMD ["python", "-m", "src.eval", "--help"]
# ---------------------------------------------------------------------------
