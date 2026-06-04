# CPU image for training/eval/benchmark on machines without a GPU.
# Uses the PyTorch CPU wheel index to avoid pulling huge CUDA wheels.
FROM python:3.12-slim

WORKDIR /app

# Install torch/torchvision from the CPU wheel index first, then the remaining
# dependencies from PyPI. (No build-essentials needed: all deps ship wheels.)
COPY requirements.txt ./
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2,<3.0" "torchvision>=0.17,<1.0" \
    && pip install --no-cache-dir -r requirements.txt

# Copy the project source.
COPY . .

# No network ports are served; this is a batch/CLI image. (No EXPOSE.)
# Default to a harmless no-op that proves the package imports cleanly.
CMD ["python", "-m", "src.eval", "--help"]

# ---------------------------------------------------------------------------
# GPU variant (commented): swap the base image for a CUDA-enabled PyTorch image
# and drop the CPU index install above, since torch ships preinstalled there.
#
#   FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
#   WORKDIR /app
#   COPY requirements.txt ./
#   RUN pip install --no-cache-dir -r requirements.txt
#   COPY . .
#   CMD ["python", "-m", "src.eval", "--help"]
# ---------------------------------------------------------------------------
