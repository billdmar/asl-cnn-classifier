# ASL Sign-Language CNN — Real-Time Image Classifier

**Full ML-engineering lifecycle for 29-class ASL recognition: PyTorch CNN + MobileNetV2 transfer learning, real-time OpenCV inference, and rigorous benchmarking.**

Recognize American Sign Language hand signs across all **29 classes** (A–Z plus
*space*, *delete*, *nothing*) with a PyTorch CNN, then run the trained model in a
**live OpenCV camera loop** for real-time classification. The project covers the
full ML-engineering lifecycle: dataset ingestion, stratified splitting,
augmentation-aware training, confusion-matrix evaluation, live inference, and a
latency/throughput benchmark with preprocessing ablations and distribution-shift
analysis.

[![CI](https://github.com/billdmar/asl-cnn-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/billdmar/asl-cnn-classifier/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Coverage](https://img.shields.io/badge/coverage-94%25%20(CI%20gate%20%E2%89%A580%25)-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Highlights

- **Two architectures** — a compact from-scratch CNN (~657K params) and a
  MobileNetV2 transfer-learning fine-tune, selectable via config.
- **Correct augmentation** — rotation, affine, color jitter, and resized-crop,
  **deliberately without horizontal flip** (ASL signs are not flip-invariant —
  b/d and p/q are mirror images).
- **Reproducible** — global seeding; file-level stratified 70/15/15 splits
  (`StratifiedShuffleSplit`) so no augmented view leaks across splits.
- **Real-time inference** — OpenCV webcam loop with an ROI box, on-screen
  prediction, confidence, and a rolling FPS counter.
- **Rigorous evaluation** — 29×29 confusion matrix, per-class F1, top-10
  confused pairs, and accuracy under five synthetic distribution shifts.
- **Benchmarked** — end-to-end latency (p50/p95/p99) and throughput on CPU and
  Apple-Silicon MPS, plus a preprocessing-stage ablation.
- **Engineered** — 94% test coverage, GitHub Actions CI, Dockerfile, Makefile,
  TensorBoard logging, and a full MODEL_CARD.

## Results

> **Accuracy status — read this.** The headline **≥98% test accuracy** is the
> target on the full ~87k-image Kaggle ASL Alphabet dataset and is reproduced
> with `make train` once the dataset is downloaded (see
> [Reproducing 98%](#reproducing-the-98-accuracy-target)). It is **not** yet
> reproduced in this checkout. Every number below that *is* measured on this
> machine is labeled as such; the tiny committed sample subset is a wiring
> sanity-check, not a meaningful accuracy.

| Metric | Value | Source |
| --- | --- | --- |
| Test accuracy — MobileNetV2, full dataset | **≥98% (target)** | reproduce via `make train` |
| Test accuracy — custom CNN, full dataset | ~95–98% (target) | reproduce via `make train` |
| Custom-CNN parameters | **656,829** | measured (`tests/test_model.py` asserts this) |
| CPU inference latency (mean) | **5.08 ms/frame** | measured, this machine |
| CPU throughput | **197 FPS** | measured, this machine |
| MPS (Apple-Silicon GPU) latency (mean) | **1.27 ms/frame** | measured, this machine |
| MPS throughput | **785 FPS** | measured, this machine |

*Latency/throughput measured with `make benchmark` (1000 frames, warm-up
excluded) on an Apple-Silicon Mac. CPU/GPU numbers are real today and do not
depend on training.*

## Quickstart

```bash
# 1. Install (creates an isolated Python 3.12 venv via uv and installs deps)
make install

# 2. Run the whole pipeline on the committed sample subset — no Kaggle needed
make sample-train     # trains 2 epochs on the 232-image sample fixture (CPU, <60s)
make eval             # confusion matrix, per-class F1, metrics.json
make benchmark        # latency/throughput + preprocessing ablation + dist-shift
make test             # pytest suite with coverage (>=80% enforced)

# 3. Real-time camera demo (needs a webcam)
make camera
# or classify a single image headlessly:
python -m src.infer_camera --source data/sample/A/0.png
```

> Uses [`uv`](https://github.com/astral-sh/uv) to manage Python 3.12 (PyTorch has
> no wheels for newer interpreters). Install uv with `brew install uv` or the
> [standalone installer](https://docs.astral.sh/uv/getting-started/installation/).

## Reproducing the 98% accuracy target

The sample subset cannot produce real accuracy. To train on the real data:

```bash
# Requires Kaggle API credentials at ~/.kaggle/kaggle.json (chmod 600).
python -m src.download_data            # downloads grassknoted/asl-alphabet (~1GB)
make train                             # full custom-CNN training
# or the transfer variant that reliably hits >=98%:
python -m src.train --config configs/train_mobilenet.yaml
make eval                              # writes the real accuracy to metrics.json
```

On Apple-Silicon MPS, full training takes roughly **30–90 minutes**. After it
finishes, update the Results table with the value from `artifacts/metrics.json`.

## Architecture

```
Input 3×128×128
  Block 1:  [Conv 3→32 → BN → ReLU] ×2 → MaxPool → Dropout2d(0.1)    → 32×64×64
  Block 2:  [Conv 32→64 → BN → ReLU] ×2 → MaxPool → Dropout2d(0.1)   → 64×32×32
  Block 3:  [Conv 64→128 → BN → ReLU] ×2 → MaxPool → Dropout2d(0.15) → 128×16×16
  Block 4:  Conv 128→256 → BN → ReLU → MaxPool → Dropout2d(0.2)      → 256×8×8
  Global Average Pool → 256
  FC 256→256 → ReLU → Dropout(0.5) → FC 256→29
```

Global average pooling keeps the classifier head tiny, so the whole network is
**~657K parameters** — fast to train and deploy. The MobileNetV2 variant
(`--arch mobilenet_v2`) freezes the ImageNet backbone for a 5-epoch warm-up,
then fine-tunes end-to-end at a 10× lower learning rate.

## Preprocessing ablation & distribution shift

`make benchmark` localizes the inference bottleneck by progressively removing
preprocessing stages. On this machine the model forward pass is roughly half of
end-to-end latency; resize and normalization account for most of the rest (see
`artifacts/benchmark_ablation.png`).

It also characterizes how accuracy degrades under five synthetic corruptions
(Gaussian blur, JPEG q20, brightness ×0.4 / ×1.8, 5% salt-and-pepper) →
`artifacts/distribution_shift.json`. Low-light (brightness ×0.4) is the harshest
shift, which mirrors real-world failure modes.

## Common confusions

ASL letters that share hand shapes are the usual error sources — **M/N/S**
(fist variants) and **A/E/S** are classic confusions. After training, the
top-10 confused pairs are written to `artifacts/per_class_errors.txt` and
visualized in `artifacts/confusion_matrix.png`.

## Real-world caveat

98% is measured on the Kaggle benchmark, whose images are highly uniform
(consistent signer, lighting, and background). **Real-world accuracy is lower**
and varies with lighting, skin tone, background clutter, and camera angle. See
[`MODEL_CARD.md`](MODEL_CARD.md) for limitations and ethical considerations.

## Project layout

```
src/
  dataset.py        # ASLDataset, stratified splits, canonical DRY transforms
  model.py          # CustomCNN, MobileNetV2/ResNet18 transfer, build_model factory
  train.py          # training loop: cosine LR, early stopping, TensorBoard, AMP
  eval.py           # confusion matrix, per-class F1, distribution shift
  infer_camera.py   # real-time OpenCV inference (ROI, FPS, snapshots)
  benchmark.py      # latency/throughput + preprocessing ablation
  download_data.py  # Kaggle download helper
  utils.py          # seeding, device selection (CUDA → MPS → CPU)
tests/              # 73 tests, 94% coverage
configs/            # train_custom_cnn.yaml, train_mobilenet.yaml
data/sample/        # 232 committed sample images (CI fixture)
```

## Dataset

[ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) by
*grassknoted* on Kaggle — ~87,000 200×200 RGB images across 29 classes.

## License

[MIT](LICENSE) © William Mar
