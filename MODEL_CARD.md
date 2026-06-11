# Model Card — ASL Sign-Language CNN

## Model Details

- **Developer:** William Mar
- **Type:** Image classifier (convolutional neural network) for static ASL
  hand-sign recognition.
- **Architectures:**
  - *Custom CNN* — 4 convolutional blocks with batch norm, dropout, and global
    average pooling followed by a 2-layer classifier head. **656,829
    parameters.** Input 3×128×128, output 29 logits.
  - *Transfer variant* — MobileNetV2 (or ResNet18) pretrained on ImageNet, with
    a replaced 29-way classifier head; frozen-backbone warm-up then full
    fine-tune at 10× lower LR.
- **Framework:** PyTorch 2.x / torchvision. Devices: CUDA, Apple-Silicon MPS, or
  CPU (auto-detected).
- **Training:** AdamW, cosine-annealing LR schedule, early stopping on
  validation loss, best-by-val-accuracy checkpointing, optional CUDA AMP.
- **Serving formats:** native PyTorch (`.pth`), ONNX export (numerically
  parity-tested against PyTorch within `atol=1e-4`), and an INT8 dynamically
  quantized variant. Exposed via an OpenCV live-camera loop and a FastAPI
  `/predict` endpoint.
- **License:** MIT.

## Intended Use

- **Primary use:** An educational / portfolio demonstration of an end-to-end
  computer-vision pipeline — training, evaluation, real-time inference, and
  benchmarking — for static ASL alphabet recognition.
- **Intended users:** Learners and engineers exploring CNN image classification
  and real-time inference patterns.
- **Out of scope:** This is **not** an accessibility or communication product.
  It classifies single still frames of the ASL *alphabet* only; it does not
  recognize continuous signing, words, grammar, or the motion-based letters
  (J, Z) that require temporal modeling. It must not be relied upon for any
  safety-critical, medical, legal, or real communication-assistance purpose.

## Training Data

- **Dataset:** [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
  (grassknoted, Kaggle) — ~87,000 200×200 RGB images across 29 classes (A–Z,
  *space*, *delete*, *nothing*), roughly balanced at ~3,000 images per class.
- **Splits:** File-level stratified 70/15/15 train/val/test via
  `StratifiedShuffleSplit` (seed 42), so augmented views never leak across
  splits.
- **Augmentation:** Random resized crop, rotation (±15°), affine
  (translate/scale/shear), and color jitter. **No horizontal flip** — ASL signs
  are not flip-invariant (b/d, p/q are mirror images).
- **Committed sample subset:** `data/sample/` holds 232 deterministic *synthetic*
  images (8 per class) used purely as a CI/wiring fixture. Accuracy on it is
  meaningless.

## Evaluation Data

- Held-out 15% test split of the Kaggle dataset (never seen during training or
  validation), recreated deterministically from the seed.
- Reported metrics: overall accuracy, macro precision/recall/F1
  (`classification_report`), a 29×29 confusion matrix, and the top-10 confused
  class pairs.

## Performance

- **Target test accuracy:** ≥98% (MobileNetV2 fine-tune) on the full Kaggle
  dataset. Reproduce with `make train`; the measured value is written to
  `artifacts/metrics.json`. **Not yet reproduced in this checkout** — see the
  README Results note.
- **Inference performance (measured, Apple-Silicon Mac):** ~5.08 ms/frame
  (197 FPS) on CPU; ~1.27 ms/frame (785 FPS) on MPS, end-to-end including
  preprocessing.
- **Robustness:** Accuracy is characterized under five synthetic distribution
  shifts (Gaussian blur, JPEG q20, brightness ×0.4 / ×1.8, salt-and-pepper) in
  `artifacts/distribution_shift.json`. Low-light degrades accuracy most.
- **Calibration:** Expected Calibration Error (ECE) and a reliability diagram are
  produced by `src/calibration.py` (`artifacts/calibration.json`). The ECE
  computation is unit-tested against analytically known values; the reported ECE
  is only meaningful once the model is trained on real data.
- **Explainability:** `src/gradcam.py` produces Grad-CAM saliency overlays for
  the predicted class. On an untrained model over the synthetic fixture these are
  wiring demonstrations, not interpretable saliency.

## Limitations

- **Benchmark optimism / data leakage:** The Kaggle dataset's images are highly
  homogeneous (same signer, consistent lighting and background per class). A
  random split places near-duplicate frames in train and test, so the ≥98%
  figure is **optimistic** and not representative of real-world use. A
  group-aware split (by signer/session) would lower it substantially.
- **Static frames only:** No temporal modeling; motion letters (J, Z) are
  captured as single frames and are inherently ambiguous.
- **Shared hand shapes:** M/N/S and A/E/S are commonly confused.
- **Real-world shift:** Lighting, skin tone, background clutter, camera angle,
  and hand position all reduce accuracy versus the benchmark.

## Ethical Considerations

- **Demographic representation:** The training data appears to feature a narrow
  range of skin tones and a single signing environment. A model trained on it
  will likely **underperform for under-represented skin tones and settings**.
  Any production or accessibility use would require a diverse, consented,
  group-split dataset and per-group fairness evaluation.
- **Respect for the Deaf community:** ASL is a complete language with its own
  grammar; alphabet fingerspelling is a small part of it. This project should
  not be presented as "translating ASL." Accessibility tools for Deaf and
  hard-of-hearing users should be built with that community, not merely for it.
- **Honest reporting:** Benchmark accuracy is reported alongside its leakage
  caveat specifically to avoid overstating real-world capability.
