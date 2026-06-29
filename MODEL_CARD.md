# Model Card — ASL Sign-Language CNN

## Model Details

- **Developer:** William Mar
- **Type:** Image classifier (convolutional neural network) for static ASL
  hand-sign recognition.
- **Deployed model:** the **MobileNetV2 transfer variant**, **26-class (A–Z)**,
  trained on a 3-source union (see Training Data). This is what the live web demo
  and the committed `web/public/model/model.onnx` run. The from-scratch Custom
  CNN is retained in the repo as a baseline/ablation, not deployed.
- **Architectures:**
  - *Transfer variant (deployed)* — MobileNetV2 pretrained on ImageNet with a
    replaced **26-way** classifier head; frozen-backbone warm-up then full
    fine-tune at 10× lower LR. Input 3×128×128.
  - *Custom CNN (baseline, not deployed)* — 4 convolutional blocks with batch
    norm, dropout, and global average pooling followed by a 2-layer classifier
    head. **656,829 parameters.** (From-scratch CNNs stall near chance on real
    hands; transfer learning is why the deployed model works — see README.)
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

- **Deployed model — 3-source union (26 classes, A–Z):** the diversity of the
  training data is the only lever that ever moved the honest cross-dataset number
  (33.4% → 47.6% → 55.5% as sources were added — see
  `docs/EXPERIMENT_supply_exhausted_closure.md`). The deployed checkpoint trains
  on the union of three HF Hub datasets (`make download-real download-diverse
  download-hemg`): **Marxulia** (single signer, plain background), **aliciiavs**
  (multi-signer, real backgrounds — the diversity that drove the biggest gain),
  and **Hemg** (adds the only static J/Z frames). Config:
  `configs/train_real_mobilenet_diverse_hemg.yaml`.
- **Splits:** File-level stratified 70/15/15 train/val/test (seed 42), so
  augmented views never leak across splits.
- **Augmentation:** Random resized crop, rotation (±15°), affine
  (translate/scale/shear), and color jitter. **No horizontal flip** — ASL signs
  are not flip-invariant (b/d, p/q are mirror images).
- **Train↔eval contamination guard:** `make check-overlap-hemg` (perceptual hash)
  confirms the training union does not overlap the held-out cross-dataset gate.
- **Committed sample subset:** `data/sample/` holds deterministic *synthetic*
  images used purely as a CI/wiring fixture. Accuracy on it is meaningless.
- **Historical note:** earlier baselines used the single 29-class Kaggle "ASL
  Alphabet" set (grassknoted); the project moved away from it precisely because
  its single-signer homogeneity inflates same-dataset accuracy (see Limitations).

## Evaluation Data

- **Deploy decider — honest cross-dataset gate:** the deployed model is judged on
  a **held-out 4th dataset** (EitanG98, `data/asl_crossval/`, 712 A–Z frames —
  different signers/backgrounds, never trained on), produced by
  `make eval-realworld-diverse-hemg` → `web/public/metrics/realworld_eval.json`.
  This cross-dataset number is the **only** metric used to decide what ships.
- **Same-dataset benchmark (for contrast only):** a held-out split of the
  training distribution (`web/public/metrics/metrics.json`) — reported solely to
  show the leakage gap, never as the goal.
- Reported metrics: overall + A–Y accuracy, macro precision/recall/F1, a 26×26
  confusion matrix, and the top confused class pairs.

## Performance

- **Honest cross-dataset accuracy (the deploy metric):** **55.5%** over all 26
  classes / **59.8%** on the A–Y headline (excluding the dynamic motion letters
  J, Z), macro-F1 0.548 / 0.603, measured on the held-out EitanG98 gate (712
  frames). This is the **only** number used to decide what ships. Reproduce with
  `make reproduce-deployed` (or `make eval-realworld-diverse-hemg` on the trained
  checkpoint) → `web/public/metrics/realworld_eval.json`.
- **Same-dataset benchmark (NOT the goal):** **96.9%** on a held-out split of the
  training distribution (`web/public/metrics/metrics.json`). This is inflated by
  near-duplicate frames across the split (see Limitations) and is reported only
  to make the ~37-point leakage gap visible — it must not be cited as the model's
  real-world accuracy.
- **Inference performance (measured, Apple-Silicon Mac):** ~5.08 ms/frame
  (197 FPS) on CPU; ~1.27 ms/frame (785 FPS) on MPS, end-to-end including
  preprocessing.
- **Robustness:** Accuracy is characterized under five synthetic distribution
  shifts (Gaussian blur, JPEG q20, brightness ×0.4 / ×1.8, salt-and-pepper) in
  `artifacts/distribution_shift.json`. Low-light degrades accuracy most.
- **Calibration:** Expected Calibration Error (ECE) and a reliability diagram are
  produced by `src/calibration.py` (`web/public/metrics/calibration.json`, shown
  on the web dashboard). The ECE computation is unit-tested against analytically
  known values.
- **Explainability:** `src/gradcam.py` produces Grad-CAM saliency overlays for
  the predicted class; pre-computed overlays for the bundled examples are shown
  in the web app (in-browser ONNX can't expose gradients, so they're generated
  offline via `make gradcam-web`).

## Limitations

- **Benchmark optimism / data leakage:** single-signer source images are highly
  homogeneous (same signer, consistent lighting/background per class), so a random
  split places near-duplicate frames in train and test. That is why the
  same-dataset **96.9%** is optimistic and the honest cross-dataset number is
  **55.5% / 59.8%** — a ~37-point gap. Always cite the cross-dataset figure as
  the model's real-world accuracy.
- **Accuracy is sourcing-bound (closed):** training-data diversity is the only
  lever that moved the honest number; the accessible clean-data supply is
  exhausted and every other lever (crop, augmentation tiers, TTA, per-class
  thresholds, temperature calibration, class-balanced loss, SWA + label smoothing,
  and two architecture swaps) was measured and rejected — documented as negatives
  in `docs/EXPERIMENT_*.md`.
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
