# ULTRACODE Enhancement Prompt — asl-cnn-classifier → best-in-class recruiter showcase

> Paste the block below into a Claude Code session opened at `~/asl-cnn-classifier`.
> It is self-contained. It assumes the repo state at `main` (real 26-class
> MobileNetV2, val 97.8% / test 96.8%, green Python suite, live HF Gradio Space).

---

ultracode

Act as a world-class product+ML engineering org with many specialized teams. Take
`~/asl-cnn-classifier` from "good ML repo with a Gradio demo" to a **best-in-class,
recruiter-facing portfolio SHOWCASE**: a custom, polished, genuinely-impressive
website backed by a model that actually works on a real webcam. Aggressively
parallelize with subagent fan-out; sequence only genuinely dependent steps.
Every line of code must be accurate, well-made, and do exactly as intended —
verified, not assumed. Never fabricate a metric.

## NORTH STAR
Audience = engineering/ML hiring managers. "Best" = polish, visible rigor, and a
live demo that wows. Optimize for being looked at and tried, not for scale.

## GROUND TRUTH FIRST (before any planning)
Read README.md, MODEL_CARD.md, src/*.py, configs/*, tests/*, .github/workflows/ci.yml,
scripts/deploy_hf.py, space/*, and artifacts/*.json. Confirm: real 26-class A–Z
MobileNetV2 checkpoint (artifacts/checkpoints/best_model.pth, val 97.8% / test
96.8%), ONNX export path (src/export_onnx.py), Grad-CAM/calibration/benchmark
artifacts, FastAPI serve.py, live HF Gradio Space. Treat closing the gap between
"impressive on paper" and "impressive when a recruiter actually uses it live" as
priority #1.

## TARGET ARCHITECTURE (locked decisions — do not relitigate)
- **Monorepo restructure** into `web/` (Next.js) and `ml/` (the existing Python).
  Refactor freely where it improves clarity; END STATE MUST BE GREEN. The existing
  HF Gradio Space + FastAPI may remain as a legacy/optional backend but the website
  does NOT depend on them.
- **Website:** Next.js + TypeScript (strict), Tailwind + shadcn/ui + Framer Motion.
  **Modern dark "AI product" aesthetic** (think Vercel/Linear/HF demos): accent
  gradient, animated confidence bars, tasteful motion.
- **Inference: 100% IN-BROWSER** via onnxruntime-web (WASM/WebGPU). No server
  inference for the site. Webcam frames never leave the browser (make this a
  privacy selling point). Deploys as a static site.
- **Hosting: Vercel** (auto-deploy the `web/` subdir on merge to main).

## SINGLE SOURCE OF TRUTH FOR PREPROCESSING (critical correctness)
The full pipeline — hand-crop → resize → ImageNet normalize → tensor layout — MUST
be defined once and reproduced identically in (a) Python train/eval, (b) ONNX
export assumptions, (c) the browser JS/TS path. Mismatch silently destroys live
accuracy. **HARD GATE:** an automated cross-language parity test that feeds the
SAME image through the Python path and the JS/ONNX path and asserts predicted
class + probabilities match within tolerance (e.g. atol 1e-3 on probs). This test
must run in CI and block merge. This is the #1 silent-bug risk — treat it as
non-negotiable.

## WORKSTREAM 1 — REAL-WORLD ROBUSTNESS (ML; do this BEFORE the live demo ships)
The model scores 96.8% on its uniform test set but will do far worse on a
recruiter's cluttered webcam. Close that gap with ALL of:
1. **MediaPipe hand detection + crop (keystone):** detect the hand and crop to a
   normalized box before classifying, in BOTH the Python pipeline (MediaPipe
   Python / Tasks) AND the browser (MediaPipe Tasks JS). This removes background
   dependence — the biggest real-world win. Preprocessing parity must include the
   crop step.
2. **Aggressive domain augmentation:** retrain with heavy background replacement,
   lighting/contrast/brightness jitter, blur, random crops/rotations, and
   skin-tone variation so the model generalizes past the dataset's uniformity.
3. **More diverse data:** pull at least one additional, more varied ASL alphabet
   dataset from the HF Hub (more signers/backgrounds/skin tones), combine with the
   existing `Marxulia/asl_sign_languages_alphabets_v03`, dedupe/balance, and
   retrain. Keep the ingestion reproducible (extend src/download_hf_data.py).
4. **Confidence calibration + "unsure" UX:** use/extend the calibration work
   (ECE/temperature scaling) so low-confidence live predictions honestly render as
   "unsure — adjust hand/lighting" instead of a confident wrong letter.
Deliverable: a retrained checkpoint with REAL measured numbers on BOTH the
held-out test set AND a small real-world/webcam-style validation set, plus a fresh
ONNX export. Report measured benchmark vs real-world accuracy separately.

## WORKSTREAM 2 — WEBSITE FEATURES (all four ship)
1. **Live webcam inference (the centerpiece):** real-time in-browser prediction,
   ROI/hand box overlay, top-5 animated confidence bars, live FPS, and the
   calibrated "unsure" state. On-screen guidance (plain background, good lighting,
   hand in box). Graceful no-camera/permission-denied handling.
2. **Image upload inference:** drag-drop / click-to-upload, same result viz. The
   fallback path so every recruiter can try it.
3. **Interactive metrics dashboard:** render the REAL artifacts (confusion matrix,
   per-class F1, calibration/reliability diagram, training curves, top-confused
   pairs) as interactive components sourced from the committed JSON. No hardcoded
   numbers.
4. **Project-story / model-card page:** the narrative recruiters read — problem,
   architecture diagram, honest accuracy story (benchmark vs real-world, leakage
   caveats, ethics), Grad-CAM explainability gallery, tech-stack, and a
   "how it works / in-browser inference" explainer.

## QUALITY GATES (acceptance criteria — all must pass; this is what "well-made" means)
Python/ML:
- Existing suite stays green; new ML logic gets tests. ruff + black + mypy clean.
- Retraining is reproducible via make targets; all displayed numbers come from code.
Web:
- **TypeScript strict** (no `any`), ESLint + Prettier clean, enforced in CI.
- **Unit tests** (Vitest + React Testing Library) for components AND the ONNX
  preprocessing/inference logic — including the cross-language parity test.
- **Playwright E2E:** load the site, upload a known hand image, assert the correct
  prediction renders; exercise the metrics dashboard. Runs in CI.
- **Lighthouse + a11y budget:** perf & accessibility ≥ 90, WCAG basics, enforced in CI.
Cross-cutting:
- Preprocessing parity gate (above) is mandatory and CI-blocking.

## ACCURACY HONESTY RULE (enforce everywhere on the site + docs)
Every displayed number must be produced by code in the repo on a real set, and
labeled with its source and whether it's benchmark vs real-world. Aspirational
numbers are allowed ONLY if explicitly marked "target". No rounding up, no
target-stated-as-fact. This is consistent with the whole project's ethos.

## EXECUTION & DELIVERY
- Decompose into phases and fan out subagents per independent scope (use git
  worktree isolation for agents touching overlapping files). Suggested phases:
  (0) monorepo restructure + canonical preprocessing module,
  (1) robustness ML retrain + new ONNX export + parity test,
  (2) Next.js scaffold + design system + in-browser inference core,
  (3) the four features,
  (4) metrics dashboard wired to real artifacts,
  (5) polish + a11y/Lighthouse + E2E + Vercel deploy.
  After each phase, integrate and re-verify before the next.
- **Auto-merge any phase whose full CI gate is green** (tests, strict TS, lint,
  parity, E2E, Lighthouse, accuracy thresholds). The gates are the guardrail —
  make them strict enough to earn it. Vercel auto-deploys `web/` on merge.
- **ONE hard exception that always pauses for the human:** anything requiring the
  Hugging Face token (the legacy HF Space redeploy) — that secret is the user's to
  supply. Everything else may land automatically when green.
- Each agent reports concise results (what changed, how verified, residual risk).
  Never fabricate metrics; if something can't be measured, say so.

## DELIVERABLE
A monorepo where: the live Vercel site is a polished dark-mode ASL classifier that
works on a real webcam (hand-crop + calibrated), every headline number is backed
by reproducible code and honestly labeled, the whole thing is covered by strict
typing + unit + parity + E2E + Lighthouse gates in CI, and the existing Python
pipeline + HF Space remain green. Provide the live Vercel URL.
