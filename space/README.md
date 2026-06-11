---
title: ASL Sign-Language CNN
emoji: 🤟
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# ASL Sign-Language CNN — demo

A PyTorch CNN that classifies static American Sign Language hand signs
(A–Z, plus *space* / *del* / *nothing*). Upload a cropped image of a single
hand sign and the app returns the predicted class with top-5 probabilities.

This Space hosts the Gradio demo (`app.py`) from the
[`asl-cnn-classifier`](https://github.com/billdmar/asl-cnn-classifier)
repository. Source, training code, model card, and CI live there.

## Honesty note

If no trained checkpoint (`artifacts/checkpoints/best_model.pth`) is bundled
with this Space, the app loads an **untrained, random-init** model and its
predictions are **meaningless** — they only demonstrate the wiring. The in-app
banner says so. To get real predictions, train a model (`make train` in the
repo) and include `best_model.pth` when deploying. See the repo's `MODEL_CARD.md`.

> This file is the *Space-side* README. The HF YAML frontmatter above
> configures the Space (SDK, hardware, entry file). It intentionally lives at
> `space/README.md` in the source repo so it never collides with the repo's own
> `README.md`; `scripts/deploy_hf.py` uploads it to the Space **as** `README.md`.
