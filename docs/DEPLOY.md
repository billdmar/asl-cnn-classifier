# Deploying the Gradio demo to Hugging Face Spaces

`app.py` is a self-contained [Gradio](https://gradio.app) app written to the
Hugging Face Spaces convention (entry file `app.py`, SDK `gradio`). These steps
deploy it as a **CPU-only** Space.

> **Status: not yet deployed.** The author does not have a Hugging Face account
> wired up in this environment, so no live Space exists yet. Follow the steps
> below to publish one, then paste the URL into the README's
> [Live demo](../README.md#live-demo) section.

> **Honesty note.** Until you upload a trained checkpoint to the Space (see
> step 5), the app loads the **untrained random-init fallback** and its
> predictions are meaningless — the in-app banner says so. Deploying the UI is
> fine for a wiring demo, but do not present it as accurate until a real
> `best_model.pth` is included.

## One-time setup

1. **Create the Space.** On <https://huggingface.co/new-space>:
   - Owner: your username/org.
   - Space name: e.g. `asl-cnn-classifier`.
   - SDK: **Gradio**. Hardware: **CPU basic** (free) is sufficient.
   - Visibility: Public (or Private).

2. **Install and authenticate the HF CLI** (locally):

   ```bash
   pip install huggingface_hub
   huggingface-cli login        # paste a token from https://huggingface.co/settings/tokens
   ```

3. **Add the Space as a git remote** in this repo and prepare the Space's
   `requirements.txt`. Spaces install from a file literally named
   `requirements.txt`, so copy the demo requirements into that name on the
   Space branch (this repo keeps demo deps in `requirements-demo.txt` to avoid
   clobbering the core file):

   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/asl-cnn-classifier
   # Build the Space's requirements.txt from the demo deps (gradio + core stack):
   cp requirements-demo.txt hf-space-requirements.txt   # optional staging copy
   ```

   When pushing to the Space, the file the Space reads as `requirements.txt`
   must contain `gradio` plus the core deps. The simplest path is to commit a
   `requirements.txt` on the Space that is just:

   ```text
   -r requirements-demo.txt
   ```

   or, if the Space disallows the `-r` indirection, inline the contents of
   `requirements.txt` + `gradio>=4.0,<6.0` into the Space's `requirements.txt`.

4. **Push the app.** From the repo root:

   ```bash
   git push space HEAD:main
   ```

   The Space needs at minimum: `app.py`, `src/`, `data/sample/` (for the
   example images), and a `requirements.txt` as described above. The Space will
   build and start automatically; first build takes a few minutes (PyTorch is
   large).

5. **(Recommended) Ship a trained checkpoint** so predictions are real. Either
   commit `artifacts/checkpoints/best_model.pth` to the Space (Git LFS for large
   files) or load it from the HF Hub at startup. Without it, the app runs the
   honest untrained fallback. Update `DEFAULT_CHECKPOINT` handling only if you
   change where the checkpoint lives.

## Local smoke test before deploying

```bash
make install                 # core deps + venv
.venv/bin/python -m pip install gradio   # or: uv pip install gradio
.venv/bin/python app.py      # launches at http://127.0.0.1:7860
```

You can also exercise the prediction path headlessly (no server):

```bash
.venv/bin/python -c "from PIL import Image; import app; print(app.predict(Image.open('data/sample/A/0.png')))"
```

## After deploying

1. Paste the Space URL into the README's **Live demo** section (replace the
   `TODO` placeholder).
2. Record a short GIF / screenshot of the app and drop it in `docs/` (or
   `artifacts/`), then update the README image placeholder.
