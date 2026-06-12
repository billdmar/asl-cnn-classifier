# Deploying the Gradio demo to Hugging Face Spaces

`app.py` is a self-contained [Gradio](https://gradio.app) app written to the
Hugging Face Spaces convention (entry file `app.py`, SDK `gradio`). It deploys
as a **CPU-only** Space.

> **Status: not yet deployed.** No Hugging Face token is wired up in this
> environment, so no live Space exists yet. Run the one command below to publish
> one, then paste the URL into the README's
> [Live demo](../README.md#live-demo) section.

> **Honesty note.** Until a trained checkpoint
> (`artifacts/checkpoints/best_model.pth`) is included, the Space loads the
> **untrained random-init fallback** and its predictions are **meaningless** —
> the in-app banner says so. The deploy script auto-detects the checkpoint and
> uploads it when present; without it you get an honest wiring demo, not an
> accurate model.

## Deploy in one command

```bash
# 1. Get a WRITE-scoped token at https://huggingface.co/settings/tokens
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export HF_SPACE=<your-username>/asl-cnn-classifier

# 2. Deploy (creates the Space if it doesn't exist, then syncs files).
make deploy-hf
```

That's it. The command prints the Space URL
(`https://huggingface.co/spaces/<user>/asl-cnn-classifier`); the Space builds
and starts automatically (first build takes a few minutes — PyTorch is large).

> **Python version matters.** `space/README.md` pins `python_version: "3.11"`.
> Do not remove it: on Python 3.13 the stdlib `audioop` module was removed
> (PEP 594), which breaks Gradio's transitive `pydub` import at startup
> (`ModuleNotFoundError: No module named 'pyaudioop'`) and the Space fails with a
> `RUNTIME_ERROR`. 3.11 matches the repo's own target and CI matrix.

The deploy is **idempotent**: `create_repo` runs with `exist_ok=True` and the
upload overwrites changed files, so re-running just re-syncs the latest state.

### Preview first (no token, no network)

```bash
export HF_SPACE=<your-username>/asl-cnn-classifier
make deploy-hf-dryrun
```

This lists exactly which files would be uploaded and the resolved Space URL
without authenticating or touching the network. You can also call the script
directly:

```bash
.venv/bin/python scripts/deploy_hf.py --space-id demo/asl --dry-run
```

### Required environment

| Variable   | Purpose                                              |
| ---------- | ---------------------------------------------------- |
| `HF_TOKEN` | Hugging Face **write** token. Never hardcode it; the script reads it from the env (or `--token`) and never logs it. |
| `HF_SPACE` | Target Space as `<user>/<space>`, consumed by the Makefile targets. |

## What gets uploaded

`scripts/deploy_hf.py` uploads the runtime essentials and ignores everything
else:

- **Uploaded:** `app.py`, `src/`, `configs/`, `data/sample/` (backs the in-app
  examples), the Space-side `README.md` and `requirements.txt` (see below), and
  `artifacts/checkpoints/best_model.pth` **if it exists**.
- **Ignored:** `.venv/`, `.git/`, `artifacts/*` (except a checkpoint),
  `__pycache__/`, `tests/`, `.github/`, `runs/`.

### Space-side metadata files (`space/`)

Hugging Face Spaces require two files at the Space root with reserved names that
would collide with this repo's own files, so they are staged under `space/` and
uploaded under their canonical names:

| Repo file               | Uploaded to Space as | Why                                                                 |
| ----------------------- | -------------------- | ------------------------------------------------------------------- |
| `space/README.md`       | `README.md`          | Holds the HF YAML frontmatter (`sdk: gradio`, `app_file: app.py`, …) that configures the Space. Kept out of the repo root so it doesn't clobber the project README. |
| `space/requirements.txt`| `requirements.txt`   | CPU-pinned runtime deps (gradio + torch + torchvision + pillow + numpy + the transitively-imported opencv-headless / scikit-learn). HF installs CPU torch by default. |

Note: the repo's `requirements-demo.txt` uses `-r requirements.txt` indirection,
which pulls in the full training/serving stack. `space/requirements.txt` is a
trimmed, self-contained list so the Space build stays lean and reliable.

## Local smoke test before deploying

```bash
make install                              # core deps + venv
.venv/bin/python -m pip install gradio    # or: uv pip install gradio
.venv/bin/python app.py                   # launches at http://127.0.0.1:7860
```

Exercise the prediction path headlessly (no server):

```bash
.venv/bin/python -c "from PIL import Image; import app; print(app.predict(Image.open('data/sample/A/0.png')))"
```

## Manual fallback (git push)

If you prefer not to use the script, you can push to the Space's git remote
directly:

1. **Create the Space** on <https://huggingface.co/new-space> (SDK: **Gradio**,
   Hardware: **CPU basic**).
2. **Authenticate the CLI:** `pip install huggingface_hub && huggingface-cli login`.
3. **Add the remote and prepare the Space files.** The Space reads a file
   literally named `requirements.txt`, so copy `space/requirements.txt` into
   place and `space/README.md` to the Space root as `README.md`:

   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/asl-cnn-classifier
   cp space/requirements.txt requirements.txt   # on the Space branch only
   cp space/README.md README.md                 # on the Space branch only
   git push space HEAD:main
   ```

   The Space needs at minimum `app.py`, `src/`, `data/sample/`, the
   `requirements.txt`, and the frontmatter `README.md`.
4. **(Recommended) Ship a trained checkpoint** so predictions are real. Commit
   `artifacts/checkpoints/best_model.pth` (Git LFS for large files) or have the
   app load it from the HF Hub at startup. Without it, the app runs the honest
   untrained fallback.

## After deploying

1. Paste the Space URL into the README's **Live demo** section (replace the
   `TODO` placeholder).
2. Record a short GIF / screenshot of the app and drop it in `docs/` (or
   `artifacts/`), then update the README image placeholder.
