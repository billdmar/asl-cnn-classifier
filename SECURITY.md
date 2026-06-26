# Security Policy

This is a portfolio project, but reports are welcome and appreciated.

## Reporting a vulnerability

Please open a [GitHub Security Advisory](https://github.com/billdmar/asl-cnn-classifier/security/advisories/new)
(preferred) or a regular issue for non-sensitive reports. Include repro steps and
the affected component (Python pipeline, FastAPI/Gradio serving, or the in-browser
web app).

## Scope notes

- **The web app runs 100% client-side.** Webcam frames and uploaded images are
  processed in the browser via onnxruntime-web and never leave the device — there
  is no inference backend receiving user media.
- **No secrets are committed.** Dataset downloads use public, credential-free
  Hugging Face datasets. Deploy tokens (Vercel, Hugging Face) live only in CI
  secrets / local env, never in the repo.
- **Datasets** are third-party; see `MODEL_CARD.md` for provenance and the
  no-declared-license caveats on the diversity sources.

## Supported

The `main` branch and the live deployment are supported. Older branches are not.
