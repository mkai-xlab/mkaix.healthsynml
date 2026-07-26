# Repository Guidelines

## Project Structure & Module Organization

`app/` contains the FastAPI and ML inference code: routes live under `app/api/v1/`, configuration under `app/core/`, model definitions and pipelines under `app/ml/`, and orchestration services under `app/services/`. `main.py` is the API entry point. Tests are in `tests/` and follow the application boundaries. Training and ablation work belongs in `notebooks/`; completed results and figures belong in `docs/report/<model>/`. Store operational utilities in `scripts/`, temporary analysis helpers in `scratch/`, and standalone tools in `tools/`. Model weights are expected under `checkpoints/<model>/` and are mounted read-only in containers.

## Build, Test, and Development Commands

```bash
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8005
pytest -q
docker compose up --build -d
```

The first command installs API, imaging, ML, and test dependencies. Uvicorn starts the local API; required YOLO/classifier checkpoints must exist first. `pytest -q` runs the regression suite. Docker Compose builds the production image, exposes port `8005`, and mounts `./checkpoints` read-only. Check readiness with `curl http://127.0.0.1:8005/api/v1/health`.

## Coding Style & Naming Conventions

Use Python with four-space indentation and PEP 8 layout. Prefer `snake_case` for modules, functions, and variables; `PascalCase` for classes; and uppercase names for constants. Add type hints to public functions and keep FastAPI schemas explicit. No formatter is enforced, so preserve surrounding style, group imports, and avoid unrelated notebook or metadata churn. Keep inference preprocessing synchronized with checkpoint metadata and training transforms.

## Testing Guidelines

Use pytest and name files `tests/test_<feature>.py` and functions `test_<behavior>()`. Add focused regression tests for API schemas, model-mode validation, preprocessing dimensions/orientation, ensemble weighting, and native-CAM geometry. Mock expensive model loading where practical. There is no configured coverage threshold; changes should cover all altered behavior. Never validate configuration choices on the repeatedly inspected test set.

## Commit & Pull Request Guidelines

Use a concise imperative subject; existing history accepts plain subjects and prefixes such as `feat:` or `fix:`. Keep commits scoped. PRs should explain behavior and model/configuration impact, list tests run, link the issue, and include screenshots or CAM/report artifacts for visual changes. CI targets pull requests to `dev`.

## Security & Clinical Safeguards

Do not commit secrets, patient data, or new checkpoint binaries. Use `.env` locally and read-only checkpoint mounts. Before modifying training, ROI, reporting, or endpoint contracts, read `AGENT_GUIDELINE.md`; notably, do not center-crop away marginal osteophytes, and archive successful notebook runs with their exact metrics and timestamp.
