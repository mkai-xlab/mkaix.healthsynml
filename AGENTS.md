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

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **mkaix.healthsynml** (697 symbols, 1063 relationships, 25 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/mkaix.healthsynml/context` | Codebase overview, check index freshness |
| `gitnexus://repo/mkaix.healthsynml/clusters` | All functional areas |
| `gitnexus://repo/mkaix.healthsynml/processes` | All execution flows |
| `gitnexus://repo/mkaix.healthsynml/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
