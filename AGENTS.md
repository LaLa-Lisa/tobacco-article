# Repository Guidelines

## Research Scope & Priorities

Follow [notes/todo.md](notes/todo.md): tobacco detection → protective regions → HSV vegetation mask → weed mask. Robotics, targeting, and physical crop safety are outside scope. Photographic streams do not establish real-video quality; exclude the historical 110-image results from new baselines. Completed writing tasks do not establish implementation or validation.

Prioritize trusted annotations and splits, then the evaluator/pipeline, resource screening, training, ablations, final evaluation, and manuscript results. Adaptive HSV and asymmetric loss remain hypotheses requiring experiments.

## Project Structure & Module Organization

- `dataset_notebooks/`: numbered preparation, labeling, training, and review notebooks; synthetic validator.
- `manual_notebooks/`: model notebooks, shared `scripts/`, provider adapters in `scripts/backends/`, and `tests/`.
- `research/`: reviews and bibliography; `notes/`: plans; `experiments/`: splits, results, and environment records.
- `data/` and `weights/`: local datasets/checkpoints, largely Git-ignored.

Read directory-specific `AGENTS.md` before editing a subtree.

## Build, Test, and Development Commands

Run from the repository root; there is no central build:

- `.venv-notebooks/bin/python -m unittest discover -s manual_notebooks/tests -v`: CPU adapter tests with mocked providers.
- `python dataset_notebooks/validate_dataset_notebooks.py`: synthetic checks requiring the SAM 3 environment.
- `python -m jupyterlab`: launch notebooks with the intended kernel; configure paths and execution flags first.

Dependency specifications: `dataset_notebooks/requirements.txt` and optional `manual_notebooks/requirements-rfdetr.txt`.

## Coding Style & Testing Guidelines

Use four-space indentation, `snake_case`, `PascalCase` classes, uppercase constants, and existing typed contracts. Keep provider behavior in adapters and imports lazy. Preserve `MODELS_REGESTRY` spelling. No repository-wide formatter or coverage threshold is configured.

Use `unittest`, `test_*.py`, temporary fixtures, and mocked providers. Check evaluator formulas, empty masks, coordinate transforms, and loss gradients with known synthetic cases when implementing them. Distinguish software tests from real-model validation.

## Data & Experiment Rules

Treat current annotations as provisional. Preserve provenance/checksums; split independent fields/campaigns before teacher tuning. Annotate whole plants, identify pseudolabels, and independently verify validation/test masks. Tune on validation; freeze configurations before final test evaluation.

Prioritize Weed IoU/precision/recall, Crop→weed error, and weeds lost inside protection; box mAP is diagnostic. Measure warmed-up batch=1 model and full-pipeline latency/FPS and peak memory separately per device.

Limit planning to 72 aggregate GPU-hours across devices; screen candidates before training 3–4. Preserve historical weights and identify reruns by configuration.

## Commit & Pull Request Guidelines

Use short, focused subjects matching history, e.g. `added rtdetr backend`. Describe changes, validation, limitations, and relevant issues; include visual outputs where useful. Preserve run metadata; exclude local datasets, checkpoints, and generated training artifacts.
