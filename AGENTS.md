# Repository guidance

## Purpose and layout

This repository supports an article on tobacco detection with YOLOv8n/m/x
and an exploratory weed-candidate mask and centroid pipeline. It is a standalone
reproducibility package; do not introduce dependencies on a sibling article checkout.

- `notebooks/01_understand_existing_results.ipynb`: inspect historical results.
- `notebooks/02_reproduce_detector_experiments.ipynb`: reproduce detector runs.
- `notebooks/03_masks_and_weed_coordinates.ipynb`: mask and coordinate generation.
- `notebooks/04_evaluate_masks_and_coordinates.ipynb`: independent manual-ground-truth evaluation.
- `notebooks/reproduction.py`: shared path, split, run, provenance, and result checks.
- Scientific algorithms live in notebook cells; the former experiment scripts
  are retired. Do not reintroduce wrappers or parallel implementations.
- `experiments/splits/`: frozen dataset membership and summary.
- `experiments/results/`: historical metrics, configurations, and provenance.
- `figures/publication/`: saved figures and qualitative examples.
- `data/README.md` and `notes/external_artifact_manifest.md`: external input details.

Read the root and notebook READMEs before changing experiment workflows.

## Setup and validation

Use Python 3.11+ in a dedicated virtual environment. From the repository root:

```bash
python -m venv .venv-notebooks
source .venv-notebooks/bin/activate
python -m pip install -r notebooks/requirements-analysis.txt
python notebooks/validate_notebooks.py
```

For training or image processing, use `notebooks/requirements-experiments.txt`
and a PyTorch build appropriate to the actual platform and GPU. The recorded
Ultralytics version is 8.4.81; document any replacement as an experimental deviation.
Do not blindly apply historical Windows/CUDA installation commands on other systems.

Run the notebook validator after changes to notebook code, shared helpers, or
scientific definitions. It executes default notebooks in fresh kernels, checks split
portability and run collisions, and verifies historical evidence remains unchanged.
OpenCV enables additional temporary synthetic integration checks. Report skipped
checks or missing dependencies explicitly. Documentation-only edits need review,
not model training.

## Implementation conventions

Follow the surrounding Python style; prefer `pathlib` and existing shared helpers.
Keep notebooks independently executable, with clear English explanations and
configuration cells. Default execution must work without raw data or a GPU and
must not train, download weights, or write experiment results. Keep training and
processing and evaluation opt-in through the existing configuration flags.
Use explicit function parameters and return values for scientific functions. Keep
definition, configuration, and execution cell tags compatible with the validator.
Notebooks exchange artifacts, not imports or kernel state; only tests may extract
notebook definitions. Keep infrastructure helpers free of scientific algorithms.

Preserve cross-platform path handling, including relocation of historical Windows
paths. Reuse frozen split membership rather than resampling during reproduction.
Check detector result statuses and required evaluation metrics: a successful
runner exit alone does not establish a successful experiment.

## Experiment integrity and outputs

- Preserve historical results and split files. Never manually edit metric CSVs
  to change reported performance or rewrite recorded paths in historical evidence.
- Write new experiments under ignored `experiments/reproductions/<run-name>/`.
  Require a fresh run directory and retain collision checks.
- Keep raw images, labels, trained `.pt` weights, environments, caches, and bulk
  generated outputs out of commits. Respect `.gitignore`.
- The original 110-image dataset, trained weights, and manual weed annotations
  are external inputs. Do not invent download links, checksums, or annotations.
- Record configuration, environment, and deviations for new experiments. The
  split seed 20260627 is not evidence of a training seed. Label CPU timing as such.
- Only tobacco detection has quantitative historical evidence. Qualitative masks,
  centroids, and synthetic software fixtures do not establish mask accuracy,
  coordinate accuracy, or laser-treatment success.
- Preserve existing saved notebook outputs unless the task calls for regenerating
  or clearing them; avoid incidental notebook and publication-figure churn.
