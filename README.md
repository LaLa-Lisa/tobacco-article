# Tobacco-field experiment notebooks

A standalone reproducibility package for the tobacco detection and weed-candidate
localization study. This folder can be copied into a new repository without the
original article repository. It does not require a sibling checkout or symlinks.

## Start here

Read the [setup guide](notebooks/README.md), then open:

1. [Understand existing results](notebooks/01_understand_existing_results.ipynb):
   inspect the dataset split, detector comparison, learning curves, and saved figures.
2. [Reproduce detector experiments](notebooks/02_reproduce_detector_experiments.ipynb):
   train and evaluate YOLOv8n/m/x with the frozen split and external dataset.
3. [Masks and weed coordinates](notebooks/03_masks_and_weed_coordinates.ipynb):
   inspect the pipeline, generate predictions, and optionally evaluate manual annotations.

The first notebook retains its saved walkthrough outputs for reading immediately.
All three can run in their default modes without the raw dataset or a GPU.
Training and image processing are opt-in. Install dependencies from the provided
requirements files rather than copying a Python virtual environment.

## Included

- `notebooks/`: notebooks, setup instructions, dependency lists, portability helpers,
  and the validation command.
- `scripts/`: the six experiment scripts used by the notebooks.
- `experiments/splits/`: original frozen image assignments and dataset summary.
- `experiments/results/`: historical CSVs, configuration, environment metadata,
  detector plots, and notes. Original recorded paths describe the historical machine;
  the notebooks relocate input paths for a new machine.
- `figures/publication/`: saved qualitative examples and result figures.
- `data/README.md` and `notes/external_artifact_manifest.md`: external input requirements.

The package was copied from `tabacco_article` on 2026-10-15. Historical evidence
is retained, including recorded environment paths; no dataset download URL or
archive checksum was available in the source manifest.

## External inputs and outputs

Raw images, labels, trained `.pt` weights, and manual weed annotations are not
included. Request them from the project owner as described in the
[external artifact manifest](notes/external_artifact_manifest.md).
New experiments go under ignored `experiments/reproductions/<run-name>/`.

Only tobacco detection has quantitative historical results. The mask and centroid
examples do not establish mask accuracy, coordinate accuracy, or laser-treatment
success. The article source, PDFs, LaTeX build products, Python environments,
cache files, and generated bulk masks are not needed and are omitted.

## Verify the copy

After installing notebook dependencies, run from this folder:

```bash
python notebooks/validate_notebooks.py
```

This executes the notebooks with default settings, checks frozen split portability,
and verifies historical evidence is unchanged. OpenCV enables additional temporary
synthetic integration checks; these are software tests, not research results.

This is a plain directory prepared for importing into another repository. No new
Git history or remote is created by this export.
