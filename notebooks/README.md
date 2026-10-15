# Reproduce the tobacco-field experiments

Start with [01_understand_existing_results.ipynb](01_understand_existing_results.ipynb).
It explains the study and regenerates comparison tables and learning curves from
saved CSV files. It needs no raw images, weights, or GPU.

Continue with [02_reproduce_detector_experiments.ipynb](02_reproduce_detector_experiments.ipynb)
to train YOLOv8n/m/x on the frozen split, then
[03_masks_and_weed_coordinates.ipynb](03_masks_and_weed_coordinates.ipynb)
to inspect and run the mask/centroid pipeline. All notebooks use accessible
English explanations and default to running without external experiment inputs.

## Set up Jupyter

Use a dedicated Python 3.11+ environment. From the repository root:

```bash
python -m venv .venv-notebooks
```

Activate it on Linux/macOS:

```bash
source .venv-notebooks/bin/activate
```

Or in Windows PowerShell:

```powershell
.venv-notebooks\Scripts\Activate.ps1
```

Then install the lightweight walkthrough dependencies and register the kernel:

```bash
python -m pip install -r notebooks/requirements-analysis.txt
python -m ipykernel install --user --name tobacco-notebooks --display-name "Tobacco experiments"
python -m jupyterlab
```

Choose **Tobacco experiments** as the notebook kernel. Open notebooks in order
and use **Restart Kernel and Run All**. Launch Jupyter from this repository;
root discovery works from its root or a subdirectory. The notebooks have no
hidden dependency on executing an earlier notebook in the same kernel.

For the tested Windows / RTX 5070 Ti setup, run inside the activated venv.
Install requirements first, then restore the working CUDA build (the default
installation may select CPU-only PyTorch):

```bash
python -m pip install -r notebooks/requirements-experiments.txt
python -m pip uninstall -y torch torchvision
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu132
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected: `2.12.1+cu132 13.2 True`. Restart the Jupyter kernel afterward.

The experiment requirements request the recorded Ultralytics version, 8.4.81.
The other package ranges are setup constraints, not an exact environment lock.
See [historical environment](../experiments/results/environment/metadata.json)
and [package inventory](../experiments/results/environment/pip-freeze.txt) for
original provenance. Those files describe the original Windows machine and
should not be installed blindly on other platforms. If the historical detector
version is unavailable in your package index, obtain it from the experiment
owner or document any replacement version as a deviation. GPU-specific PyTorch
installation is deliberately separate from the requirements files.

## Supply external inputs

| Task | Required inputs |
| --- | --- |
| Read saved results | Only this checkout and analysis dependencies |
| Retrain detectors | Original 110 images and YOLO tobacco labels; pretrained initial weights can download on first use |
| Generate masks/centers | Images and a trained single-class tobacco checkpoint |
| Measure mask quality | Manual weed masks with matching stems, size, and values 0/255 |
| Measure center error | Manual CSVs with `x,y` pixel coordinates and matching stems |

The original dataset and trained weights are absent from this checkout. Request
them from the project owner; [the external artifact manifest](../notes/external_artifact_manifest.md)
does not yet provide a download URL or checksum. Ground-truth weed masks and
centers were not available for the historical experiment. No synthetic data is
presented as article evidence.

Notebook 2 verifies counts and frozen membership and relocates the Windows paths
without resampling. This is not a byte-level verification of the original images;
obtain archive checksums when available. Class ID checks cannot verify semantics:
the user must confirm that class 0 means tobacco.

Set the configuration cells before enabling `RUN_TRAINING` or `RUN_PROCESSING`.
The mask notebook can read the new `best.pt` produced by the detector notebook.
It can also use externally supplied historical weights without retraining.

## Outputs and reproducibility limits

New outputs go to `experiments/reproductions/<RUN_NAME>/`, which is ignored by
Git. Each enabled experiment requires a fresh run name. To retry, change the
name and restart the notebook; it will not silently overwrite a previous run.
Default execution does not train, download weights, or write experiment results.
Notebook outputs are intentionally cleared in the source files; execute them to
see fresh tables and plots. Existing publication figures are loaded locally.

The historical settings are 50 epochs, size 1024, batch 14, SGD with learning
rate 0.005, momentum 0.937, weight decay 0.0005, cosine scheduling, and AMP.
The split seed is 20260627; it is not evidence of a training seed. The saved
artifacts do not include all resolved framework defaults or image-content
checksums. Results may differ with software, randomness, and hardware. For an
out-of-memory failure, reduce batch size consistently for all detectors and
record that deviation. CPU timing must be labeled as CPU timing.

Only the detector has quantitative historical evidence. The mask notebook
explains the current evaluators' empty-input and nearest-neighbor conventions;
coordinate metrics do not measure false targets, actuation, or treatment success.

## Validation for maintainers

With the analysis environment installed:

```bash
python notebooks/validate_notebooks.py
```

This validates notebook format, runs all three with their default settings in
fresh kernels, checks manifest relocation and output collision handling, and
verifies that historical results and split files remain unchanged. With OpenCV
installed it additionally checks mask and coordinate integration using temporary
synthetic fixtures. These fixtures test software only, never scientific results.
