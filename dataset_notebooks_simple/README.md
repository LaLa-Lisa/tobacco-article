# Train SAM 3 and compare results

Two notebooks, using your existing V2 images/masks and local original `sam3.pt`:

1. **01_train_sam3.ipynb** — set `DATA`, `CHECKPOINT`, and `RUN_NAME`, then enable
   `RUN_TRAINING` and run all cells. One run saves `best.pt`, `last.pt`, training
   history, configuration, and environment under
   `experiments/reproductions/sam3-simple-training-01/`.
2. **02_review_results.ipynb** — point `CHECKPOINT` at `best.pt`, enable
   `RUN_INFERENCE`, and run all cells. Displays and saves six comparisons:
   original image, dataset labels, and trained SAM 3. Change `EXAMPLE_IMAGES`
   or `CONFIDENCE` to inspect other results; use a new `RUN_NAME` when saving again.

No prior audit, annotation editor, exported dataset, or smoke report is needed.
Neither notebook depends on `dataset_notebooks/`. Launch Jupyter inside this
repository; the only shared helper is the existing fresh-run directory utility.
Defaults do not train, download weights, read raw data, or write results.

## Windows GPU setup

Use Python 3.12, Git, and an NVIDIA driver supporting CUDA 12.8. From the repository
root in PowerShell, create a separate environment (do not copy the CPU environment):

```powershell
py -3.12 -m venv .venv-sam3-simple
.\.venv-sam3-simple\Scripts\python.exe -m pip install --upgrade pip
.\.venv-sam3-simple\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv-sam3-simple\Scripts\python.exe -m pip install -r dataset_notebooks_simple\requirements.txt
.\.venv-sam3-simple\Scripts\python.exe -m pip check
.\.venv-sam3-simple\Scripts\python.exe -m ipykernel install --user --name sam3-simple --display-name "SAM 3 simple"
.\.venv-sam3-simple\Scripts\python.exe -m jupyterlab
```

Select **SAM 3 simple** as the kernel. Copy this repository, the V2 dataset at its
existing relative path, and your `sam3.pt` to the GPU machine. The checkpoint path
can point anywhere on that machine. No Hugging Face login or weight download is
needed. The pinned upstream SAM 3 training code still requires Triton, even with
compilation disabled. Native Windows GPU execution has not been verified on this
CPU machine; report any installation or CUDA traceback.

## What the training does

Masks use 0 background, 1 tobacco, and 2 weed. Each 8-connected region of one
class is a training instance, without dropping small regions. Touching plants
can remain grouped; disconnected parts can become separate instances. These
are automatically derived targets, not individual-plant ground truth. The 656
campaign-2 patches exceeding 200 regions in a class are automatically split along
the longest side until each crop fits SAM 3's query limit. All pixels are retained;
only these generated crops are written inside the run. Validation and examples
use whole patches. The generated training JSON is local to that run and is
rebuilt when starting training on another machine.

Campaigns 1–5 train, campaign 6 selects the best checkpoint by mean tobacco/weed
IoU, and campaigns 7–8 supply held-out examples. Exact duplicate training images
keep their first copy in campaign/numeric order (12 excluded in the inspected V2;
some duplicates have different labels). No historical detector split is changed.

The single recipe trains heads and the last four vision blocks for 60 epochs,
with a frozen text encoder, resolution 1008, batch size 1, accumulation 8, BF16,
activation checkpointing, AdamW, warmup/cosine scheduling, and flip/color
augmentation. SAM 3 uses classification, box, focal-mask, and Dice losses.
Validation uses confidence 0.5; overlapping predicted masks assign pixels to the
highest-confidence mask. There is no hyperparameter sweep or inference ensemble.

Training may take a long time on the GPU. Epoch progress and timings appear in
the notebook. The checkpoints contain model weights and training settings;
`last.pt` is usable for inference but does not contain optimizer-resume state.
Use a fresh run name for a new attempt. Outputs are ignored by Git.

CPU checks validate notebook execution, connected-region targets, and comparison
rendering. They do not establish GPU compatibility or segmentation accuracy.
No trained model or trained examples have been produced on this CPU machine.
