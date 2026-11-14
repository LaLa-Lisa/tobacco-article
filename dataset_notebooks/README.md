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

For a smaller training run, set `MAX_TRAIN_IMAGES = 200` in notebook 1. The
limit counts unique source images before dense-patch cropping; each selected
image may produce multiple training samples. Selection uses `SEED`, and validation
stays unchanged. `None` (default) uses all training images. Limits larger than
the available count use all images; the chosen limit is recorded in `config.json`.

Set `SAVE_EVERY_N_EPOCHS = 5` to retain `epoch_005.pt`, `epoch_010.pt`, etc.
Use `1` to save every epoch, or any positive integer such as `2` or `10`.
These numbered checkpoints use the same format and atomic saving as `best.pt`
and `last.pt`; point notebook 2 at any of them to compare results. `last.pt` still
updates every epoch, including a final epoch outside the chosen interval.

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

Checkpoint saves write a temporary file first and replace `best.pt`/`last.pt`
only when writing finishes. If an older checkpoint reports “failed finding
central directory”, wait for any saving/copying to finish or recopy a complete
file. If `best.pt` remains damaged, set notebook 2's `CHECKPOINT` to that run's
`last.pt`. This evaluates the last saved epoch, which may differ from the best
validation epoch. If neither file is readable, a complete backup or a new training
run is needed; the loader cannot reconstruct an incomplete archive.

CPU checks validate notebook execution, connected-region targets, and comparison
rendering. They do not establish GPU compatibility or segmentation accuracy.
No trained model or trained examples have been produced on this CPU machine.

## Migrate and relabel both dataset versions

**Correction and quality warning:** in the inspected V2 campaign 1, patch images
are numbered by numerical original-image order, but masks are numbered by
lexicographical order. Notebook 3 now remaps those masks and refuses unexpected
campaign-1 layouts. Runs created with the previous notebook, including
`tobacco-image-migration-02` and `tobacco-pseudolabels-01`, contain mispaired
campaign-1 masks and should not be used as reliable training labels. SAM 3 runs
67 and 68 were trained using the old pairs; their performance on corrected data
has not been established. A small pilot found many low-confidence overlapping
predictions, so simply lowering the detection threshold is not a validated fix.
Review representative predictions and evaluate against independently checked
labels before using any new pseudo-labels for training.

Notebook 3 can omit V1 with `INCLUDE_V1 = False`. V1 `RGB22` contains processed
rather than original RGB imagery, and its annotations need further alignment and
quality checks. The V2-only migration excludes these images and V1 original-size
images, whose binary masks do not identify weeds; it is not the entire dataset.
The correction addresses pairing, not annotation accuracy.

The additional notebooks are independent of the first two notebooks' kernel
state. Disable their execution flags when inspecting them without processing.

1. `03_migrate_dataset.ipynb` reads paired V1 and V2 images and masks, assigns
   neutral names, and writes a manifest and source masks in a fresh run under
   `experiments/reproductions/`. Exact RGB duplicates with identical weed masks
   are merged, keeping V2 first and recording every source path.
2. `04_create_labels.ipynb` reads a completed migration and local original or
   trained SAM 3 weights. It writes a new, unsplit run with `images/`, 0/1/2 PNGs in
   `semantic_masks/`, COCO RLE instances in `annotations/instances.json`, and
   tobacco-only YOLO detection boxes in `labels/` (class 0). Each image has a
   box file, including an empty file if no tobacco was detected. A train/val/test
   split must be created separately before using the data to train YOLO.
   Labeling is opt-in: set `RUN_LABELING = True` and choose a fresh `RUN_NAME`.
   For tobacco, `PROMPT_MODE = 'reference_box'` uses the same reference image and
   pixel box as notebook 6; set `REFERENCE_IMAGE` and `REFERENCE_BOX_XYWH` to your
   chosen example. Use `PROMPT_MODE = 'text'` to use `TOBACCO_PROMPT` instead.
   Weed predictions for binary masks remain text-prompted. A progress bar shows
   completed images and estimated time remaining. Configuration records the prompt
   mode and reference image hash and box, or the tobacco text prompt.

To inspect a completed labeled run, open `05_review_dataset.ipynb`, set `DATASET`
and `IMAGE_NUMBER`, enable `RUN_REVIEW`, and run the notebook. It displays the
original image, a colored semantic-mask overlay, and tobacco detection boxes.
The notebook reads existing files without changing them; rerun the configuration
and display cells to select another image.

Notebook 6 inspects SAM 3 predictions using either text or an experimental
cross-image reference-box prompt. Set `IMAGE`, `CHECKPOINT`, and `PROMPT_MODE`,
then enable `RUN_INFERENCE`. The reference defaults to the local ignored file
`data/sam3_reference/image_000001.png` and its `(300, 165, 175, 180)` pixel
box in `(x, y, width, height)` format. If that external image is unavailable,
copy the original example there or choose another image and its matching box.
The reference and target can be different images. The adapter transfers prompt
embeddings through SAM 3 internal methods, not a supported cross-image API;
results and compatibility require inspection and are not accuracy evidence.
Text mode does not require the reference image.

Three-class patch masks supply weed pixels; the 60 original-sized V1 masks are
binary and cannot identify weeds, so SAM 3 predicts weeds for those images.
SAM 3 predicts tobacco everywhere. Tobacco takes precedence on overlapping
pixels; the final masks determine the detection boxes. Connected weed regions
from the original masks are not verified plant instances. The model and source
annotations can both be inaccurate; the generated labels are pseudo-labels,
not independent ground truth or proof of segmentation performance.

In the inspected inputs, some images have identical RGB pixels but different
original weed masks. Choose `CONFLICT_POLICY` in notebook 3:

- `error` (default): stop on the first conflict without completing the run.
- `keep_separate`: retain each distinct image/weed-label variant.
- `keep_first`: retain one variant per identical RGB image, automatically
  choosing the first source in deterministic order: V2 before V1, then the
  order in which the notebook discovers directories and filenames. This does not
  determine which weed mask is more accurate. The manifest records discarded
  source paths and hashes, and the summary counts discarded sources.

Use a fresh run name after any failed attempt. The manifest identifies matching RGB
hashes, and the summary counts conflict groups. Group identical RGB hashes
together when later splitting the data to avoid leakage; related patches,
alternate crops, and full images can also depict the same scene. The default
`error` policy never writes a completed summary for a conflict.

Configure `SOURCE`, `RUN_NAME`, and `RUN_MIGRATION` in notebook 3; then configure
`MIGRATION`, `CHECKPOINT`, `RUN_NAME`, `CONFIDENCE`, and `RUN_LABELING` in
notebook 4. Checkpoint weights are local and are never downloaded. Labeling
requires the SAM 3 environment, a CUDA GPU with BF16 support, and local original
or trained weights. Original weights use resolution 1008 and record no training epoch;
trained checkpoints retain their recorded resolution and epoch. Failed
or interrupted runs lack a `summary.json` with `status: complete`; use a new run
name to retry, and never treat a partial run as a completed dataset.

With the SAM 3 environment active, run the synthetic software checks:

```powershell
.\.venv-sam3-simple\Scripts\python.exe dataset_notebooks_simple\validate_dataset_notebooks.py
```

The validator executes the migration, labeling, review, and inspection notebooks in fresh
kernels with their flags disabled in memory, without changing saved settings. It
checks migration, overlap resolution, COCO, YOLO, fake-model labeling, review,
and reference-prompt routing with a fake model.
It does not train, use real images, or evaluate model accuracy.
