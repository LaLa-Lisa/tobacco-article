# SAM 3 tobacco and weed instances

This is a new experiment, separate from the historical YOLO reproduction. Start
with the four numbered notebooks here. Scientific functions live in notebook
cells; `runtime.py` only handles paths, provenance, and strict model loading.
Default **Restart Kernel and Run All** does not read raw data, download weights,
train, or create experiment outputs.

**Current status:** CPU workflow and synthetic tests are implemented. No model
has been fine-tuned and no trained segmentation examples are claimed. Real
training needs reviewed individual-plant annotations and your GPU. Native Windows
CUDA/Triton execution must pass notebook 3's smoke test on that machine.

## Dataset and annotation work

`original_article.pdf`, section 2.1 and Table 1, describes eight campaigns,
1920×1080 original images, and 2,520 patches of 480×352 pixels. The patch masks
are semantic classes (0 background, 1 tobacco, 2 weed), **not instance IDs**.
The article's weed-detection methodology is not reproduced here.

V2 is canonical. Campaigns 1–5 train (2,160 nominal patches), campaign 6 validates
(120), and campaigns 7–8 test (240). Repeated images are excluded from export.
Existing `experiments/splits/` belongs to a different 110-image detector
experiment and remains unchanged.

The initial local audit found 12 duplicate image pairs in campaign 5, with
different semantic mask files. It found no exact cross-split image duplicates.
V1 contains 2,520 PNG files byte-identical to V2 input patches; merging versions
would duplicate data. Some campaign-1 full-resolution masks contain values
3/4/5 as well as 0/127/255; do not silently interpret these as instances or
rescale them into classes. Notebook 1 inventories these anomalies. Approximate
appearance matching links patches to original frames without assuming filename
order. Its similarity screen is not proof that drone scenes never overlap;
inspect source scenes and record that review before approving the split.

Individual plant review can be substantial, especially for dense touching weeds.
SAM proposals and connected components are only starting points. Notebook 2
supports painting, erase, new instance/split, merge, class correction, undo,
uncertainty marking, and versioned approvals. The canvas uses source-pixel
coordinates. Use **New / split**, then paint over the pixels assigned to the new
plant; **Merge IDs** combines listed IDs into the active ID. Disconnected visible
parts of one plant may share an ID. Do not label invisible/occluded pixels.
Resolve magenta uncertainty regions with the **resolve** brush after inspecting
them, or **Mark all inspected** only after inspecting the entire image. Review
missed plants and background too, then approve the complete image.

Every validation/test image must be approved. Unreviewed training images are
excluded and listed explicitly. Duplicate images use the lowest-key canonical
copy, which must be corrected against RGB, especially when semantic guides
conflict. There is no automatic approval. Save a draft after edits to revoke a
previous approval. Exported COCO RLE annotations preserve holes and disconnected
parts, and record reviewer, source hashes, and revision provenance.

## Native Windows 11 setup (RTX 5070 Ti)

Use Python **3.12**, Git, a current NVIDIA driver, and a fresh environment.
Do not copy the Linux virtual environment or install into the old detector env.
From the copied repository root in PowerShell:

```powershell
py -3.12 -m venv .venv-sam3
.\.venv-sam3\Scripts\python.exe -m pip install --upgrade pip
.\.venv-sam3\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv-sam3\Scripts\python.exe -m pip install -r dataset_notebooks\requirements-windows.txt
.\.venv-sam3\Scripts\python.exe -m pip check
.\.venv-sam3\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'NO CUDA')"
.\.venv-sam3\Scripts\python.exe -m ipykernel install --user --name tobacco-sam3 --display-name "Tobacco SAM 3"
.\.venv-sam3\Scripts\python.exe -m jupyterlab
```

Choose **Tobacco SAM 3** in Jupyter. The dependency file pins SAM 3 to commit
`f1647e8d0c1d280991f4d1b7d974c2704537fe66`. NumPy/OpenCV/setuptools constraints
match upstream imports. Windows uses Triton 3.6 with PyTorch 2.10. The notebook
loop avoids NCCL/Slurm/multiprocessing, but upstream SAM 3 imports and mask losses
still use Triton; simply disabling compilation does not remove that dependency.
No WSL installation is required or performed.

Request access to [facebook/sam3](https://huggingface.co/facebook/sam3), then use
`.\.venv-sam3\Scripts\hf.exe auth login` locally. Never put tokens in notebooks.
Set `ALLOW_WEIGHT_DOWNLOAD=True` deliberately, or point `PRETRAINED_CHECKPOINT`
to your downloaded official `sam3.pt` and leave downloads disabled. Each enabled
GPU notebook has its own checkpoint configuration. Saved fine-tuned checkpoints
are loaded strictly using their full state dict; the upstream loader's
`detector.` prefix filtering must not be applied to these checkpoints.

Sources: [SAM 3 setup](https://github.com/facebookresearch/sam3),
[PyTorch 2.10 CUDA wheels](https://pytorch.org/get-started/previous-versions/#v2100),
[official training configuration at the pinned revision](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml),
[Windows Triton compatibility table](https://github.com/woct0rdho/triton-windows#3-pytorch).
The Windows dependency combination is a documented candidate until your smoke
test succeeds, not a locally verified environment lock.

## Run order and GPU handoff

1. **Notebook 3, smoke test only:** set checkpoint access, a fresh
   `SMOKE_RUN_NAME`, and `RUN_SMOKE_TEST=True`; keep `RUN_TRAINING=False`.
   Synthetic positive and negative queries test the real mask loss, gradients,
   optimizer, prediction, saving/reloading, time, and VRAM for four profiles.
   OOM profiles are recorded; other failures stop with a traceback. This does
   not require reviewed data. Return `smoke.json` and any traceback first.
2. **Notebook 1:** enable `RUN_AUDIT`, then inspect RGB/semantic previews,
   duplicate conflicts, and original scenes for overlap. Enter reviewer/notes
   and enable `APPROVE_SPLIT` only after this review. The locally generated audit
   is under `experiments/reproductions/sam3-audit-local-20261031-v2/`; it remains
   unapproved. Notebook defaults create a new `sam3-audit-v1/` audit instead.
   To review the existing audit, set `REVIEW_AUDIT_PATH` to its `audit.json`,
   leave `RUN_AUDIT=False`, and run the preview and approval cells.
3. **Notebook 2:** set `AUDIT_PATH`, create the annotation workspace once, then
   turn `CREATE_WORKSPACE` off. Enable proposals on the GPU (begin with a few
   `PROPOSAL_KEYS`), then turn proposals off and enable `OPEN_REVIEW`. Set
   `REVIEW_KEY` and `REVIEWER`; review each image. Export to a fresh run after
   validation/test reviews are complete and sufficient training images are ready.
4. **Notebook 3, training:** point `DATA` to the reviewed export and
   `SMOKE_REPORT` to the successful hardware report. Set `RUN_SMOKE_TEST=False`,
   `RUN_TRAINING=True`, and a fresh `RUN_NAME`. It uses microbatch 1, accumulation
   8, BF16, activation checkpointing, frozen text encoder, and memory-tested
   image-encoder profiles. AdamW, warmup/cosine decay, augmentation, and
   field-balanced sampling are explicit. Selection uses validation mask AP;
   test data are never loaded. The default sweep has up to 24 trials (four
   profiles × two learning rates × three seeds), each capped at 60 epochs with
   patience 10. **This can be long training.** Use the smoke timings to budget
   it; reduce the explicit grid for a pilot and record that deviation. Do not
   start the whole sweep merely to check installation.
5. **Notebook 4:** set `TRAINING_RUN` and enable validation. It evaluates the
   pretrained baseline, tiles, eight rotation/flip views, prompt variants,
   threshold/NMS settings, and ensembles. It freezes the best *trained* recipe
   by validation AP and reports whether it beats the pretrained baseline.
   Then turn validation off, set `SELECTION`, and enable test with a fresh name.
   Two deterministic held-out examples (one per test campaign), numeric metrics,
   per-instance masks, scores, and provenance are saved under `examples/`.
   Optional `FULL_IMAGE_PATHS` generates qualitative full-resolution examples;
   this does not create full-resolution ground truth or an accuracy claim.

A passed synthetic memory test is necessary but not sufficient for dense real
images. Real image instance counts can increase memory. Training fails visibly
on OOM, non-finite loss, changed data, or unsupported instance counts rather than
silently skipping images or changing resolution. For a training OOM, select a
shallower passed profile and a new run name. More than 200 instances per class
per image needs a separately reviewed smaller-crop dataset; this workflow stops
rather than truncating annotations. Epoch checkpoints allow a trial to resume
into a new run: set `RESUME_CHECKPOINT` to `last.pt`, retain its sibling `best.pt`,
and leave its recorded configuration unchanged. Epoch-boundary RNG and optimizer
states are restored; an interrupted partial epoch is repeated.

All output directories are fresh by default. Only in-progress annotation
revisions, training checkpoints, and run logs are updated within their owning
workspace. Exported annotations and frozen selections should be treated as
immutable; make a new export/run for corrections.

## What to transfer

Keep the repository layout. Copy the small repository checkout (including
`dataset_notebooks/`, `notebooks/reproduction.py`, and `experiments/splits/`),
`data/tobacco_original/`, and optionally `original_article.pdf` for reference.
The local audit folder contains `transfer_manifest.json` with exact relative
file paths, byte counts, and SHA-256 hashes for this handoff.
No sibling article repository is used. V1 is needed only for the audit comparison;
V2 is needed for proposals and review. When reusing a local audit, also copy its
entire ignored run directory and update `AUDIT_PATH` to the relocated path.

After review, the GPU training machine needs the complete reviewed export
(`train/`, `val/`, `test/`, `provenance.json`), the notebook source, checkpoint
access, and its smoke report. Copy annotation workspace revisions too for future
correction. Do not transfer `.venv*`, Jupyter caches, HF tokens, or unrelated
historical weights. Do not commit raw inputs or generated experiment outputs.

Return the training run's `environment.json`, `options.json`, `trials.json`,
`selection.json`, trial `history.json`/`status.json`, selected `best.pt` files,
validation `locked_selection.json`/`comparison.json`, and test `metrics.json`,
`predictions.json`, and `examples/`. Preserve directory names when relocating
checkpoints. The evaluation notebook resolves trial folders under `TRAINING_RUN`.
Prediction caches can be large and need not be transferred after evaluation.

## CPU validation

```bash
python -m venv .venv-notebooks
.venv-notebooks/bin/python -m pip install -r notebooks/requirements-analysis.txt -r dataset_notebooks/requirements-analysis.txt
.venv-notebooks/bin/python dataset_notebooks/validate_notebooks.py
.venv-notebooks/bin/python notebooks/validate_notebooks.py
```

Tests exercise notebook definitions, approval/revocation, RLE round trips,
perfect/empty COCO predictions, inverse transforms, tile coverage, duplicate
mask suppression, cross-class ownership, editor construction, and fresh-run
collisions. All four new notebooks execute in fresh kernels without SAM 3.
The original validator verifies the four named reproduction workflows and their
historical evidence. `notebooks/sam.ipynb` is a separate pre-existing exploratory
notebook and is not executed by either validator.

Interactive mouse rendering still requires checking in your browser. CPU tests
and synthetic GPU fixtures do not establish tobacco/weed segmentation accuracy.
