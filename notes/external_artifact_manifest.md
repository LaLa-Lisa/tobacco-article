# External Artifact Manifest

Large or private research artifacts must not be transferred through Git. Store
them in external storage, a release artifact, institutional storage, USB media,
or an archive shared directly between machines. Commit only this manifest,
small summaries, split manifests, and publication figures.

## Required external artifacts

| Artifact | Required | Suggested filename | Status | Notes |
|---|---:|---|---|---|
| Raw YOLO dataset | Yes | `dataset_tobacco_yolo_20260630.tar.zst` | External only | Contains tobacco images and YOLO labels. Not committed. |
| YOLO model weights | Recommended | `model_weights_yolov8_20260630.tar.zst` | External only | `.pt` files are ignored by Git. |
| Full YOLO run folders | Optional | `yolo_runs_full_20260630.tar.zst` | External only | Useful for audit, but CSV/figures are enough for manuscript. |
| Weed-mask ground truth | Needed for quantitative mask claims | `mask_validation_gt_YYYYMMDD.tar.zst` | Missing | Current experiment did not include ground-truth masks. |
| Weed-center annotations | Needed for coordinate accuracy claims | `weed_centers_gt_YYYYMMDD.tar.zst` | Missing | Current experiment did not include center ground truth. |

## Small artifacts that should be committed

| Artifact | Current status |
|---|---|
| `experiments/results/dataset_preflight.json` | Present |
| `experiments/results/experiment_notes.md` | Present |
| `experiments/results/yolo/model_comparison.csv` | Present |
| `experiments/results/yolo/run_metadata.json` | Present |
| `experiments/results/mask_metrics.csv` | Present, but records zero evaluated masks |
| `figures/publication/` | Present |
| `experiments/splits/train.txt` | Present |
| `experiments/splits/val.txt` | Present |
| `experiments/splits/test.txt` | Present |
| `experiments/splits/items.csv` | Present |
| `experiments/splits/summary.json` | Present |

## Checksum policy

For every external archive, create a checksum file:

```bash
sha256sum dataset_tobacco_yolo_20260630.tar.zst > dataset_tobacco_yolo_20260630.tar.zst.sha256
```

Record the checksum, file size, creation date, machine path, and sharing status
in this manifest before submission.

## Current detector evidence

The detector experiment used 110 images, 4436 tobacco boxes, class `0=tobacco`,
YOLOv8n/m/x, 50 epochs, image size 1024, batch size 14, and an NVIDIA GeForce
RTX 5070 Ti. The strongest detector by test mAP@0.5:0.95 was YOLOv8x.
