#!/usr/bin/env python3
"""Evaluate binary predicted masks against binary ground-truth masks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--predicted", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=int, default=127)
    return parser.parse_args()


def read_mask(path: Path, threshold: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read mask: {path}")
    return image > threshold


def find_masks(directory: Path) -> dict[str, Path]:
    masks = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in MASK_EXTENSIONS:
            masks[path.stem] = path
    return masks


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if gt.shape != pred.shape:
        raise ValueError(f"Mask shape mismatch: {gt.shape} vs {pred.shape}")
    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    tn = int(np.logical_and(~gt, ~pred).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "dice": dice,
    }


def main() -> None:
    args = parse_args()
    gt_masks = find_masks(args.ground_truth)
    pred_masks = find_masks(args.predicted)
    if not gt_masks:
        raise SystemExit(f"No ground-truth masks found in {args.ground_truth}")
    if not pred_masks:
        raise SystemExit(f"No predicted masks found in {args.predicted}")

    rows = []
    for stem, gt_path in gt_masks.items():
        pred_path = pred_masks.get(stem)
        if pred_path is None:
            rows.append({"image": stem, "status": "missing_prediction"})
            continue
        metrics = compute_metrics(read_mask(gt_path, args.threshold), read_mask(pred_path, args.threshold))
        row = {"image": stem, "status": "ok"}
        row.update({key: f"{value:.6f}" if isinstance(value, float) else value for key, value in metrics.items()})
        rows.append(row)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    if ok_rows:
        mean_row = {"image": "__mean__", "status": "ok"}
        for key in ["iou", "precision", "recall", "dice"]:
            mean_row[key] = f"{sum(float(row[key]) for row in ok_rows) / len(ok_rows):.6f}"
        rows.append(mean_row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "status", "tp", "fp", "fn", "tn", "iou", "precision", "recall", "dice"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
