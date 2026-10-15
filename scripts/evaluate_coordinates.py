#!/usr/bin/env python3
"""Evaluate predicted weed-center coordinates against manual center annotations."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", default=Path("data/mask_validation/centers"), type=Path)
    parser.add_argument("--predicted", default=Path("experiments/results/masks/centers"), type=Path)
    parser.add_argument("--output", default=Path("experiments/results/coordinate_metrics.csv"), type=Path)
    parser.add_argument("--not-available-output", default=Path("experiments/results/coordinate_metrics_not_available.md"), type=Path)
    return parser.parse_args()


def read_points(path: Path) -> list[tuple[float, float]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [(float(row["x"]), float(row["y"])) for row in reader if row.get("x") and row.get("y")]


def nearest_distances(predicted: list[tuple[float, float]], truth: list[tuple[float, float]]) -> list[float]:
    distances: list[float] = []
    for gt_x, gt_y in truth:
        if not predicted:
            distances.append(math.inf)
            continue
        distances.append(min(math.hypot(gt_x - pred_x, gt_y - pred_y) for pred_x, pred_y in predicted))
    return distances


def hit_rate(distances: list[float], threshold: float) -> float:
    if not distances:
        return 1.0
    return sum(distance <= threshold for distance in distances) / len(distances)


def main() -> None:
    args = parse_args()
    if not args.ground_truth.exists() or not list(args.ground_truth.glob("*.csv")):
        args.not_available_output.parent.mkdir(parents=True, exist_ok=True)
        args.not_available_output.write_text(
            "Coordinate accuracy was not evaluated because manually annotated weed-center\n"
            "points were not available. The article must restrict quantitative claims to\n"
            "mask quality.\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.not_available_output}")
        return
    if not args.predicted.exists():
        raise SystemExit(f"Missing predicted coordinate directory: {args.predicted}")

    rows: list[dict[str, str]] = []
    for gt_path in sorted(args.ground_truth.glob("*.csv")):
        truth = read_points(gt_path)
        predicted = read_points(args.predicted / gt_path.name)
        distances = nearest_distances(predicted, truth)
        finite = [distance for distance in distances if math.isfinite(distance)]
        mean_error = sum(finite) / len(finite) if finite else math.inf
        rows.append(
            {
                "image": gt_path.stem,
                "predicted_points": str(len(predicted)),
                "ground_truth_points": str(len(truth)),
                "mean_nearest_error_px": f"{mean_error:.6f}" if math.isfinite(mean_error) else "inf",
                "hit_rate_10px": f"{hit_rate(distances, 10):.6f}",
                "hit_rate_20px": f"{hit_rate(distances, 20):.6f}",
                "hit_rate_50px": f"{hit_rate(distances, 50):.6f}",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "predicted_points",
                "ground_truth_points",
                "mean_nearest_error_px",
                "hit_rate_10px",
                "hit_rate_20px",
                "hit_rate_50px",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
