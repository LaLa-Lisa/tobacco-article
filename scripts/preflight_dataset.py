#!/usr/bin/env python3
"""Validate the tobacco YOLO dataset before freezing splits or training."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="YOLO dataset root")
    parser.add_argument("--output", default=Path("experiments/results/dataset_preflight.json"), type=Path)
    parser.add_argument("--expected-class-zero", default="tobacco")
    parser.add_argument("--allow-missing-labels", action="store_true")
    parser.add_argument("--group-regex", default=None, help="Optional video-frame grouping regex to audit leakage risk")
    return parser.parse_args()


def image_paths(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def parse_label(path: Path) -> tuple[list[int], list[str]]:
    class_ids: list[int] = []
    errors: list[str] = []
    if not path.exists():
        return class_ids, errors
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_number}: expected 5 columns, found {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric YOLO value")
            continue
        if class_id < 0:
            errors.append(f"{path}:{line_number}: negative class id {class_id}")
        for value in values:
            if value < 0 or value > 1:
                errors.append(f"{path}:{line_number}: normalized coordinate outside [0, 1]: {value}")
        class_ids.append(class_id)
    return class_ids, errors


def infer_prefixes(stems: list[str]) -> dict[str, int]:
    patterns = [
        re.compile(r"^(.*?)(?:[_-]?frame[_-]?\d+)$", re.IGNORECASE),
        re.compile(r"^(.*?)(?:[_-]?\d{3,})$"),
    ]
    counts: Counter[str] = Counter()
    for stem in stems:
        for pattern in patterns:
            match = pattern.match(stem)
            if match and match.group(1):
                counts[match.group(1).rstrip("_-")] += 1
                break
    return dict(counts.most_common(20))


def main() -> int:
    args = parse_args()
    image_dir = args.dataset / "images"
    label_dir = args.dataset / "labels"
    failures: list[str] = []

    if not image_dir.exists():
        failures.append(f"missing image directory: {image_dir}")
    if not label_dir.exists():
        failures.append(f"missing label directory: {label_dir}")
    if failures:
        report = {
            "dataset": str(args.dataset),
            "expected_class_zero": args.expected_class_zero,
            "images": 0,
            "labels": 0,
            "boxes": 0,
            "class_counts": {},
            "missing_labels": [],
            "missing_labels_count": 0,
            "extra_label_files": [],
            "extra_label_files_count": 0,
            "malformed_rows": [],
            "malformed_rows_count": 0,
            "likely_frame_prefixes": {},
            "group_regex": args.group_regex,
            "group_regex_matches": {},
            "status": "fail",
            "failures": failures,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("\n".join(failures))
        return 1

    images = image_paths(image_dir)
    label_files = sorted(label_dir.glob("*.txt"))
    label_stems = {path.stem for path in label_files}
    image_stems = [path.stem for path in images]
    missing_labels: list[str] = []
    extra_labels = sorted(label_stems - set(image_stems))
    class_counts: Counter[int] = Counter()
    total_boxes = 0
    malformed: list[str] = []

    for image in images:
        label_path = label_dir / f"{image.stem}.txt"
        if not label_path.exists():
            missing_labels.append(image.stem)
            continue
        class_ids, errors = parse_label(label_path)
        malformed.extend(errors)
        class_counts.update(class_ids)
        total_boxes += len(class_ids)

    if not images:
        failures.append(f"no images found in {image_dir}")
    if missing_labels and not args.allow_missing_labels:
        failures.append(f"{len(missing_labels)} images are missing label files")
    if malformed:
        failures.append(f"{len(malformed)} malformed label rows")
    if total_boxes == 0:
        failures.append("no bounding boxes found")
    if 0 not in class_counts and total_boxes > 0:
        failures.append("class id 0 is absent, but class 0 is expected to be tobacco/crop")

    group_matches: Counter[str] = Counter()
    if args.group_regex:
        regex = re.compile(args.group_regex)
        for stem in image_stems:
            match = regex.search(stem)
            if match:
                group_matches[match.group(1)] += 1

    report = {
        "dataset": str(args.dataset),
        "expected_class_zero": args.expected_class_zero,
        "images": len(images),
        "labels": len(label_files),
        "boxes": total_boxes,
        "class_counts": dict(sorted(class_counts.items())),
        "missing_labels": missing_labels[:100],
        "missing_labels_count": len(missing_labels),
        "extra_label_files": extra_labels[:100],
        "extra_label_files_count": len(extra_labels),
        "malformed_rows": malformed[:100],
        "malformed_rows_count": len(malformed),
        "likely_frame_prefixes": infer_prefixes(image_stems),
        "group_regex": args.group_regex,
        "group_regex_matches": dict(group_matches.most_common(20)),
        "status": "fail" if failures else "ok",
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
