#!/usr/bin/env python3
"""Generate HSV/crop-exclusion weed-mask ablation predictions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


os.environ.setdefault("YOLO_CONFIG_DIR", str((Path("experiments") / ".ultralytics").resolve()))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
VARIANTS = ["hsv_only", "crop_exclusion_only", "hsv_crop_exclusion", "final_contour_filtering"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default=Path("data/mask_validation/images"), type=Path)
    parser.add_argument("--output", default=Path("experiments/results/masks"), type=Path)
    parser.add_argument("--model", type=Path, help="YOLO crop detector weights for crop-exclusion variants")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--h-min", type=int, default=35)
    parser.add_argument("--h-max", type=int, default=85)
    parser.add_argument("--s-min", type=int, default=40)
    parser.add_argument("--s-max", type=int, default=255)
    parser.add_argument("--v-min", type=int, default=40)
    parser.add_argument("--v-max", type=int, default=255)
    parser.add_argument("--min-area", type=float, default=50.0)
    parser.add_argument("--max-area", type=float, default=0.0, help="0 disables fixed maximum contour area")
    parser.add_argument("--box-padding", type=int, default=0)
    return parser.parse_args()


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        raise SystemExit(f"Missing image directory: {directory}")
    images = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"No images found in {directory}")
    return images


def load_detector(model_path: Path | None):
    if model_path is None:
        return None
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    return YOLO(str(model_path))


def hsv_mask(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([args.h_min, args.s_min, args.v_min], dtype=np.uint8)
    upper = np.array([args.h_max, args.s_max, args.v_max], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def crop_mask(detector, image_path: Path, shape: tuple[int, int], args: argparse.Namespace) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if detector is None:
        return mask
    results = detector.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        verbose=False,
    )
    height, width = shape
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for xyxy in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = [int(round(value)) for value in xyxy[:4]]
            x1 = max(0, x1 - args.box_padding)
            y1 = max(0, y1 - args.box_padding)
            x2 = min(width - 1, x2 + args.box_padding)
            y2 = min(height - 1, y2 + args.box_padding)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def contour_filter(mask: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[int, int]]]:
    filtered = np.zeros_like(mask)
    centers: list[tuple[int, int]] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < args.min_area:
            continue
        if args.max_area and area > args.max_area:
            continue
        cv2.drawContours(filtered, [contour], -1, 255, thickness=-1)
        moments = cv2.moments(contour)
        if moments["m00"]:
            centers.append((int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])))
    return filtered, centers


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)


def main() -> None:
    args = parse_args()
    images = list_images(args.images)
    detector = load_detector(args.model)
    if detector is None:
        print("No --model supplied; crop-exclusion variants will contain diagnostic empty/inverse masks.")

    for variant in VARIANTS:
        (args.output / variant).mkdir(parents=True, exist_ok=True)

    centers_dir = args.output / "centers"
    centers_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read image: {image_path}")
        vegetation = hsv_mask(image, args)
        crops = crop_mask(detector, image_path, vegetation.shape, args)
        non_crop = cv2.bitwise_not(crops)
        hsv_crop = cv2.bitwise_and(vegetation, non_crop)
        final, centers = contour_filter(hsv_crop, args)

        save_mask(args.output / "hsv_only" / f"{image_path.stem}.png", vegetation)
        save_mask(args.output / "crop_exclusion_only" / f"{image_path.stem}.png", non_crop)
        save_mask(args.output / "hsv_crop_exclusion" / f"{image_path.stem}.png", hsv_crop)
        save_mask(args.output / "final_contour_filtering" / f"{image_path.stem}.png", final)
        with (centers_dir / f"{image_path.stem}.csv").open("w", encoding="utf-8", newline="") as handle:
            handle.write("x,y\n")
            for x, y in centers:
                handle.write(f"{x},{y}\n")

    config = {
        "images": str(args.images),
        "model": str(args.model) if args.model else None,
        "device": args.device,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "hsv": {"h": [args.h_min, args.h_max], "s": [args.s_min, args.s_max], "v": [args.v_min, args.v_max]},
        "min_area": args.min_area,
        "max_area": args.max_area,
        "box_padding": args.box_padding,
        "variants": VARIANTS,
    }
    (args.output / "mask_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote masks for {len(images)} images to {args.output}")


if __name__ == "__main__":
    main()
