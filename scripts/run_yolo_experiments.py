#!/usr/bin/env python3
"""Run comparable Ultralytics YOLO training and validation experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path


os.environ.setdefault("YOLO_CONFIG_DIR", str((Path("experiments") / ".ultralytics").resolve()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="YOLO dataset root")
    parser.add_argument("--splits", required=True, type=Path, help="Directory with train.txt/val.txt/test.txt")
    parser.add_argument("--models", nargs="+", required=True, help="Model weights, e.g. yolov8n.pt yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=14)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--class-name", default="tobacco")
    return parser.parse_args()


def require_ultralytics() -> None:
    try:
        from ultralytics import YOLO  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise SystemExit("Ultralytics is not installed. Install project requirements before running.") from exc


def write_data_yaml(args: argparse.Namespace, output: Path) -> Path:
    yaml_path = output / "data.generated.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {args.dataset.resolve()}",
                f"train: {Path(args.splits / 'train.txt').resolve()}",
                f"val: {Path(args.splits / 'val.txt').resolve()}",
                f"test: {Path(args.splits / 'test.txt').resolve()}",
                "names:",
                f"  0: {args.class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def collect_metrics(run_dir: Path) -> dict[str, str]:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return {}
    with results_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    return {key.strip(): value for key, value in rows[-1].items()}


def flatten_metrics(prefix: str, metrics: object) -> dict[str, str]:
    output: dict[str, str] = {}
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            output[f"{prefix}_{key}".replace(" ", "_")] = str(value)
    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict):
        for key, value in speed.items():
            output[f"{prefix}_speed_{key}_ms"] = str(value)
    return output


def model_file_size_mb(path: Path) -> str:
    if not path.exists():
        return ""
    return f"{path.stat().st_size / (1024 * 1024):.3f}"


def parameter_count(detector: object) -> str:
    model = getattr(detector, "model", None)
    if model is None:
        return ""
    try:
        return str(sum(parameter.numel() for parameter in model.parameters()))
    except Exception:
        return ""


def copy_artifacts(run_dir: Path, output: Path, model_name: str) -> None:
    for filename in ["results.csv", "results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png"]:
        source = run_dir / filename
        if source.exists():
            shutil.copy2(source, output / f"{model_name}_{filename}")


def main() -> None:
    args = parse_args()
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    require_ultralytics()
    from ultralytics import YOLO

    args.output.mkdir(parents=True, exist_ok=True)
    data_yaml = write_data_yaml(args, args.output)

    comparison_rows: list[dict[str, str]] = []
    for model in args.models:
        model_name = Path(model).stem
        run_name = model_name
        start = time.time()
        run_dir = args.output / run_name
        row = {
            "model": model,
            "model_name": model_name,
            "run_dir": str(run_dir),
            "epochs": str(args.epochs),
            "imgsz": str(args.imgsz),
            "batch": str(args.batch),
            "device": str(args.device),
            "status": "started",
            "notes": "",
        }
        try:
            print(f"Training {model}", flush=True)
            detector = YOLO(model)
            row["initial_parameters"] = parameter_count(detector)
            train_metrics = detector.train(
                data=str(data_yaml),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                optimizer="SGD",
                lr0=0.005,
                momentum=0.937,
                weight_decay=0.0005,
                cos_lr=True,
                amp=True,
                plots=True,
                project=str(args.output.resolve()),
                name=run_name,
                exist_ok=True,
            )
            elapsed = time.time() - start
            row["train_seconds"] = f"{elapsed:.2f}"
            row.update(flatten_metrics("train_return", train_metrics))

            metrics = collect_metrics(run_dir)
            row.update(metrics)
            weights_dir = run_dir / "weights"
            best_weights = weights_dir / "best.pt"
            last_weights = weights_dir / "last.pt"
            row["best_weights"] = str(best_weights) if best_weights.exists() else ""
            row["best_weights_mb"] = model_file_size_mb(best_weights)
            row["last_weights"] = str(last_weights) if last_weights.exists() else ""

            eval_detector = YOLO(str(best_weights if best_weights.exists() else last_weights))
            row["trained_parameters"] = parameter_count(eval_detector)
            for split in ["val", "test"]:
                try:
                    split_metrics = eval_detector.val(
                        data=str(data_yaml),
                        split=split,
                        imgsz=args.imgsz,
                        batch=args.batch,
                        device=args.device,
                        plots=(split == "val"),
                        project=str(args.output.resolve()),
                        name=f"{run_name}_{split}",
                        exist_ok=True,
                    )
                    row.update(flatten_metrics(split, split_metrics))
                except Exception as exc:
                    row[f"{split}_status"] = "failed"
                    row[f"{split}_error"] = str(exc)
            row["status"] = "ok"
            copy_artifacts(run_dir, args.output, model_name)
        except Exception as exc:
            elapsed = time.time() - start
            row["train_seconds"] = f"{elapsed:.2f}"
            row["status"] = "failed"
            row["notes"] = str(exc)
            copy_artifacts(run_dir, args.output, model_name)
        comparison_rows.append(row)

    fieldnames = sorted({key for row in comparison_rows for key in row})
    with (args.output / "model_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    metadata = {
        "dataset": str(args.dataset),
        "splits": str(args.splits),
        "models": args.models,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "model_status": {row["model_name"]: row["status"] for row in comparison_rows},
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
