#!/usr/bin/env python3
"""Capture software and hardware metadata for the experiment record."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


os.environ.setdefault("YOLO_CONFIG_DIR", str((Path("experiments") / ".ultralytics").resolve()))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=Path("experiments/results/environment"), type=Path)
    return parser.parse_args()


def run_text(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        return f"COMMAND_NOT_FOUND: {exc}\n"
    text = completed.stdout
    if completed.stderr:
        text += ("\nSTDERR:\n" + completed.stderr)
    if completed.returncode:
        text += f"\nRETURN_CODE: {completed.returncode}\n"
    return text


def module_version(name: str) -> str | None:
    try:
        if name == "ultralytics":
            Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        module = __import__(name)
    except Exception:
        return None
    return getattr(module, "__version__", "unknown")


def torch_info() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    cuda_available = torch.cuda.is_available()
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    (args.output / "nvidia-smi.txt").write_text(run_text(["nvidia-smi"]), encoding="utf-8")
    (args.output / "python-version.txt").write_text(sys.version + "\n", encoding="utf-8")
    (args.output / "pip-freeze.txt").write_text(
        run_text([sys.executable, "-m", "pip", "freeze"]),
        encoding="utf-8",
    )
    metadata = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch": torch_info(),
        "ultralytics_version": module_version("ultralytics"),
        "opencv_version": module_version("cv2"),
        "numpy_version": module_version("numpy"),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote environment metadata to {args.output}")


if __name__ == "__main__":
    main()
