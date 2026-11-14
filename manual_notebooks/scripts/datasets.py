"""Проверка локального датасета YOLO и вычисление хешей для воспроизводимости."""
import hashlib
import math
from pathlib import Path

from .contracts import DatasetSpec

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def inspect_dataset(dataset: DatasetSpec, task: str) -> dict:
    import yaml

    if dataset.task != task or task not in {"detect", "segment"}:
        raise ValueError(f"Dataset task {dataset.task!r} does not match model task {task!r}")
    path = Path(dataset.yaml).resolve()
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or not data.get("names"):
        raise ValueError(f"{path}: expected YOLO YAML with class names")
    names = data["names"]
    if isinstance(names, list):
        class_ids = set(range(len(names)))
    elif isinstance(names, dict):
        class_ids = set(names)
        if class_ids != set(range(len(names))):
            raise ValueError("Dataset class IDs must be contiguous integers starting at zero")
    else:
        raise ValueError("Dataset names must be a list or integer-keyed mapping")
    root = Path(data.get("path") or path.parent)
    # Как в Ultralytics: существующий относительный путь считается от рабочего каталога;
    # иначе — от настроенного каталога датасетов.
    if not root.is_absolute():
        from . import common as cmn
        root = root.resolve() if root.exists() else (cmn.DATASETS_DIR / root).resolve()
    result = {"yaml": str(path), "yaml_sha256": hashlib.sha256(raw).hexdigest(),
              "task": task, "names": names, "splits": {}}
    for split in ("train", "val", "test"):
        value = data.get(split)
        if not value:
            if split == "test":
                continue
            raise ValueError(f"Dataset is missing {split!r}")
        sources = value if isinstance(value, list) else [value]
        images = []
        for source in sources:
            source = Path(source)
            relative_source = str(source)
            source = source if source.is_absolute() else root / source
            if not source.exists() and isinstance(value, str) and relative_source.startswith("../"):
                source = root / relative_source[3:]
            if source.is_dir():
                images.extend(sorted(p.resolve() for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES))
            elif source.is_file() and source.suffix == ".txt":
                for line in source.read_text(encoding="utf-8-sig").strip().splitlines():
                    if not line.strip():
                        continue
                    image = Path(line.strip())
                    if not image.is_absolute():
                        image = (source.parent / image) if line.startswith("./") else image
                    if image.suffix.lower() in IMAGE_SUFFIXES:
                        images.append(image.resolve())
            else:
                raise ValueError(f"Expected an image directory or split .txt file: {source}")
        images.sort()
        if not images:
            raise ValueError(f"Empty {split} split")
        digest = hashlib.sha256()
        annotation_count = 0
        for image in images:
            if not image.is_file():
                raise FileNotFoundError(image)
            parts = list(image.parts)
            if "images" not in parts:
                raise ValueError(f"YOLO image path must contain an images directory: {image}")
            index = len(parts) - 1 - parts[::-1].index("images")
            parts[index] = "labels"
            label = Path(*parts).with_suffix(".txt")
            labels = label.read_bytes() if label.is_file() else b""
            digest.update(str(image).encode() + b"\0" + labels + b"\0")
            for line in labels.decode().splitlines():
                if not line.strip():
                    continue
                fields = [float(v) for v in line.split()]
                valid_shape = len(fields) == 5 if task == "detect" else len(fields) >= 7 and len(fields) % 2 == 1
                if (not valid_shape or not all(math.isfinite(v) for v in fields)
                        or fields[0] not in class_ids or any(v < 0 or v > 1 for v in fields[1:])):
                    raise ValueError(f"Invalid {task} annotation in {label}: {line}")
                annotation_count += 1
        if split == "train" and not annotation_count:
            raise ValueError("Training split has no annotations")
        result["splits"][split] = {"images": len(images), "annotations": annotation_count,
                                    "paths_and_labels_sha256": digest.hexdigest()}
    return result
