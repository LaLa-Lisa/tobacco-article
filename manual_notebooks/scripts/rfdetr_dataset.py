"""Подготовка YOLO-каталога без изменения исходных данных и разбиений."""
import hashlib
import math
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def stage_dataset(dataset, destination):
    import yaml
    source = Path(dataset.yaml).resolve()
    data = yaml.safe_load(source.read_text())
    names = data["names"]
    if isinstance(names, dict):
        indexed = {int(k): v for k, v in names.items()}
        if len(indexed) != len(names) or sorted(indexed) != list(range(len(indexed))):
            raise ValueError("Class IDs must be contiguous from zero")
        names = [indexed[k] for k in range(len(indexed))]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names):
        raise ValueError("Dataset names must be a nonempty list of class names")
    base = Path(data.get("path", source.parent))
    if not base.is_absolute():
        base = source.parent / base
    base = base.resolve()
    record = {"yaml": str(source), "sha256": digest(source), "task": dataset.task,
              "names": names, "splits": {}}
    output = {"names": names, "nc": len(names)}
    for split, folder in [("train", "train"), ("val", "valid"), ("test", "test")]:
        value = data.get(split)
        if not value:
            if split != "test":
                raise ValueError(f"Missing required split: {split}")
            continue
        images, lists = [], []
        for entry in value if isinstance(value, list) else [value]:
            path = Path(entry)
            path = path if path.is_absolute() else base / path
            if path.is_dir():
                images.extend(sorted(p.resolve() for p in path.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES))
            elif path.is_file() and path.suffix.lower() == '.txt':
                lists.append({"path": str(path.resolve()), "sha256": digest(path)})
                for line in path.read_text().splitlines():
                    line = line.strip()
                    if line:
                        p = Path(line)
                        # Совместимость с Ultralytics: ./ относительно списка, остальные — cwd.
                        images.append((path.parent / line if line.startswith('./') else p).resolve())
            else:
                raise ValueError(f"Invalid split path: {path}")
        if not images:
            raise ValueError(f"Empty split: {split}")
        image_dir, label_dir = destination / folder / 'images', destination / folder / 'labels'
        image_dir.mkdir(parents=True)
        label_dir.mkdir()
        members = []
        for index, image in enumerate(images):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Invalid image: {image}")
            parts = list(image.parts)
            if 'images' not in parts:
                raise ValueError(f"Cannot resolve YOLO labels without an images directory: {image}")
            position = len(parts) - 1 - parts[::-1].index('images')
            parts[position] = 'labels'
            label = Path(*parts).with_suffix('.txt')
            if label.exists():
                for line in label.read_text().splitlines():
                    if not line.strip():
                        continue
                    values = [float(v) for v in line.split()]
                    if not all(math.isfinite(v) for v in values):
                        raise ValueError(f"Non-finite annotation: {label}")
                    cls, coords = values[0], values[1:]
                    if cls != int(cls) or not 0 <= cls < len(names) or any(v < 0 or v > 1 for v in coords):
                        raise ValueError(f"Invalid class or coordinates: {label}")
                    if dataset.task == 'detect':
                        valid = len(coords) == 4 and coords[2] > 0 and coords[3] > 0
                    else:
                        valid = len(coords) >= 6 and len(coords) % 2 == 0
                    if not valid:
                        raise ValueError(f"Invalid {dataset.task} annotation: {label}")
            stem = f'{index:08d}'
            (image_dir / (stem + image.suffix.lower())).symlink_to(image)
            target = label_dir / (stem + '.txt')
            if label.exists():
                target.symlink_to(label)
            else:
                target.write_text('')
            members.append({"image": str(image), "image_sha256": digest(image),
                            "label": str(label), "label_sha256": digest(label) if label.exists() else None})
        record['splits'][split] = {"lists": lists, "members": members}
        output[split] = f'{folder}/images'
    (destination / 'data.yaml').write_text(yaml.safe_dump(output, allow_unicode=True, sort_keys=False))
    return record
