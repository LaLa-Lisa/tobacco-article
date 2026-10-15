"""Small, dependency-free helpers shared by the reproducibility notebooks."""
from pathlib import Path, PureWindowsPath
import csv
import json
import subprocess
import sys
import os
import platform


def repository_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / 'notebooks').is_dir() and (candidate / 'experiments/splits/items.csv').is_file():
            return candidate
    raise FileNotFoundError('Open Jupyter from the repository root or notebooks directory.')


def new_run(root, name):
    if not name or Path(name).name != name or name in {'.', '..'} or '\\' in name:
        raise ValueError('Use a single directory name for RUN_NAME.')
    output = root / 'experiments/reproductions' / name
    output.mkdir(parents=True, exist_ok=False)
    return output


def frozen_assignments(root):
    """Read historical paths portably and verify membership against items.csv."""
    source = root / 'experiments/splits'
    assignments = {}
    seen = set()
    for split in ('train', 'val', 'test'):
        names = [PureWindowsPath(line.strip()).name
                 for line in (source / f'{split}.txt').read_text().splitlines() if line.strip()]
        if len(names) != len(set(names)) or seen.intersection(names):
            raise ValueError('Duplicate image or overlap in frozen splits.')
        seen.update(names)
        assignments[split] = names
    with (source / 'items.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(seen):
        raise ValueError('items.csv does not match frozen manifests.')
    for split, names in assignments.items():
        expected = {PureWindowsPath(row['image']).name for row in rows if row['split'] == split}
        if set(names) != expected:
            raise ValueError(f'{split} assignments disagree with items.csv.')
    summary = json.loads((source / 'summary.json').read_text())
    if {key: len(value) for key, value in assignments.items()} != summary['split_counts']:
        raise ValueError('Split counts disagree with summary.json.')
    return assignments


def write_local_splits(root, dataset, output):
    """Relocate frozen manifests without resampling; validate before writing."""
    dataset = Path(dataset).expanduser().resolve()
    assignments = frozen_assignments(root)
    for names in assignments.values():
        for name in names:
            image = dataset / 'images' / name
            label = dataset / 'labels' / f'{Path(name).stem}.txt'
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(f'Required image/label pair missing: {image}, {label}')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for split, names in assignments.items():
        (output / f'{split}.txt').write_text(
            ''.join(f'{(dataset / "images" / name).as_posix()}\n' for name in names), encoding='utf-8')
    return output


def check_detector_results(path):
    """Reject recorded training/evaluation failures and missing required metrics."""
    with Path(path).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    problems = []
    for row in rows:
        name = row['model_name']
        if row.get('status') != 'ok':
            problems.append(f'{name}: {row.get("notes", "training failed")}')
        for split in ('val', 'test'):
            if row.get(f'{split}_status') == 'failed' or not row.get(f'{split}_metrics/mAP50-95(B)'):
                problems.append(f'{name} {split}: {row.get(f"{split}_error", "metrics missing")}')
    if not rows or problems:
        raise RuntimeError('Incomplete detector experiment: ' + '; '.join(problems))
    return rows


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


def capture_environment(output):
    output.mkdir(parents=True, exist_ok=True)

    (output / "nvidia-smi.txt").write_text(run_text(["nvidia-smi"]), encoding="utf-8")
    (output / "python-version.txt").write_text(sys.version + "\n", encoding="utf-8")
    (output / "pip-freeze.txt").write_text(
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
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote environment metadata to {output}")
