from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = ROOT / "data"
DATASET_PATH = DATASETS_DIR / "tobacco_yolo"
DATA_YAML = DATASET_PATH / "data.generated.yaml"

WEIGHTS_DIR = ROOT / "weights"
ORIGINAL_WEIGHTS_DIR = WEIGHTS_DIR / "original"
TRAINED_WEIGHTS_DIR = WEIGHTS_DIR / "trained"

RUNS_DIR = ROOT / "experiments" / "reproductions"
YOLO_CONFIG_DIR = ROOT / "experiments" / ".ultralytics"

# Catalog from research/sota_models/report.md (sections 4–9).
# Keys are local identifiers, not a list of Ultralytics-compatible checkpoints.
# Family entries need a concrete variant before training. inspect_weights checks
# the local convention <identifier>.pt; upstream filenames/formats may differ.
MODELS = {False
    # YOLO families, including the older baselines discussed in the report.
    "yolov5": False,
    "yolov6": False,  # v3.0
    "yolov7": False,
    "damo-yolo": False,
    "yolov8": False,
    "yolov9": False,
    "yolov10": False,
    "yolo11": False,
    "yolov12": False,
    "yolov13": False,
    "yolo26": False,
    "yolo-world": False,
    "yoloe": False,

    # Explicit detection variants named in the comparisons and recommendations.
    "yolov8n": True,
    "yolov8s": False,
    "yolov12n": False,
    "yolo26n": True,
    "yolo26s": False,
    "yolo26m": False,

    # Alternative detectors.
    "rtmdet": False,
    "rtmdet-x": False,
    "rt-detr": False,
    "rt-detrv2": False,
    "rt-detrv2-s": False,
    "d-fine": False,
    "deim": False,
    "deimv2": False,
    "rt-detrv4": False,
    "rf-detr": False,
    "rf-detr-m": False,
    "faster-rcnn": False,

    # Instance and semantic segmentation.
    "yolov8-seg": False,
    "yolov8n-seg": False,
    "yolo11-seg": False,
    "yolo11n-seg": False,
    "yolo11m-seg": False,
    "yolo26-seg": False,
    "rtmdet-ins": False,
    "rf-detr-seg": False,
    "pidnet": False,
    "pidnet-s": False,
    "efficientvit": False,

    # Promptable image and video segmentation.
    "sam": False,
    "fastsam": False,
    "mobilesam": False,
    "efficientvit-sam": False,
    "sam2": False,
    "sam2-hiera-b-plus": False,
    "sam2-hiera-l": False,
    "sam3": False,

    # Models from the agricultural studies (section 8).
    "tia-yolov5": False,
    "yoloc-tiny": False,
    # "lightweight-multispectral-crop-weed-segmentation": False,
    "weedsense": False,
}

def inspect_weights():
    """Display checkpoint availability table."""
    import pandas as pd
    from IPython.display import display

    rows = []
    for model_name, model_enabled in MODELS.items():
        if not model_enabled:
            continue

        filename = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
        rows.append({
            "model": model_name,
            "original": (ORIGINAL_WEIGHTS_DIR / filename).is_file(),
            "trained": (TRAINED_WEIGHTS_DIR / filename).is_file(),
        })

    table = pd.DataFrame(
        rows,
        columns=["model", "original", "trained"],
    )
    display(table)

def download_weights(*, return_paths=False):
    """Download enabled checkpoints and display a live availability table.

    Existing files are kept. Set return_paths=True to also return a mapping
    of model names to paths. Enabled keys must name concrete checkpoints.
    """
    import os
    import shutil

    import pandas as pd
    from IPython.display import display

    paths = {}
    for model_name, enabled in MODELS.items():
        if not enabled:
            continue
        filename = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
        paths[model_name] = ORIGINAL_WEIGHTS_DIR / filename

    if not paths:
        print("No models enabled in MODELS.")
        return {} if return_paths else None

    missing = {name: path for name, path in paths.items() if not path.is_file()}
    rows = {
        name: {
            "model": name,
            "status": "Pending" if name in missing else "Already exists",
            "size (MiB)": None if name in missing else round(path.stat().st_size / 2**20, 2),
            "file": path.name,
        }
        for name, path in paths.items()
    }

    def table():
        return pd.DataFrame(rows.values()).set_index("model").fillna("")

    output = display(table(), display_id=True)

    def refresh():
        if output is not None:
            output.update(table())
        else:
            display(table())

    if missing:
        # Set this before importing Ultralytics, as in first.ipynb.
        os.environ["YOLO_CONFIG_DIR"] = str(YOLO_CONFIG_DIR)
        YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        ORIGINAL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

        from ultralytics import settings
        from ultralytics.utils.downloads import attempt_download_asset

        settings.update({"weights_dir": str(ORIGINAL_WEIGHTS_DIR.resolve())})
        for model_name, path in missing.items():
            rows[model_name]["status"] = "Downloading…"
            refresh()
            try:
                downloaded = Path(attempt_download_asset(str(path), progress=False))
                if not downloaded.is_file():
                    raise FileNotFoundError(
                        f"No weights downloaded for {model_name!r}. "
                        "Use a concrete Ultralytics checkpoint name in MODELS."
                    )
                # Ultralytics may normalize a legacy checkpoint name.
                if downloaded.resolve() != path.resolve():
                    shutil.copy2(downloaded, path)
                rows[model_name]["size (MiB)"] = round(path.stat().st_size / 2**20, 2)
                rows[model_name]["status"] = "Downloaded"
            except Exception:
                rows[model_name]["status"] = "Failed"
                for row in rows.values():
                    if row["status"] == "Pending":
                        row["status"] = "Not attempted"
                refresh()
                raise
            refresh()

    return paths if return_paths else None
