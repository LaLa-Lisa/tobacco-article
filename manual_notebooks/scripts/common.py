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

MODELS = {
    "yolo26n": True
}

def inspect_weights():
    """Display checkpoint availability table."""
    import pandas as pd
    from IPython.display import display

    rows = []
    for model_name in MODELS:
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
