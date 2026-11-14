"""Weight inspection and downloads for the manual notebooks."""

from pathlib import Path

from . import common as cmn


def inspect_weights():
    """Display checkpoint availability table."""
    import pandas as pd
    from IPython.display import display

    rows = []
    for model_name, enabled in cmn.MODELS.items():
        if not enabled:
            continue

        filename = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
        rows.append({
            "model": model_name,
            "original": (cmn.ORIGINAL_WEIGHTS_DIR / filename).is_file(),
            "trained": (cmn.TRAINED_WEIGHTS_DIR / filename).is_file(),
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
    for model_name, enabled in cmn.MODELS.items():
        if not enabled:
            continue
        filename = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
        paths[model_name] = cmn.ORIGINAL_WEIGHTS_DIR / filename

    if not paths:
        print("No models enabled in cmn.MODELS.")
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
        os.environ["YOLO_CONFIG_DIR"] = str(cmn.YOLO_CONFIG_DIR)
        cmn.YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        cmn.ORIGINAL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

        from ultralytics import settings
        from ultralytics.utils.downloads import attempt_download_asset

        settings.update({"weights_dir": str(cmn.ORIGINAL_WEIGHTS_DIR.resolve())})
        for model_name, path in missing.items():
            rows[model_name]["status"] = "Downloading…"
            refresh()
            try:
                downloaded = Path(attempt_download_asset(str(path), progress=False))
                if not downloaded.is_file():
                    raise FileNotFoundError(
                        f"No weights downloaded for {model_name!r}. "
                        "Use a concrete Ultralytics checkpoint name in cmn.MODELS."
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
