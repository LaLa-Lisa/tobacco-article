"""Проверка и скачивание весов по реестру моделей для ручных ноутбуков."""
from . import common as cmn
from .backends import get_backend
from .workflows import enabled_specs



def weight_status():
    """Вернуть сведения о весах без импорта библиотек провайдеров."""
    rows = []
    for spec in enabled_specs():
        original = cmn.ORIGINAL_WEIGHTS_DIR / spec.checkpoint if spec.checkpoint else None
        trained = cmn.TRAINED_WEIGHTS_DIR / f"{spec.id}.pt"
        rows.append({"model": spec.id, "backend": spec.backend or "outside scope", "task": spec.task,
                     "status": spec.unavailable_reason("download") or "Ready",
                     "original": bool(original and original.is_file() and original.stat().st_size),
                     "trained": trained.is_file(), "checkpoint": spec.checkpoint or "variant required"})
    return rows


def inspect_weights():
    """Показать наличие весов и причины недоступности реализаций или вариантов."""
    import pandas as pd
    from IPython.display import display
    display(pd.DataFrame(weight_status(), columns=["model", "backend", "task", "status", "original", "trained", "checkpoint"]))


def download_weights(*, return_paths=False):
    """Скачать веса включённых моделей, сохранив существующие файлы и обновляя статусы.

    Весь список проверяется до начала скачивания. Словарь путей с идентификаторами
    моделей в качестве ключей возвращается только при return_paths=True.
    """
    specs = enabled_specs()
    problems = [f"{s.id}: {reason}" for s in specs if (reason := s.unavailable_reason("download"))]
    if problems:
        raise ValueError("Cannot download selected models:\n" + "\n".join(problems))
    if not specs:
        print("No models enabled in cmn.MODELS.")
        return {} if return_paths else None
    import pandas as pd
    from IPython.display import display
    rows = {s.id: {"model": s.id, "status": "Pending", "size (MiB)": None,
                   "file": s.checkpoint} for s in specs}
    paths = {}

    def table():
        return pd.DataFrame(rows.values()).set_index("model").fillna("")

    output = display(table(), display_id=True)

    def refresh():
        if output is not None:
            output.update(table())
        else:
            display(table())

    for spec in specs:
        path = cmn.ORIGINAL_WEIGHTS_DIR / spec.checkpoint
        existed = path.is_file() and path.stat().st_size > 0
        rows[spec.id]["status"] = "Already exists" if existed else "Downloading…"
        refresh()
        try:
            artifact = get_backend(spec).download(spec)
            paths[spec.id] = artifact.checkpoint
            rows[spec.id].update(status="Already exists" if existed else "Downloaded",
                                 **{"size (MiB)": round(artifact.checkpoint.stat().st_size / 2**20, 2)})
        except Exception:
            rows[spec.id]["status"] = "Failed"
            for row in rows.values():
                if row["status"] == "Pending":
                    row["status"] = "Not attempted"
            refresh()
            raise
        refresh()
    return paths if return_paths else None
