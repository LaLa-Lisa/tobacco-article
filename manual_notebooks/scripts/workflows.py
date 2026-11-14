"""Управление запуском из ноутбука; выбор моделей всегда берётся из common.MODELS."""
from dataclasses import dataclass, field

from . import common as cmn
from .backends import get_backend
from .contracts import DatasetSpec, ModelArtifacts, TrainConfig, TrainingRun, UnsupportedOperation


@dataclass
class TrainingBatch:
    runs: dict[str, TrainingRun] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


def enabled_specs():
    cmn.validate_registry(cmn.MODELS, cmn.MODELS_REGESTRY)
    return [cmn.MODELS_REGESTRY[name] for name, enabled in cmn.MODELS.items() if enabled]


def train_models(dataset: DatasetSpec, config: TrainConfig | None = None) -> TrainingBatch:
    """Обучить включённые модели без сохранённых обученных весов.

    Существующие weights/trained/<model-id>.pt и неподдерживаемые модели пропускаются.
    Некорректные данные и непредвиденные ошибки провайдера останавливают выполнение.
    Описания завершённых запусков остаются на диске при сбое следующей модели.
    """
    batch = TrainingBatch()
    config = config or TrainConfig()
    for spec in enabled_specs():
        trained = cmn.TRAINED_WEIGHTS_DIR / f"{spec.id}.pt"
        if trained.is_file():
            reason = f"Trained weights already exist: {trained}"
            batch.skipped[spec.id] = reason
            print(f"Skipped {spec.id}: {reason}")
            continue
        reason = spec.unavailable_reason("train")
        if reason:
            batch.skipped[spec.id] = reason
            print(f"Skipped {spec.id}: {reason}")
            continue
        try:
            backend = get_backend(spec)
            initial = backend.download(spec)
            batch.runs[spec.id] = backend.train(spec, initial, dataset, config)
            print(f"Completed {spec.id}: {batch.runs[spec.id].directory}")
        except UnsupportedOperation as exc:
            batch.skipped[spec.id] = str(exc)
            print(f"Skipped {spec.id}: {exc}")
        except Exception as exc:
            raise RuntimeError(f"Training failed for {spec.id}: {exc}") from exc
    if not batch.runs:
        print("No training runs completed. Check MODELS and the reported skip reasons.")
    return batch


def load_model(model_id: str, artifacts: ModelArtifacts, *, device="cpu"):
    """Загрузить явно выбранные веса без автоматической подмены предобученными."""
    cmn.validate_registry(cmn.MODELS, cmn.MODELS_REGESTRY)
    spec = cmn.MODELS_REGESTRY[model_id]
    return get_backend(spec).load(spec, artifacts, device)
