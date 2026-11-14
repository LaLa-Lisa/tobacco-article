"""Структуры данных, независимые от провайдера. Импорт не загружает библиотеки машинного обучения."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class UnsupportedOperation(NotImplementedError):
    """Операция недоступна; ошибка определяется до начала обучения."""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    backend: str | None
    task: str
    constructor: str | None = None
    checkpoint: str | None = None
    operations: frozenset[str] = frozenset()
    note: str = ""

    def unavailable_reason(self, operation: str) -> str | None:
        if self.backend != "ultralytics":
            return self.note or f"Бэкенд {self.backend!r} пока не реализован"
        if not self.checkpoint:
            return "Укажите конкретный вариант и файл весов в MODELS_REGESTRY"
        if operation not in self.operations:
            return f"Операция {operation} не поддерживается этой интеграцией"
        return None


@dataclass(frozen=True)
class ModelArtifacts:
    checkpoint: Path
    config: Path | None = None


@dataclass(frozen=True)
class DatasetSpec:
    yaml: Path
    task: str = "detect"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    image_size: int = 640
    batch: int = 1
    device: str = "cpu"
    seed: int = 0
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingRun:
    model_id: str
    directory: Path
    best_artifacts: ModelArtifacts
    last_artifacts: ModelArtifacts
    manifest: Path
    metrics: dict[str, Any]


@dataclass
class Prediction:
    """Массивы NumPy на CPU: рамки xyxy и булевы маски в координатах исходного изображения.

    Для сегментации по подсказкам без семантических классов class_ids/names равны None.
    native хранит исходный результат провайдера, включая средства визуализации.
    """
    image_shape: tuple[int, int]
    boxes_xyxy: Any
    scores: Any
    class_ids: Any
    names: dict[int, str] | None
    masks: Any
    native: Any


class Predictor(Protocol):
    def predict(self, image: Any, *, prompts: dict | None = None, **options) -> list[Prediction]: ...


class Backend(Protocol):
    def download(self, spec: ModelSpec) -> ModelArtifacts: ...
    def train(self, spec: ModelSpec, initial: ModelArtifacts,
              dataset: DatasetSpec, config: TrainConfig) -> TrainingRun: ...
    def load(self, spec: ModelSpec, artifacts: ModelArtifacts, device: str) -> Predictor: ...
