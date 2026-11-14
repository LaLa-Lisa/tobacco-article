"""Интеграция Ultralytics. Библиотека провайдера импортируется только при выполнении операций."""
import hashlib
import importlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
from uuid import uuid4

from .. import common as cmn
from ..contracts import ModelArtifacts, ModelSpec, Prediction, TrainingRun, UnsupportedOperation
from ..datasets import inspect_dataset


def _provider():
    os.environ["YOLO_CONFIG_DIR"] = str(cmn.YOLO_CONFIG_DIR)
    cmn.YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        module = importlib.import_module("ultralytics")
    except ImportError as exc:
        raise ImportError("Для запуска моделей установите ultralytics в Python-окружение этого ноутбука") from exc
    module.settings.update({"weights_dir": str(cmn.ORIGINAL_WEIGHTS_DIR.resolve()),
                            "datasets_dir": str(cmn.DATASETS_DIR.resolve()),
                            "runs_dir": str(cmn.RUNS_DIR.resolve())})
    return module


def _require(spec, operation):
    reason = spec.unavailable_reason(operation)
    if reason:
        raise ValueError(f"{spec.id}: {reason}")
    if Path(spec.checkpoint).name != spec.checkpoint:
        raise ValueError("В реестре указываются имена файлов весов исходного проекта, а не пути")


def _construct(spec, path):
    module = _provider()
    constructor = getattr(module, spec.constructor, None)
    if constructor is None:
        raise RuntimeError(f"В установленной версии Ultralytics нет {spec.constructor}; установите версию с поддержкой {spec.id}")
    return constructor(str(path))


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class UltralyticsBackend:
    def download(self, spec: ModelSpec) -> ModelArtifacts:
        _require(spec, "download")
        path = cmn.ORIGINAL_WEIGHTS_DIR / spec.checkpoint
        if path.is_file() and path.stat().st_size:
            return ModelArtifacts(path)
        _provider()
        cmn.ORIGINAL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        # Скачиваем во временный каталог, чтобы незавершённые файлы не считались готовыми весами.
        from tempfile import TemporaryDirectory
        downloads = importlib.import_module("ultralytics.utils.downloads")
        with TemporaryDirectory(prefix="download-", dir=cmn.ORIGINAL_WEIGHTS_DIR) as temporary:
            requested = Path(temporary) / spec.checkpoint
            downloaded = Path(downloads.attempt_download_asset(str(requested), progress=False))
            if not downloaded.is_file() or not downloaded.stat().st_size:
                raise FileNotFoundError(f"Не удалось скачать веса {spec.id}: {spec.checkpoint}. "
                                        "Если веса требуют разрешения или ручного скачивания, поместите полученный файл по пути " + str(path))
            if downloaded.resolve() != requested.resolve():
                shutil.copy2(downloaded, requested)
            requested.replace(path)
        return ModelArtifacts(path)

    def load(self, spec, artifacts, device="cpu"):
        _require(spec, "predict")
        if not artifacts.checkpoint.is_file():
            raise FileNotFoundError(artifacts.checkpoint)
        return UltralyticsPredictor(spec, artifacts, device)

    def train(self, spec, initial, dataset, config):
        _require(spec, "train")
        dataset_info = inspect_dataset(dataset, spec.task)
        if config.epochs <= 0 or config.image_size <= 0 or config.batch <= 0:
            raise ValueError("epochs, image_size и batch должны быть положительными")
        reserved = {"data", "epochs", "imgsz", "batch", "device", "seed", "project", "name", "exist_ok", "resume", "save", "model", "task"}
        if reserved.intersection(config.options):
            raise ValueError(f"Задайте {reserved.intersection(config.options)} через основные поля конфигурации, а не options")
        if not initial.checkpoint.is_file():
            raise FileNotFoundError(initial.checkpoint)
        model = _construct(spec, initial.checkpoint)
        if model.task != spec.task:
            raise ValueError(f"Задача файла весов {model.task!r} не соответствует задаче в реестре {spec.task!r}")
        if "trainer" not in model.task_map.get(model.task, {}):
            raise UnsupportedOperation(f"{spec.id}: установленная версия Ultralytics не поддерживает обучение этой модели")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"{spec.id}-{stamp}-{uuid4().hex[:8]}"
        directory = cmn.RUNS_DIR / run_name
        directory.mkdir(parents=True, exist_ok=False)
        args = dict(config.options, data=str(Path(dataset.yaml).resolve()), epochs=config.epochs,
                    imgsz=config.image_size, batch=config.batch, device=config.device, seed=config.seed,
                    project=str(cmn.RUNS_DIR.resolve()), name=run_name, exist_ok=True, save=True)
        manifest = directory / "manifest.json"
        with initial.checkpoint.open("rb") as source:
            checkpoint_hash = hashlib.file_digest(source, "sha256").hexdigest()
        record = {"status": "running", "initial_checkpoint_sha256": checkpoint_hash, "model": asdict(spec), "initial_checkpoint": initial.checkpoint,
                  "ultralytics_version": _provider().__version__, "requested_config": args,
                  "dataset": dataset_info}

        def persist():
            manifest.write_text(json.dumps(_jsonable(record), indent=2) + "\n")

        persist()
        try:
            results = model.train(**args)
            best, last = Path(model.trainer.best), Path(model.trainer.last)
            for path in (best, last):
                if not path.is_file():
                    raise FileNotFoundError(f"После обучения не найден файл {path}")
            metrics = dict(getattr(results, "results_dict", {}) or {})
            effective = getattr(model.trainer, "args", {})
            effective = vars(effective) if hasattr(effective, "__dict__") else effective
            record.update(status="complete", best_checkpoint=str(best.resolve()), last_checkpoint=str(last.resolve()),
                          effective_config=effective, metrics=metrics)
            persist()
            cmn.TRAINED_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            alias = cmn.TRAINED_WEIGHTS_DIR / f"{spec.id}.pt"
            shutil.copy2(best, alias)
            return TrainingRun(spec.id, directory, ModelArtifacts(best), ModelArtifacts(last), manifest, metrics)
        except Exception as exc:
            record.update(status="failed", error=str(exc))
            persist()
            raise


class UltralyticsPredictor:
    def __init__(self, spec, artifacts, device):
        self.spec, self.artifacts, self.device = spec, artifacts, device
        self.model = None  # Загрузка при первом предсказании; у SAM3 два отдельных интерфейса.
        self.semantic = None
        self.semantic_options = None

    def predict(self, image, *, prompts=None, **options):
        import numpy as np

        if isinstance(image, (str, Path)):
            from ..datasets import IMAGE_SUFFIXES
            if Path(image).suffix.lower() not in IMAGE_SUFFIXES or not Path(image).is_file():
                raise ValueError("predict ожидает локальный файл изображения; видео, потоки и URL не поддерживаются")
        elif isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError("Массив изображения должен иметь форму H×W×3, тип uint8 и порядок каналов BGR")
        else:
            from PIL.Image import Image
            if not isinstance(image, Image):
                raise TypeError("Передайте путь к изображению, массив NumPy в BGR или изображение PIL")
        forbidden = {"source", "stream", "device", "model", "task", "mode", "predictor", "retina_masks",
                     "text", "texts", "points", "labels", "bboxes", "visual_prompts", "refer_image"}
        if forbidden.intersection(options):
            raise ValueError("Задайте подсказки через prompts, а устройство — при загрузке; потоки и переопределение интерфейса не поддерживаются")
        prompts = dict(prompts or {})
        constructor = self.spec.constructor
        sam3 = self.spec.checkpoint == "sam3.pt"
        allowed = {"SAM": {"points", "labels", "bboxes"},
                   "FastSAM": {"points", "labels", "bboxes", "text"},
                   "YOLOWorld": {"text"}, "YOLOE": {"text", "visual_prompts", "refer_image"}}.get(constructor, set())
        if sam3:
            allowed = allowed | {"text", "exemplars"}
        unknown = prompts.keys() - allowed
        if unknown:
            raise ValueError(f"{self.spec.id} не поддерживает подсказки: {sorted(unknown)}")
        if "labels" in prompts and "points" not in prompts:
            raise ValueError("Метки labels требуют указания точек points")
        if "refer_image" in prompts and "visual_prompts" not in prompts:
            raise ValueError("Для refer_image необходимо задать visual_prompts")
        if "text" in prompts:
            texts = prompts["text"]
            texts = [texts] if isinstance(texts, str) else texts
            if not isinstance(texts, list) or not texts or not all(isinstance(t, str) and t.strip() for t in texts):
                raise ValueError("text должен быть непустой строкой или непустым списком строк")
            prompts["text"] = texts
        if constructor == "YOLOE" and "text" in prompts and "visual_prompts" in prompts:
            raise ValueError("Для каждого предсказания YOLOE выберите текстовые или визуальные подсказки")
        if sam3 and ({"text", "exemplars"} & prompts.keys()) and ({"points", "labels", "bboxes"} & prompts.keys()):
            raise ValueError("В SAM3 подсказки для поиска по понятиям нельзя объединять с интерактивными подсказками")
        kwargs = {"verbose": False, "save": False, **options}
        kwargs["device"] = self.device
        if self.spec.task in {"segment", "prompt_segment"}:
            kwargs["retina_masks"] = True
        if sam3 and ({"text", "exemplars"} & prompts.keys()):
            _provider()
            module = importlib.import_module("ultralytics.models.sam")
            cls = getattr(module, "SAM3SemanticPredictor", None)
            if cls is None:
                raise RuntimeError("В установленной версии Ultralytics нет SAM3SemanticPredictor; обновите её для поиска по понятиям в SAM3")
            overrides = dict(kwargs, model=str(self.artifacts.checkpoint), task="segment", mode="predict")
            if self.semantic is None or overrides != self.semantic_options:
                self.semantic = cls(overrides=overrides)
                self.semantic_options = overrides
            self.semantic.set_image(image)
            semantic_prompts = dict(prompts)
            if "exemplars" in semantic_prompts:
                semantic_prompts["bboxes"] = semantic_prompts.pop("exemplars")
            results = self.semantic(**semantic_prompts)
        else:
            if self.model is None:
                self.model = _construct(self.spec, self.artifacts.checkpoint)
            if constructor in {"YOLOWorld", "YOLOE"} and "text" in prompts:
                texts = prompts.pop("text")
                if constructor == "YOLOWorld":
                    self.model.set_classes(texts)
                else:
                    self.model.set_classes(texts, self.model.get_text_pe(texts))
            if constructor == "FastSAM" and "text" in prompts:
                prompts["texts"] = prompts.pop("text")
            results = self.model.predict(source=image, **prompts, **kwargs)
        return [normalize_result(r, semantic_classes=self.spec.task != "prompt_segment") for r in results]


def normalize_result(result, *, semantic_classes=True):
    import numpy as np

    def array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    shape = tuple(result.orig_shape)
    mask_data = None
    if masks is not None:
        mask_data = array(masks.data).astype(bool)
        # Нужен retina_masks=True: изменение размера масок с отступами нарушило бы координаты.
        if tuple(mask_data.shape[-2:]) != shape:
            raise ValueError("Несмотря на retina_masks=True, провайдер вернул маски не в координатах исходного изображения")
    return Prediction(shape,
                      array(boxes.xyxy).reshape(-1, 4) if boxes is not None else np.empty((0, 4)),
                      array(boxes.conf) if boxes is not None else np.empty(0),
                      array(boxes.cls).astype(int) if boxes is not None and semantic_classes else None,
                      dict(result.names) if semantic_classes and getattr(result, "names", None) is not None else None,
                      mask_data, result)
