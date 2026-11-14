"""Адаптер RF-DETR 1.12.1 с отложенным импортом SDK."""
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from uuid import uuid4

from .. import common as cmn
from ..contracts import ModelArtifacts, Prediction, TrainingRun, UnsupportedOperation
from ..rfdetr_dataset import digest, stage_dataset


def _provider():
    try:
        module = importlib.import_module('rfdetr')
        installed = version('rfdetr')
    except ImportError as exc:
        raise ImportError('Install rfdetr[train]==1.12.1 in the notebook environment') from exc
    if installed != '1.12.1':
        raise RuntimeError(f'Expected rfdetr==1.12.1, found {installed}')
    return module


def _require(spec, operation):
    reason = spec.unavailable_reason(operation)
    if reason:
        raise UnsupportedOperation(f'{spec.id}: {reason}')
    if spec.backend != 'rfdetr' or spec.task not in {'detect', 'segment'} or not spec.constructor:
        raise ValueError('Specify a concrete RF-DETR constructor and task')
    if Path(spec.checkpoint).name != spec.checkpoint:
        raise ValueError('Checkpoint must be an upstream filename')


def _check_model(spec, model):
    if type(model).__name__ != spec.constructor:
        raise ValueError(f'Checkpoint variant does not match {spec.constructor}')
    segment = bool(model.model_config.segmentation_head)
    if segment != (spec.task == 'segment'):
        raise ValueError('Checkpoint task does not match registry task')


class RFDETRBackend:
    def download(self, spec):
        _require(spec, 'download')
        path = cmn.ORIGINAL_WEIGHTS_DIR / spec.checkpoint
        if path.is_file() and path.stat().st_size:
            return ModelArtifacts(path)
        _provider()
        downloader = importlib.import_module('rfdetr.assets.model_weights').download_pretrain_weights
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='download-', dir=path.parent) as temporary:
            staged = Path(temporary) / path.name
            downloader(str(staged))
            if not staged.is_file() or not staged.stat().st_size:
                raise FileNotFoundError(f'Download produced no checkpoint: {path.name}')
            staged.replace(path)
        return ModelArtifacts(path)

    def load(self, spec, artifacts, device='cpu'):
        _require(spec, 'predict')
        if not artifacts.checkpoint.is_file():
            raise FileNotFoundError(artifacts.checkpoint)
        return RFDETRPredictor(spec, artifacts, device)

    def train(self, spec, initial, dataset, config):
        _require(spec, 'train')
        if dataset.task != spec.task:
            raise UnsupportedOperation(f'{spec.id}: dataset task {dataset.task} does not match {spec.task}')
        if min(config.epochs, config.batch, config.image_size) <= 0:
            raise ValueError('epochs, batch and image_size must be positive')
        if not initial.checkpoint.is_file():
            raise FileNotFoundError(initial.checkpoint)
        module = _provider()
        configs = importlib.import_module('rfdetr.config')
        model_config = getattr(configs, spec.constructor + 'Config')()
        divisor = model_config.patch_size * model_config.num_windows
        if config.image_size % divisor:
            raise ValueError(f'{spec.id}: image_size must be divisible by {divisor}')
        options = config.options_for('rfdetr')
        reserved = {'dataset_dir', 'dataset_file', 'output_dir', 'epochs', 'batch_size', 'resolution',
                    'device', 'seed', 'resume', 'pretrain_weights', 'class_names', 'accelerator',
                    'devices', 'num_nodes', 'callbacks', 'model', 'trainer', 'logger'}
        train_cls = configs.SegmentationTrainConfig if spec.task == 'segment' else configs.TrainConfig
        invalid = (options.keys() - train_cls.model_fields.keys()) | (options.keys() & reserved)
        if invalid:
            raise ValueError(f'Unsupported or reserved RF-DETR options: {sorted(invalid)}')
        directory = cmn.RUNS_DIR / f'{spec.id}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}'
        directory.mkdir(parents=True)
        manifest = directory / 'manifest.json'
        args = dict(options, dataset_dir=str((directory / 'dataset').resolve()), dataset_file='yolo',
                    output_dir=str(directory.resolve()), epochs=config.epochs, batch_size=config.batch,
                    resolution=config.image_size, device=config.device, seed=config.seed)
        record = {'status': 'running', 'model': asdict(spec), 'rfdetr_version': '1.12.1',
                  'initial_checkpoint': str(initial.checkpoint.resolve()),
                  'initial_checkpoint_sha256': digest(initial.checkpoint), 'requested_config': args}

        def persist():
            manifest.write_text(json.dumps(record, indent=2, default=lambda x: sorted(x) if isinstance(x, (set, frozenset)) else str(x)) + '\n')

        persist()
        try:
            record['dataset'] = stage_dataset(dataset, directory / 'dataset')
            persist()
            model = module.RFDETR.from_checkpoint(str(initial.checkpoint), device=config.device)
            _check_model(spec, model)
            model.train(**args)
            best, last = directory / 'checkpoint_best_total.pth', directory / 'last.ckpt'
            for path in (best, last):
                if not path.is_file() or not path.stat().st_size:
                    raise FileNotFoundError(f'Training did not produce {path}')
            effective = json.loads((directory / 'training_config.json').read_text())
            metrics = {}
            for path in sorted(directory.rglob('metrics.csv')):
                with path.open(newline='') as source:
                    for row in csv.DictReader(source):
                        for key, value in row.items():
                            if value:
                                try:
                                    metrics[key] = float(value)
                                except ValueError:
                                    pass
            record.update(status='complete', effective_config=effective, metrics=metrics,
                          best_checkpoint=str(best.resolve()), last_checkpoint=str(last.resolve()))
            persist()
            cmn.TRAINED_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            alias = cmn.TRAINED_WEIGHTS_DIR / f'{spec.id}.pt'
            with TemporaryDirectory(dir=alias.parent) as temporary:
                staged = Path(temporary) / alias.name
                shutil.copy2(best, staged)
                staged.replace(alias)
            return TrainingRun(spec.id, directory, ModelArtifacts(best), ModelArtifacts(last), manifest, metrics)
        except Exception as exc:
            record.update(status='failed', error=str(exc))
            persist()
            raise


class RFDETRPredictor:
    def __init__(self, spec, artifacts, device):
        self.spec, self.artifacts, self.device = spec, artifacts, device
        self.model = None

    def predict(self, image, *, prompts=None, **options):
        import numpy as np
        from PIL import Image
        if prompts:
            raise ValueError('RF-DETR does not support prompts')
        unknown = options.keys() - {'conf', 'threshold', 'imgsz', 'antialias'}
        if unknown or ('conf' in options and 'threshold' in options):
            raise ValueError(f'Unsupported or conflicting prediction options: {sorted(options)}')
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                rgb = source.convert('RGB')
        elif isinstance(image, Image.Image):
            rgb = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError('Array images must be H×W×3 uint8 in BGR order')
            rgb = Image.fromarray(image[..., ::-1])
        else:
            raise TypeError('Use a local image path, PIL image or BGR numpy array')
        if self.model is None:
            model = _provider().RFDETR.from_checkpoint(str(self.artifacts.checkpoint), device=self.device)
            _check_model(self.spec, model)
            self.model = model
        size = options.pop('imgsz', self.model.model_config.resolution)
        if size != self.model.model_config.resolution:
            raise ValueError(f'imgsz must match loaded resolution {self.model.model_config.resolution}')
        threshold = options.pop('conf', options.pop('threshold', 0.25))
        if not 0 <= threshold <= 1:
            raise ValueError('Confidence threshold must be between zero and one')
        result = self.model.predict(rgb, threshold=threshold, **options)
        shape = (rgb.height, rgb.width)
        masks = None if result.mask is None else np.asarray(result.mask, dtype=bool)
        if masks is not None and masks.shape != (len(result.xyxy), *shape):
            raise ValueError('Provider masks must use original-image coordinates')
        names = dict(enumerate(self.model.class_names))
        return [Prediction(shape, np.asarray(result.xyxy).reshape(-1, 4),
                           np.asarray(result.confidence), np.asarray(result.class_id, dtype=int),
                           names, masks, result)]
