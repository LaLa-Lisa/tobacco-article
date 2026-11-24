"""Адаптер RTMDet-Tiny для MMDetection 3.3.0."""
from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
from uuid import uuid4

from .. import common as cmn
from ..artifacts import ready, resolve_artifacts
from ..contracts import ModelArtifacts, Prediction, TrainingRun, UnsupportedOperation
from ..mmdet_dataset import export_coco
from ..yolo_dataset import digest


def _provider():
    import mmdet
    if mmdet.__version__ != '3.3.0':
        raise RuntimeError(f'Expected mmdet==3.3.0, found {mmdet.__version__}')
    return mmdet


def _require(spec, operation):
    if spec != cmn.MODELS_REGESTRY['rtmdet-tiny'] or spec.unavailable_reason(operation):
        raise UnsupportedOperation(f'{spec.id}: only RTMDet-Tiny detection is supported')


def _pair(spec, artifacts):
    pair = resolve_artifacts(spec, artifacts)
    if not ready(pair):
        raise FileNotFoundError(f'Expected nonempty checkpoint and config: {pair}')
    return pair


def _training_config(initial, record, directory, config, workers):
    from mmengine.config import Config
    cfg = Config.fromfile(initial.config)
    size = (config.image_size, config.image_size)
    cfg.model.backbone.init_cfg = None
    cfg.model.bbox_head.num_classes = len(record['names'])
    cfg.load_from = str(initial.checkpoint.resolve())
    cfg.work_dir = str(directory)
    cfg.randomness = dict(seed=config.seed)
    cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=config.epochs, val_interval=1)
    # Короткий пилот использует постоянный шаг и пайплайн без смешивания изображений.
    cfg.param_scheduler = []
    cfg.custom_hooks = [hook for hook in cfg.custom_hooks if hook.type != 'PipelineSwitchHook']
    cfg.default_hooks.checkpoint.update(interval=1, save_best='coco/bbox_mAP', rule='greater')
    cfg.train_pipeline = [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True),
                          dict(type='Resize', scale=size, keep_ratio=True), dict(type='RandomFlip', prob=0.5),
                          dict(type='Pad', size=size, pad_val=dict(img=(114, 114, 114))),
                          dict(type='PackDetInputs')]
    for transform in cfg.test_pipeline:
        if transform.type == 'Resize':
            transform.scale = size
        if transform.type == 'Pad':
            transform.size = size
    for split in ('train', 'val', 'test'):
        # Для вывода SDK ожидает test_dataloader даже при отсутствии тестовой выборки.
        info = record['splits'].get(split, record['splits']['val'])
        loader = cfg[f'{split}_dataloader']
        loader.update(batch_size=config.batch if split == 'train' else 1,
                      num_workers=workers, persistent_workers=workers > 0, pin_memory=config.device != 'cpu')
        loader.dataset = dict(type='CocoDataset', ann_file=info['ann_file'], data_prefix=dict(img=''),
                              metainfo=dict(classes=tuple(record['names'])), test_mode=split != 'train',
                              filter_cfg=None, pipeline=cfg.train_pipeline if split == 'train' else cfg.test_pipeline)
        if split != 'train':
            cfg[f'{split}_evaluator'].ann_file = info['ann_file']
    return cfg


class MMDetectionBackend:
    def download(self, spec):
        _require(spec, 'download')
        checkpoint = cmn.ORIGINAL_WEIGHTS_DIR / spec.checkpoint
        pair = resolve_artifacts(spec, checkpoint)
        if ready(pair):
            return pair
        mmdet = _provider()
        from mmengine.config import Config
        upstream = Path(mmdet.__file__).parent / '.mim' / 'configs' / 'rtmdet' / f'{spec.constructor}.py'
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=checkpoint.parent) as tmp:
            cfg = Path(tmp) / pair.config.name
            Config.fromfile(upstream).dump(cfg)
            if not checkpoint.is_file() or not checkpoint.stat().st_size:
                staged = Path(tmp) / checkpoint.name
                url = f'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/{spec.constructor}/{checkpoint.name}'
                urlretrieve(url, staged)
                if not digest(staged).startswith(checkpoint.stem.rsplit('-', 1)[1]):
                    raise ValueError('Downloaded checkpoint checksum mismatch')
                staged.replace(checkpoint)
            cfg.replace(pair.config)
        return pair

    def load(self, spec, artifacts, device):
        _require(spec, 'predict')
        pair = _pair(spec, artifacts)
        _provider()
        from mmdet.apis import init_detector
        model = init_detector(pair.config, str(pair.checkpoint), device=device, palette='random')
        if len(model.dataset_meta['classes']) != model.bbox_head.num_classes:
            raise ValueError('Checkpoint classes do not match the model configuration')
        return MMDetectionPredictor(model)

    def train(self, spec, initial, dataset, config):
        _require(spec, 'train')
        if dataset.task != 'detect':
            raise UnsupportedOperation('RTMDet-Tiny requires a detection dataset')
        initial = _pair(spec, initial)
        options = config.options_for(spec.backend)
        workers = options.get('num_workers', 0)
        if options.keys() - {'num_workers'} or type(workers) is not int or workers < 0:
            raise ValueError('MMDetection supports only num_workers (a nonnegative integer)')
        if min(config.batch, config.epochs, config.image_size) <= 0 or config.image_size % 32:
            raise ValueError('Use positive batch and epochs and an image size divisible by 32')
        if config.device != 'cpu' and not (config.device.startswith('cuda:') and config.device[5:].isdigit()):
            raise ValueError('Use cpu or cuda:N')
        _provider()
        directory = (cmn.RUNS_DIR / f'{spec.id}-{uuid4().hex}').resolve()
        directory.mkdir(parents=True)
        manifest = directory / 'manifest.json'
        cfg_path = directory / 'config.py'
        metadata = dict(model_id=spec.id, backend=spec.backend, requested=asdict(config),
                        packages={p: version(p) for p in ('mmdet', 'mmengine', 'mmcv', 'torch')},
                        initial_checkpoint=str(initial.checkpoint), initial_checkpoint_sha256=digest(initial.checkpoint),
                        initial_config_sha256=digest(initial.config), config=str(cfg_path))
        try:
            record = export_coco(dataset, directory / 'dataset')
            metadata['dataset'] = record
            _training_config(initial, record, directory, config, workers).dump(cfg_path)
            environment = os.environ.copy()
            if config.device == 'cpu':
                environment['CUDA_VISIBLE_DEVICES'] = ''
            subprocess.run([sys.executable, '-m', 'scripts.backends.mmdetection', str(cfg_path), config.device],
                           cwd=Path(__file__).resolve().parents[2], env=environment, check=True)
            result = json.loads((directory / 'validation.json').read_text())
            best = _pair(spec, ModelArtifacts(Path(result['best']), cfg_path))
            last = _pair(spec, ModelArtifacts(Path(result['last']), cfg_path))
            metadata.update(status='complete', **result)
            manifest.write_text(json.dumps(metadata, indent=2))
            # Один указатель публикует согласованную пару после успешного запуска.
            cmn.TRAINED_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=cmn.TRAINED_WEIGHTS_DIR) as tmp:
                pointer = Path(tmp) / f'{spec.id}.json'
                pointer.write_text(json.dumps(dict(checkpoint=str(best.checkpoint), config=str(cfg_path))))
                pointer.replace(cmn.TRAINED_WEIGHTS_DIR / pointer.name)
        except Exception as exc:
            metadata.update(status='failed', error=str(exc))
            manifest.write_text(json.dumps(metadata, indent=2))
            raise
        return TrainingRun(spec.id, directory, best, last, manifest, result['metrics'])


class MMDetectionPredictor:
    def __init__(self, model):
        self.model = model
        pipeline = model.cfg.test_dataloader.dataset.pipeline
        self.scale = tuple(next(t.scale for t in pipeline if t.type == 'Resize'))

    def benchmark_metadata(self):
        return dict(configured_input_size=list(self.scale), actual_tensor_shape=None)

    def predict(self, image, *, prompts=None, conf=0.0, imgsz=None, **options):
        import numpy as np
        from PIL import Image
        from mmdet.apis import inference_detector
        if prompts or options:
            raise UnsupportedOperation('RTMDet-Tiny supports conf and imgsz only')
        if imgsz is not None and self.scale != (imgsz, imgsz):
            raise ValueError(f'imgsz must match the config scale {self.scale}')
        if not 0 <= conf <= 1:
            raise ValueError('conf must be between zero and one')
        if isinstance(image, (str, Path)):
            if not Path(image).is_file():
                raise FileNotFoundError(image)
            image = str(image)
        elif isinstance(image, Image.Image):
            image = np.asarray(image.convert('RGB'))[..., ::-1].copy()
        elif not (isinstance(image, np.ndarray) and image.dtype == np.uint8
                  and image.ndim == 3 and image.shape[2] == 3):
            raise ValueError('Expected an image path, PIL image or uint8 HxWx3 BGR array')
        native = inference_detector(self.model, image)
        instances = native.pred_instances.cpu().numpy()
        keep = instances.scores >= conf
        return [Prediction(tuple(native.ori_shape[:2]), instances.bboxes[keep], instances.scores[keep],
                           instances.labels[keep], dict(enumerate(self.model.dataset_meta['classes'])), None, native)]


def _run_training(config_path, device):
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    if device != 'cpu':
        torch.cuda.set_device(device)
    cfg = Config.fromfile(config_path)
    runner = Runner.from_cfg(cfg)
    runner.train()
    hub = runner.message_hub
    metrics = {key: float(history.current()) for key, history in hub.log_scalars.items() if key.startswith('val/')}
    result = dict(best=hub.get_info('best_ckpt'), last=hub.get_info('last_ckpt'), metrics=metrics)
    (Path(cfg.work_dir) / 'validation.json').write_text(json.dumps(result))


if __name__ == '__main__':
    _run_training(sys.argv[1], sys.argv[2])
