"""Автономная проверка RF-DETR без скачивания весов и обучения."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import common as cmn
from scripts.backends import get_backend
from scripts.backends.rfdetr import RFDETRBackend
from scripts.contracts import DatasetSpec, ModelArtifacts, TrainConfig, UnsupportedOperation
from scripts.rfdetr_dataset import stage_dataset


class RFDETRTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ('ORIGINAL_WEIGHTS_DIR', 'TRAINED_WEIGHTS_DIR', 'RUNS_DIR'):
            self.enterContext(patch.object(cmn, name, self.root / name))
        self.spec = cmn.MODELS_REGESTRY['rf-detr-m']
        self.backend = get_backend(self.spec)
        self.checkpoint = self.root / self.spec.checkpoint
        self.checkpoint.write_bytes(b'weights')
        self.artifacts = ModelArtifacts(self.checkpoint)

    def dataset(self, task='detect', label='0 0.5 0.5 0.2 0.2'):
        for sub in ('images/a', 'images/b', 'labels/a', 'labels/b'):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        for sub in ('a', 'b'):
            Image.new('RGB', (20, 10)).save(self.root / 'images' / sub / 'same.png')
            (self.root / 'labels' / sub / 'same.txt').write_text(label)
        (self.root / 'train.txt').write_text('./images/a/same.png\n./images/b/same.png\n')
        (self.root / 'val.txt').write_text(str(self.root / 'images/a/same.png') + '\n')
        path = self.root / 'data.yaml'
        path.write_text(yaml.safe_dump({'path': '.', 'train': 'train.txt', 'val': 'val.txt', 'names': {0: 'tobacco'}}))
        return DatasetSpec(path, task)

    def provider(self, spec=None):
        spec = spec or self.spec
        model = type(spec.constructor, (), {})()
        model.model_config = SimpleNamespace(segmentation_head=spec.task == 'segment', resolution=640)
        model.class_names = ['tobacco']
        model.predict = Mock(return_value=SimpleNamespace(xyxy=np.array([[1, 2, 6, 8]]),
                             confidence=np.array([0.8]), class_id=np.array([0]), mask=None))
        model.train = Mock()
        provider = SimpleNamespace(RFDETR=SimpleNamespace(from_checkpoint=Mock(return_value=model)))
        config = SimpleNamespace(model_fields={'num_workers': None, 'run_test': None})
        configs = SimpleNamespace(TrainConfig=config, SegmentationTrainConfig=config)
        setattr(configs, spec.constructor + 'Config', lambda: SimpleNamespace(patch_size=12 if spec.task == 'segment' else 16, num_windows=1 if spec.id == 'rf-detr-seg-n' else 2))
        self.enterContext(patch('scripts.backends.rfdetr._provider', return_value=provider))
        self.enterContext(patch('scripts.backends.rfdetr.importlib.import_module', return_value=configs))
        return model, provider

    def test_registry_and_lazy_factory(self):
        self.assertIsInstance(self.backend, RFDETRBackend)
        for suffix in 'nsml':
            for prefix in ('rf-detr-', 'rf-detr-seg-'):
                spec = cmn.MODELS_REGESTRY[prefix + suffix]
                self.assertIsNone(spec.unavailable_reason('train'))
                self.assertFalse(cmn.MODELS[spec.id])
        code = "from scripts import common, weight; from scripts.backends import get_backend; import sys; get_backend(common.MODELS_REGESTRY['rf-detr-m']); weight.weight_status(); assert not {'rfdetr', 'torch', 'ultralytics'} & sys.modules.keys()"
        subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1], check=True)

    def test_staging_preserves_splits_and_colliding_names(self):
        dataset = self.dataset()
        (self.root / 'labels/b/same.txt').unlink()
        destination = self.root / 'staged'
        record = stage_dataset(dataset, destination)
        self.assertEqual(len(record['splits']['train']['members']), 2)
        self.assertEqual(len(record['splits']['val']['members']), 1)
        self.assertNotIn('test', record['splits'])
        self.assertEqual(len(list((destination / 'train/images').iterdir())), 2)
        self.assertEqual((destination / 'train/labels/00000001.txt').read_text(), '')
        self.assertTrue(record['splits']['train']['members'][0]['image_sha256'])
        self.assertEqual(yaml.safe_load((destination / 'data.yaml').read_text())['names'], ['tobacco'])

    def test_polygon_and_malformed_labels(self):
        dataset = self.dataset('segment', '0 0.1 0.1 0.9 0.1 0.5 0.9')
        stage_dataset(dataset, self.root / 'polygons')
        for i, label in enumerate(('0 0.5 0.5 0.2 0.2', '2 0 0 1 0 1 1', '0 nan 0 1 0 1 1')):
            (self.root / 'labels/a/same.txt').write_text(label)
            with self.assertRaises(ValueError):
                stage_dataset(dataset, self.root / f'invalid{i}')

    def test_download_atomic_and_reuse(self):
        downloader = Mock(side_effect=lambda p: Path(p).write_bytes(b'complete'))
        with patch('scripts.backends.rfdetr._provider') as provider, patch('scripts.backends.rfdetr.importlib.import_module', return_value=SimpleNamespace(download_pretrain_weights=downloader)):
            artifact = self.backend.download(self.spec)
            self.assertEqual(artifact.checkpoint.read_bytes(), b'complete')
            provider.reset_mock()
            self.backend.download(self.spec)
            provider.assert_not_called()
        artifact.checkpoint.unlink()
        def fail(path):
            Path(path).write_bytes(b'partial')
            raise OSError('interrupted')
        with patch('scripts.backends.rfdetr._provider'), patch('scripts.backends.rfdetr.importlib.import_module', return_value=SimpleNamespace(download_pretrain_weights=fail)):
            with self.assertRaises(OSError):
                self.backend.download(self.spec)
        self.assertEqual(list(cmn.ORIGINAL_WEIGHTS_DIR.iterdir()), [])

    def test_training_mapping_artifacts_and_manifest(self):
        dataset = self.dataset()
        model, provider = self.provider()
        def train(**args):
            directory = Path(args['output_dir'])
            for name in ('checkpoint_best_total.pth', 'last.ckpt'):
                (directory / name).write_bytes(b'trained')
            (directory / 'training_config.json').write_text(json.dumps({'resolution': 640}))
            (directory / 'metrics.csv').write_text('epoch,val/mAP_50_95\n0,0.7\n1,0.8\n')
        model.train.side_effect = train
        config = TrainConfig(epochs=2, backend_options={'rfdetr': {'num_workers': 0}, 'ultralytics': {'plots': True}})
        run = self.backend.train(self.spec, self.artifacts, dataset, config)
        args = model.train.call_args.kwargs
        self.assertEqual(args['num_workers'], 0)
        self.assertNotIn('plots', args)
        self.assertEqual(args['resolution'], 640)
        self.assertEqual(run.metrics['val/mAP_50_95'], 0.8)
        self.assertEqual(run.last_artifacts.checkpoint.name, 'last.ckpt')
        record = json.loads(run.manifest.read_text())
        self.assertEqual(record['status'], 'complete')
        self.assertTrue(record['initial_checkpoint_sha256'])
        self.assertEqual((cmn.TRAINED_WEIGHTS_DIR / 'rf-detr-m.pt').read_bytes(), b'trained')

    def test_training_failure_and_preflight(self):
        dataset = self.dataset()
        model, _ = self.provider()
        for config in (TrainConfig(image_size=641), TrainConfig(options={'plots': True}), TrainConfig(options={'epochs': 10})):
            with self.assertRaises(ValueError):
                self.backend.train(self.spec, self.artifacts, dataset, config)
        with self.assertRaises(UnsupportedOperation):
            self.backend.train(self.spec, self.artifacts, replace(dataset, task='segment'), TrainConfig())
        model.train.side_effect = RuntimeError('training failed')
        with self.assertRaises(RuntimeError):
            self.backend.train(self.spec, self.artifacts, dataset, TrainConfig())
        manifests = list(cmn.RUNS_DIR.glob('*/manifest.json'))
        self.assertEqual(json.loads(manifests[0].read_text())['status'], 'failed')
        self.assertFalse(cmn.TRAINED_WEIGHTS_DIR.exists())

    def test_prediction_reuses_model_rgb_and_coordinates(self):
        model, provider = self.provider()
        predictor = self.backend.load(self.spec, self.artifacts)
        bgr = np.full((10, 20, 3), [1, 2, 3], dtype=np.uint8)
        result = predictor.predict(bgr, conf=0.3, imgsz=640)[0]
        np.testing.assert_array_equal(np.asarray(model.predict.call_args.args[0])[0, 0], [3, 2, 1])
        self.assertEqual(result.image_shape, (10, 20))
        self.assertEqual(result.names, {0: 'tobacco'})
        predictor.predict(Image.new('RGB', (20, 10)))
        provider.RFDETR.from_checkpoint.assert_called_once()
        with self.assertRaises(ValueError):
            predictor.predict(bgr, imgsz=384)
        with self.assertRaises(ValueError):
            predictor.predict(bgr, prompts={'text': 'leaf'})

    def test_masks_empty_and_wrong_checkpoint(self):
        spec = cmn.MODELS_REGESTRY['rf-detr-seg-m']
        model, _ = self.provider(spec)
        model.predict.return_value.mask = np.ones((1, 10, 20))
        predictor = self.backend.load(spec, self.artifacts)
        image = Image.new('RGB', (20, 10))
        self.assertEqual(predictor.predict(image)[0].masks.dtype, bool)
        model.predict.return_value.mask = np.ones((1, 5, 10))
        with self.assertRaises(ValueError):
            predictor.predict(image)
        model.predict.return_value = SimpleNamespace(xyxy=np.empty((0, 4)), confidence=np.empty(0), class_id=np.empty(0), mask=np.empty((0, 10, 20)))
        self.assertEqual(predictor.predict(image)[0].boxes_xyxy.shape, (0, 4))
        with self.assertRaises(ValueError):
            self.backend.load(self.spec, self.artifacts).predict(image)

    def test_segmentation_training_and_directory_splits(self):
        spec = cmn.MODELS_REGESTRY['rf-detr-seg-s']
        dataset = self.dataset('segment', '0 0.1 0.1 0.9 0.1 0.5 0.9')
        data = yaml.safe_load(dataset.yaml.read_text())
        data.update(train=['images/a', 'images/b'], val='images/a', test='images/b')
        dataset.yaml.write_text(yaml.safe_dump(data))
        model, _ = self.provider(spec)
        def train(**args):
            directory = Path(args['output_dir'])
            for name in ('checkpoint_best_total.pth', 'last.ckpt'):
                (directory / name).write_bytes(b'segmentation')
            (directory / 'training_config.json').write_text('{}')
        model.train.side_effect = train
        with self.assertRaises(ValueError):
            self.backend.train(spec, self.artifacts, dataset, TrainConfig(image_size=640))
        run = self.backend.train(spec, self.artifacts, dataset, TrainConfig(image_size=384))
        record = json.loads(run.manifest.read_text())
        self.assertEqual(len(record['dataset']['splits']['test']['members']), 1)
        self.assertEqual(model.train.call_args.kwargs['resolution'], 384)

    def test_missing_artifact_and_training_output(self):
        with self.assertRaises(FileNotFoundError):
            self.backend.load(self.spec, ModelArtifacts(self.root / 'missing'))
        dataset = self.dataset()
        self.provider()
        with self.assertRaises(FileNotFoundError):
            self.backend.train(self.spec, self.artifacts, dataset, TrainConfig())
        self.assertFalse(cmn.TRAINED_WEIGHTS_DIR.exists())

    def test_options_merge(self):
        config = TrainConfig(options={'num_workers': 4}, backend_options={'rfdetr': {'num_workers': 0}})
        self.assertEqual(config.options_for('rfdetr'), {'num_workers': 0})
        self.assertEqual(config.options_for('ultralytics'), {'num_workers': 4})


if __name__ == '__main__':
    unittest.main()
