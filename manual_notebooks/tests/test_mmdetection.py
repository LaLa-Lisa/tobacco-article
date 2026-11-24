"""Автономные проверки интеграции RTMDet-Tiny."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import common as cmn
from scripts.artifacts import ready, trained_artifacts
from scripts.backends import get_backend
from scripts.benchmark.core import BenchmarkConfig, run_benchmarks
from scripts.benchmark.worker import run_worker
from scripts.contracts import DatasetSpec, ModelArtifacts, TrainConfig, UnsupportedOperation
from scripts.mmdet_dataset import export_coco
from scripts.workflows import load_model


class MMDetectionTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        for name in ('ORIGINAL_WEIGHTS_DIR', 'TRAINED_WEIGHTS_DIR', 'RUNS_DIR'):
            self.enterContext(patch.object(cmn, name, self.root / name))
        self.spec = cmn.MODELS_REGESTRY['rtmdet-tiny']
        self.backend = get_backend(self.spec)
        self.checkpoint = self.root / 'model.pth'
        self.config = self.root / 'config.py'
        self.checkpoint.write_bytes(b'weights')
        self.config.write_text('model = {}')
        self.artifacts = ModelArtifacts(self.checkpoint, self.config)

    def dataset(self):
        (self.root / 'images').mkdir()
        (self.root / 'labels').mkdir()
        for name in ('a', 'b', 'c'):
            Image.new('RGB', (40, 20)).save(self.root / f'images/{name}.png')
        (self.root / 'labels/a.txt').write_text('1 .5 .5 .5 .2')
        (self.root / 'train.txt').write_text('./images/b.png\n./images/a.png\n')
        (self.root / 'val.txt').write_text('./images/c.png\n')
        path = self.root / 'data.yaml'
        path.write_text('names: [weed, tobacco]\ntrain: train.txt\nval: val.txt\n')
        return DatasetSpec(path)

    def test_coco_split_identity_negatives_and_pixel_units(self):
        dataset = self.dataset()
        (self.root / 'test.txt').write_text('./images/a.png\n')
        with dataset.yaml.open('a') as stream:
            stream.write('test: test.txt\n')
        record = export_coco(dataset, self.root / 'export')
        self.assertEqual(list(record['splits']), ['train', 'val', 'test'])
        split = record['splits']['train']
        self.assertEqual(len(split['lists'][0]['sha256']), 64)
        coco = json.loads(Path(split['ann_file']).read_text())
        self.assertEqual([Path(x['file_name']).name for x in coco['images']], ['b.png', 'a.png'])
        self.assertEqual([(c['id'], c['name']) for c in coco['categories']], [(1, 'weed'), (2, 'tobacco')])
        self.assertEqual(coco['annotations'], [dict(id=1, image_id=2, category_id=2, bbox=[10, 8, 20, 4],
                                                    area=80, iscrowd=0)])
        self.assertIsNone(split['members'][0]['label_sha256'])
        self.assertEqual(json.loads((self.root / 'export/val.json').read_text())['annotations'], [])

    def test_download_failure_retry_and_lazy_reuse(self):
        backend = 'scripts.backends.mmdetection'
        config = Mock()
        config.fromfile.return_value.dump.side_effect = lambda p: p.write_text('model = {}')
        with patch(f'{backend}._provider', return_value=SimpleNamespace(__file__=__file__)), patch.dict(
                sys.modules, {'mmengine.config': SimpleNamespace(Config=config)}), patch(
                f'{backend}.urlretrieve', side_effect=lambda url, p: p.write_bytes(b'weights')), patch(
                f'{backend}.digest', side_effect=['invalid', '78e30dcc1234']):
            with self.assertRaisesRegex(ValueError, 'checksum'):
                self.backend.download(self.spec)
            self.assertEqual(list(cmn.ORIGINAL_WEIGHTS_DIR.iterdir()), [])
            pair = self.backend.download(self.spec)
        with patch(f'{backend}._provider', side_effect=AssertionError):
            self.assertEqual(self.backend.download(self.spec), pair)
        self.assertTrue(ready(pair))

    def test_predictor_coordinates_channels_and_reuse(self):
        instances = Mock()
        instances.cpu.return_value.numpy.return_value = SimpleNamespace(
            bboxes=np.array([[1, 2, 8, 6]]), scores=np.array([0.5]), labels=np.array([0]))
        native = SimpleNamespace(ori_shape=(10, 20), pred_instances=instances)
        model = Mock(dataset_meta={'classes': ['tobacco']})
        model.bbox_head.num_classes = 1
        model.cfg.test_dataloader.dataset.pipeline = [SimpleNamespace(type='Resize', scale=(640, 640))]
        apis = SimpleNamespace(init_detector=Mock(return_value=model), inference_detector=Mock(return_value=native))
        with patch('scripts.backends.mmdetection._provider'), patch.dict(sys.modules, {'mmdet.apis': apis}):
            predictor = load_model(self.spec.id, self.artifacts)
            pred = predictor.predict(Image.new('RGB', (20, 10), 'red'), imgsz=640)[0]
            array = apis.inference_detector.call_args.args[1]
            self.assertEqual(array[0, 0].tolist(), [0, 0, 255])
            self.assertEqual(pred.boxes_xyxy.tolist(), [[1, 2, 8, 6]])
            self.assertEqual(pred.image_shape, (10, 20))
            self.assertEqual(pred.names, {0: 'tobacco'})
            self.assertIs(pred.native, native)
            empty = predictor.predict(array, conf=1.0)[0]
            self.assertEqual(empty.boxes_xyxy.shape, (0, 4))
            self.assertEqual(empty.class_ids.shape, (0,))
            apis.init_detector.assert_called_once()
            with self.assertRaises(UnsupportedOperation):
                predictor.predict(array, prompts={'points': []})

    def test_training_publication_and_failure(self):
        dataset = self.dataset()
        backend = 'scripts.backends.mmdetection'

        def train(args, **kwargs):
            cfg = Path(args[-2])
            best, last = cfg.parent / 'best.pth', cfg.parent / 'epoch.pth'
            best.write_bytes(b'best')
            last.write_bytes(b'last')
            (cfg.parent / 'validation.json').write_text(json.dumps(dict(
                best=str(best), last=str(last), metrics={'val/coco/bbox_mAP': 0.5})))

        with patch(f'{backend}._provider'), patch(f'{backend}.version', return_value='mock'), patch(
                f'{backend}._training_config') as config, patch(f'{backend}.subprocess.run', side_effect=train) as worker:
            config.return_value.dump.side_effect = lambda p: p.write_text('model = {}')
            run = self.backend.train(self.spec, self.artifacts, dataset, TrainConfig())
            self.assertEqual(trained_artifacts(self.spec), run.best_artifacts)
            self.assertEqual(run.last_artifacts.checkpoint.read_bytes(), b'last')
            self.assertEqual(run.metrics['val/coco/bbox_mAP'], 0.5)
            worker.side_effect = RuntimeError('interrupted')
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                self.backend.train(self.spec, self.artifacts, dataset, TrainConfig())
            self.assertEqual(trained_artifacts(self.spec), run.best_artifacts)

    def test_benchmark_worker_receives_and_verifies_run_config(self):
        self.enterContext(patch.dict(cmn.MODELS, dict.fromkeys(cmn.MODELS, False)))
        cmn.MODELS[self.spec.id] = True
        image = self.root / 'frame.png'
        Image.new('RGB', (20, 10)).save(image)

        def launch(command, **kwargs):
            request = Path(command[-1])
            payload = json.loads(request.read_text())
            self.assertEqual(payload['model_config'], str(self.config))
            devices = Mock(cuda=False, device='cpu')
            with patch('scripts.benchmark.worker.DeviceMetrics', return_value=devices), patch(
                    'scripts.benchmark.worker.write_json'), patch('scripts.workflows.load_model',
                    side_effect=RuntimeError('loading detector')) as load:
                with self.assertRaisesRegex(RuntimeError, 'loading detector'):
                    run_worker(request)
                load.assert_called_once_with(self.spec.id, self.artifacts, device='cpu')
                self.config.write_text('modified')
                with self.assertRaisesRegex(ValueError, 'configuration changed'):
                    run_worker(request)
            return Mock(returncode=0, poll=Mock(return_value=0))

        with patch('scripts.benchmark.core.subprocess.Popen', side_effect=launch):
            run_benchmarks({self.spec.id: self.artifacts}, [image], config=BenchmarkConfig(),
                           progress=False, output_root=self.root / 'benchmarks')

    def test_benchmark_requires_paired_config_and_lazy_import(self):
        self.enterContext(patch.dict(cmn.MODELS, dict.fromkeys(cmn.MODELS, False)))
        cmn.MODELS[self.spec.id] = True
        with self.assertRaisesRegex(FileNotFoundError, 'configuration'):
            run_benchmarks({self.spec.id: self.checkpoint}, [self.checkpoint])
        subprocess.run([sys.executable, '-c', "from scripts import common, weight; "
                        "from scripts.backends import get_backend; import sys; "
                        "get_backend(common.MODELS_REGESTRY['rtmdet-tiny']); weight.weight_status(); "
                        "assert not {'torch', 'mmcv', 'mmdet', 'mmengine'} & sys.modules.keys()"],
                       cwd=Path(__file__).resolve().parents[1], check=True)
