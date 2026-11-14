"""Автономные тесты интерфейса адаптера. Запускаются через unittest без весов и GPU."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import common as cmn, weight, workflows
from scripts.backends import get_backend
from scripts.backends.ultralytics import UltralyticsBackend, normalize_result
from scripts.contracts import DatasetSpec, ModelArtifacts, TrainConfig, UnsupportedOperation
from scripts.datasets import inspect_dataset
from scripts.registry import validate_registry


class ModelsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name, folder in [("ORIGINAL_WEIGHTS_DIR", "original"), ("TRAINED_WEIGHTS_DIR", "trained"),
                             ("RUNS_DIR", "runs"), ("YOLO_CONFIG_DIR", "config")]:
            self.enterContext(patch.object(cmn, name, self.root / folder))
        self.spec = cmn.MODELS_REGESTRY["yolov8n"]
        self.backend = UltralyticsBackend()
        self.weights = self.root / "model.pt"
        self.weights.write_bytes(b"test checkpoint")
        self.artifacts = ModelArtifacts(self.weights)

    def dataset(self, label="0 0.5 0.5 0.2 0.2", task="detect"):
        (self.root / "images").mkdir(exist_ok=True)
        (self.root / "labels").mkdir(exist_ok=True)
        (self.root / "images" / "a.jpg").write_bytes(b"fixture")
        (self.root / "labels" / "a.txt").write_text(label)
        path = self.root / "data.yaml"
        path.write_text(f"path: {self.root}\ntrain: images\nval: images\nnames: [tobacco]\n")
        return DatasetSpec(path, task)

    def selection(self, *names):
        self.enterContext(patch.object(cmn, "MODELS", {k: k in names for k in cmn.MODELS}))

    def test_catalog_coverage_and_mapping(self):
        validate_registry(cmn.MODELS)
        self.assertEqual(len(cmn.MODELS), 53)
        self.assertEqual(cmn.MODELS_REGESTRY["yolov12n"].checkpoint, "yolo12n.pt")
        self.assertEqual(cmn.MODELS_REGESTRY["sam2-hiera-b-plus"].checkpoint, "sam2_b.pt")
        self.assertIsNotNone(cmn.MODELS_REGESTRY["yolov8"].unavailable_reason("download"))
        with self.assertRaises(ValueError):
            validate_registry({"absent": True})

    def test_catalog_and_inspection_are_lazy(self):
        code = "import sys; from scripts import common, weight; weight.weight_status(); assert 'ultralytics' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], check=True)

    def test_unimplemented_factory(self):
        with self.assertRaises(NotImplementedError):
            get_backend(cmn.MODELS_REGESTRY["rtmdet"])

    def test_inspection_explains_unavailable_selection(self):
        self.selection("yolov8", "rtmdet", "pidnet")
        rows = weight.weight_status()
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["status"] != "Ready" for row in rows))

    def test_download_reuses_file_without_provider(self):
        cmn.ORIGINAL_WEIGHTS_DIR.mkdir()
        expected = cmn.ORIGINAL_WEIGHTS_DIR / "yolov8n.pt"
        expected.write_bytes(b"existing")
        with patch("scripts.backends.ultralytics._provider", side_effect=AssertionError("must not import")):
            self.assertEqual(self.backend.download(self.spec).checkpoint, expected)

    def test_download_stages_and_maps_upstream_name(self):
        spec = cmn.MODELS_REGESTRY["yolov12n"]
        def download(path, **kwargs):
            path = Path(path)
            self.assertEqual(path.name, "yolo12n.pt")
            path.write_bytes(b"downloaded")
            return str(path)
        module = SimpleNamespace(attempt_download_asset=download)
        with patch("scripts.backends.ultralytics._provider"), patch("scripts.backends.ultralytics.importlib.import_module", return_value=module):
            artifact = self.backend.download(spec)
        self.assertEqual(artifact.checkpoint.read_bytes(), b"downloaded")
        self.assertEqual(list(cmn.ORIGINAL_WEIGHTS_DIR.iterdir()), [artifact.checkpoint])

    def test_download_failure_leaves_no_cached_partial(self):
        def download(path, **kwargs):
            Path(path).write_bytes(b"partial")
            raise OSError("network interrupted")
        with patch("scripts.backends.ultralytics._provider"), patch("scripts.backends.ultralytics.importlib.import_module", return_value=SimpleNamespace(attempt_download_asset=download)):
            with self.assertRaises(OSError):
                self.backend.download(self.spec)
        self.assertEqual(list(cmn.ORIGINAL_WEIGHTS_DIR.iterdir()), [])

    def test_download_preflight_before_side_effects(self):
        self.selection("yolov8n", "rtmdet")
        with patch("scripts.weight.get_backend") as factory:
            with self.assertRaisesRegex(ValueError, "rtmdet"):
                weight.download_weights()
            factory.assert_not_called()

    def test_download_failure_table(self):
        self.selection("yolov8n", "yolo26n")
        display = Mock()
        with patch("IPython.display.display", return_value=display), patch("scripts.weight.get_backend") as factory:
            factory.return_value.download.side_effect = OSError("offline")
            with self.assertRaises(OSError):
                weight.download_weights()
        table = display.update.call_args.args[0]
        self.assertEqual(table.loc["yolov8n", "status"], "Failed")
        self.assertEqual(table.loc["yolo26n", "status"], "Not attempted")

    def test_download_preserves_return_paths_api(self):
        self.selection("yolov8n")
        cmn.ORIGINAL_WEIGHTS_DIR.mkdir()
        path = cmn.ORIGINAL_WEIGHTS_DIR / "yolov8n.pt"
        path.write_bytes(b"cached")
        with patch("IPython.display.display", return_value=Mock()):
            self.assertEqual(weight.download_weights(return_paths=True), {"yolov8n": path})
            self.assertIsNone(weight.download_weights())

    def test_split_file_relative_paths_and_bom(self):
        data = self.dataset()
        split = self.root / "split.txt"
        split.write_text("\ufeff./images/a.jpg\n", encoding="utf-8")
        data.yaml.write_text(f"path: {self.root}\ntrain: split.txt\nval: split.txt\nnames: [tobacco]\n")
        info = inspect_dataset(data, "detect")
        self.assertEqual(info["splits"]["train"]["images"], 1)

    def test_training_checkpoint_task_mismatch(self):
        with patch("scripts.backends.ultralytics._construct", return_value=SimpleNamespace(task="segment")):
            with self.assertRaisesRegex(ValueError, "Checkpoint task"):
                self.backend.train(self.spec, self.artifacts, self.dataset(), TrainConfig())

    def test_dataset_task_and_annotation_validation(self):
        data = self.dataset()
        info = inspect_dataset(data, "detect")
        self.assertEqual(info["splits"]["train"]["annotations"], 1)
        with self.assertRaises(ValueError):
            inspect_dataset(data, "segment")
        with self.assertRaises(ValueError):
            inspect_dataset(replace(data, task="segment"), "segment")
        polygon = self.dataset("0 0.1 0.1 0.8 0.1 0.8 0.8", task="segment")
        self.assertEqual(inspect_dataset(polygon, "segment")["task"], "segment")

    def test_dataset_split_hash_changes_with_annotations(self):
        data = self.dataset()
        before = inspect_dataset(data, "detect")
        (self.root / "labels/a.txt").write_text("0 0.4 0.4 0.2 0.2")
        after = inspect_dataset(data, "detect")
        self.assertNotEqual(before["splits"], after["splits"])

    def fake_trainer(self, fail=False):
        model = Mock()
        model.task = "detect"
        model.task_map = {"detect": {"trainer": object}}
        def train(**args):
            if fail:
                raise RuntimeError("GPU failure")
            directory = Path(args["project"]) / args["name"] / "weights"
            directory.mkdir()
            best, last = directory / "best.pt", directory / "last.pt"
            best.write_bytes(b"best")
            last.write_bytes(b"last")
            model.trainer = SimpleNamespace(best=best, last=last, args=SimpleNamespace(**args))
            return SimpleNamespace(results_dict={"map": 0.5})
        model.train.side_effect = train
        return model

    def test_training_runs_and_alias(self):
        model = self.fake_trainer()
        with patch("scripts.backends.ultralytics._construct", return_value=model), patch("scripts.backends.ultralytics._provider", return_value=SimpleNamespace(__version__="test")):
            first = self.backend.train(self.spec, self.artifacts, self.dataset(), TrainConfig())
            second = self.backend.train(self.spec, self.artifacts, self.dataset(), TrainConfig())
        self.assertNotEqual(first.directory, second.directory)
        record = json.loads(first.manifest.read_text())
        self.assertEqual(record["status"], "complete")
        self.assertIn("effective_config", record)
        self.assertEqual((cmn.TRAINED_WEIGHTS_DIR / "yolov8n.pt").read_bytes(), b"best")
        self.assertTrue(first.last_artifacts.checkpoint.is_file())

    def test_training_failure_manifest(self):
        with patch("scripts.backends.ultralytics._construct", return_value=self.fake_trainer(fail=True)), patch("scripts.backends.ultralytics._provider", return_value=SimpleNamespace(__version__="test")):
            with self.assertRaisesRegex(RuntimeError, "GPU failure"):
                self.backend.train(self.spec, self.artifacts, self.dataset(), TrainConfig())
        manifest = next(cmn.RUNS_DIR.glob("*/manifest.json"))
        self.assertEqual(json.loads(manifest.read_text())["status"], "failed")

    def test_training_rejects_reserved_options(self):
        with self.assertRaises(ValueError):
            self.backend.train(self.spec, self.artifacts, self.dataset(), TrainConfig(options={"resume": True}))

    def test_missing_installed_trainer_is_capability_error(self):
        model = SimpleNamespace(task="detect", task_map={"detect": {}})
        with patch("scripts.backends.ultralytics._construct", return_value=model):
            with self.assertRaises(UnsupportedOperation):
                self.backend.train(self.spec, self.artifacts, self.dataset(), TrainConfig())

    def test_batch_skips_inference_only_and_future_backends(self):
        self.selection("mobilesam", "rtmdet", "yolov8", "yolov8n")
        backend = Mock()
        backend.train.return_value = SimpleNamespace(directory=self.root)
        with patch("scripts.workflows.get_backend", return_value=backend), contextlib.redirect_stdout(io.StringIO()):
            result = workflows.train_models(self.dataset())
        self.assertEqual(set(result.skipped), {"mobilesam", "rtmdet", "yolov8"})
        self.assertEqual(set(result.runs), {"yolov8n"})
        backend.train.assert_called_once()

    def test_unexpected_notimplemented_error_is_not_silently_skipped(self):
        self.selection("yolov8n")
        backend = Mock()
        backend.train.side_effect = NotImplementedError("unexpected kernel failure")
        with patch("scripts.workflows.get_backend", return_value=backend):
            with self.assertRaisesRegex(RuntimeError, "yolov8n"):
                workflows.train_models(self.dataset())

    def test_constructor_routing_and_reuse(self):
        import numpy as np
        for constructor in ("YOLO", "RTDETR", "YOLOWorld", "YOLOE", "SAM", "FastSAM"):
            with self.subTest(constructor=constructor):
                module = Mock()
                fake = Mock()
                fake.predict.return_value = []
                getattr(module, constructor).return_value = fake
                spec = replace(self.spec, constructor=constructor)
                with patch("scripts.backends.ultralytics._provider", return_value=module):
                    predictor = self.backend.load(spec, self.artifacts)
                    predictor.predict(np.zeros((4, 8, 3), dtype=np.uint8))
                    predictor.predict(np.zeros((4, 8, 3), dtype=np.uint8))
                getattr(module, constructor).assert_called_once_with(str(self.weights))

    def test_text_and_visual_prompt_routing(self):
        import numpy as np
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        for constructor in ("YOLOWorld", "YOLOE", "FastSAM"):
            fake = Mock()
            fake.predict.return_value = []
            spec = replace(self.spec, constructor=constructor)
            with patch("scripts.backends.ultralytics._construct", return_value=fake):
                predictor = self.backend.load(spec, self.artifacts)
                predictor.predict(image, prompts={"text": "tobacco"})
            if constructor == "FastSAM":
                self.assertEqual(fake.predict.call_args.kwargs["texts"], ["tobacco"])
            else:
                self.assertEqual(fake.set_classes.call_args.args[0], ["tobacco"])
        with patch("scripts.backends.ultralytics._construct", return_value=fake):
            predictor = self.backend.load(replace(self.spec, constructor="YOLOE"), self.artifacts)
            vp = {"bboxes": [[0, 0, 2, 2]], "cls": [0]}
            predictor.predict(image, prompts={"visual_prompts": vp})
            self.assertEqual(fake.predict.call_args.kwargs["visual_prompts"], vp)

    def test_invalid_prompts_fail_before_loading(self):
        import numpy as np
        with patch("scripts.backends.ultralytics._construct") as constructor:
            predictor = self.backend.load(self.spec, self.artifacts)
            with self.assertRaises(ValueError):
                predictor.predict(np.zeros((4, 8, 3), dtype=np.uint8), prompts={"points": [[1, 2]]})
            constructor.assert_not_called()

    def test_sam3_concept_interface(self):
        import numpy as np
        semantic = Mock(return_value=[])
        factory = Mock(return_value=semantic)
        module = SimpleNamespace(SAM3SemanticPredictor=factory)
        spec = cmn.MODELS_REGESTRY["sam3"]
        with patch("scripts.backends.ultralytics._provider"), patch("scripts.backends.ultralytics.importlib.import_module", return_value=module):
            predictor = self.backend.load(spec, self.artifacts)
            image = np.zeros((4, 8, 3), dtype=np.uint8)
            predictor.predict(image, prompts={"text": "tobacco"})
            predictor.predict(image, prompts={"exemplars": [[0, 0, 2, 2]]})
        factory.assert_called_once()
        self.assertEqual(semantic.call_args.kwargs, {"bboxes": [[0, 0, 2, 2]]})
        self.assertEqual(semantic.set_image.call_count, 2)

    def test_normalization_original_coordinates_and_classless_masks(self):
        import numpy as np
        result = SimpleNamespace(orig_shape=(4, 8), names={0: "tobacco"},
                                 boxes=SimpleNamespace(xyxy=np.array([[1, 2, 3, 4]]), conf=np.array([.8]), cls=np.array([0.])),
                                 masks=SimpleNamespace(data=np.ones((1, 4, 8))))
        prediction = normalize_result(result)
        np.testing.assert_array_equal(prediction.boxes_xyxy, [[1, 2, 3, 4]])
        self.assertEqual(prediction.masks.dtype, np.bool_)
        self.assertEqual(prediction.names, {0: "tobacco"})
        prediction = normalize_result(result, semantic_classes=False)
        self.assertIsNone(prediction.class_ids)
        self.assertIsNone(prediction.names)
        result.masks.data = np.ones((1, 8, 8))
        with self.assertRaises(ValueError):
            normalize_result(result)

    def test_empty_prediction(self):
        result = SimpleNamespace(orig_shape=(4, 8), boxes=None, masks=None, names={})
        prediction = normalize_result(result)
        self.assertEqual(prediction.boxes_xyxy.shape, (0, 4))
        self.assertIsNone(prediction.masks)


if __name__ == "__main__":
    unittest.main()
