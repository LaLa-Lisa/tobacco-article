"""Проверки измерительной логики без GPU, SDK и загрузки весов."""
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark import core as bm
from scripts import common as cmn
from scripts.benchmark.plots import completion_fps_bins


class BenchmarkTest(unittest.TestCase):
    def test_fps_bins_include_empty_and_partial_windows(self):
        run = dict(initial={'timestamp': 10}, final={'timestamp': 12.5},
                   records=[{'completed_at': t} for t in (10.2, 10.8, 12.5)])
        edges, fps = completion_fps_bins(run)
        self.assertEqual(edges.tolist(), [0, 1, 2, 2.5])
        self.assertEqual(fps.tolist(), [2, 0, 2])
        self.assertEqual(float(np.sum(fps * np.diff(edges))), 3)

    def test_fps_bins_short_run_and_exact_boundary(self):
        for elapsed, times, expected in [(0.25, [0.25], [4]), (2, [1, 2], [0, 2])]:
            run = dict(initial={'timestamp': 0}, final={'timestamp': elapsed},
                       records=[{'completed_at': t} for t in times])
            self.assertEqual(completion_fps_bins(run)[1].tolist(), expected)
        for width in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                completion_fps_bins(run, width)

    def test_known_summary_uses_elapsed_not_reciprocal_frame_latency(self):
        rows = [dict(prediction_ms=x, frame_latency_ms=x + 100) for x in (10, 20, 30, 40)]
        value = bm.summarize(rows, 2)
        self.assertEqual(value['processing_fps'], 2)
        self.assertEqual(value['prediction_ms_mean'], 25)
        self.assertEqual(value['prediction_ms_median'], 25)
        self.assertAlmostEqual(value['prediction_ms_p95'], 38.5)
        self.assertAlmostEqual(value['frame_latency_ms_p99'], 139.7)

    def test_invalid_configuration(self):
        for kwargs in ({'duration_s': 0}, {'duration_s': float('nan')},
                       {'warmup_frames': 0}, {'warmup_frames': True},
                       {'memory_interval_s': -1}, {'input_rates': (0,)},
                       {'stream': True, 'input_rates': ()}, {'direct': False},
                       {'input_rates': (10, 10)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bm.BenchmarkConfig(**kwargs).validate()

    def test_schedule_reproducible_and_complete(self):
        config = bm.BenchmarkConfig(stream=True)
        self.assertEqual(bm.make_schedule(config), bm.make_schedule(config))
        self.assertCountEqual(bm.make_schedule(config), [None, 10, 20, 30, 60, 300])
        self.assertEqual(bm.make_schedule(bm.BenchmarkConfig()), [None])
        self.assertCountEqual(bm.make_schedule(bm.BenchmarkConfig(direct=False, stream=True)),
                              [10, 20, 30, 60, 300])

    def test_queue_replacement_and_close(self):
        with patch.object(bm.LatestFrameStream, '_produce'):
            stream = bm.LatestFrameStream([np.zeros((2, 2, 3), dtype=np.uint8)], 30)
        self.addCleanup(stream.close)
        stream._publish('first', 1)
        stream._publish('latest', 2)
        self.assertEqual(stream.read(), ('latest', 2, 2))
        snapshot = stream.snapshot()
        self.assertEqual((snapshot['generated'], snapshot['consumed'], snapshot['dropped']), (2, 1, 1))
        stream.close()
        with self.assertRaises(StopIteration):
            stream.read()

    def test_producer_error_propagates(self):
        image = Mock()
        image.copy.side_effect = ValueError('copy failed')
        stream = bm.LatestFrameStream([image], 30)
        self.addCleanup(stream.close)
        with self.assertRaisesRegex(RuntimeError, 'producer failed'):
            stream.read()

    def test_close_unblocks_consumer(self):
        with patch.object(bm.LatestFrameStream, '_produce'):
            stream = bm.LatestFrameStream([object()], 30)
        stopped = threading.Event()

        def consume():
            try:
                stream.read()
            except StopIteration:
                stopped.set()

        thread = threading.Thread(target=consume, daemon=True)
        thread.start()
        stream.close()
        thread.join(timeout=1)
        self.assertTrue(stopped.is_set())

    def test_memory_boundary_samples(self):
        sampler = bm.MemorySampler(100, read_rss=Mock(side_effect=[100, 250]))
        sampler.start()
        sampler.close()
        self.assertEqual([r['rss_bytes'] for r in sampler.records], [100, 250])
        self.assertFalse(sampler.thread.is_alive())

    def test_device_synchronization_and_peak_api(self):
        cuda = Mock()
        cuda.memory_allocated.return_value = 10
        cuda.memory_reserved.return_value = 20
        cuda.max_memory_allocated.return_value = 30
        cuda.max_memory_reserved.return_value = 40
        metrics = object.__new__(bm.DeviceMetrics)
        metrics.torch, metrics.device, metrics.cuda = SimpleNamespace(cuda=cuda), 'cuda:1', True
        metrics.reset()
        cuda.synchronize.assert_called_once_with('cuda:1')
        cuda.reset_peak_memory_stats.assert_called_once_with('cuda:1')
        self.assertEqual(metrics.memory(), dict(cuda_allocated_bytes=10, cuda_reserved_bytes=20))
        self.assertEqual(metrics.memory(peak=True), dict(cuda_allocated_bytes=30, cuda_reserved_bytes=40))
        metrics.cuda = False
        self.assertIsNone(metrics.memory()['cuda_allocated_bytes'])

    def test_warmup_excluded_and_identical_options(self):
        events = []
        predictor = Mock()
        predictor.predict.side_effect = lambda *a, **kw: events.append('predict')
        metrics = Mock()
        metrics.synchronize.side_effect = lambda: events.append('sync')
        metrics.reset.side_effect = lambda: events.append('reset')
        metrics.memory.return_value = dict(cuda_allocated_bytes=None, cuda_reserved_bytes=None)
        sampler = Mock(records=[{'timestamp': 0, 'rss_bytes': 100}])
        # Начало окна, начало предсказания, конец предсказания, конец окна.
        with patch.object(bm.time, 'perf_counter', side_effect=[10, 10, 12, 12]):
            run = bm.measure_run(predictor, [object()], metrics,
                                 bm.BenchmarkConfig(warmup_frames=2, duration_s=1),
                                 options={'conf': 0.25}, sampler_factory=lambda _: sampler)
        self.assertEqual(events, ['predict', 'sync', 'predict', 'sync', 'reset', 'sync', 'predict', 'sync'])
        self.assertEqual(run['summary']['processed'], 1)
        self.assertEqual(run['summary']['processing_fps'], 0.5)
        self.assertEqual(run['summary']['prediction_ms_mean'], 2000)
        self.assertTrue(all(call.kwargs == {'prompts': None, 'conf': 0.25}
                            for call in predictor.predict.call_args_list))
        sampler.close.assert_called_once()

    def test_measurement_failure_closes_stream_and_sampler(self):
        predictor = Mock()
        predictor.predict.side_effect = [[], RuntimeError('inference failed')]
        metrics = Mock()
        sampler = Mock()
        stream = Mock()
        stream.snapshot.return_value = {'timestamp': 0}
        stream.read.return_value = (object(), 1, 0)
        with patch.object(bm, 'LatestFrameStream', return_value=stream):
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                bm.measure_run(predictor, [object()], metrics,
                               bm.BenchmarkConfig(warmup_frames=1), input_fps=30,
                               sampler_factory=lambda _: sampler)
        sampler.close.assert_called_once()
        stream.close.assert_called_once()

    def test_stream_counters_and_queue_latency(self):
        predictor, metrics, sampler, stream = Mock(), Mock(), Mock(), Mock()
        metrics.memory.return_value = {}
        sampler.records = [{'timestamp': 10, 'rss_bytes': 100}]
        stream.snapshot.side_effect = [
            dict(timestamp=10, generated=5, consumed=2, dropped=2),
            dict(timestamp=12, generated=9, consumed=3, dropped=5),
        ]
        stream.read.return_value = (object(), 6, 9.5)
        with patch.object(bm, 'LatestFrameStream', return_value=stream), \
             patch.object(bm.time, 'perf_counter', side_effect=[10.5, 12]):
            run = bm.measure_run(predictor, [object()], metrics,
                                 bm.BenchmarkConfig(warmup_frames=1, duration_s=1),
                                 input_fps=30, sampler_factory=lambda _: sampler)
        self.assertEqual(run['summary']['generation_fps'], 2)
        self.assertEqual(run['summary']['drop_rate'], 1.5)
        self.assertEqual(run['summary']['prediction_ms_mean'], 1500)
        self.assertEqual(run['summary']['frame_latency_ms_mean'], 2500)
        self.assertEqual(run['summary']['processing_fps'], 0.5)

    def test_interrupt_terminates_worker_and_retains_completed_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, image = root / 'weights.pt', root / 'image.png'
            checkpoint.write_bytes(b'weights')
            image.write_bytes(b'image')
            process = Mock()
            process.poll.return_value = None

            def started(*args, **kwargs):
                request = Path(args[0][-1])
                bm.save_run(request.parent, 0, dict(
                    model_id='yolov8n', device='cpu', mode='direct', input_fps=None,
                    summary={'processing_fps': 1}, records=[]))
                return process

            with patch('scripts.workflows.enabled_specs', return_value=[cmn.MODELS_REGESTRY['yolov8n']]), \
                 patch.object(bm.subprocess, 'Popen', side_effect=started), \
                 patch.object(bm.time, 'sleep', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    bm.run_benchmarks({'yolov8n': checkpoint}, [image], output_root=root / 'results')
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=5)
            output = next((root / 'results').iterdir())
            self.assertEqual(len(bm.load_runs(output)), 1)
            self.assertEqual(json.loads((output / 'manifest.json').read_text())['status'], 'interrupted')

    def test_persistence_keeps_completed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = dict(model_id='test', device='cpu', mode='direct', input_fps=None,
                       summary={'processing_fps': 42}, records=[])
            bm.save_run(directory, 0, run)
            self.assertEqual(bm.load_runs(directory), [run])
            self.assertIn('42', (Path(directory) / 'summary.csv').read_text())

    def test_worker_failure_records_status_and_relay_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, image = root / 'weights.pt', root / 'image.png'
            checkpoint.write_bytes(b'weights')
            image.write_bytes(b'image')
            process = Mock(returncode=1)
            process.poll.return_value = 1
            with patch('scripts.workflows.enabled_specs', return_value=[cmn.MODELS_REGESTRY['yolov8n']]), \
                 patch.object(bm.subprocess, 'Popen', return_value=process):
                with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                    bm.run_benchmarks({'yolov8n': checkpoint}, [image], output_root=root / 'results')
            output = next((root / 'results').iterdir())
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            request = json.loads(next(output.rglob('request.json')).read_text())
            self.assertEqual(request['spec']['id'], 'yolov8n')
            self.assertEqual(request['checkpoint'], str(checkpoint))


if __name__ == '__main__':
    unittest.main()
