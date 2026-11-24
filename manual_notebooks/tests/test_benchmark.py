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
from scripts.benchmark.plots import completion_fps_bins, configuration_table, summary_table
from scripts.benchmark.progress import BenchmarkProgress


class BenchmarkTest(unittest.TestCase):
    def test_model_footprint_counts_shared_storage_once_in_loaded_dtype(self):
        import torch
        from scripts.benchmark.footprint import model_footprint

        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.ones(6, dtype=torch.float16))
        module.tied = torch.nn.Parameter(module.weight[:2], requires_grad=False)
        module.register_buffer('view', module.weight.detach()[1:])
        module.register_buffer('counter', torch.zeros(2, dtype=torch.int64), persistent=False)
        self.assertEqual(model_footprint(module), dict(
            model_parameter_bytes=12, model_buffer_bytes=16, model_tensor_bytes=28,
            model_bytes_by_dtype={'torch.float16': 12, 'torch.int64': 16}))

    def test_progress_counts_saved_runs_once_across_workers(self):
        bar = Mock(n=0)
        progress = BenchmarkProgress(4, bar_factory=Mock(return_value=bar))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for worker in ('first', 'second'):
                path = root / worker
                path.mkdir()
                progress.start_worker(path, worker, 'cpu')
                progress.refresh()  # Загрузка: файл статуса ещё не создан.
                for completed in (0, 1, 1, 2):
                    bm.write_json(path / 'progress.json', dict(
                        completed=completed, stage='Measuring', mode='Direct', stage_started=10))
                    with patch('scripts.benchmark.progress.time.monotonic', return_value=13):
                        progress.refresh()
            self.assertEqual(bar.n, 4)
            self.assertIn('Measuring (3s)', bar.set_postfix_str.call_args.args[0])
        progress.close('completed')
        bar.close.assert_called_once()

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
                       {'input_rates': (10, 10)},
                       {'memory_passes': -1}, {'memory_passes': 1.5},
                       {'torch_num_threads': 0}, {'torch_num_threads': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bm.BenchmarkConfig(**kwargs).validate()

    def test_image_selection_is_reproducible_and_resolves_from_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = [f'image-{index}.png' for index in range(10)]
            for name in names:
                (root / name).write_bytes(b'image')
            source = root / 'images.txt'
            source.write_text('\n'.join(names))
            chosen = bm.select_images(source, count=4, seed=7)
            self.assertEqual(chosen, bm.select_images(source, count=4, seed=7))
            self.assertEqual(len(set(chosen)), 4)
            self.assertTrue(all(path.parent == root for path in chosen))
            self.assertEqual(bm.select_images(source), [root / name for name in names])
            for count in (0, 11, True):
                with self.assertRaises(ValueError):
                    bm.select_images(source, count=count)
            source.write_text(names[0] + '\n' + names[0])
            with self.assertRaisesRegex(ValueError, 'duplicates'):
                bm.select_images(source)
            source.write_text('missing.png')
            with self.assertRaisesRegex(ValueError, 'existing files'):
                bm.select_images(source)

    def test_jobs_have_every_model_device_and_rate_once(self):
        specs = [{'spec': {'id': name}} for name in ('first', 'second')]
        config = bm.BenchmarkConfig(stream=True, input_rates=(10, 30))
        jobs = bm.make_jobs(specs, ['cpu', 'cuda:0'], config)
        self.assertEqual(jobs, bm.make_jobs(specs, ['cpu', 'cuda:0'], config))
        keys = [(j['model_id'], j['device'], j['input_fps']) for j in jobs]
        self.assertCountEqual(keys, [(name, device, rate)
                                    for name in ('first', 'second') for device in ('cpu', 'cuda:0')
                                    for rate in (None, 10, 30)])

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

    def test_producer_paces_with_sleep_and_absolute_deadlines(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        with patch.object(bm.threading, 'Thread'):
            stream = bm.LatestFrameStream([image], 300)
        clock = [0.0]
        published = []

        def sleep(duration):
            self.assertGreater(duration, 0)
            self.assertLessEqual(duration, 0.01)
            clock[0] += duration

        def publish(frame, generated_at):
            self.assertIsNot(frame, image)
            published.append(generated_at)
            clock[0] += 0.0002
            if len(published) == 3:
                stream.stop.set()

        with patch.object(bm.time, 'perf_counter', side_effect=lambda: clock[0]), \
                patch.object(bm.time, 'sleep', side_effect=sleep), \
                patch.object(stream, '_publish', side_effect=publish), \
                patch.object(stream.stop, 'wait', side_effect=AssertionError('Coarse timer used')):
            stream._produce()
        self.assertIsNone(stream.error)
        np.testing.assert_allclose(published, [0, 1 / 300, 2 / 300])

    def test_producer_low_rate_checks_stop_between_short_sleeps(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        with patch.object(bm.threading, 'Thread'):
            stream = bm.LatestFrameStream([image], 0.1)
        with patch.object(bm.time, 'perf_counter', return_value=0), \
                patch.object(bm.time, 'sleep', side_effect=lambda duration: stream.stop.set()) as sleep:
            stream._produce()
        self.assertIsNone(stream.error)
        self.assertEqual(stream.generated, 1)
        sleep.assert_called_once_with(0.01)

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

    def test_memory_probe_sampler_keeps_peak_without_growing_records(self):
        sampler = bm.MemorySampler(.01, read_rss=Mock(side_effect=[100, 400, 200] * 1000),
                                   retain_records=False)
        with patch.object(bm.time, 'perf_counter', side_effect=range(3000)):
            for _ in range(3000):
                sampler.capture()
        self.assertEqual(sampler.records, [])
        self.assertEqual(sampler.sample_count, 3000)
        self.assertEqual(sampler.first, dict(timestamp=0, rss_bytes=100))
        self.assertEqual(sampler.last, dict(timestamp=2999, rss_bytes=200))
        self.assertEqual(sampler.peak, dict(timestamp=1, rss_bytes=400))
        self.assertEqual(sampler.max_gap_s, 1)

    def test_memory_probe_visits_same_images_and_releases_results(self):
        import weakref
        references, seen = [], []

        class Result:
            pass

        class Predictor:
            def predict(self, image, **options):
                self_outer.assertTrue(all(ref() is None for ref in references))
                self_outer.assertEqual(options, dict(prompts=None, conf=.3))
                seen.append(image)
                result = Result()
                references.append(weakref.ref(result))
                return result

        self_outer = self
        metrics = Mock()
        metrics.memory.side_effect = [dict(cuda_allocated_bytes=50, cuda_reserved_bytes=80),
                                      dict(cuda_allocated_bytes=100, cuda_reserved_bytes=120)]
        sampler = bm.MemorySampler(100, read_rss=Mock(side_effect=[200, 300]), retain_records=False)
        factory = Mock(return_value=sampler)
        probe = bm.measure_memory(Predictor(), ['a', 'b', 'c'], metrics,
                                  bm.BenchmarkConfig(memory_passes=2), options={'conf': .3},
                                  sampler_factory=factory)
        self.assertEqual(seen, ['a', 'b', 'c'] * 2)
        self.assertTrue(all(ref() is None for ref in references))
        self.assertEqual(probe['frames'], 6)
        self.assertEqual(probe['summary']['rss_sampled_peak_bytes'], 300)
        self.assertEqual(probe['summary']['cuda_allocated_bytes_peak'], 100)
        self.assertEqual(sampler.records, [])
        self.assertFalse(sampler.thread.is_alive())
        factory.assert_called_once_with(.01, retain_records=False)

    def test_memory_probe_failure_closes_sampler(self):
        predictor = Mock()
        predictor.predict.side_effect = RuntimeError('failed')
        sampler = Mock()
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            bm.measure_memory(predictor, [object()], Mock(), bm.BenchmarkConfig(memory_passes=1),
                              sampler_factory=Mock(return_value=sampler))
        sampler.close.assert_called_once()

    def test_all_images_warmed_before_separate_memory_probe(self):
        events = []
        predictor = Mock()
        predictor.predict.side_effect = lambda image, **kw: events.append(image)
        metrics = Mock()
        metrics.memory.return_value = {}
        sampler = Mock(records=[{'timestamp': 0, 'rss_bytes': 100}])

        def probe(*args, **kwargs):
            self.assertEqual(events, ['a', 'b', 'c', 'memory'])
            return dict(summary={'rss_sampled_peak_bytes': 50})

        with patch.object(bm, 'measure_memory', side_effect=probe), \
             patch.object(bm.time, 'perf_counter', side_effect=[10, 10, 12, 12]):
            run = bm.measure_run(predictor, ['a', 'b', 'c'], metrics,
                                 bm.BenchmarkConfig(warmup_frames=1, duration_s=1, memory_passes=2),
                                 sampler_factory=lambda _: sampler,
                                 on_memory_start=lambda: events.append('memory'),
                                 on_measurement_start=lambda: events.append('timing'))
        self.assertEqual(events, ['a', 'b', 'c', 'memory', 'timing', 'a'])
        self.assertEqual(run['summary']['probe_rss_sampled_peak_bytes'], 50)
        self.assertEqual(run['summary']['rss_sampled_peak_bytes'], 100)

    def test_thread_count_restored_after_lazy_provider_setup_and_rewarmed(self):
        count = [1]
        observed = []

        def predict(*args, **kwargs):
            if not observed:
                count[0] = 8
            observed.append(count[0])

        metrics = Mock()
        metrics.memory.return_value = {}
        metrics.torch.get_num_threads.side_effect = lambda: count[0]
        metrics.torch.set_num_threads.side_effect = lambda n: count.__setitem__(0, n)
        sampler = Mock(records=[{'timestamp': 0, 'rss_bytes': 100}])
        with patch.object(bm.time, 'perf_counter', side_effect=[10, 10, 12, 12]):
            bm.measure_run(Mock(predict=Mock(side_effect=predict)), [object()], metrics,
                           bm.BenchmarkConfig(warmup_frames=2, duration_s=1, torch_num_threads=1),
                           sampler_factory=lambda _: sampler)
        self.assertEqual(observed, [8, 8, 1, 1, 1])

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
                                 options={'conf': 0.25}, sampler_factory=lambda _: sampler,
                                 on_measurement_start=lambda: events.append('progress'))
        self.assertEqual(events, ['predict', 'sync', 'predict', 'sync', 'progress',
                                  'reset', 'sync', 'predict', 'sync'])
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
                 patch.object(bm, 'BenchmarkProgress') as reporter, \
                 patch.object(bm.time, 'sleep', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    bm.run_benchmarks({'yolov8n': checkpoint}, [image], output_root=root / 'results')
            reporter.return_value.close.assert_called_once_with('interrupted')
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

    def test_summary_table_keeps_memory_scopes_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            bm.save_run(directory, 0, dict(
                model_id='test', device='cpu', mode='direct', input_fps=None,
                summary=dict(rss_sampled_peak_bytes=10 * 2**20,
                             probe_rss_sampled_peak_bytes=2**20), records=[]))
            row = summary_table(directory).iloc[0]
            self.assertEqual(row['rss_sampled_peak_mib'], 10)
            self.assertEqual(row['probe_rss_sampled_peak_mib'], 1)

    def test_configuration_table_preserves_unknown_runtime_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / 'worker'
            worker.mkdir()
            bm.write_json(root / 'manifest.json', dict(status='completed'))
            bm.write_json(worker / 'manifest.json', dict(
                model_id='rf-detr-m', device='cpu', backend='rfdetr', status='completed',
                runtime=dict(configured_input_size=384, precision=None, actual_tensor_shape=None),
                inputs=[{'shape': [100, 200, 3]}, {'shape': [100, 200, 3]}], torch_threads=1))
            table = configuration_table(root)
            self.assertEqual(len(table), 1)
            self.assertEqual(table.iloc[0]['configured_input_size'], 384)
            self.assertIsNone(table.iloc[0]['precision'])
            self.assertIsNone(table.iloc[0]['actual_tensor_shape'])
            self.assertEqual(table.iloc[0]['source_shapes'], [(100, 200, 3)])

    def test_runner_launches_fresh_process_for_each_mode_and_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, image = root / 'weights.pt', root / 'image.png'
            checkpoint.write_bytes(b'weights')
            image.write_bytes(b'image')
            requests = []

            def started(command, **kwargs):
                request = Path(command[-1])
                payload = json.loads(request.read_text())
                requests.append((request, payload))
                self.assertEqual(payload['input_hashes'][str(image)], bm.sha256(image))
                self.assertEqual(payload['checkpoint_sha256'], bm.sha256(checkpoint))
                bm.save_run(request.parent, 0, dict(
                    model_id='yolov8n', device=payload['device'],
                    mode='direct' if payload['input_fps'] is None else 'stream',
                    input_fps=payload['input_fps'],
                    summary={'processing_fps': 1}, records=[]))
                return Mock(returncode=0, poll=Mock(return_value=0))

            with patch('scripts.workflows.enabled_specs', return_value=[cmn.MODELS_REGESTRY['yolov8n']]), \
                 patch.object(bm.subprocess, 'Popen', side_effect=started):
                output = bm.run_benchmarks(
                    {'yolov8n': checkpoint}, [image], output_root=root / 'results', progress=False,
                    config=bm.BenchmarkConfig(stream=True, input_rates=(30,)))
            self.assertEqual(len({path.parent for path, _ in requests}), 2)
            self.assertCountEqual([p['input_fps'] for _, p in requests], [None, 30])
            self.assertEqual(len(bm.load_runs(output)), 2)
            self.assertEqual(json.loads((output / 'manifest.json').read_text())['status'], 'completed')

    def test_worker_failure_records_status_and_relay_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, image = root / 'weights.pt', root / 'image.png'
            checkpoint.write_bytes(b'weights')
            image.write_bytes(b'image')
            process = Mock(returncode=1)
            process.poll.return_value = 1
            with patch('scripts.workflows.enabled_specs', return_value=[cmn.MODELS_REGESTRY['yolov8n']]), \
                 patch.object(bm, 'BenchmarkProgress') as reporter, \
                 patch.object(bm.subprocess, 'Popen', return_value=process):
                with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                    bm.run_benchmarks({'yolov8n': checkpoint}, [image], output_root=root / 'results')
            reporter.return_value.close.assert_called_once_with('failed')
            output = next((root / 'results').iterdir())
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            request = json.loads(next(output.rglob('request.json')).read_text())
            self.assertEqual(request['spec']['id'], 'yolov8n')
            self.assertEqual(request['checkpoint'], str(checkpoint))


if __name__ == '__main__':
    unittest.main()
