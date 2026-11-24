"""Проверки выбора сохранённых результатов и их отображения."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark.review import comparison_figures, full_table, latest_measurements, simple_table


class ReviewTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.addCleanup(plt.close, 'all')

    def save(self, session, worker, index=0, rate=None, device='cpu', model='m', peak=None):
        folder = self.root / f'20261121T00000{session}Z-id' / worker
        folder.mkdir(parents=True, exist_ok=True)
        (folder.parent / 'manifest.json').write_text('{"status":"running"}')
        (folder / 'manifest.json').write_text('{"checkpoint_sha256":"abc","inputs":[]}')
        summary = dict(processing_fps=1, frame_latency_ms_p95=8)
        if peak is not None:
            summary.update(probe_rss_sampled_peak_bytes=peak * 2**20)
        if device.startswith('cuda'):
            summary.update(probe_cuda_allocated_bytes_peak=2**20)
        run = dict(model_id=model, device=device, mode='direct' if rate is None else 'stream',
                   input_fps=rate, initial={'timestamp': 20}, final={'timestamp': 21},
                   records=[dict(completed_at=20.7, prediction_ms=5, frame_latency_ms=8)],
                   memory_records=[dict(timestamp=19.9, rss_bytes=0),
                                   dict(timestamp=20.5, rss_bytes=3 * 2**20)], summary=summary)
        path = folder / f'run-{index:04d}.json'
        path.write_text(json.dumps(run))
        return path

    def test_selection_and_tables(self):
        self.save(1, 'old', peak=2)
        slow = self.save(1, 'old', 1, rate=5)
        self.save(1, 'old', 2, rate=30)
        self.save(2, '0000-worker-r1', peak=9)
        self.save(2, '0001-worker-r2', peak=10)
        newest = self.save(2, '0001-worker-r2', 1, peak=11)
        self.save(2, 'gpu', device='cuda:0', peak=8)
        bad = self.save(3, 'broken')
        bad.write_text('{')
        with self.assertWarnsRegex(UserWarning, 'broken/run-0000.json'):
            chosen = latest_measurements(self.root)
        self.assertEqual(len(chosen), 4)
        detail = full_table(chosen)
        self.assertEqual(detail.source.tolist()[:2], [str(newest), str(slow)])
        self.assertTrue((detail.worker_checkpoint_sha256 == 'abc').all())
        self.assertTrue((detail.frame_latency_ms_p95 == 8).all())
        brief = simple_table(chosen)
        self.assertEqual(brief.columns.get_level_values(0).unique().tolist(),
                         ['Direct', 'Stream 5 FPS', 'Stream 30 FPS'])
        self.assertEqual(brief.loc[('m', 'cpu'), ('Direct', 'Peak RAM (MiB)')], 11)
        self.assertEqual(brief.loc[('m', 'cuda:0'), ('Direct', 'Peak GPU allocated (MiB)')], 1)
        self.assertTrue(pd.isna(brief.loc[('m', 'cpu'), ('Stream 5 FPS', 'Peak RAM (MiB)')]))

    def test_plots_and_empty_root(self):
        self.assertEqual(latest_measurements(self.root), [])
        self.save(1, 'a')
        self.save(1, 'b', rate=30)
        self.save(1, 'c', rate=10)
        self.save(1, 'd', rate=10, model='stream-only')
        figures = list(comparison_figures(latest_measurements(self.root)))
        self.assertEqual(len(figures), 3)
        fps, stream, latency, queued, ram, stream_ram = figures[0].axes
        self.assertEqual([fps.get_title(), stream.get_title()], ['Direct', 'Stream 10 FPS'])
        self.assertEqual(figures[1].axes[1].get_title(), 'Stream 30 FPS')
        self.assertEqual(figures[2].axes[0].get_title(), 'Direct (unmeasured)')
        self.assertEqual(len(latency.lines), 1)
        self.assertEqual(len(queued.lines), 2)
        self.assertEqual(ram.lines[0].get_xdata().tolist(), [.5])
        self.assertEqual(ram.lines[0].get_ydata().tolist(), [3])
        self.assertTrue(ram.get_shared_y_axes().joined(ram, stream_ram))
