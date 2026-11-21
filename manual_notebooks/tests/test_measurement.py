"""Синтетические проверки временных рядов, срезов и совместимости графиков."""
from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark import MeasurementRun, MeasurementSeries, load_measurement_runs
from scripts.benchmark.plots import (
    completion_fps_bins, plot_fps_over_time, plot_latency_over_time, plot_series,
)


def saved_run():
    return dict(model_id='test', device='cpu', mode='stream', input_fps=30,
                initial={'timestamp': 100}, final={'timestamp': 110},
                records=[dict(completed_at=t, prediction_ms=10 + i, frame_latency_ms=20 + i)
                         for i, t in enumerate((100.5, 101, 105, 106, 110))],
                summary={'processing_fps': 0.5})


class MeasurementTest(unittest.TestCase):
    def test_binned_fps_counts_empty_partial_and_final_bins(self):
        run = MeasurementRun([0.2, 0.8, 2.5], end_s=2.5)
        series = run.fps()
        self.assertEqual(series.bin_edges, (0, 1, 2, 2.5))
        self.assertEqual(series.values, (2, 0, 2))
        self.assertEqual(series.times, (0.5, 1.5, 2.25))
        self.assertAlmostEqual(sum(np.diff(series.bin_edges) * series.values), 3)
        self.assertEqual(run.processing_fps, 1.2)
        edges, values = completion_fps_bins(run)
        np.testing.assert_equal(edges, series.bin_edges)
        np.testing.assert_equal(values, series.values)

    def test_adjacent_slices_and_nested_slices_keep_original_axis(self):
        run = MeasurementRun([0, 1, 5, 6, 10], end_s=10)
        first, second = run.slice(0, 5), run.slice(5, 10)
        self.assertEqual(first.completed_at, (0, 1))
        self.assertEqual(second.completed_at, (5, 6, 10))
        self.assertEqual(len(first.completed_at) + len(second.completed_at), len(run.completed_at))
        self.assertEqual(second.processing_fps, 0.6)
        nested = first.slice(1, 5)
        self.assertEqual(nested.completed_at, (1,))
        self.assertFalse(nested.include_end)
        self.assertEqual(second.fps(bin_seconds=3).bin_edges, (5, 8, 10))
        self.assertEqual(second.fps(bin_seconds=3).values, (2 / 3, 0.5))

    def test_instantaneous_fps_uses_only_gaps_inside_slice(self):
        run = MeasurementRun([1, 2, 2.5, 4], end_s=6)
        series = run.fps(kind='instantaneous')
        self.assertEqual(series.times, (2, 2.5, 4))
        np.testing.assert_allclose(series.values, [1, 2, 2 / 3])
        sliced = run.slice(2, 5).fps(kind='instantaneous')
        self.assertEqual(sliced.times, (2.5, 4))
        np.testing.assert_allclose(sliced.values, [2, 2 / 3])
        self.assertEqual(run.processing_fps, 4 / 6)

    def test_empty_and_single_completion_runs(self):
        for times in ([], [0.5]):
            run = MeasurementRun(times, end_s=1)
            self.assertEqual(run.fps(kind='instantaneous').values, ())
        empty = MeasurementRun([1, 4], end_s=5).slice(2, 3)
        self.assertEqual(empty.processing_fps, 0)
        self.assertEqual(empty.fps(bin_seconds=0.75).values, (0, 0))
        self.assertEqual(empty.fps(kind='instantaneous').filter_outliers().included, ())

    def test_loading_preserves_latency_and_ignores_historical_extra_fields(self):
        raw = saved_run()
        raw['repeat'] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'worker'
            path.mkdir()
            file = path / 'run-0000.json'
            file.write_text(json.dumps(raw))
            run, = load_measurement_runs(directory)
            self.assertEqual(json.loads(file.read_text()), raw)
        self.assertEqual(run.completed_at, (0.5, 1, 5, 6, 10))
        self.assertEqual(run.processing_fps, raw['summary']['processing_fps'])
        self.assertEqual((run.model_id, run.device, run.mode, run.input_fps), ('test', 'cpu', 'stream', 30))
        selected = run.slice(5, 10)
        self.assertEqual(selected.latency().times, (5, 6, 10))
        self.assertEqual(selected.latency().values, (12, 13, 14))
        self.assertEqual(selected.latency('frame_latency_ms').values, (22, 23, 24))
        raw['records'][2]['prediction_ms'] = 999
        self.assertEqual(run.records[2]['prediction_ms'], 12)
        with self.assertRaises(TypeError):
            run.records[0]['prediction_ms'] = 999

    def test_construction_and_transformations_do_not_mutate_inputs(self):
        timestamps = [1, 2, 3, 4, 4.01, 5.01, 6.01]
        run = MeasurementRun(timestamps, end_s=7)
        original = run.fps(kind='instantaneous')
        filtered = original.filter_outliers()
        self.assertEqual(filtered.included, (True, True, True, False, True, True))
        self.assertEqual(filtered.values, original.values)
        self.assertEqual(filtered.times, original.times)
        self.assertTrue(math.isnan(filtered.plot_values[3]))
        self.assertEqual(original.included, (True,) * 6)
        self.assertEqual(run.processing_fps, 1)
        timestamps.append(6.5)
        self.assertEqual(len(run.completed_at), 7)
        with self.assertRaises(FrozenInstanceError):
            run.end_s = 8

    def test_bounds_are_inclusive_and_filters_compose(self):
        series = MeasurementSeries([1, 2, 3, 4], [1, 2, 3, 100], 0, 5)
        bounded = series.filter_outliers(method='bounds', minimum=2, maximum=3)
        self.assertEqual(bounded.included, (False, True, True, False))
        again = bounded.filter_outliers(method='bounds', minimum=0, maximum=100)
        self.assertEqual(again.included, bounded.included)
        self.assertEqual(bounded.filter_outliers().included, bounded.included)
        self.assertEqual(series.filter_outliers(method='bounds', maximum=0).filter_outliers().included,
                         (False,) * 4)

    def test_iqr_zero_spread_and_single_observation(self):
        series = MeasurementSeries([1, 2, 3, 4, 5], [2, 2, 2, 2, 100], 0, 6)
        self.assertEqual(series.filter_outliers().included, (True, True, True, True, False))
        single = MeasurementSeries([1], [2], 0, 2)
        self.assertEqual(single.filter_outliers().included, (True,))

    def test_series_slice_retains_values_and_mask(self):
        points = MeasurementSeries([1, 2, 3], [10, 20, 30], 0, 4)
        selected = points.filter_outliers(method='bounds', maximum=10).slice(1, 3)
        self.assertEqual(selected.times, (1, 2))
        self.assertEqual(selected.values, (10, 20))
        self.assertEqual(selected.included, (True, False))
        run = MeasurementRun([0.1, 0.8, 1.5, 2], end_s=3)
        series = run.fps().filter_outliers(method='bounds', maximum=1).slice(0.5, 2.5)
        self.assertEqual(series.bin_edges, (0.5, 1, 2, 2.5))
        self.assertEqual(series.times, (0.75, 1.5, 2.25))
        self.assertEqual(series.values, (2, 1, 1))
        self.assertEqual(series.included, (False, True, True))
        self.assertEqual(run.slice(0.5, 2.5).fps().values, (1, 2))

    def test_invalid_runs_and_windows(self):
        for times, end in [([2, 1], 3), ([1, 1], 3), ([math.nan], 3),
                           ([math.inf], 3), ([-1], 3), ([4], 3), ([], 0), ([], math.inf)]:
            with self.subTest(times=times, end=end), self.assertRaises(ValueError):
                MeasurementRun(times, end_s=end)
        run = MeasurementRun([1, 2], end_s=3)
        for start, end in [(1, 1), (2, 1), (-1, 1), (0, 4), (math.nan, 2)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                run.slice(start, end)
        for width in [0, -1, math.nan, math.inf]:
            with self.assertRaises(ValueError):
                run.fps(bin_seconds=width)
        with self.assertRaises(ValueError):
            run.fps(kind='unknown')

    def test_invalid_series_and_filters(self):
        for times, values in [([1], []), ([1], [math.inf]), ([2, 1], [1, 2])]:
            with self.assertRaises(ValueError):
                MeasurementSeries(times, values, 0, 3)
        with self.assertRaises(ValueError):
            MeasurementSeries([1], [2], 0, 3, bin_edges=(0, 1))
        series = MeasurementSeries([1], [2], 0, 3)
        for options in [dict(method='unknown'), dict(factor=-1), dict(factor=math.nan),
                        dict(method='bounds'), dict(method='bounds', minimum=math.inf),
                        dict(method='bounds', minimum=3, maximum=2), dict(minimum=1)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                series.filter_outliers(**options)


class MeasurementPlotsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')

    def setUp(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.addCleanup(plt.close, 'all')

    def test_point_filter_has_gaps_and_original_time_axis(self):
        run = MeasurementRun([5, 6, 6.01, 7.01], start_s=5, end_s=10)
        series = run.fps(kind='instantaneous')
        filtered = series.filter_outliers(method='bounds', maximum=10)
        _, ax = self.plt.subplots()
        self.assertIs(plot_series([series, filtered], ax=ax), ax)
        self.assertTrue(math.isnan(ax.lines[1].get_ydata()[1]))
        np.testing.assert_equal(ax.lines[1].get_xdata(), [6, 6.01, 7.01])
        self.assertEqual(ax.get_xlim(), (5, 10))
        ax.figure.canvas.draw()

    def test_binned_filters_empty_and_single_runs_plot(self):
        run = MeasurementRun([0.5, 2.5], end_s=3)
        with patch.object(self.plt, 'show'):
            plot_series(run).figure.canvas.draw()
            series = run.fps().filter_outliers(method='bounds', maximum=0)
            ax = plot_series(series)
            np.testing.assert_equal(ax.patches[0].get_data().values, [math.nan, 0, math.nan])
            ax.figure.canvas.draw()
            plot_series(run.slice(1, 2).fps(kind='instantaneous')).figure.canvas.draw()
            plot_series([]).figure.canvas.draw()
        with self.assertRaises(ValueError):
            plot_series([run.fps(), run.latency()])

    def test_directory_and_object_plot_entrypoints(self):
        raw = saved_run()
        with tempfile.TemporaryDirectory() as directory, patch.object(self.plt, 'show'):
            (Path(directory) / 'run-0000.json').write_text(json.dumps(raw))
            run, = load_measurement_runs(directory)
            for source in (directory, run.slice(5, 10), [run], []):
                plot_fps_over_time(source)
                plot_latency_over_time(source)
            for number in self.plt.get_fignums():
                self.plt.figure(number).canvas.draw()


if __name__ == '__main__':
    unittest.main()
