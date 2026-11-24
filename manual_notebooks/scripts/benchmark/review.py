"""Обзор последних сохранённых замеров по всем моделям."""
from dataclasses import dataclass
from itertools import groupby
import json
import math
from pathlib import Path
import warnings

from .measurement import MeasurementRun


@dataclass
class SavedMeasurement:
    path: Path
    run: MeasurementRun
    summary: dict
    manifest: dict
    memory: list
    repeat: int | None

    @property
    def model_device(self):
        return self.run.model_id, self.run.device

    @property
    def mode_label(self):
        if self.run.mode == 'direct':
            return 'Direct'
        return f'Stream {self.run.input_fps:g} FPS'


def _read_measurement(path):
    raw = json.loads(path.read_text(encoding='utf-8'))
    run = MeasurementRun.from_dict(raw)
    if not run.model_id or not run.device or run.mode not in ('direct', 'stream'):
        raise ValueError('Missing model, device or valid mode')
    if run.mode == 'stream' and not (math.isfinite(run.input_fps) and run.input_fps > 0):
        raise ValueError('Invalid stream rate')
    summary = raw['summary']
    if not math.isfinite(summary['processing_fps']):
        raise ValueError('Invalid processing FPS')
    origin = raw['initial']['timestamp']
    memory = [(sample['timestamp'] - origin, sample) for sample in raw.get('memory_records', [])
              if origin <= sample['timestamp'] <= raw['final']['timestamp']]
    return SavedMeasurement(path, run, summary, {}, memory, raw.get('repeat', raw.get('repetition')))


def latest_measurements(root):
    """Выбрать последние записанные результаты, включая завершённые задания прерванных сессий."""
    selected = {}
    # UTC в имени сессии, затем порядковые номера worker и run в порядке записи.
    for path in sorted(Path(root).glob('????????T??????Z-*/*/run-*.json')):
        try:
            item = _read_measurement(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.warn(f'Skipping measurement {path}: {exc}', stacklevel=2)
            continue
        selected[(*item.model_device, item.run.mode, item.run.input_fps)] = item

    for item in selected.values():
        path = item.path.with_name('manifest.json')
        try:
            item.manifest = dict(json.loads(path.read_text(encoding='utf-8')))
        except (OSError, TypeError, ValueError) as exc:
            warnings.warn(f'Unavailable worker metadata {path}: {exc}', stacklevel=2)
    return [selected[key] for key in sorted(selected)]


def simple_table(measurements):
    """Сопоставить FPS и пиковую память отдельного прохода рядом для каждого режима."""
    import pandas as pd

    rows = {}
    columns = {}
    for item in measurements:
        row = rows.setdefault(item.model_device, {})
        label = item.mode_label
        metrics = {'FPS': item.summary['processing_fps']}
        fields = {'Peak RAM (MiB)': 'probe_rss_sampled_peak_bytes'}
        if item.run.device.startswith('cuda'):
            fields['Peak GPU allocated (MiB)'] = 'probe_cuda_allocated_bytes_peak'
        for name, field in fields.items():
            value = item.summary.get(field)
            metrics[name] = None if value is None else value / 2**20
        for index, (name, value) in enumerate(metrics.items()):
            column = (label, name)
            columns[column] = (item.run.mode, item.run.input_fps, index)
            row[column] = value
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame.from_dict(rows, orient='index')
    table.index.names = ['Model', 'Device']
    return table.reindex(columns=sorted(columns, key=columns.get)).sort_index()


def full_table(measurements):
    """Показать сводку, настройки, происхождение входов и путь к каждому замеру."""
    import pandas as pd

    rows = []
    for item in measurements:
        rows.append(dict(model_id=item.run.model_id, device=item.run.device, mode=item.run.mode,
                         input_fps=item.run.input_fps, **item.summary))
        rows[-1].update(source=str(item.path), saved_repeat=item.repeat)
        rows[-1].update({f'worker_{key}': value for key, value in item.manifest.items()
                         if key not in ('model_id', 'device')})
    return pd.DataFrame(rows)


def _plot_column(axes, item):
    fps, latency, memory = axes
    series = item.run.fps()
    fps.stairs(series.values, series.bin_edges)
    for key, label, style in [('prediction_ms', 'Prediction', '-'),
                              ('frame_latency_ms', 'Frame incl. queue', '--')]:
        if key == 'frame_latency_ms' and item.run.mode == 'direct':
            continue
        series = item.run.latency(key)
        latency.plot(series.times, series.values, style, marker='.', markersize=2, label=label)
    latency.legend()
    for field, label in [('rss_bytes', 'Process RSS'), ('cuda_allocated_bytes', 'GPU allocated'),
                         ('cuda_reserved_bytes', 'GPU reserved')]:
        points = [(t, sample[field] / 2**20) for t, sample in item.memory
                  if sample.get(field) is not None]
        if points:
            memory.plot(*zip(*points), label=label, marker='.', markersize=2)
    if memory.lines:
        memory.legend()
    else:
        memory.text(.5, .5, 'No timing-window samples', transform=memory.transAxes, ha='center')


def comparison_figures(measurements):
    """Для каждой частоты потока сопоставить модели на одинаковых шкалах."""
    import matplotlib.pyplot as plt

    for (model, device), group in groupby(sorted(measurements, key=lambda m: m.model_device),
                                          key=lambda m: m.model_device):
        modes = list(group)
        direct = next((m for m in modes if m.run.mode == 'direct'), None)
        streams = sorted((m for m in modes if m.run.mode == 'stream'), key=lambda m: m.run.input_fps)
        for stream in streams or [None]:
            fig, axes = plt.subplots(3, 2, sharex=True, sharey='row', figsize=(12, 8),
                                     layout='constrained')
            fig.suptitle(f'{model} / {device}')
            for column, (name, item) in enumerate([('Direct', direct), ('Stream', stream)]):
                axes[0, column].set_title(item.mode_label if item else f'{name} (unmeasured)')
                if item:
                    _plot_column(axes[:, column], item)
                axes[2, column].set_xlabel('Seconds since measurement start')
            for row, label in zip(axes, ['Processing FPS', 'Latency (ms)', 'Timing-window memory (MiB)']):
                row[0].set_ylabel(label)
                row[0].set_ylim(bottom=0)
                for ax in row:
                    ax.grid(alpha=.25)
            axes[0, 0].set_xlim(0, max(m.run.duration_s for m in (direct, stream) if m))
            yield fig
