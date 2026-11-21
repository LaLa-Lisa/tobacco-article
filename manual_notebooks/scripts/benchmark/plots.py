"""Таблицы и графики сохранённых измерений; предсказания здесь не запускаются."""
from pathlib import Path

from .core import load_runs, positive
from .measurement import MeasurementRun, MeasurementSeries, load_measurement_runs


def completion_fps_bins(run, bin_seconds=1.0):
    """Совместимый интерфейс для словарей и объектов замеров."""
    import numpy as np
    if not isinstance(run, MeasurementRun):
        run = MeasurementRun.from_dict(run)
    series = run.fps(bin_seconds=bin_seconds)
    return np.asarray(series.bin_edges), np.asarray(series.values)


def _measurement_runs(source):
    if isinstance(source, (str, Path)):
        return load_measurement_runs(source)
    if isinstance(source, MeasurementRun):
        return [source]
    return list(source)


def plot_series(series, *, ax=None, kind='binned', bin_seconds=1.0):
    """Показать замеры или готовые ряды; вернуть Axes для дальнейшей настройки."""
    import matplotlib.pyplot as plt
    if isinstance(series, (MeasurementRun, MeasurementSeries)):
        series = [series]
    series = [value.fps(kind=kind, bin_seconds=bin_seconds)
              if isinstance(value, MeasurementRun) else value for value in series]
    if any(not isinstance(value, MeasurementSeries) for value in series):
        raise TypeError('Expected MeasurementRun or MeasurementSeries objects')
    units = {value.unit for value in series}
    if len(units) > 1:
        raise ValueError('Plot series with different units on separate axes')
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    for value in series:
        if value.bin_edges is None:
            ax.plot(value.times, value.plot_values, marker='.', markersize=3,
                    linewidth=1, label=value.label)
        else:
            ax.stairs(value.plot_values, value.bin_edges, baseline=None, label=value.label)
    ax.set(xlabel='Time since measurement start (s)', ylabel=next(iter(units), 'FPS'))
    ax.grid(alpha=0.25)
    if series:
        ax.set_xlim(min(value.start_s for value in series), max(value.end_s for value in series))
        ax.legend()
    if show:
        ax.figure.tight_layout()
        plt.show()
    return ax


def plot_fps_over_time(directory, bin_seconds=1.0):
    """Отдельная фигура для модели/устройства, панели по режиму и входной частоте."""
    import matplotlib.pyplot as plt
    positive(bin_seconds, 'bin_seconds')
    runs = _measurement_runs(directory)
    if not runs:
        print('No completed runs found')
        return
    groups = {}
    for run in runs:
        groups.setdefault((run.model_id, run.device), {}).setdefault(
            (run.mode, run.input_fps), []).append(run)
    for (model_id, device), modes in groups.items():
        fig, axes = plt.subplots(len(modes), 1, figsize=(11, 3.5 * len(modes)), squeeze=False)
        for ax, ((mode, rate), measurements) in zip(axes[:, 0], modes.items()):
            for index, run in enumerate(measurements, 1):
                edges, fps = completion_fps_bins(run, bin_seconds)
                ax.stairs(fps, edges, baseline=None, label="Processing FPS" if len(measurements) == 1 else f"Run {index}")
            title = 'Direct inference' if mode == 'direct' else f'Synthetic stream — input {rate:g} FPS'
            ax.set(title=title, xlabel='Time since measurement start (s)', ylabel='Processing FPS')
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            ax.legend()
        fig.suptitle(f'{model_id} / {device} — FPS over time ({bin_seconds:g} s bins)')
        fig.tight_layout()
        plt.show()


def plot_latency_over_time(directory):
    """Задержки каждого кадра по времени завершения; без фильтрации и усреднения."""
    import matplotlib.pyplot as plt
    runs = _measurement_runs(directory)
    if not runs:
        print('No completed runs found')
        return
    groups = {}
    for run in runs:
        groups.setdefault((run.model_id, run.device), {}).setdefault(
            (run.mode, run.input_fps), []).append(run)
    for (model_id, device), modes in groups.items():
        fig, axes = plt.subplots(len(modes), 1, figsize=(11, 3.5 * len(modes)), squeeze=False)
        for ax, ((mode, rate), measurements) in zip(axes[:, 0], modes.items()):
            for index, run in enumerate(measurements, 1):
                prediction = run.latency()
                line, = ax.plot(prediction.times, prediction.values,
                                marker='.', markersize=3, linewidth=1,
                                label="Prediction" if len(measurements) == 1 else f"Prediction — run {index}")
                if mode == 'stream':
                    frame = run.latency('frame_latency_ms')
                    ax.plot(frame.times, frame.values,
                            color=line.get_color(), linestyle='--', linewidth=1,
                            marker='.', markersize=3,
                            label="Frame incl. queue" if len(measurements) == 1 else f"Frame incl. queue — run {index}")
            title = 'Direct inference' if mode == 'direct' else f'Synthetic stream — input {rate:g} FPS'
            ax.set(title=title, xlabel='Time since measurement start (s)', ylabel='Latency (ms)')
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            ax.legend()
        fig.suptitle(f'{model_id} / {device} — latency over time')
        fig.tight_layout()
        plt.show()


def summary_table(directory):
    import pandas as pd
    rows = [{**{key: run[key] for key in ('model_id', 'device', 'mode', 'input_fps')},
             **run['summary']} for run in load_runs(directory)]
    table = pd.DataFrame(rows)
    for key in list(table.columns):
        if 'bytes' in key:
            table[key.replace('bytes', 'mib')] = table.pop(key) / 2**20
    return table


def plot_benchmarks(directory):
    import matplotlib.pyplot as plt
    import numpy as np

    runs = load_runs(directory)
    if not runs:
        print('No completed runs found')
        return
    groups = {}
    for run in runs:
        key = (run['model_id'], run['device'], run['mode'], run['input_fps'])
        groups.setdefault(key, []).append(run)
    keys = list(groups)
    labels = [' / '.join(str(v) for v in key if v is not None) for key in keys]

    def comparison(ax, metric, label, scale=1):
        for index, key in enumerate(keys):
            values = [r['summary'].get(metric) for r in groups[key]]
            values = np.array([v / scale for v in values if v is not None])
            if not len(values):
                continue
            ax.scatter(index + (np.linspace(-0.05, 0.05, len(values)) if len(values) > 1 else 0), values, alpha=0.5)
        ax.set(xticks=range(len(keys)), xticklabels=labels, ylabel=label)
        ax.tick_params(axis='x', labelrotation=75)
        ax.set_title('Measured values')

    for metrics in [
        [('processing_fps', 'Processing FPS', 1), ('prediction_ms_median', 'Median prediction (ms)', 1),
         ('prediction_ms_p95', 'P95 prediction (ms)', 1)],
        [('rss_sampled_peak_bytes', 'Sampled peak process RSS (MiB)', 2**20),
         ('cuda_allocated_bytes_peak', 'Peak CUDA allocated (MiB)', 2**20),
         ('cuda_reserved_bytes_peak', 'Peak CUDA reserved (MiB)', 2**20)],
    ]:
        fig, axes = plt.subplots(1, len(metrics), figsize=(17, 6))
        for ax, (metric, label, scale) in zip(axes, metrics):
            comparison(ax, metric, label, scale)
        fig.tight_layout()
        plt.show()
    if any(r['mode'] == 'stream' for r in runs):
        fig, axes = plt.subplots(1, 3, figsize=(17, 6))
        for ax, metric, label in zip(axes, ('generation_fps', 'drop_rate', 'frame_latency_ms_p95'),
                                     ('Actual producer FPS', 'Replaced frames/s', 'P95 frame latency (ms)')):
            comparison(ax, metric, label)
        fig.tight_layout()
        plt.show()
    for mode in ('direct', 'stream'):
        selected = [r for r in runs if r['mode'] == mode]
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        metric = 'prediction_ms' if mode == 'direct' else 'frame_latency_ms'
        for run in selected:
            values = np.sort([row[metric] for row in run['records']])
            label = f"{run['model_id']} / {run['device']} / {run['input_fps']}"
            ax.step(np.r_[values[0], values], np.r_[0, np.arange(1, len(values) + 1) / len(values)],
                    where='post', alpha=0.7, label=label)
        ax.set(xlabel='Prediction latency (ms)' if mode == 'direct' else 'Frame latency (ms)',
               ylabel='Fraction of completed frames', title=f'{mode.title()} latency ECDF — all observations')
        ax.legend(fontsize='x-small')
        fig.tight_layout()
        plt.show()
