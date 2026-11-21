"""Неизменяемые измерения и производные ряды без зависимостей от SDK и графиков."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


def _window(start: float, end: float) -> None:
    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
        raise ValueError('Window must satisfy 0 <= start < end with finite bounds')


def _slice_window(start: float, end: float, lower: float, upper: float) -> None:
    _window(start, end)
    if start < lower or end > upper:
        raise ValueError('Slice must lie within the current measurement window')


def _contains(time: float, start: float, end: float, include_end: bool) -> bool:
    return start <= time and (time < end or (include_end and time == end))


@dataclass(frozen=True)
class MeasurementSeries:
    """Ряд значений; маска исключения не удаляет наблюдения и не меняет их время."""

    times: tuple[float, ...]
    values: tuple[float, ...]
    start_s: float
    end_s: float
    label: str = 'Processing FPS'
    unit: str = 'FPS'
    bin_edges: tuple[float, ...] | None = None
    included: tuple[bool, ...] | None = None
    include_end: bool = True

    def __post_init__(self) -> None:
        _window(self.start_s, self.end_s)
        times, values = tuple(self.times), tuple(self.values)
        if len(times) != len(values):
            raise ValueError('Times and values must have equal lengths')
        if not all(math.isfinite(t) and _contains(t, self.start_s, self.end_s, self.include_end)
                   for t in times):
            raise ValueError('Series times must be finite and inside the window')
        if any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError('Series times must be strictly increasing')
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Series values must be finite')
        included = (True,) * len(times) if self.included is None else tuple(self.included)
        if len(included) != len(times) or any(not isinstance(value, bool) for value in included):
            raise ValueError('Inclusion mask must contain one boolean per observation')
        edges = None if self.bin_edges is None else tuple(self.bin_edges)
        if edges is not None:
            if (len(edges) != len(values) + 1 or not values
                    or not all(math.isfinite(edge) for edge in edges)
                    or edges[0] != self.start_s or edges[-1] != self.end_s
                    or any(a >= b for a, b in zip(edges, edges[1:]))
                    or any(not a <= t < b for a, t, b in zip(edges, times, edges[1:]))):
                raise ValueError('Bin edges must cover the window and enclose each observation')
        object.__setattr__(self, 'times', times)
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'included', included)
        object.__setattr__(self, 'bin_edges', edges)

    @property
    def plot_values(self) -> tuple[float, ...]:
        """Исключённые точки представлены NaN, чтобы график показывал разрывы."""
        return tuple(value if keep else math.nan for value, keep in zip(self.values, self.included))

    def slice(self, start: float, end: float) -> MeasurementSeries:
        """Обрезать отображение; значения готовых интервалов не пересчитываются."""
        _slice_window(start, end, self.start_s, self.end_s)
        include_end = self.include_end and end == self.end_s
        if self.bin_edges is not None:
            indices = [i for i, (a, b) in enumerate(zip(self.bin_edges, self.bin_edges[1:]))
                       if a < end and b > start]
            edges = (start, *(edge for edge in self.bin_edges if start < edge < end), end)
            times = tuple((a + b) / 2 for a, b in zip(edges, edges[1:]))
        else:
            indices = [i for i, t in enumerate(self.times) if _contains(t, start, end, include_end)]
            edges = None
            times = tuple(self.times[i] for i in indices)
        return replace(self, times=times, values=tuple(self.values[i] for i in indices),
                       included=tuple(self.included[i] for i in indices), bin_edges=edges,
                       start_s=start, end_s=end, include_end=include_end)

    def filter_outliers(self, *, method: str = 'iqr', factor: float = 1.5,
                        minimum: float | None = None, maximum: float | None = None) -> MeasurementSeries:
        """Исключить выбросы по IQR или включительным границам; исходный ряд сохранён."""
        if method == 'iqr':
            if minimum is not None or maximum is not None:
                raise ValueError('Use method="bounds" with minimum or maximum')
            if not math.isfinite(factor) or factor < 0:
                raise ValueError('IQR factor must be finite and nonnegative')
            import numpy as np
            active = [value for value, keep in zip(self.values, self.included) if keep]
            if active:
                q1, q3 = np.percentile(active, [25, 75])
                minimum, maximum = q1 - factor * (q3 - q1), q3 + factor * (q3 - q1)
        elif method == 'bounds':
            if minimum is None and maximum is None:
                raise ValueError('Provide minimum or maximum for bounds filtering')
            if any(value is not None and not math.isfinite(value) for value in (minimum, maximum)):
                raise ValueError('Filter bounds must be finite')
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError('Minimum must not exceed maximum')
        else:
            raise ValueError('Filter method must be "iqr" or "bounds"')
        included = tuple(keep and (minimum is None or value >= minimum)
                         and (maximum is None or value <= maximum)
                         for value, keep in zip(self.values, self.included))
        return replace(self, included=tuple(bool(keep) for keep in included),
                       label=f'{self.label} — filtered ({method})')


@dataclass(frozen=True)
class MeasurementRun:
    """Один замер модели; времена завершения заданы в секундах от начала замера."""

    completed_at: tuple[float, ...]
    end_s: float
    start_s: float = 0.0
    model_id: str = ''
    device: str = ''
    mode: str = 'direct'
    input_fps: float | None = None
    repeat: int | None = None
    records: tuple[Mapping[str, object], ...] = field(default=(), repr=False)
    include_end: bool = True

    def __post_init__(self) -> None:
        _window(self.start_s, self.end_s)
        times = tuple(self.completed_at)
        if not all(math.isfinite(t) and _contains(t, self.start_s, self.end_s, self.include_end)
                   for t in times):
            raise ValueError('Completion times must be finite and inside the measurement window')
        if any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError('Completion times must be strictly increasing')
        if self.records and len(self.records) != len(times):
            raise ValueError('Records must match completion times')
        records = tuple(MappingProxyType(dict(record)) for record in self.records)
        object.__setattr__(self, 'completed_at', times)
        object.__setattr__(self, 'records', records)

    @classmethod
    def from_dict(cls, run: Mapping[str, object]) -> MeasurementRun:
        """Прочитать существующий JSON-формат без изменения исходных записей."""
        origin = run['initial']['timestamp']
        end = run['final']['timestamp']
        if not math.isfinite(origin) or not math.isfinite(end):
            raise ValueError('Measurement boundaries must be finite')
        records = tuple(run['records'])
        return cls(completed_at=tuple(row['completed_at'] - origin for row in records),
                   end_s=end - origin, records=records,
                   **{key: run[key] for key in ('model_id', 'device', 'mode', 'input_fps', 'repeat') if key in run})

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def processing_fps(self) -> float:
        """Пропускная способность выбранного окна, включая периоды без завершений."""
        return len(self.completed_at) / self.duration_s

    @property
    def label(self) -> str:
        parts = [self.model_id, self.device, self.mode]
        if self.input_fps is not None:
            parts.append(f'input {self.input_fps:g} FPS')
        if self.repeat is not None:
            parts.append(f'repeat {self.repeat}')
        return ' / '.join(part for part in parts if part)

    def slice(self, start: float, end: float) -> MeasurementRun:
        """Выбрать [start, end); исходная конечная граница включается."""
        _slice_window(start, end, self.start_s, self.end_s)
        include_end = self.include_end and end == self.end_s
        indices = [i for i, t in enumerate(self.completed_at) if _contains(t, start, end, include_end)]
        return replace(self, completed_at=tuple(self.completed_at[i] for i in indices),
                       records=tuple(self.records[i] for i in indices) if self.records else (),
                       start_s=start, end_s=end, include_end=include_end)

    def fps(self, *, kind: str = 'binned', bin_seconds: float = 1.0) -> MeasurementSeries:
        """FPS по окнам либо обратный интервал между соседними завершениями."""
        import numpy as np
        if kind == 'binned':
            if not math.isfinite(bin_seconds) or bin_seconds <= 0:
                raise ValueError('Bin width must be finite and positive')
            edges = np.append(self.start_s + np.arange(0, self.duration_s, bin_seconds), self.end_s)
            counts, _ = np.histogram(self.completed_at, bins=edges)
            times = (edges[:-1] + edges[1:]) / 2
            values = counts / np.diff(edges)
            label = f'{self.label} — Processing FPS ({bin_seconds:g} s bins)'
        elif kind == 'instantaneous':
            edges = None
            times = self.completed_at[1:]
            values = 1 / np.diff(self.completed_at)
            label = f'{self.label} — Inter-completion FPS'
        else:
            raise ValueError('FPS kind must be "binned" or "instantaneous"')
        return MeasurementSeries(times=tuple(times), values=tuple(values),
                                 bin_edges=None if edges is None else tuple(edges),
                                 start_s=self.start_s, end_s=self.end_s,
                                 include_end=self.include_end, label=label)

    def latency(self, metric: str = 'prediction_ms') -> MeasurementSeries:
        """Сохранённые задержки; отсутствие записей даёт пустой ряд."""
        if metric not in ('prediction_ms', 'frame_latency_ms'):
            raise ValueError('Latency metric must be prediction_ms or frame_latency_ms')
        points = [(time, row[metric]) for time, row in zip(self.completed_at, self.records)
                  if row.get(metric) is not None]
        return MeasurementSeries(times=tuple(t for t, _ in points), values=tuple(v for _, v in points),
                                 start_s=self.start_s, end_s=self.end_s, include_end=self.include_end,
                                 label=f'{self.label} — {metric}', unit='ms')


def load_measurement_runs(directory: str | Path) -> list[MeasurementRun]:
    """Загрузить завершённые замеры, сохранив совместимость с прежними JSON."""
    from .core import load_runs
    return [MeasurementRun.from_dict(run) for run in load_runs(directory)]
