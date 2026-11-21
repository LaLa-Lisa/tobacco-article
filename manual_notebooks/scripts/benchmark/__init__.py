"""Публичный интерфейс измерений FPS, задержки и памяти."""
from .core import BenchmarkConfig, load_runs, run_benchmarks

__all__ = ['BenchmarkConfig', 'load_runs', 'run_benchmarks']
