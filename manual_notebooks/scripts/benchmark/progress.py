"""Редкое обновление интерфейса в родительском процессе, вне измерительного цикла."""
import json
import time


class BenchmarkProgress:
    """Считает сохранённые замеры; время стадии служит только подсказкой интерфейса."""

    def __init__(self, total, bar_factory=None):
        if bar_factory is None:
            from tqdm.auto import tqdm
            bar_factory = tqdm
        self.bar = bar_factory(total=total, desc='Benchmarks', unit='run', mininterval=1.0,
                               bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                                          '[{elapsed}{postfix}]')
        self.completed = 0
        self.worker_completed = 0

    def start_worker(self, directory, model_id, device):
        self.directory = directory
        self.label = f'{model_id} / {device}'
        self.completed += self.worker_completed
        self.worker_completed = 0
        self.bar.set_postfix_str(f'{self.label} / Loading')

    def refresh(self):
        try:
            status = json.loads((self.directory / 'progress.json').read_text(encoding='utf-8'))
        except FileNotFoundError:
            self.bar.refresh()
            return
        self.worker_completed = status['completed']
        # Абсолютный счётчик и одна перерисовка, без дополнительного refresh из update.
        self.bar.n = self.completed + self.worker_completed
        elapsed = max(0, time.monotonic() - status['stage_started'])
        self.bar.set_postfix_str(
            f"{self.label} / {status['mode']} / {status['stage']} ({elapsed:.0f}s)")

    def close(self, status):
        self.bar.set_postfix_str(status.capitalize())
        self.bar.close()
