"""Измерение предсказаний и синтетического потока без зависимости от SDK провайдера."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
import uuid

from .progress import BenchmarkProgress


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup_frames: int = 30
    duration_s: float = 30.0
    input_rates: tuple[float, ...] = (10, 20, 30, 60, 300)
    direct: bool = True
    stream: bool = False
    seed: int = 20261028
    memory_interval_s: float = 0.01
    repeats: int = 1
    memory_passes: int = 0
    torch_num_threads: int | None = None

    def validate(self):
        if type(self.warmup_frames) is not int or self.warmup_frames < 1:
            raise ValueError('warmup_frames must be a positive integer')
        for name, minimum in (('repeats', 1), ('memory_passes', 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.torch_num_threads is not None and (
                type(self.torch_num_threads) is not int or self.torch_num_threads < 1):
            raise ValueError('torch_num_threads must be a positive integer or None')
        for name in ('duration_s', 'memory_interval_s'):
            positive(getattr(self, name), name)
        if not self.direct and not self.stream:
            raise ValueError('Enable at least one benchmark mode')
        if self.stream and not self.input_rates:
            raise ValueError('Stream mode requires input_rates')
        for rate in self.input_rates:
            positive(rate, 'input_fps')
        if len(set(self.input_rates)) != len(self.input_rates):
            raise ValueError('input_rates must be unique')


def positive(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


class LatestFrameStream:
    """Очередь из одного кадра; непрочитанный кадр заменяется новым."""

    def __init__(self, images, fps):
        positive(fps, 'input_fps')
        if not images:
            raise ValueError('Provide at least one image')
        self.images, self.fps = images, fps
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.pending = None
        self.error = None
        self.generated = self.consumed = self.dropped = 0
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.thread.start()

    def _publish(self, frame, generated_at):
        with self.condition:
            if self.stop.is_set():
                return
            self.generated += 1
            if self.pending is not None:
                self.dropped += 1
            self.pending = (frame, self.generated, generated_at)
            self.condition.notify_all()

    def _produce(self):
        try:
            next_time = time.perf_counter()
            index = 0
            while not self.stop.wait(max(0, next_time - time.perf_counter())):
                generated_at = time.perf_counter()
                frame = self.images[index % len(self.images)].copy()
                self._publish(frame, generated_at)
                index += 1
                next_time = max(next_time + 1 / self.fps, time.perf_counter())
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def read(self):
        with self.condition:
            self.condition.wait_for(lambda: self.pending is not None or self.error or self.stop.is_set())
            if self.error:
                raise RuntimeError('Synthetic frame producer failed') from self.error
            if self.stop.is_set():
                raise StopIteration
            value, self.pending = self.pending, None
            self.consumed += 1
            return value

    def snapshot(self):
        with self.condition:
            return dict(timestamp=time.perf_counter(), generated=self.generated,
                        consumed=self.consumed, dropped=self.dropped)

    def close(self):
        with self.condition:
            self.stop.set()
            self.condition.notify_all()
        self.thread.join()


class DeviceMetrics:
    """Синхронизация и статистика аллокатора только выбранного устройства."""

    def __init__(self, device):
        import torch
        self.torch = torch
        parsed = torch.device(device)
        if parsed.type not in {'cpu', 'cuda'}:
            raise ValueError('Only cpu and cuda:N are supported')
        if parsed.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError(f'CUDA is unavailable: {device}')
            parsed = torch.device('cuda', parsed.index if parsed.index is not None else 0)
            torch.cuda.set_device(parsed)
        self.device = str(parsed)
        self.cuda = parsed.type == 'cuda'

    def synchronize(self):
        if self.cuda:
            self.torch.cuda.synchronize(self.device)

    def reset(self):
        self.synchronize()
        if self.cuda:
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def memory(self, peak=False):
        if not self.cuda:
            return {'cuda_allocated_bytes': None, 'cuda_reserved_bytes': None}
        api = self.torch.cuda
        allocated = api.max_memory_allocated if peak else api.memory_allocated
        reserved = api.max_memory_reserved if peak else api.memory_reserved
        return dict(cuda_allocated_bytes=allocated(self.device),
                    cuda_reserved_bytes=reserved(self.device))


class MemorySampler:
    """RSS рабочего процесса: выборочный пик, не точный максимум между отсчётами."""

    def __init__(self, interval_s, read_rss=None, *, retain_records=True):
        positive(interval_s, 'memory_interval_s')
        if read_rss is None:
            import psutil
            process = psutil.Process()
            read_rss = lambda: process.memory_info().rss
        self.read_rss = read_rss
        self.interval = interval_s
        self.records = []
        self.retain_records = retain_records
        self.first = self.last = self.peak = None
        self.sample_count = 0
        self.max_gap_s = 0.0
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def capture(self):
        record = dict(timestamp=time.perf_counter(), rss_bytes=self.read_rss())
        if self.first is None:
            self.first = record
        if self.last is not None:
            self.max_gap_s = max(self.max_gap_s, record['timestamp'] - self.last['timestamp'])
        if self.peak is None or record['rss_bytes'] > self.peak['rss_bytes']:
            self.peak = record
        self.last = record
        self.sample_count += 1
        if self.retain_records:
            self.records.append(record)

    def _sample(self):
        try:
            while not self.stop.wait(self.interval):
                self.capture()
        except Exception as exc:
            self.error = exc

    def start(self):
        self.capture()
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join()
        if self.error:
            raise RuntimeError('Memory sampler failed') from self.error
        self.capture()


def summarize(records, elapsed_s):
    import numpy as np
    if not records:
        raise ValueError('No completed predictions')
    positive(elapsed_s, 'elapsed_s')
    summary = dict(processed=len(records), elapsed_s=elapsed_s,
                   processing_fps=len(records) / elapsed_s)
    for key in ('prediction_ms', 'frame_latency_ms'):
        values = [row[key] for row in records if row.get(key) is not None]
        if values:
            summary.update({f'{key}_{label}': float(value) for label, value in zip(
                ('mean', 'median', 'p95', 'p99'),
                (np.mean(values), *np.percentile(values, [50, 95, 99])))})
    return summary


def measure_memory(predictor, images, metrics, config, *, options=None, prompts=None,
                   sampler_factory=MemorySampler):
    """Одинаковое число проходов по изображениям; память записей не растёт с FPS."""
    metrics.reset()
    warm_memory = metrics.memory()
    sampler = sampler_factory(config.memory_interval_s, retain_records=False)
    sampler.start()
    try:
        for _ in range(config.memory_passes):
            for image in images:
                result = predictor.predict(image, prompts=prompts, **(options or {}))
                metrics.synchronize()
                del result
        peak_memory = metrics.memory(peak=True)
    finally:
        sampler.close()
    summary = dict(rss_warm_bytes=sampler.first['rss_bytes'],
                   rss_sampled_peak_bytes=sampler.peak['rss_bytes'])
    summary.update({f'{key}_warm': value for key, value in warm_memory.items()})
    summary.update({f'{key}_peak': value for key, value in peak_memory.items()})
    return dict(scope='fixed_workload_direct', frames=len(images) * config.memory_passes,
                image_count=len(images), passes=config.memory_passes,
                sample_count=sampler.sample_count, max_sample_gap_s=sampler.max_gap_s,
                first=sampler.first, last=sampler.last, peak=sampler.peak, summary=summary)


def measure_run(predictor, images, metrics, config, *, input_fps=None,
                options=None, prompts=None, sampler_factory=MemorySampler,
                on_measurement_start=None, on_memory_start=None):
    """Прогрев и отдельное окно измерения; объекты предсказаний не накапливаются."""
    config.validate()
    if not images:
        raise ValueError('Provide at least one image')
    options = options or {}

    def warmup():
        for index in range(max(config.warmup_frames, len(images))):
            result = predictor.predict(images[index % len(images)], prompts=prompts, **options)
            metrics.synchronize()
            del result

    def check_threads():
        if config.torch_num_threads is not None and metrics.torch.get_num_threads() != config.torch_num_threads:
            raise RuntimeError('Provider changed the requested PyTorch thread count during inference')

    warmup()
    # SDK может изменить число потоков при ленивой загрузке. Повторяем прогрев
    # с запрошенным числом до прохода памяти и окна FPS.
    if config.torch_num_threads is not None and metrics.torch.get_num_threads() != config.torch_num_threads:
        metrics.torch.set_num_threads(config.torch_num_threads)
        warmup()
    check_threads()
    memory_probe = None
    if config.memory_passes:
        if on_memory_start is not None:
            on_memory_start()
        memory_probe = measure_memory(predictor, images, metrics, config,
                                      options=options, prompts=prompts)
        check_threads()
    # Уведомление выполняется до сброса пиков памяти и начала измеряемого окна.
    if on_measurement_start is not None:
        on_measurement_start()
    metrics.reset()
    warm_memory = metrics.memory()
    sampler = sampler_factory(config.memory_interval_s)
    stream = None
    records = []
    sampler.start()
    try:
        stream = LatestFrameStream(images, input_fps) if input_fps is not None else None
        initial = stream.snapshot() if stream else {'timestamp': time.perf_counter()}
        while True:
            if stream:
                frame, frame_id, generated_at = stream.read()
            else:
                frame_id = len(records)
                frame, generated_at = images[frame_id % len(images)], None
            metrics.synchronize()
            started = time.perf_counter()
            result = predictor.predict(frame, prompts=prompts, **options)
            metrics.synchronize()
            completed = time.perf_counter()
            del result
            records.append(dict(frame_id=frame_id, generated_at=generated_at,
                                started_at=started, completed_at=completed,
                                prediction_ms=(completed - started) * 1000,
                                frame_latency_ms=None if generated_at is None else
                                (completed - generated_at) * 1000))
            if completed - initial['timestamp'] >= config.duration_s:
                break
        final = stream.snapshot() if stream else {'timestamp': time.perf_counter()}
        peak_memory = metrics.memory(peak=True)
    finally:
        if stream:
            stream.close()
        sampler.close()
    check_threads()
    summary = summarize(records, final['timestamp'] - initial['timestamp'])
    summary.update(rss_warm_bytes=sampler.records[0]['rss_bytes'],
                   rss_sampled_peak_bytes=max(r['rss_bytes'] for r in sampler.records))
    summary.update({f'{key}_warm': value for key, value in warm_memory.items()})
    summary.update({f'{key}_peak': value for key, value in peak_memory.items()})
    if memory_probe is not None:
        summary.update({f'probe_{key}': value for key, value in memory_probe['summary'].items()})
    if stream:
        for name in ('generated', 'consumed', 'dropped'):
            summary[name] = final[name] - initial[name]
        summary['generation_fps'] = summary['generated'] / summary['elapsed_s']
        summary['drop_rate'] = summary['dropped'] / summary['elapsed_s']
    return dict(mode='stream' if stream else 'direct', input_fps=input_fps,
                initial=initial, final=final, records=records,
                memory_records=sampler.records, memory_probe=memory_probe, summary=summary)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def select_images(image_list, *, count=None, seed=20261028):
    """Фиксированная выборка из явного списка; относительные пути — от списка."""
    source = Path(image_list).resolve()
    paths = [Path(line.strip()) for line in source.read_text(encoding='utf-8').splitlines()
             if line.strip()]
    paths = [(path if path.is_absolute() else source.parent / path).resolve() for path in paths]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError('Image list must contain existing files; relative paths are relative to the list')
    if len(set(paths)) != len(paths):
        raise ValueError('Image list must not contain duplicates')
    if count is None:
        return paths
    if type(count) is not int or not 1 <= count <= len(paths):
        raise ValueError(f'Image count must be an integer between 1 and {len(paths)}')
    return random.Random(seed).sample(paths, count)


def write_json(path, value):
    """Замена через временный файл не оставляет частично записанный JSON."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_run(directory, index, run):
    write_json(Path(directory) / f'run-{index:04d}.json', run)
    export_summary(directory)


def load_runs(directory):
    return [json.loads(p.read_text(encoding='utf-8'))
            for p in sorted(Path(directory).rglob('run-*.json'))]


def export_summary(directory):
    rows = [{**{k: run[k] for k in ('model_id', 'device', 'mode', 'input_fps')},
             'repeat': run.get('repeat'),
             **run['summary']} for run in load_runs(directory)]
    write_json(Path(directory) / 'summary.json', rows)
    if rows:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        path = Path(directory) / 'summary.csv'
        temporary = path.with_suffix('.csv.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def make_schedule(config):
    rng = random.Random(config.seed)
    schedule = [None] if config.direct else []
    rates = list(config.input_rates) if config.stream else []
    rng.shuffle(rates)
    schedule.extend(rates)
    return schedule


def make_jobs(specs, devices, config):
    """Перемешивание внутри повторов; каждый режим запускается в свежем процессе."""
    rng = random.Random(config.seed)
    jobs = []
    for repeat in range(1, config.repeats + 1):
        block = [dict(model_id=item['spec']['id'], device=device, repeat=repeat, input_fps=rate)
                 for item in specs for device in devices for rate in make_schedule(config)]
        rng.shuffle(block)
        jobs.extend(block)
    return jobs


def run_benchmarks(checkpoints, image_paths, *, devices=('cpu',), config=None,
                   prediction_options=None, prompts=None, output_root=None, notes='',
                   progress=True):
    """Снимок MODELS и реестра передаётся свежему процессу для каждой пары модель/устройство."""
    from .. import common as cmn
    from ..workflows import enabled_specs
    config = config or BenchmarkConfig()
    config.validate()
    devices = list(devices)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError('Provide unique devices')
    for device in devices:
        if device != 'cpu' and not (device.startswith('cuda:') and device[5:].isdigit()):
            raise ValueError('Use cpu or cuda:N explicitly')
    images = [str(Path(p).resolve()) for p in image_paths]
    if not images or any(not Path(p).is_file() for p in images):
        raise FileNotFoundError('Provide existing benchmark images')
    specs, skipped = [], {}
    for spec in enabled_specs():
        reason = spec.unavailable_reason('predict')
        if reason:
            skipped[spec.id] = reason
            print(f'Skipped {spec.id}: {reason}')
            continue
        path = Path(checkpoints.get(spec.id, ''))
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Configure an existing nonempty checkpoint for {spec.id}: {path}')
        options = dict((prediction_options or {}).get(spec.id, {}))
        forbidden = {'batch', 'stream', 'source', 'device', 'save', 'show', 'save_txt', 'save_crop', 'save_conf', 'verbose'}
        if forbidden.intersection(options):
            raise ValueError(f'Benchmark controls these options: {sorted(forbidden.intersection(options))}')
        item = asdict(spec)
        item['operations'] = sorted(spec.operations)
        specs.append(dict(spec=item, checkpoint=str(path.resolve()), checkpoint_sha256=sha256(path), options=options,
                          prompts=(prompts or {}).get(spec.id)))
    if not specs:
        raise ValueError('No enabled supported models to benchmark')
    jobs = make_jobs(specs, devices, config)
    input_hashes = {path: sha256(path) for path in images}
    root = Path(output_root or cmn.ROOT / 'experiments' / 'benchmarks').resolve()
    directory = root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    manifest = dict(schema_version=3, status='running', config=asdict(config), devices=devices,
                    images=images, models=specs, skipped=skipped, notes=notes,
                    input_hashes=input_hashes, jobs=jobs,
                    python_executable=sys.executable, completed_workers=[])
    write_json(directory / 'manifest.json', manifest)
    print(f'Benchmark directory: {directory}', flush=True)
    process = None
    reporter = None
    try:
        if progress:
            reporter = BenchmarkProgress(len(jobs))
        by_id = {item['spec']['id']: item for item in specs}
        for job_index, job in enumerate(jobs):
            item, device = by_id[job['model_id']], job['device']
            worker_dir = directory / f"{job_index:04d}-{job['model_id']}-{device.replace(':', '-')}-r{job['repeat']}"
            worker_dir.mkdir()
            payload = dict(item, device=device, images=images, config=asdict(config), notes=notes,
                           progress=progress, repeat=job['repeat'], input_fps=job['input_fps'],
                           input_hashes=input_hashes)
            request = worker_dir / 'request.json'
            write_json(request, payload)
            print(f"Running {job['model_id']} on {device}, repeat {job['repeat']}; "
                  f"log: {worker_dir / 'worker.log'}", flush=True)
            if reporter:
                reporter.start_worker(worker_dir, f"{job['model_id']} / repeat {job['repeat']}", device)
            with (worker_dir / 'worker.log').open('w', encoding='utf-8') as log:
                environment = dict(os.environ, YOLO_CONFIG_DIR=str(cmn.YOLO_CONFIG_DIR))
                process = subprocess.Popen(
                    [sys.executable, '-m', 'scripts.benchmark.worker', str(request)],
                    cwd=Path(__file__).resolve().parents[2], env=environment,
                    stdout=log, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    if reporter:
                        reporter.refresh()
                    time.sleep(1.0)
                if reporter:
                    reporter.refresh()
                if process.returncode:
                    raise RuntimeError(f'Benchmark worker failed; see {worker_dir / "worker.log"}')
            manifest['completed_workers'].append(worker_dir.name)
            write_json(directory / 'manifest.json', manifest)
            export_summary(directory)
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if reporter:
            reporter.close(manifest['status'])
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        export_summary(directory)
        write_json(directory / 'manifest.json', manifest)
    return directory
