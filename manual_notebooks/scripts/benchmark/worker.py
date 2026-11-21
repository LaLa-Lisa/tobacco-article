"""Изолированный запуск одной модели на одном устройстве из JSON-конфигурации."""
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

from .core import (BenchmarkConfig, DeviceMetrics, make_schedule, measure_run,
                        save_run, sha256, write_json)


def run_worker(request):
    import numpy as np
    import psutil
    from PIL import Image
    from .. import common as cmn
    from ..contracts import ModelArtifacts, ModelSpec
    from ..workflows import load_model

    directory = Path(request).parent
    payload = json.loads(Path(request).read_text(encoding='utf-8'))
    config = BenchmarkConfig(**payload['config'])
    config.validate()
    item = payload['spec']
    item['operations'] = frozenset(item['operations'])
    spec = ModelSpec(**item)
    cmn.MODELS_REGESTRY[spec.id] = spec
    metrics = DeviceMetrics(payload['device'])
    if config.torch_num_threads is not None:
        metrics.torch.set_num_threads(config.torch_num_threads)
        metrics.torch.set_num_interop_threads(1)
    images, inputs = [], []
    for path in payload['images']:
        checksum = sha256(path)
        if payload.get('input_hashes', {}).get(path, checksum) != checksum:
            raise ValueError(f'Benchmark image changed after preflight: {path}')
        with Image.open(path) as image:
            array = np.asarray(image.convert('RGB'))[..., ::-1].copy()
        images.append(array)
        inputs.append(dict(path=path, sha256=checksum, shape=list(array.shape)))
    packages = {}
    for name in ('torch', 'torchvision', 'ultralytics', 'rfdetr', 'numpy', 'psutil', 'Pillow'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    torch = metrics.torch
    checkpoint_hash = sha256(payload['checkpoint'])
    if payload.get('checkpoint_sha256', checkpoint_hash) != checkpoint_hash:
        raise ValueError('Benchmark checkpoint changed after preflight')
    manifest = dict(schema_version=3, status='running', model_id=spec.id, device=metrics.device,
                    backend=spec.backend, checkpoint=payload['checkpoint'],
                    checkpoint_sha256=checkpoint_hash, inputs=inputs, repeat=payload.get('repeat'),
                    config=asdict(config), options=payload['options'], prompts=payload['prompts'],
                    notes=payload['notes'], platform=platform.platform(), processor=platform.processor(),
                    python=platform.python_version(), packages=packages,
                    torch_threads=torch.get_num_threads(),
                    torch_interop_threads=torch.get_num_interop_threads(),
                    cuda_version=torch.version.cuda,
                    cudnn_version=torch.backends.cudnn.version(),
                    cuda_device=torch.cuda.get_device_name(metrics.device) if metrics.cuda else None,
                    float32_matmul_precision=torch.get_float32_matmul_precision(),
                    cudnn_benchmark=torch.backends.cudnn.benchmark,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                    cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    rss_baseline_bytes=psutil.Process().memory_info().rss,
                    cuda_baseline=metrics.memory(),
                    schedule=[payload['input_fps']] if 'input_fps' in payload else make_schedule(config))
    write_json(directory / 'manifest.json', manifest)
    try:
        predictor = load_model(spec.id, ModelArtifacts(Path(payload['checkpoint'])), device=metrics.device)
        for index, rate in enumerate(manifest['schedule']):
            def report(stage, completed=index):
                if payload.get('progress', False):
                    write_json(directory / 'progress.json', dict(
                        stage=stage, completed=completed, stage_started=time.monotonic(),
                        mode='Direct' if rate is None else f'Stream {rate:g} FPS'))

            print('Running direct inference' if rate is None else f'Running stream, input FPS {rate}', flush=True)
            report('Warming up')
            run = measure_run(predictor, images, metrics, config, input_fps=rate,
                              options=payload['options'], prompts=payload['prompts'],
                              on_measurement_start=lambda: report('Measuring'),
                              on_memory_start=lambda: report('Measuring memory'))
            metadata = getattr(predictor, 'benchmark_metadata', None)
            manifest['runtime'] = metadata() if callable(metadata) else {}
            manifest.update(torch_threads=torch.get_num_threads(),
                            torch_interop_threads=torch.get_num_interop_threads(),
                            float32_matmul_precision=torch.get_float32_matmul_precision(),
                            cudnn_benchmark=torch.backends.cudnn.benchmark,
                            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
            run.update(model_id=spec.id, device=metrics.device, repeat=payload.get('repeat'))
            run['summary']['rss_baseline_bytes'] = manifest['rss_baseline_bytes']
            save_run(directory, index, run)
            report('Saved', completed=index + 1)
            print(run['summary'], flush=True)
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(directory / 'manifest.json', manifest)


if __name__ == '__main__':
    run_worker(sys.argv[1])
