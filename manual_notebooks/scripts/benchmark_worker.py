"""Изолированный запуск одной модели на одном устройстве из JSON-конфигурации."""
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from .benchmark import (BenchmarkConfig, DeviceMetrics, make_schedule, measure_run,
                        save_run, sha256, write_json)


def run_worker(request):
    import numpy as np
    import psutil
    from PIL import Image
    from . import common as cmn
    from .contracts import ModelArtifacts, ModelSpec
    from .workflows import load_model

    directory = Path(request).parent
    payload = json.loads(Path(request).read_text(encoding='utf-8'))
    config = BenchmarkConfig(**payload['config'])
    config.validate()
    item = payload['spec']
    item['operations'] = frozenset(item['operations'])
    spec = ModelSpec(**item)
    cmn.MODELS_REGESTRY[spec.id] = spec
    metrics = DeviceMetrics(payload['device'])
    images, inputs = [], []
    for path in payload['images']:
        with Image.open(path) as image:
            array = np.asarray(image.convert('RGB'))[..., ::-1].copy()
        images.append(array)
        inputs.append(dict(path=path, sha256=sha256(path), shape=list(array.shape)))
    packages = {}
    for name in ('torch', 'torchvision', 'ultralytics', 'rfdetr', 'numpy', 'psutil', 'Pillow'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    torch = metrics.torch
    manifest = dict(schema_version=2, status='running', model_id=spec.id, device=metrics.device,
                    backend=spec.backend, checkpoint=payload['checkpoint'],
                    checkpoint_sha256=sha256(payload['checkpoint']), inputs=inputs,
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
                    cuda_baseline=metrics.memory(), schedule=make_schedule(config))
    write_json(directory / 'manifest.json', manifest)
    try:
        predictor = load_model(spec.id, ModelArtifacts(Path(payload['checkpoint'])), device=metrics.device)
        for index, rate in enumerate(manifest['schedule']):
            print('Running direct inference' if rate is None else f'Running stream, input FPS {rate}', flush=True)
            run = measure_run(predictor, images, metrics, config, input_fps=rate,
                              options=payload['options'], prompts=payload['prompts'])
            run.update(model_id=spec.id, device=metrics.device)
            run['summary']['rss_baseline_bytes'] = manifest['rss_baseline_bytes']
            save_run(directory, index, run)
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
