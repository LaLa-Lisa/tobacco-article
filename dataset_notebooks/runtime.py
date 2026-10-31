"""Paths, provenance, and strict model loading; scientific code lives in notebooks."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import sys

SAM3_REVISION = 'f1647e8d0c1d280991f4d1b7d974c2704537fe66'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def fresh_run(root, name):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notebooks'))
    from reproduction import new_run
    return new_run(Path(root), name)


def environment():
    result = {'python': sys.version, 'platform': platform.platform(),
              'sam3_revision': SAM3_REVISION,
              'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}
    if 'torch' in sys.modules:
        import torch
        result.update(torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    return result


def load_model(checkpoint=None, allow_download=False, training=False):
    """Load official detector weights or our complete state dict, never partial silently."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for enabled SAM 3 operations. Run on the GPU machine.')
    direct = importlib.metadata.distribution('sam3').read_text('direct_url.json')
    if direct is None or json.loads(direct).get('vcs_info', {}).get('commit_id') != SAM3_REVISION:
        raise RuntimeError('Install the pinned SAM 3 VCS revision from requirements-windows.txt.')
    from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
    if checkpoint is None:
        if not allow_download:
            raise ValueError('Set a local pretrained checkpoint or explicitly allow authenticated downloads.')
        checkpoint = download_ckpt_from_hf(version='sam3')
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = payload.get('model', payload)
    if payload.get('format') == 'tobacco-sam3-v1':
        if payload['sam3_revision'] != SAM3_REVISION:
            raise ValueError('Checkpoint revision mismatch.')
    else:
        state = {k.removeprefix('detector.'): v for k, v in state.items() if k.startswith('detector.')}
        if not state:
            raise ValueError('Not an official SAM 3 detector checkpoint or tobacco checkpoint.')
    model = build_sam3_image_model(device='cpu', eval_mode=not training,
                                  load_from_HF=False, enable_segmentation=True,
                                  enable_inst_interactivity=False, compile=False)
    model.load_state_dict(state, strict=True)
    del payload, state
    model._source_checkpoint_path = str(checkpoint.resolve())
    model._source_checkpoint_sha256 = sha256(checkpoint)
    return model.cuda().train(training)
