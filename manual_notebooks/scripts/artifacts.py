"""Разрешение путей артефактов без импорта SDK."""
import json
from pathlib import Path

from . import common as cmn
from .contracts import ModelArtifacts


def resolve_artifacts(spec, value):
    artifacts = value if isinstance(value, ModelArtifacts) else ModelArtifacts(Path(value))
    if spec.backend == 'mmdetection' and artifacts.config is None:
        return ModelArtifacts(artifacts.checkpoint, artifacts.checkpoint.with_suffix('.py'))
    return artifacts


def ready(artifacts):
    return all(p.is_file() and p.stat().st_size for p in (artifacts.checkpoint, artifacts.config)
               if p is not None)


def trained_artifacts(spec):
    if spec.backend == 'mmdetection':
        pointer = cmn.TRAINED_WEIGHTS_DIR / f'{spec.id}.json'
        if not pointer.is_file():
            return None
        paths = json.loads(pointer.read_text())
        artifacts = ModelArtifacts(Path(paths['checkpoint']), Path(paths['config']))
        return artifacts if ready(artifacts) else None
    checkpoint = cmn.TRAINED_WEIGHTS_DIR / f'{spec.id}.pt'
    return ModelArtifacts(checkpoint) if checkpoint.is_file() else None
