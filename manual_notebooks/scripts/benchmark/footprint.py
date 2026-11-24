"""Размер хранилищ зарегистрированных параметров и буферов до первого предсказания."""
from ..artifacts import resolve_artifacts
from .. import common as cmn
from ..workflows import load_model


def model_footprint(module):
    """Общее хранилище нескольких тензоров учитывается один раз, сначала у параметров."""
    seen = set()
    by_dtype = {}
    sizes = {}
    for group, tensors in [('parameter', module.parameters()), ('buffer', module.buffers())]:
        size = 0
        for tensor in tensors:
            storage = tensor.untyped_storage()
            key = (tensor.device, storage.data_ptr())
            if key in seen:
                continue
            seen.add(key)
            nbytes = storage.nbytes()
            size += nbytes
            dtype = str(tensor.dtype)
            by_dtype[dtype] = by_dtype.get(dtype, 0) + nbytes
        sizes[f'model_{group}_bytes'] = size
    return dict(**sizes, model_tensor_bytes=sum(sizes.values()), model_bytes_by_dtype=by_dtype)


def inspect_checkpoint(model_id, checkpoint):
    """Загрузить модель на CPU и подсчитать размер без подачи изображения."""
    artifacts = resolve_artifacts(cmn.MODELS_REGESTRY[model_id], checkpoint)
    predictor = load_model(model_id, artifacts, device='cpu')
    return model_footprint(predictor.footprint_module())
