"""Каталог реализаций; выбор моделей остаётся в common.MODELS.

Для семейств файл весов намеренно не задан. Перед включением семейства
укажите конкретный файл через dataclasses.replace. Список операций отражает
возможности адаптера, а не все возможности исходного исследовательского кода.
"""
from pathlib import Path

from .contracts import ModelSpec

FULL = frozenset({"download", "train", "predict"})
INFERENCE = frozenset({"download", "predict"})
MODELS_REGESTRY: dict[str, ModelSpec] = {}


def _add(names, *, backend="ultralytics", task="detect", constructor="YOLO",
         operations=FULL, note=""):
    for name in names.split():
        MODELS_REGESTRY[name] = ModelSpec(name, backend, task, constructor,
                                        operations=operations, note=note)


_add("yolov8 yolov9 yolov10 yolo11 yolov12 yolo26")
_add("yolov8-seg yolo11-seg yolo26-seg", task="segment")
_add("rt-detr", constructor="RTDETR")
_add("yolo-world", constructor="YOLOWorld")
_add("yoloe", constructor="YOLOE", task="segment")
_add("sam sam2", constructor="SAM", task="prompt_segment", operations=INFERENCE)
_add("fastsam", constructor="FastSAM", task="prompt_segment", operations=INFERENCE)

for name, checkpoint, task, constructor, operations in [
    ("yolov8n", "yolov8n.pt", "detect", "YOLO", FULL),
    ("yolov8s", "yolov8s.pt", "detect", "YOLO", FULL),
    ("yolov12n", "yolo12n.pt", "detect", "YOLO", FULL),
    ("yolo26n", "yolo26n.pt", "detect", "YOLO", FULL),
    ("yolo26s", "yolo26s.pt", "detect", "YOLO", FULL),
    ("yolo26m", "yolo26m.pt", "detect", "YOLO", FULL),
    ("yolov8n-seg", "yolov8n-seg.pt", "segment", "YOLO", FULL),
    ("yolo11n-seg", "yolo11n-seg.pt", "segment", "YOLO", FULL),
    ("yolo11m-seg", "yolo11m-seg.pt", "segment", "YOLO", FULL),
    ("mobilesam", "mobile_sam.pt", "prompt_segment", "SAM", INFERENCE),
    ("sam2-hiera-b-plus", "sam2_b.pt", "prompt_segment", "SAM", INFERENCE),
    ("sam2-hiera-l", "sam2_l.pt", "prompt_segment", "SAM", INFERENCE),
    ("sam3", "sam3.pt", "prompt_segment", "SAM", INFERENCE),
]:
    MODELS_REGESTRY[name] = ModelSpec(name, "ultralytics", task, constructor, checkpoint, operations)

_add("rtmdet rtmdet-x faster-rcnn", backend="mmdetection", constructor=None,
     operations=frozenset(), note="Disabled: future mmdetection backend is not implemented")
_add("rtmdet-ins", backend="mmdetection", task="segment", constructor=None,
     operations=frozenset(), note="Disabled: future mmdetection backend is not implemented")
_add("rf-detr", backend="rfdetr", constructor=None)
_add("rf-detr-seg", backend="rfdetr", task="segment", constructor=None)
for suffix, size in [("n", "Nano"), ("s", "Small"), ("m", "Medium"), ("l", "Large")]:
    for segment in (False, True):
        identifier = f"rf-detr-{'seg-' if segment else ''}{suffix}"
        filename = (f"rf-detr-seg-{size.lower()}.pt" if segment else
                    f"rf-detr-{size.lower()}{'-2026' if size == 'Large' else ''}.pth")
        MODELS_REGESTRY[identifier] = ModelSpec(
            identifier, "rfdetr", "segment" if segment else "detect",
            f"RFDETR{'Seg' if segment else ''}{size}", filename, FULL)
_add("yolov5 yolov6 yolov7 damo-yolo yolov13 rt-detrv2 rt-detrv2-s d-fine deim deimv2 rt-detrv4 tia-yolov5 yoloc-tiny",
     backend=None, constructor=None, operations=frozenset(),
     note="Disabled: requires an implementation outside the three planned backends")
_add("pidnet pidnet-s efficientvit weedsense", backend=None, constructor=None,
     task="semantic", operations=frozenset(), note="Disabled: backend outside scope")
_add("efficientvit-sam", backend=None, constructor=None, task="prompt_segment",
     operations=frozenset(), note="Disabled: backend outside scope")


def validate_registry(models, registry=None):
    registry = MODELS_REGESTRY if registry is None else registry
    if models.keys() != registry.keys():
        raise ValueError(f"MODELS/registry mismatch: missing={set(models) - set(registry)}, "
                         f"extra={set(registry) - set(models)}")
    for name, enabled in models.items():
        if type(enabled) is not bool or registry[name].id != name:
            raise ValueError(f"Invalid selection or registry identity for {name}")
        spec = registry[name]
        if spec.checkpoint and (Path(spec.checkpoint).name != spec.checkpoint):
            raise ValueError(f"{name}: checkpoint must be an explicit upstream filename")

