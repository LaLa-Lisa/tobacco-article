"""Фабрика бэкендов: импорт каталога не загружает пакеты провайдеров."""
from ..contracts import ModelSpec


def get_backend(spec: ModelSpec):
    if spec.backend == "ultralytics":
        from .ultralytics import UltralyticsBackend
        return UltralyticsBackend()
    if spec.backend == "rfdetr":
        from .rfdetr import RFDETRBackend
        return RFDETRBackend()
    raise NotImplementedError(spec.note or f"Backend {spec.backend!r} is not implemented")
