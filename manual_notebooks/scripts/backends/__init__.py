"""Фабрика бэкендов: импорт каталога не загружает пакеты провайдеров."""
from ..contracts import ModelSpec


def get_backend(spec: ModelSpec):
    if spec.backend == "ultralytics":
        from .ultralytics import UltralyticsBackend
        return UltralyticsBackend()
    raise NotImplementedError(spec.note or f"Бэкенд {spec.backend!r} пока не реализован")
