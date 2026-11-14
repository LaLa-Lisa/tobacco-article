from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = ROOT / "data"
DATASET_PATH = DATASETS_DIR / "tobacco_yolo"
DATA_YAML = DATASET_PATH / "data.generated.yaml"

WEIGHTS_DIR = ROOT / "weights"
ORIGINAL_WEIGHTS_DIR = WEIGHTS_DIR / "original"
TRAINED_WEIGHTS_DIR = WEIGHTS_DIR / "trained"

RUNS_DIR = ROOT / "experiments" / "reproductions"
YOLO_CONFIG_DIR = ROOT / "experiments" / ".ultralytics"

# Каталог из research/sota_models/report.md (разделы 4–9).
# Ключи — локальные идентификаторы, а не имена совместимых с Ultralytics файлов весов.
# Перед запуском семейства нужно задать конкретный файл весов в реестре.
# MODELS управляет включением моделей; реестр описывает их реализации.
MODELS = {
    # Семейства YOLO, включая более ранние базовые модели из обзора.
    "yolov5": False,  # Отключено: нужна реализация вне трёх запланированных бэкендов
    "yolov6": False,  # v3.0; Отключено: нужна реализация вне трёх запланированных бэкендов
    "yolov7": False,  # Отключено: нужна реализация вне трёх запланированных бэкендов
    "damo-yolo": False,  # Отключено: нужна реализация вне трёх запланированных бэкендов
    "yolov8": False,
    "yolov9": False,
    "yolov10": False,
    "yolo11": False,
    "yolov12": False,
    "yolov13": False,  # Отключено: нужна реализация вне трёх запланированных бэкендов
    "yolo26": False,
    "yolo-world": False,
    "yoloe": False,

    # Конкретные варианты детекторов из сравнений и рекомендаций.
    "yolov8n": True,
    "yolov8s": False,
    "yolov12n": False,
    "yolo26n": True,
    "yolo26s": False,
    "yolo26m": False,

    # Альтернативные детекторы.
    "rtmdet": False,  # Отключено: бэкенд mmdetection запланирован, но пока не реализован
    "rtmdet-x": False,  # Отключено: бэкенд mmdetection запланирован, но пока не реализован
    "rt-detr": False,
    "rt-detrv2": False,  # Отключено: нужна реализация вне трёх запланированных бэкендов
    "rt-detrv2-s": False,  # Отключено: нужна реализация вне трёх запланированных бэкендов
    "d-fine": False,  # Отключено: нужна реализация вне трёх запланированных бэкендов
    "deim": False,  # Отключено: нужна реализация вне трёх запланированных бэкендов
    "deimv2": False,  # Отключено: нужна реализация вне трёх запланированных бэкендов
    "rt-detrv4": False,  # Отключено: нужна реализация вне трёх запланированных бэкендов
    "rf-detr": False,  # Отключено: бэкенд rfdetr запланирован, но пока не реализован
    "rf-detr-m": False,  # Отключено: бэкенд rfdetr запланирован, но пока не реализован
    "faster-rcnn": False,  # Отключено: бэкенд mmdetection запланирован, но пока не реализован

    # Сегментация экземпляров и семантическая сегментация.
    "yolov8-seg": False,
    "yolov8n-seg": False,
    "yolo11-seg": False,
    "yolo11n-seg": False,
    "yolo11m-seg": False,
    "yolo26-seg": False,
    "rtmdet-ins": False,  # Отключено: бэкенд mmdetection запланирован, но пока не реализован
    "rf-detr-seg": False,  # Отключено: бэкенд rfdetr запланирован, но пока не реализован
    "pidnet": False,  # Отключено: этот бэкенд не входит в текущий план
    "pidnet-s": False,  # Отключено: этот бэкенд не входит в текущий план
    "efficientvit": False,  # Отключено: этот бэкенд не входит в текущий план

    # Сегментация изображений и видео по подсказкам.
    "sam": False,
    "fastsam": False,
    "mobilesam": False,
    "efficientvit-sam": False,  # Отключено: этот бэкенд не входит в текущий план
    "sam2": False,
    "sam2-hiera-b-plus": False,
    "sam2-hiera-l": False,
    "sam3": False,

    # Модели из сельскохозяйственных исследований (раздел 8).
    "tia-yolov5": False,  # Отключено: нужна реализация вне трёх запланированных бэкендов
    "yoloc-tiny": False,  # Отключено: нужна реализация вне трёх запланированных бэкендов
    # "lightweight-multispectral-crop-weed-segmentation": False,
    "weedsense": False,  # Отключено: этот бэкенд не входит в текущий план
}

# Сохраняем согласованное написание имени для API ноутбука.
from .registry import MODELS_REGESTRY, validate_registry

validate_registry(MODELS, MODELS_REGESTRY)
