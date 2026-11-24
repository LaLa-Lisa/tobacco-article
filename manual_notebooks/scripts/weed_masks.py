"""Маски растительности и попиксельная оценка в разрешении исходного снимка."""
import math

import cv2
import numpy as np

from .contracts import Prediction


def vegetation_mask(rgb: np.ndarray, lower: tuple, upper: tuple) -> np.ndarray:
    """Выделить растительность по включительным границам OpenCV HSV."""
    if not all(0 <= lo <= hi <= limit for lo, hi, limit in
               zip(lower, upper, (179, 255, 255), strict=True)):
        raise ValueError("Invalid OpenCV HSV bounds")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, lower, upper) != 0


def protection_mask(prediction: Prediction, image_shape: tuple, tobacco_id: int,
                    confidence: float, padding: int) -> np.ndarray:
    """Объединить рамки в координатах xyxy; отступ задан в пикселях с каждой стороны."""
    if tuple(prediction.image_shape) != tuple(image_shape):
        raise ValueError("Prediction shape differs from source image")
    if not 0 <= confidence <= 1 or not isinstance(padding, int) or padding < 0:
        raise ValueError("Invalid confidence or padding")
    height, width = image_shape
    mask = np.zeros(image_shape, dtype=bool)
    for box, score, class_id in zip(prediction.boxes_xyxy, prediction.scores,
                                    prediction.class_ids, strict=True):
        if class_id != tobacco_id or score < confidence:
            continue
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        left = max(0, math.floor(x1) - padding)
        top = max(0, math.floor(y1) - padding)
        right = min(width, math.ceil(x2) + padding)
        bottom = min(height, math.ceil(y2) + padding)
        if left < right and top < bottom:
            mask[top:bottom, left:right] = True
    return mask


def metric_counts(truth: np.ndarray, weeds: np.ndarray, protection: np.ndarray,
                  ignore: int | None = None) -> dict[str, int]:
    """Вернуть слагаемые для агрегации по пикселям до вычисления отношений."""
    if truth.ndim != 2 or truth.shape != weeds.shape or truth.shape != protection.shape:
        raise ValueError("Expected three equally sized 2D masks")
    if weeds.dtype != bool or protection.dtype != bool:
        raise ValueError("Prediction and protection must be boolean masks")
    if ignore in (0, 1, 2):
        raise ValueError("Ignore label overlaps a semantic class")
    labels = (0, 1, 2) if ignore is None else (0, 1, 2, ignore)
    if not np.isin(truth, labels).all():
        raise ValueError("Unexpected semantic label")
    if ignore is not None:
        weeds = weeds & (truth != ignore)
    crop = truth == 1
    weed = truth == 2
    return {
        "weed_intersection": int((weeds & weed).sum()),
        "weed_union": int((weeds | weed).sum()),
        "predicted_weed": int(weeds.sum()),
        "gt_weed": int(weed.sum()),
        "gt_crop": int(crop.sum()),
        "crop_as_weed": int((weeds & crop).sum()),
        "weed_in_protection": int((weed & protection).sum()),
        "crop_outside": int((crop & ~protection).sum()),
    }


METRIC_TERMS = {
    "weed_iou": ("weed_intersection", "weed_union"),
    "weed_precision": ("weed_intersection", "predicted_weed"),
    "weed_recall": ("weed_intersection", "gt_weed"),
    "crop_to_weed_error": ("crop_as_weed", "gt_crop"),
    "weed_lost_in_protection": ("weed_in_protection", "gt_weed"),
    "crop_outside_protection": ("crop_outside", "gt_crop"),
}


def with_metrics(counts):
    """Добавить метрики к таблице счётчиков; пустой знаменатель даёт NaN."""
    result = counts.copy()
    for metric, (numerator, denominator) in METRIC_TERMS.items():
        result[metric] = counts[numerator].div(counts[denominator].replace(0, np.nan))
    return result
