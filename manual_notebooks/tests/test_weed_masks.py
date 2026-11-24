"""Проверки растеризации и пиксельных метрик на маленьких примерах."""
import unittest

import numpy as np
import pandas as pd

from manual_notebooks.scripts.contracts import Prediction
from manual_notebooks.scripts.weed_masks import (
    metric_counts, protection_mask, vegetation_mask, with_metrics,
)


class WeedMaskTests(unittest.TestCase):
    def test_rgb_hsv(self):
        rgb = np.array([[[0, 255, 0], [255, 0, 0], [0, 0, 0]]], dtype=np.uint8)
        mask = vegetation_mask(rgb, (35, 40, 40), (85, 255, 255))
        np.testing.assert_array_equal(mask, [[True, False, False]])

    def test_box_union_and_filtering(self):
        boxes = [[-2, -1, 1.1, 2], [1.7, 1, 4, 4], [0, 0, 3, 3],
                 [0, 0, 3, 3], [-8, -8, -1, -1]]
        pred = Prediction((3, 3), boxes, [1, 1, 0.1, 1, 1], [0, 0, 0, 1, 0], None, None, None)
        mask = protection_mask(pred, (3, 3), 0, 0.25, 0)
        np.testing.assert_array_equal(mask, [[1, 1, 0], [1, 1, 1], [0, 1, 1]])
        self.assertTrue(protection_mask(pred, (3, 3), 0, 0.25, 1).all())
        with self.assertRaises(ValueError):
            protection_mask(pred, (2, 3), 0, 0.25, 1)

    def test_paired_metrics_and_ignore(self):
        truth = np.array([[1, 1, 2], [2, 0, 255]])
        hsv = np.array([[1, 1, 1], [1, 1, 1]], dtype=bool)
        protection = np.array([[1, 0, 0], [1, 0, 1]], dtype=bool)
        counts = metric_counts(truth, hsv & ~protection, protection, ignore=255)
        scores = with_metrics(pd.DataFrame([counts])).iloc[0]
        for name, value in dict(weed_iou=1/4, weed_precision=1/3, weed_recall=1/2,
                                crop_to_weed_error=1/2, weed_lost_in_protection=1/2,
                                crop_outside_protection=1/2).items():
            self.assertAlmostEqual(scores[name], value)
        without = metric_counts(truth, hsv, np.zeros_like(protection), ignore=255)
        self.assertEqual(without["predicted_weed"], 5)
        self.assertEqual(without["weed_in_protection"], 0)

    def test_empty_prediction_and_empty_classes(self):
        shape = (2, 3)
        prediction = Prediction(shape, [], [], [], None, None, None)
        empty = protection_mask(prediction, shape, 0, 0.25, 0)
        self.assertFalse(empty.any())
        counts = metric_counts(np.zeros(shape), empty, empty)
        scores = with_metrics(pd.DataFrame([counts])).iloc[0]
        self.assertTrue(scores.filter(items=["weed_iou", "weed_precision", "weed_recall",
                                            "crop_to_weed_error", "weed_lost_in_protection",
                                            "crop_outside_protection"]).isna().all())
