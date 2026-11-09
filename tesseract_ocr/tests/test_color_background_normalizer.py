import unittest

import cv2
import numpy as np

from tesseract_ocr.color_background_normalizer import (
    color_difference_map,
    detect_colored_background,
    neutralize_color_background,
)


class ColorBackgroundNormalizerTests(unittest.TestCase):
    def test_black_text_on_blue_background_becomes_dark_on_white(self):
        image = np.full((360, 600, 3), (220, 190, 120), dtype=np.uint8)
        cv2.putText(
            image,
            "BAO HIEM",
            (75, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (15, 15, 15),
            5,
            cv2.LINE_AA,
        )
        neutral, metadata = neutralize_color_background(image)
        background = neutral[40:90, 40:120]
        text_region = neutral[150:220, 60:400]

        self.assertTrue(metadata["applied"])
        self.assertGreater(float(np.median(background)), 245.0)
        self.assertLess(float(np.percentile(text_region, 10)), 80.0)

    def test_yellow_text_on_red_background_is_not_lost(self):
        image = np.full((420, 680, 3), (35, 45, 210), dtype=np.uint8)
        cv2.putText(
            image,
            "PCCC VIET NAM",
            (65, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.7,
            (25, 220, 245),
            5,
            cv2.LINE_AA,
        )
        neutral, _ = neutralize_color_background(image)
        background = neutral[50:100, 50:150]
        text_region = neutral[180:250, 50:560]

        self.assertGreater(float(np.median(background)), 245.0)
        self.assertLess(float(np.percentile(text_region, 10)), 100.0)

    def test_small_color_texture_is_suppressed(self):
        rng = np.random.default_rng(42)
        image = np.full((280, 420, 3), (210, 225, 235), dtype=np.int16)
        image += rng.integers(-3, 4, size=image.shape, dtype=np.int16)
        image = np.clip(image, 0, 255).astype(np.uint8)
        neutral, metadata = neutralize_color_background(image)

        self.assertGreater(metadata["near_white_ratio"], 0.95)
        self.assertGreater(float(np.median(neutral)), 250.0)

    def test_difference_map_rejects_shape_mismatch(self):
        image = np.zeros((30, 40, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            color_difference_map(image, np.zeros((31, 40, 3), dtype=np.uint8))

    def test_rejects_grayscale_input(self):
        with self.assertRaises(ValueError):
            neutralize_color_background(np.zeros((40, 40), dtype=np.uint8))

    def test_detects_large_colored_background(self):
        image = np.full((300, 500, 3), (210, 190, 120), dtype=np.uint8)
        result = detect_colored_background(image)
        self.assertTrue(result["detected"])

    def test_does_not_treat_small_colored_stamp_as_background(self):
        image = np.full((500, 500, 3), 245, dtype=np.uint8)
        cv2.circle(image, (250, 250), 35, (20, 20, 220), -1)
        result = detect_colored_background(image)
        self.assertFalse(result["detected"])


if __name__ == "__main__":
    unittest.main()
