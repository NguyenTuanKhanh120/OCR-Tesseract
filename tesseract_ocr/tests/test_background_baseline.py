import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from tesseract_ocr.background_baseline import (
    image_quality_metrics,
    text_quality_metrics,
)


class BackgroundBaselineTests(unittest.TestCase):
    def test_background_range_detects_uneven_illumination(self):
        image = np.full((320, 640, 3), 245, dtype=np.uint8)
        image[:, :320] = 130
        metrics = image_quality_metrics(image)
        self.assertGreater(metrics["background_range"], 80.0)

    def test_color_metrics_detect_saturated_band(self):
        image = np.full((200, 500, 3), 255, dtype=np.uint8)
        image[95:105, :] = (255, 255, 0)
        metrics = image_quality_metrics(image)
        self.assertGreater(metrics["max_saturated_row_ratio"], 0.95)
        self.assertGreater(metrics["bright_chroma_p99"], 20.0)
        self.assertGreater(metrics["colored_bright_area_ratio"], 0.01)

    def test_text_metrics_flag_symbols_and_non_latin_scripts(self):
        metrics = text_quality_metrics("Nội dung đúng\n--- ___\n测试")
        self.assertGreater(metrics["suspicious_token_ratio"], 0.0)
        self.assertGreater(metrics["non_latin_letter_ratio"], 0.0)
        self.assertGreater(metrics["symbol_only_line_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
