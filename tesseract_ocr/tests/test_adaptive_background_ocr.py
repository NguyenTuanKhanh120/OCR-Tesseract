import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from tesseract_ocr.adaptive_background_ocr import run_adaptive_background_ocr
from tesseract_ocr.config import TesseractConfig
from tesseract_ocr.schemas import PageResult


def page_result(confidence, text, word_count):
    return PageResult(
        page_number=1,
        status="success",
        text=text,
        confidence=confidence,
        boxes=[object()] * word_count,
        preprocessing={},
    )


class AdaptiveBackgroundOcrTests(unittest.TestCase):
    def setUp(self):
        self.config = TesseractConfig(
            input_dir=Path("."),
            output_dir=Path("test-output"),
            preprocessing="grayscale",
        )

    @patch("tesseract_ocr.adaptive_background_ocr.run_tesseract_on_image")
    def test_disabled_runs_only_current_pipeline(self, ocr_mock):
        ocr_mock.return_value = page_result(0.70, "Nội dung", 20)
        image = np.full((300, 500, 3), 245, dtype=np.uint8)

        result = run_adaptive_background_ocr(
            image, self.config, 1, "grayscale", enabled=False
        )

        self.assertEqual(ocr_mock.call_count, 1)
        self.assertEqual(result["selected_variant"], "current")
        self.assertEqual(result["decision"]["reason"], "adaptive_background_disabled")

    @patch("tesseract_ocr.adaptive_background_ocr.run_tesseract_on_image")
    def test_selects_better_illumination_candidate(self, ocr_mock):
        ocr_mock.side_effect = [
            page_result(0.60, "Nội dung ---", 50),
            page_result(0.78, "Nội dung rõ ràng", 48),
        ]
        gradient = np.tile(
            np.linspace(130, 250, 420, dtype=np.uint8)[:, None], (1, 650)
        )
        image = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
        cv2.putText(image, "OCR", (200, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 5)

        result = run_adaptive_background_ocr(
            image, self.config, 1, "grayscale", enabled=True
        )

        self.assertEqual(ocr_mock.call_count, 2)
        self.assertEqual(result["candidate_kind"], "illumination_balanced")
        self.assertEqual(result["selected_variant"], "processed")

    @patch("tesseract_ocr.adaptive_background_ocr.run_tesseract_on_image")
    def test_rejects_harmful_colored_candidate(self, ocr_mock):
        ocr_mock.side_effect = [
            page_result(0.70, "Nội dung rõ ràng", 80),
            page_result(0.40, "--- ___", 60),
        ]
        image = np.full((360, 600, 3), (220, 190, 120), dtype=np.uint8)
        cv2.putText(image, "BAO HIEM", (80, 190), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 5)

        result = run_adaptive_background_ocr(
            image, self.config, 1, "grayscale", enabled=True
        )

        self.assertEqual(result["candidate_kind"], "color_normalized")
        self.assertEqual(result["selected_variant"], "current")
        self.assertEqual(result["decision"]["reason"], "candidate_confidence_drop")


if __name__ == "__main__":
    unittest.main()
