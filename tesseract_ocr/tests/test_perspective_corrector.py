import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tesseract_ocr.document_renderer import save_png
from tesseract_ocr.perspective_corrector import (
    detect_panel_quad,
    rectify_from_quad,
    rectify_panels,
)


def synthetic_perspective_panel() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((900, 700, 3), 255, dtype=np.uint8)
    quad = np.array(
        [[55, 45], [650, 72], [665, 855], [32, 830]], dtype=np.int32
    )
    cv2.polylines(image, [quad.reshape(-1, 1, 2)], True, (30, 70, 70), 8)
    for row in range(5):
        y = 220 + (row * 95)
        cv2.line(image, (120, y), (585, y + 12), (70, 70, 70), 3)
    cv2.putText(
        image,
        "BAO HIEM",
        (160, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    return image, quad.astype(np.float32)


class TestPerspectiveCorrector(unittest.TestCase):
    def test_detects_four_corners(self):
        image, expected = synthetic_perspective_panel()
        detected = detect_panel_quad(image)

        self.assertGreaterEqual(detected.confidence, 0.65)
        for actual_point, expected_point in zip(detected.points, expected):
            self.assertLess(float(np.linalg.norm(actual_point - expected_point)), 15.0)

    def test_rectifies_trapezoid_to_rectangle(self):
        image, _ = synthetic_perspective_panel()
        detected = detect_panel_quad(image)
        rectified = rectify_from_quad(image, detected.points, output_margin=8)

        height, width = rectified.shape[:2]
        self.assertGreater(width, 590)
        self.assertGreater(height, 770)
        self.assertLess(abs(detected.max_tilt_degrees), 12.0)

    def test_rectifies_all_manifest_panels(self):
        with tempfile.TemporaryDirectory(prefix="perspective_test_") as temp_dir:
            root = Path(temp_dir)
            panels_dir = root / "panels"
            output_dir = root / "rectified"
            page_dir = panels_dir / "page_0001"
            page_dir.mkdir(parents=True)
            image, _ = synthetic_perspective_panel()
            save_png(page_dir / "panel_01.png", image)
            (panels_dir / "panels_manifest.json").write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "page_number": 1,
                                "panels": [
                                    {
                                        "panel_index": 1,
                                        "file": "page_0001/panel_01.png",
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = rectify_panels(panels_dir, output_dir)

            self.assertEqual(result["successful_panels"], 1)
            self.assertEqual(result["needs_review_panels"], 0)
            self.assertTrue(
                (output_dir / "page_0001" / "panel_01_rectified.png").is_file()
            )
            self.assertTrue(
                (
                    output_dir
                    / "page_0001"
                    / "panel_01_perspective_overlay.png"
                ).is_file()
            )
            self.assertTrue((output_dir / "perspective_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
