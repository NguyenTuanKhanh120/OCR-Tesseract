import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tesseract_ocr.document_renderer import save_png
from tesseract_ocr.panel_splitter import detect_panel_boxes, split_rendered_pages


def synthetic_panel_page(panel_count: int) -> np.ndarray:
    image = np.full((800, 1200, 3), 255, dtype=np.uint8)
    margin = 40
    gap = 30
    available_width = 1200 - (2 * margin) - ((panel_count - 1) * gap)
    panel_width = available_width // panel_count
    for index in range(panel_count):
        left = margin + index * (panel_width + gap)
        cv2.rectangle(image, (left, 80), (left + panel_width, 700), (30, 70, 70), 6)
        cv2.putText(
            image,
            f"PANEL {index + 1}",
            (left + 25, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            3,
        )
    return image


class TestPanelSplitter(unittest.TestCase):
    def test_detects_two_panels(self):
        boxes = detect_panel_boxes(synthetic_panel_page(2), expected_panels=2)
        self.assertEqual(len(boxes), 2)
        self.assertLess(boxes[0][0], boxes[1][0])

    def test_detects_three_panels(self):
        boxes = detect_panel_boxes(synthetic_panel_page(3), expected_panels=3)
        self.assertEqual(len(boxes), 3)
        self.assertEqual(boxes, sorted(boxes, key=lambda box: box[0]))

    def test_splits_all_rendered_pages_and_writes_manifest(self):
        with tempfile.TemporaryDirectory(prefix="panel_split_test_") as temp_dir:
            root = Path(temp_dir)
            render_dir = root / "rendered"
            output_dir = root / "panels"
            render_dir.mkdir()
            page = synthetic_panel_page(3)
            save_png(render_dir / "page_0001.png", page)
            (render_dir / "render_manifest.json").write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "page_number": 1,
                                "file": "page_0001.png",
                                "width": 1200,
                                "height": 800,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = split_rendered_pages(
                render_dir,
                output_dir,
                expected_panels=3,
                padding=10,
            )

            self.assertEqual(result["successful_pages"], 1)
            self.assertEqual(result["pages"][0]["detected_panels"], 3)
            self.assertTrue((output_dir / "page_0001" / "panel_01.png").is_file())
            self.assertTrue((output_dir / "page_0001" / "panel_02.png").is_file())
            self.assertTrue((output_dir / "page_0001" / "panel_03.png").is_file())
            self.assertTrue(
                (output_dir / "page_0001" / "page_0001_panels_overlay.png").is_file()
            )
            self.assertTrue((output_dir / "panels_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
