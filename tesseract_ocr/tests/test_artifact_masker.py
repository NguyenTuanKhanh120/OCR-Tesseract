import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tesseract_ocr.artifact_masker import (
    detect_barcode_regions,
    detect_logo_regions,
    detect_qr_regions,
    mask_panel_artifacts,
    mask_regions,
)
from tesseract_ocr.document_renderer import save_png


def synthetic_qr_panel() -> np.ndarray:
    image = np.full((800, 600, 3), 255, dtype=np.uint8)
    qr = cv2.QRCodeEncoder_create().encode("OCR-STEP-4-TEST")
    qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    qr = cv2.resize(qr, (210, 210), interpolation=cv2.INTER_NEAREST)
    image[250:460, 195:405] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
    return image


def synthetic_barcode_panel() -> np.ndarray:
    image = np.full((800, 600, 3), 255, dtype=np.uint8)
    x = 110
    for index in range(42):
        bar_width = 2 + (index % 4)
        cv2.rectangle(image, (x, 330), (x + bar_width, 450), (0, 0, 0), -1)
        x += bar_width + 4 + (index % 2)
    return image


def synthetic_logo_panel() -> np.ndarray:
    image = np.full((900, 700, 3), 255, dtype=np.uint8)
    cv2.circle(image, (95, 95), 52, (160, 100, 0), -1)
    cv2.rectangle(image, (150, 55), (315, 135), (120, 80, 0), -1)
    cv2.putText(
        image,
        "VNI",
        (162, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (255, 255, 255),
        5,
        cv2.LINE_AA,
    )
    return image


class TestArtifactMasker(unittest.TestCase):
    def test_detects_qr_after_scan_style_processing(self):
        regions = detect_qr_regions(synthetic_qr_panel())
        self.assertGreaterEqual(len(regions), 1)
        self.assertEqual(regions[0].kind, "qr")

    def test_detects_one_dimensional_barcode(self):
        regions = detect_barcode_regions(synthetic_barcode_panel())
        self.assertGreaterEqual(len(regions), 1)
        self.assertEqual(regions[0].kind, "barcode")

    def test_detects_colored_top_edge_logo(self):
        regions = detect_logo_regions(synthetic_logo_panel())
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].kind, "logo")

    def test_masks_regions_and_writes_manifest(self):
        with tempfile.TemporaryDirectory(prefix="artifact_mask_test_") as temp_dir:
            root = Path(temp_dir)
            rectified_dir = root / "rectified"
            output_dir = root / "masked"
            page_dir = rectified_dir / "page_0001"
            page_dir.mkdir(parents=True)
            image = synthetic_qr_panel()
            save_png(page_dir / "panel_01_rectified.png", image)
            (rectified_dir / "perspective_manifest.json").write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "page_number": 1,
                                "panels": [
                                    {
                                        "panel_index": 1,
                                        "rectified_file": (
                                            "page_0001/panel_01_rectified.png"
                                        ),
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = mask_panel_artifacts(rectified_dir, output_dir, padding=8)

            self.assertEqual(result["successful_panels"], 1)
            self.assertEqual(result["region_counts"]["qr"], 1)
            masked_path = output_dir / "page_0001" / "panel_01_masked.png"
            self.assertTrue(masked_path.is_file())
            self.assertTrue(
                (output_dir / "page_0001" / "panel_01_mask_overlay.png").is_file()
            )
            self.assertTrue((output_dir / "artifact_mask_manifest.json").is_file())
            masked = cv2.imdecode(
                np.fromfile(str(masked_path), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            self.assertTrue(np.all(masked[260:450, 205:395] == 255))

    def test_mask_regions_preserves_source_image(self):
        image = synthetic_logo_panel()
        original = image.copy()
        regions = detect_logo_regions(image)
        masked, _ = mask_regions(image, regions, padding=5)

        self.assertTrue(np.array_equal(image, original))
        self.assertFalse(np.array_equal(masked, original))


if __name__ == "__main__":
    unittest.main()
