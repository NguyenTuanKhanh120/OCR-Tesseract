import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tesseract_ocr.background_preprocessor import (
    apply_clahe,
    estimate_yellow_fraction,
    process_masked_panels,
    sauvola_binarize,
    select_document_channel,
)
from tesseract_ocr.document_renderer import save_png


def synthetic_yellow_document() -> np.ndarray:
    image = np.full((600, 450, 3), (20, 220, 225), dtype=np.uint8)
    cv2.putText(
        image,
        "BAO HIEM",
        (45, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (10, 10, 10),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Viet Nam",
        (65, 300),
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        1.5,
        (120, 50, 30),
        3,
        cv2.LINE_AA,
    )
    return image


class TestBackgroundPreprocessor(unittest.TestCase):
    def test_auto_selects_green_for_yellow_background(self):
        image = synthetic_yellow_document()
        channel, selected, yellow_fraction = select_document_channel(image)

        self.assertEqual(selected, "green")
        self.assertGreater(yellow_fraction, 0.80)
        self.assertGreater(float(np.median(channel)), 200.0)

    def test_auto_selects_grayscale_for_neutral_document(self):
        image = np.full((300, 300, 3), 245, dtype=np.uint8)
        cv2.putText(image, "OCR", (70, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

        _, selected, yellow_fraction = select_document_channel(image)

        self.assertEqual(selected, "grayscale")
        self.assertLess(yellow_fraction, 0.01)

    def test_clahe_increases_low_contrast_variation(self):
        gradient = np.tile(np.linspace(120, 140, 320, dtype=np.uint8), (240, 1))
        enhanced = apply_clahe(gradient, clip_limit=2.2, tile_grid_size=8)

        self.assertGreater(float(np.std(enhanced)), float(np.std(gradient)))

    def test_sauvola_returns_binary_and_keeps_text(self):
        image = np.full((300, 500), 235, dtype=np.uint8)
        cv2.putText(image, "TEST OCR", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, 30, 5)
        binary = sauvola_binarize(image, window_size=41, k=0.20)

        self.assertEqual(set(np.unique(binary).tolist()), {0, 255})
        self.assertGreater(float(np.mean(binary < 128)), 0.005)
        self.assertLess(float(np.mean(binary < 128)), 0.30)

    def test_processes_manifest_and_writes_both_outputs(self):
        with tempfile.TemporaryDirectory(prefix="background_test_") as temp_dir:
            root = Path(temp_dir)
            masked_dir = root / "masked"
            output_dir = root / "preprocessed"
            page_dir = masked_dir / "page_0001"
            page_dir.mkdir(parents=True)
            image = synthetic_yellow_document()
            save_png(page_dir / "panel_01_masked.png", image)
            (masked_dir / "artifact_mask_manifest.json").write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "page_number": 1,
                                "panels": [
                                    {
                                        "panel_index": 1,
                                        "masked_file": "page_0001/panel_01_masked.png",
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = process_masked_panels(masked_dir, output_dir)

            self.assertEqual(result["successful_panels"], 1)
            self.assertEqual(result["selected_channel_counts"]["green"], 1)
            self.assertTrue(
                (output_dir / "page_0001" / "panel_01_enhanced.png").is_file()
            )
            self.assertTrue(
                (output_dir / "page_0001" / "panel_01_binary.png").is_file()
            )
            self.assertTrue((output_dir / "background_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
