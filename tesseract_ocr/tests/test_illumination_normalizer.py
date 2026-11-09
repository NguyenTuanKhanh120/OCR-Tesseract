import unittest

import cv2
import numpy as np

from tesseract_ocr.illumination_normalizer import (
    background_range,
    normalize_illumination,
    thin_line_retention,
)


class IlluminationNormalizerTests(unittest.TestCase):
    def test_reduces_gradient_background_and_preserves_dark_text(self):
        height, width = 500, 800
        gradient = np.tile(
            np.linspace(135, 250, height, dtype=np.uint8)[:, None], (1, width)
        )
        image = gradient.copy()
        cv2.putText(
            image,
            "DON DAT HANG",
            (90, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            20,
            4,
            cv2.LINE_AA,
        )
        before = background_range(image)
        balanced, metadata = normalize_illumination(image)

        self.assertTrue(metadata["applied"])
        self.assertLess(background_range(balanced), before * 0.30)
        self.assertGreater(thin_line_retention(image, balanced), 0.95)

    def test_skips_uniform_document(self):
        image = np.full((300, 500), 245, dtype=np.uint8)
        cv2.putText(image, "OCR", (150, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 4)
        balanced, metadata = normalize_illumination(image)

        self.assertFalse(metadata["applied"])
        np.testing.assert_array_equal(balanced, image)

    def test_defers_colored_background_to_step2(self):
        image = np.tile(
            np.linspace(120, 250, 300, dtype=np.uint8)[:, None], (1, 500)
        )
        balanced, metadata = normalize_illumination(
            image,
            color_chroma_p99=45.0,
        )

        self.assertFalse(metadata["applied"])
        self.assertEqual(metadata["reason"], "deferred_colored_background_to_step2")
        np.testing.assert_array_equal(balanced, image)

    def test_preserves_light_text_on_dark_region(self):
        image = np.full((400, 600), 248, dtype=np.uint8)
        image[70:330, 80:520] = 75
        cv2.putText(
            image,
            "PCCC",
            (185, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            220,
            5,
            cv2.LINE_AA,
        )
        balanced, metadata = normalize_illumination(image)

        self.assertTrue(metadata["applied"])
        self.assertTrue(metadata["reverse_content_detected"])
        text_source = image[170:235, 170:435]
        text_patch = balanced[170:235, 170:435]
        light_text_pixels = text_patch[text_source >= 150]
        dark_patch = balanced[90:130, 100:160]
        self.assertGreater(
            float(np.median(light_text_pixels)),
            float(np.median(dark_patch)) + 15,
        )

    def test_rejects_mismatched_shapes_for_retention(self):
        with self.assertRaises(ValueError):
            thin_line_retention(
                np.zeros((20, 20), dtype=np.uint8),
                np.zeros((21, 20), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
