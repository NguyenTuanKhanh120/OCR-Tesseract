import unittest

from tesseract_ocr.ocr_variant_selector import select_ocr_variant


def variant(confidence, suspicious, words, symbols=0.02):
    return {
        "confidence": confidence,
        "word_count": words,
        "text_metrics": {
            "suspicious_token_ratio": suspicious,
            "symbol_character_ratio": symbols,
            "non_latin_letter_ratio": 0.0,
        },
    }


class OcrVariantSelectorTests(unittest.TestCase):
    def test_identical_pixels_keep_current(self):
        result = select_ocr_variant(
            variant(0.70, 0.10, 100),
            variant(0.80, 0.05, 100),
            pixel_difference_ratio=0.0,
        )
        self.assertEqual(result["selected"], "current")
        self.assertEqual(result["reason"], "identical_pixels_keep_current")

    def test_selects_candidate_with_better_confidence_and_text(self):
        result = select_ocr_variant(
            variant(0.70, 0.15, 100),
            variant(0.75, 0.08, 95),
            pixel_difference_ratio=0.40,
        )
        self.assertEqual(result["selected"], "candidate")
        self.assertTrue(result["guardrail_passed"])

    def test_rejects_confidence_drop(self):
        result = select_ocr_variant(
            variant(0.70, 0.10, 100),
            variant(0.55, 0.11, 105),
            pixel_difference_ratio=0.30,
        )
        self.assertEqual(result["selected"], "current")
        self.assertEqual(result["reason"], "candidate_confidence_drop")

    def test_rejects_suspicious_token_spike(self):
        result = select_ocr_variant(
            variant(0.60, 0.04, 300),
            variant(0.64, 0.09, 360),
            pixel_difference_ratio=0.45,
        )
        self.assertEqual(result["selected"], "current")
        self.assertEqual(result["reason"], "candidate_suspicious_tokens_increased")

    def test_rejects_candidate_with_too_little_text(self):
        result = select_ocr_variant(
            variant(0.50, 0.20, 100),
            variant(0.80, 0.05, 20),
            pixel_difference_ratio=0.20,
        )
        self.assertEqual(result["selected"], "current")
        self.assertEqual(result["reason"], "candidate_text_too_short")

    def test_validates_pixel_ratio(self):
        with self.assertRaises(ValueError):
            select_ocr_variant(
                variant(0.50, 0.20, 100),
                variant(0.60, 0.10, 100),
                pixel_difference_ratio=1.2,
            )


if __name__ == "__main__":
    unittest.main()
