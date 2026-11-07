import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from tesseract_ocr.document_renderer import save_png
from tesseract_ocr.region_ocr import (
    candidate_metrics,
    ocr_preprocessed_panels,
    parse_psm_values,
    score_candidates,
    text_quality_ratio,
)
from tesseract_ocr.schemas import BoundingBox, PageResult


def fake_result(text: str, confidence: float, psm: int) -> PageResult:
    words = text.split()
    boxes = [
        BoundingBox(
            text=word,
            confidence=confidence,
            x=index * 20,
            y=20,
            w=18,
            h=12,
        )
        for index, word in enumerate(words)
    ]
    return PageResult(
        page_number=1,
        status="success" if text else "empty",
        text=text,
        confidence=confidence,
        image_width=300,
        image_height=200,
        duration_seconds=0.1 + (psm / 100.0),
        boxes=boxes,
    )


class TestRegionOcr(unittest.TestCase):
    def test_parse_psm_values_deduplicates_and_validates(self):
        self.assertEqual(parse_psm_values([3, 6, 3, 11]), (3, 6, 11))
        with self.assertRaises(ValueError):
            parse_psm_values([14])

    def test_text_quality_supports_vietnamese(self):
        ratio = text_quality_ratio("Cộng hòa xã hội chủ nghĩa Việt Nam")
        self.assertGreater(ratio, 0.98)

    def test_candidate_metrics_counts_lines_and_diacritics(self):
        result = fake_result("Cộng hòa Việt Nam\nĐộc lập Tự do", 0.82, 6)
        metrics = candidate_metrics(result)
        self.assertEqual(metrics["line_count"], 2)
        self.assertGreater(metrics["vietnamese_diacritic_characters"], 0)

    def test_scoring_does_not_choose_short_high_confidence_result(self):
        short = {
            "candidate_id": "short",
            "input_variant": "enhanced",
            "psm": 3,
            "metrics": candidate_metrics(fake_result("Việt Nam", 0.95, 3)),
        }
        complete = {
            "candidate_id": "complete",
            "input_variant": "binary",
            "psm": 6,
            "metrics": candidate_metrics(
                fake_result(
                    "Cộng hòa xã hội chủ nghĩa Việt Nam Độc lập Tự do Hạnh phúc",
                    0.76,
                    6,
                )
            ),
        }
        ranked = score_candidates([short, complete])
        self.assertEqual(ranked[0]["candidate_id"], "complete")

    def test_scoring_penalizes_fragmented_sparse_output(self):
        fragmented_result = fake_result(
            "Một\nhai\nba\nbốn\nnăm\nsáu\nbảy\ntám\nchín\nmười", 0.80, 11
        )
        coherent_result = fake_result(
            "Một hai ba bốn năm\nsáu bảy tám chín mười", 0.78, 6
        )
        fragmented = {
            "candidate_id": "fragmented",
            "input_variant": "binary",
            "psm": 11,
            "metrics": candidate_metrics(fragmented_result),
        }
        coherent = {
            "candidate_id": "coherent",
            "input_variant": "enhanced",
            "psm": 6,
            "metrics": candidate_metrics(coherent_result),
        }
        ranked = score_candidates([fragmented, coherent])
        self.assertEqual(ranked[0]["candidate_id"], "coherent")

    def test_runs_matrix_and_writes_selected_output(self):
        with tempfile.TemporaryDirectory(prefix="region_ocr_test_") as temp_dir:
            root = Path(temp_dir)
            preprocessed = root / "preprocessed"
            output = root / "ocr"
            tessdata = root / "tessdata_best"
            page_dir = preprocessed / "page_0001"
            page_dir.mkdir(parents=True)
            tessdata.mkdir()
            (tessdata / "vie.traineddata").write_bytes(b"vie-test")
            (tessdata / "eng.traineddata").write_bytes(b"eng-test")
            image = np.full((200, 300), 255, dtype=np.uint8)
            cv2.putText(image, "OCR", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 4)
            save_png(page_dir / "panel_01_enhanced.png", image)
            save_png(page_dir / "panel_01_binary.png", image)
            (preprocessed / "background_manifest.json").write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "page_number": 1,
                                "panels": [
                                    {
                                        "panel_index": 1,
                                        "enhanced_file": "page_0001/panel_01_enhanced.png",
                                        "binary_file": "page_0001/panel_01_binary.png",
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            def mocked_ocr(image, config, page_number, prep_meta):
                if prep_meta["input_variant"] == "binary" and config.psm == 6:
                    return fake_result(
                        "Cộng hòa xã hội chủ nghĩa Việt Nam Độc lập Tự do",
                        0.82,
                        config.psm,
                    )
                return fake_result("Việt Nam", 0.70, config.psm)

            with patch(
                "tesseract_ocr.region_ocr.find_tesseract_executable",
                return_value="fake-tesseract.exe",
            ), patch(
                "tesseract_ocr.region_ocr.verify_tesseract_environment",
                return_value="tesseract 5.test",
            ), patch(
                "tesseract_ocr.region_ocr.run_tesseract_on_image",
                side_effect=mocked_ocr,
            ):
                result = ocr_preprocessed_panels(
                    preprocessed,
                    output,
                    tessdata_dir=tessdata,
                    psm_values=(3, 6),
                )

            self.assertEqual(result["candidate_run_count"], 4)
            panel = result["pages"][0]["panels"][0]
            self.assertEqual(panel["selected_candidate_id"], "binary_psm6")
            self.assertTrue(
                (output / "page_0001" / "panel_01" / "selected.txt").is_file()
            )
            self.assertTrue(
                (
                    output
                    / "page_0001"
                    / "panel_01"
                    / "selected_overlay.png"
                ).is_file()
            )
            self.assertTrue((output / "region_ocr_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
