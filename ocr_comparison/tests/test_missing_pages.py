"""
Unit test kiểm tra xử lý trường hợp trang bị thiếu, rỗng hoặc số lượng trang không đồng đều.
"""

import unittest

from ocr_comparison.loaders import LoadedDocument, LoadedPage
from ocr_comparison.matching import match_documents
from ocr_comparison.metrics import evaluate_page_pair


class TestMissingPages(unittest.TestCase):
    def test_mismatched_page_counts(self):
        # Apple có 3 trang, Tesseract chỉ có 2 trang
        apple_doc = LoadedDocument(
            relative_path="HOP_DONG/hd_01_ocr.txt",
            category="HOP_DONG",
            stem_name="hd_01",
            engine="apple_vision",
            total_pages=3,
            pages={
                1: LoadedPage(page_number=1, text="Nội dung trang 1", status="success"),
                2: LoadedPage(page_number=2, text="Nội dung trang 2", status="success"),
                3: LoadedPage(page_number=3, text="Nội dung trang 3", status="success"),
            },
        )
        tess_doc = LoadedDocument(
            relative_path="HOP_DONG/hd_01_ocr.json",
            category="HOP_DONG",
            stem_name="hd_01",
            engine="tesseract",
            total_pages=2,
            pages={
                1: LoadedPage(page_number=1, text="Nội dung trang 1", status="success"),
                2: LoadedPage(page_number=2, text="Nội dung trang 2", status="success"),
            },
        )

        apple_docs = {"hop_dong/hd_01": apple_doc}
        tess_docs = {"hop_dong/hd_01": tess_doc}

        matching = match_documents(apple_docs, tess_docs)
        self.assertEqual(matching.total_matched, 1)

        pair = matching.matched_pairs[0]
        # Tổng số trang được ghép là 3
        self.assertEqual(len(pair.page_pairs), 3)

        # Trang 1 và 2 đầy đủ
        self.assertTrue(pair.page_pairs[0].is_complete)
        self.assertTrue(pair.page_pairs[1].is_complete)

        # Trang 3: Tesseract bị thiếu
        p3 = pair.page_pairs[2]
        self.assertEqual(p3.page_number, 3)
        self.assertEqual(p3.apple_status, "success")
        self.assertEqual(p3.tesseract_status, "missing")

    def test_evaluate_missing_page(self):
        # Đánh giá metric khi một bên bị missing
        metric = evaluate_page_pair(
            doc_id="D1",
            relative_path="PCCC/d1",
            category="PCCC",
            page_number=3,
            apple_text="Trang 3 Apple",
            tesseract_text="",
            apple_status="success",
            tesseract_status="missing",
            ground_truth_text="Trang 3 Apple Chuẩn",
        )
        self.assertEqual(metric.tesseract_status, "missing")
        self.assertEqual(metric.tesseract_chars, 0)
        self.assertEqual(metric.tesseract_cer_pct, 100.0)
        self.assertEqual(metric.winner, "apple")


if __name__ == "__main__":
    unittest.main()
