"""
Unit test kiểm tra cơ chế khớp nối tài liệu và trang (Matching Logic).
"""

import unittest
from pathlib import Path

from ocr_comparison.loaders import LoadedDocument, LoadedPage, normalize_relative_stem
from ocr_comparison.matching import match_documents


class TestMatching(unittest.TestCase):
    def test_normalize_relative_stem(self):
        # 1. Bỏ extension và hậu tố _ocr
        p1 = Path(r"PCCC\document_ocr.json")
        key1, cat1, stem1 = normalize_relative_stem(p1)
        self.assertEqual(key1, "pccc/document")
        self.assertEqual(cat1, "PCCC")
        self.assertEqual(stem1, "document")

        # 2. Xử lý file TXT chữ hoa chữ thường
        p2 = Path(r"PCCC/document_OCR.TXT")
        key2, cat2, stem2 = normalize_relative_stem(p2)
        self.assertEqual(key2, "pccc/document")
        self.assertEqual(stem2, "document")

        # 3. File ở thư mục gốc
        p3 = Path(r"single_file_ocr.txt")
        key3, cat3, stem3 = normalize_relative_stem(p3)
        self.assertEqual(key3, "single_file")
        self.assertEqual(cat3, "root")

    def test_match_documents_exact_and_unmatched(self):
        apple_docs = {
            "pccc/doc_a": LoadedDocument(
                relative_path="PCCC/doc_a_ocr.txt",
                category="PCCC",
                stem_name="doc_a",
                engine="apple_vision",
                total_pages=2,
                pages={1: LoadedPage(page_number=1, text="A1"), 2: LoadedPage(page_number=2, text="A2")},
            ),
            "bao_hiem/doc_b": LoadedDocument(
                relative_path="BAO_HIEM/doc_b_ocr.txt",
                category="BAO_HIEM",
                stem_name="doc_b",
                engine="apple_vision",
                total_pages=1,
                pages={1: LoadedPage(page_number=1, text="B1")},
            ),
        }

        tess_docs = {
            "pccc/doc_a": LoadedDocument(
                relative_path="PCCC/doc_a_ocr.json",
                category="PCCC",
                stem_name="doc_a",
                engine="tesseract",
                total_pages=2,
                pages={1: LoadedPage(page_number=1, text="A1"), 2: LoadedPage(page_number=2, text="A2")},
            ),
            "khac/doc_c": LoadedDocument(
                relative_path="KHAC/doc_c_ocr.json",
                category="KHAC",
                stem_name="doc_c",
                engine="tesseract",
                total_pages=1,
                pages={1: LoadedPage(page_number=1, text="C1")},
            ),
        }

        result = match_documents(apple_docs, tess_docs)

        # 1 tài liệu khớp
        self.assertEqual(result.total_matched, 1)
        self.assertEqual(result.matched_pairs[0].normalized_key, "pccc/doc_a")
        self.assertEqual(len(result.matched_pairs[0].page_pairs), 2)

        # Apple unmatched có doc_b
        self.assertEqual(len(result.unmatched_apple), 1)
        self.assertEqual(result.unmatched_apple[0].stem_name, "doc_b")

        # Tesseract unmatched có doc_c
        self.assertEqual(len(result.unmatched_tesseract), 1)
        self.assertEqual(result.unmatched_tesseract[0].stem_name, "doc_c")

    def test_do_not_match_different_categories(self):
        # Trùng tên file nhưng khác category: không được ghép!
        apple_docs = {
            "cat1/contract": LoadedDocument(
                relative_path="cat1/contract_ocr.txt",
                category="cat1",
                stem_name="contract",
                engine="apple_vision",
                total_pages=1,
            )
        }
        tess_docs = {
            "cat2/contract": LoadedDocument(
                relative_path="cat2/contract_ocr.json",
                category="cat2",
                stem_name="contract",
                engine="tesseract",
                total_pages=1,
            )
        }
        res = match_documents(apple_docs, tess_docs)
        self.assertEqual(res.total_matched, 0)
        self.assertEqual(len(res.unmatched_apple), 1)
        self.assertEqual(len(res.unmatched_tesseract), 1)


if __name__ == "__main__":
    unittest.main()
