import tempfile
import unittest
from pathlib import Path

from tesseract_ocr.benchmark_variants import (
    count_missed_lines,
    iter_variants,
    load_reference_text,
)


class TestBenchmarkVariants(unittest.TestCase):
    def test_default_matrix_has_twelve_variants(self):
        variants = list(iter_variants())
        self.assertEqual(len(variants), 12)
        self.assertEqual(len({item["variant_id"] for item in variants}), 12)

    def test_count_missed_lines(self):
        reference = "Cộng hòa xã hội chủ nghĩa Việt Nam\nĐộc lập Tự do Hạnh phúc"
        hypothesis = "Cộng hòa xã hội chủ nghĩa Việt Nam"
        missed, total = count_missed_lines(reference, hypothesis)
        self.assertEqual(total, 2)
        self.assertEqual(missed, 1)

    def test_load_reference_text_removes_ios_page_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sample_ocr.txt").write_text(
                "--- [TRANG 1/1] ---\nCộng hòa xã hội chủ nghĩa Việt Nam\n",
                encoding="utf-8",
            )
            result = load_reference_text(root, Path("sample.pdf"))
            self.assertEqual(result, "Cộng hòa xã hội chủ nghĩa Việt Nam")


if __name__ == "__main__":
    unittest.main()
