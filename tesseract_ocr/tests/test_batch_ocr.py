import io
import unittest
import zipfile
from types import SimpleNamespace

from tesseract_ocr.batch_ocr import (
    build_results_zip,
    combine_page_results,
    safe_text_filename,
    unique_text_filenames,
)


class BatchOcrTests(unittest.TestCase):
    def test_safe_text_filename_removes_path_and_invalid_characters(self):
        self.assertEqual(
            safe_text_filename(r"C:\upload\Hồ sơ: PCCC?.pdf"),
            "Hồ sơ_ PCCC__ocr.txt",
        )

    def test_unique_names_do_not_overwrite_case_insensitive_duplicates(self):
        names = unique_text_filenames(["Mẫu.pdf", "mẫu.PDF", "Mẫu.png"])
        self.assertEqual(names, ["Mẫu_ocr.txt", "mẫu_ocr_2.txt", "Mẫu_ocr_3.txt"])

    def test_combines_pages_in_numeric_order(self):
        records = [
            {
                "page_number": 2,
                "result": SimpleNamespace(status="success", text="Hai", error=None),
            },
            {
                "page_number": 1,
                "result": SimpleNamespace(status="success", text="Một", error=None),
            },
        ]
        self.assertEqual(
            combine_page_results(records, 2),
            "--- [TRANG 1/2] ---\nMột\n\n--- [TRANG 2/2] ---\nHai",
        )

    def test_zip_contains_each_utf8_text_file(self):
        payload = build_results_zip(
            [("Một_ocr.txt", "Nội dung một"), ("Hai_ocr.txt", "Nội dung hai")]
        )
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertEqual(sorted(archive.namelist()), ["Hai_ocr.txt", "Một_ocr.txt"])
            self.assertEqual(
                archive.read("Một_ocr.txt").decode("utf-8"), "Nội dung một"
            )


if __name__ == "__main__":
    unittest.main()
