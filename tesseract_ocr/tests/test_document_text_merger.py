import json
import tempfile
import unicodedata
import unittest
from pathlib import Path

from tesseract_ocr.document_renderer import sha256_file
from tesseract_ocr.document_text_merger import merge_region_ocr_results


class DocumentTextMergerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.region_dir = self.root / "step6"
        self.output_dir = self.root / "step7"
        self.source_pdf = self.root / "Tài liệu thử.pdf"
        self.source_pdf.write_bytes(b"%PDF-1.4\nmock\n")

    def tearDown(self):
        self.temporary.cleanup()

    def write_text(self, relative_path: str, text: str) -> None:
        path = self.region_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def write_region_manifest(self, pages, **overrides) -> Path:
        manifest = {
            "schema_version": "1.0",
            "operation": "ocr_panels_with_psm_selection",
            "source_file": str(self.source_pdf),
            "source_sha256": sha256_file(self.source_pdf),
            "page_count": len(pages),
            "panel_count": sum(len(page["panels"]) for page in pages),
            "pages": pages,
        }
        manifest.update(overrides)
        path = self.region_dir / "region_ocr_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        return path

    @staticmethod
    def panel(index: int, relative_path: str, status: str = "success"):
        return {
            "panel_index": index,
            "status": status,
            "selected_candidate_id": "enhanced_psm6",
            "selected_text_file": relative_path,
        }

    def test_merges_pages_and_panels_in_numeric_order(self):
        self.write_text("page_0001/panel_01/selected.txt", "Một\r\n")
        self.write_text("page_0001/panel_02/selected.txt", "Hai\n")
        self.write_text("page_0002/panel_01/selected.txt", "Ba\n")
        pages = [
            {
                "page_number": 2,
                "panel_count": 1,
                "panels": [self.panel(1, "page_0002/panel_01/selected.txt")],
            },
            {
                "page_number": 1,
                "panel_count": 2,
                "panels": [
                    self.panel(2, "page_0001/panel_02/selected.txt"),
                    self.panel(1, "page_0001/panel_01/selected.txt"),
                ],
            },
        ]
        result = merge_region_ocr_results(
            self.write_region_manifest(pages), self.output_dir
        )
        output_path = Path(result["output_text_file"])
        self.assertEqual(output_path.name, "Tài liệu thử_ocr.txt")
        self.assertEqual(
            output_path.read_text(encoding="utf-8"),
            "--- [TRANG 1/2] ---\nMột\n\nHai\n\n"
            "--- [TRANG 2/2] ---\nBa\n",
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["page_count"], 2)
        self.assertEqual(result["panel_count"], 3)
        self.assertTrue((self.output_dir / "merge_manifest.json").is_file())

    def test_normalizes_unicode_to_nfc(self):
        decomposed = unicodedata.normalize("NFD", "Công ty")
        self.write_text("page_0001/panel_01/selected.txt", decomposed)
        pages = [
            {
                "page_number": 1,
                "panel_count": 1,
                "panels": [self.panel(1, "page_0001/panel_01/selected.txt")],
            }
        ]
        result = merge_region_ocr_results(
            self.write_region_manifest(pages), self.output_dir
        )
        text = Path(result["output_text_file"]).read_text(encoding="utf-8")
        self.assertEqual(text, unicodedata.normalize("NFC", text))
        self.assertIn("Công ty", text)

    def test_follows_manifest_lineage_to_source_pdf(self):
        self.write_text("page_0001/panel_01/selected.txt", "Nội dung")
        render_manifest = self.root / "render_manifest.json"
        render_manifest.write_text(
            json.dumps(
                {
                    "source_file": str(self.source_pdf),
                    "source_sha256": sha256_file(self.source_pdf),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        pages = [
            {
                "page_number": 1,
                "panel_count": 1,
                "panels": [self.panel(1, "page_0001/panel_01/selected.txt")],
            }
        ]
        path = self.write_region_manifest(
            pages,
            source_file=None,
            source_sha256=None,
            source_render_manifest=str(render_manifest),
            source_render_manifest_sha256=sha256_file(render_manifest),
        )
        result = merge_region_ocr_results(path, self.output_dir)
        self.assertEqual(result["source_pdf"], str(self.source_pdf.resolve()))
        self.assertEqual(len(result["manifest_lineage"]), 1)

    def test_rejects_missing_page(self):
        self.write_text("page_0001/panel_01/selected.txt", "Nội dung")
        pages = [
            {
                "page_number": 1,
                "panel_count": 1,
                "panels": [self.panel(1, "page_0001/panel_01/selected.txt")],
            }
        ]
        path = self.write_region_manifest(pages, page_count=2)
        with self.assertRaisesRegex(ValueError, "Số trang"):
            merge_region_ocr_results(path, self.output_dir)

    def test_rejects_missing_selected_text(self):
        pages = [
            {
                "page_number": 1,
                "panel_count": 1,
                "panels": [self.panel(1, "page_0001/panel_01/selected.txt")],
            }
        ]
        with self.assertRaises(FileNotFoundError):
            merge_region_ocr_results(
                self.write_region_manifest(pages), self.output_dir
            )

    def test_rejects_selected_text_outside_region_directory(self):
        outside = self.root / "outside.txt"
        outside.write_text("Không được đọc", encoding="utf-8")
        pages = [
            {
                "page_number": 1,
                "panel_count": 1,
                "panels": [self.panel(1, "../outside.txt")],
            }
        ]
        with self.assertRaisesRegex(ValueError, "nằm ngoài"):
            merge_region_ocr_results(
                self.write_region_manifest(pages), self.output_dir
            )


if __name__ == "__main__":
    unittest.main()
