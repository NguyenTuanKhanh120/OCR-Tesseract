import json
import tempfile
import unittest
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from tesseract_ocr.document_renderer import render_pdf_to_pngs


class TestDocumentRenderer(unittest.TestCase):
    def test_renders_every_pdf_page_and_writes_manifest(self):
        with tempfile.TemporaryDirectory(prefix="ocr_render_test_") as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            output_dir = root / "rendered"
            source_dir.mkdir()
            pdf_path = source_dir / "two_pages.pdf"

            document = fitz.open()
            for page_number in (1, 2):
                page = document.new_page(width=300, height=200)
                page.insert_text((30, 80), f"PAGE {page_number}", fontsize=24)
            document.save(pdf_path)
            document.close()

            manifest = render_pdf_to_pngs(pdf_path, output_dir, dpi=150)

            self.assertEqual(manifest["page_count"], 2)
            self.assertEqual(manifest["dpi"], 150)
            self.assertTrue((output_dir / "page_0001.png").is_file())
            self.assertTrue((output_dir / "page_0002.png").is_file())
            self.assertTrue((output_dir / "render_manifest.json").is_file())

            saved_manifest = json.loads(
                (output_dir / "render_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_manifest["page_count"], 2)
            self.assertEqual([page["page_number"] for page in saved_manifest["pages"]], [1, 2])
            self.assertTrue(all(page["png_sha256"] for page in saved_manifest["pages"]))

    def test_rejects_non_pdf_input(self):
        with tempfile.TemporaryDirectory(prefix="ocr_render_test_") as temp_dir:
            root = Path(temp_dir)
            text_path = root / "input.txt"
            text_path.write_text("not a pdf", encoding="utf-8")
            with self.assertRaises(ValueError):
                render_pdf_to_pngs(text_path, root / "output")


if __name__ == "__main__":
    unittest.main()
