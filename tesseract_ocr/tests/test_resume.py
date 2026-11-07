"""
Unit test kiểm tra cơ chế Checkpoint, Resume và nhận diện thay đổi cấu hình.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from tesseract_ocr.config import TesseractConfig
from tesseract_ocr.pipeline import TesseractPipeline, atomic_write_text
from tesseract_ocr.scanner import ScannedDocument
from tesseract_ocr.schemas import DocumentResult, PageResult


class TestResumeAndCheckpoint(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="tess_test_"))
        self.input_dir = self.test_dir / "input"
        self.output_dir = self.test_dir / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_atomic_write_text(self):
        file_path = self.output_dir / "test_atomic.txt"
        atomic_write_text(file_path, "Dữ liệu thử nghiệm")
        self.assertTrue(file_path.exists())
        self.assertEqual(file_path.read_text(encoding="utf-8"), "Dữ liệu thử nghiệm")

    def test_should_skip_document(self):
        config = TesseractConfig(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            lang="vie+eng",
            dpi=300,
        )
        pipeline = TesseractPipeline(config)

        scanned = ScannedDocument(
            path=self.input_dir / "sample.pdf",
            relative_path=Path("sample.pdf"),
            file_type="pdf",
            file_hash="hash_12345",
            file_size_bytes=1024,
        )

        json_path, txt_path = pipeline.get_output_paths(scanned)

        # 1. Chưa có file kết quả -> không skip
        self.assertFalse(pipeline.should_skip_document(scanned, json_path, txt_path))

        # 2. Tạo kết quả hoàn chỉnh hợp lệ
        doc_res = DocumentResult(
            schema_version="1.0",
            engine="tesseract",
            engine_version="5.0",
            source_file="sample.pdf",
            source_type="pdf",
            source_hash="hash_12345",
            config_hash=config.compute_config_hash(),
            status="success",
            total_pages=1,
            processed_pages=1,
            configuration=config.to_ocr_dict(),
            pages=[PageResult(page_number=1, text="Test", status="success")],
        )
        atomic_write_text(json_path, doc_res.to_json())
        atomic_write_text(txt_path, doc_res.to_txt())

        # Kiểm tra: file khớp hash và thành công -> nên skip
        self.assertTrue(pipeline.should_skip_document(scanned, json_path, txt_path))

        # 3. Thay đổi cấu hình OCR (ví dụ đổi lang sang 'eng') -> config_hash thay đổi -> không skip!
        config_different = TesseractConfig(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            lang="eng",
            dpi=300,
        )
        pipeline_diff = TesseractPipeline(config_different)
        self.assertFalse(pipeline_diff.should_skip_document(scanned, json_path, txt_path))

        # 4. Khi bật cờ --force -> không bao giờ skip
        pipeline.config.force = True
        self.assertFalse(pipeline.should_skip_document(scanned, json_path, txt_path))

    def test_metadata_dir_keeps_json_separate_from_txt_output(self):
        metadata_dir = self.test_dir / "metadata"
        config = TesseractConfig(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            metadata_dir=metadata_dir,
        )
        pipeline = TesseractPipeline(config)
        scanned = ScannedDocument(
            path=self.input_dir / "nested" / "document.pdf",
            relative_path=Path("nested/document.pdf"),
            file_type="pdf",
            file_hash="abc123",
            file_size_bytes=1024,
        )

        json_path, txt_path = pipeline.get_output_paths(scanned)

        self.assertEqual(
            json_path,
            metadata_dir.resolve() / "nested" / "document_ocr.json",
        )
        self.assertEqual(
            txt_path,
            self.output_dir.resolve() / "nested" / "document_ocr.txt",
        )

    def test_output_paths_preserve_dots_in_source_filename(self):
        metadata_dir = self.test_dir / "metadata"
        config = TesseractConfig(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            metadata_dir=metadata_dir,
        )
        pipeline = TesseractPipeline(config)
        scanned = ScannedDocument(
            path=self.input_dir / "HDKT 04.2020-ABC.pdf",
            relative_path=Path("contracts/HDKT 04.2020-ABC.pdf"),
            file_type="pdf",
            file_hash="dot-name",
            file_size_bytes=1024,
        )

        json_path, txt_path = pipeline.get_output_paths(scanned)

        self.assertEqual(
            json_path.name,
            "HDKT 04.2020-ABC_ocr.json",
        )
        self.assertEqual(
            txt_path.name,
            "HDKT 04.2020-ABC_ocr.txt",
        )


if __name__ == "__main__":
    unittest.main()
