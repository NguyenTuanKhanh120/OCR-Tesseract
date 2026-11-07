"""
Unit test kiểm tra tính toàn vẹn của JSON Schemas và TXT export.
"""

import json
import unittest

from tesseract_ocr.schemas import BoundingBox, DocumentResult, PageResult, PreprocessingMeta


class TestSchemas(unittest.TestCase):
    def test_bounding_box_serialization(self):
        box = BoundingBox(
            text="CỘNG HÒA",
            confidence=0.952,
            x=100,
            y=150,
            w=200,
            h=30,
            block_num=1,
            par_num=1,
            line_num=1,
            word_num=1,
        )
        data = box.to_dict()
        self.assertEqual(data["text"], "CỘNG HÒA")
        self.assertEqual(data["confidence"], 0.952)
        self.assertEqual(data["x"], 100)

        restored = BoundingBox.from_dict(data)
        self.assertEqual(restored.text, box.text)
        self.assertEqual(restored.confidence, box.confidence)
        self.assertEqual(restored.w, box.w)

    def test_page_and_document_serialization(self):
        page1 = PageResult(
            page_number=1,
            status="success",
            method="tesseract",
            text="Trang 1 nội dung",
            confidence=0.88,
            image_width=2000,
            image_height=3000,
            duration_seconds=1.25,
            preprocessing={"deskew_angle": 1.2},
            boxes=[
                BoundingBox(text="Trang", confidence=0.9, x=10, y=20, w=50, h=20),
                BoundingBox(text="1", confidence=0.86, x=65, y=20, w=20, h=20),
            ],
        )
        page2 = PageResult(
            page_number=2,
            status="success",
            method="tesseract",
            text="Trang 2 nội dung",
            confidence=0.92,
            image_width=2000,
            image_height=3000,
            duration_seconds=1.1,
        )

        doc = DocumentResult(
            schema_version="1.0",
            engine="tesseract",
            engine_version="5.3.0",
            source_file="PCCC/test.pdf",
            source_type="pdf",
            source_hash="abcdef123456",
            config_hash="cfg987654321",
            status="success",
            total_pages=2,
            processed_pages=2,
            configuration={"dpi": 300, "lang": "vie+eng"},
            pages=[page1, page2],
        )

        json_str = doc.to_json()
        restored_doc = DocumentResult.from_json(json_str)

        self.assertEqual(restored_doc.engine, "tesseract")
        self.assertEqual(restored_doc.total_pages, 2)
        self.assertEqual(restored_doc.processed_pages, 2)
        self.assertEqual(len(restored_doc.pages), 2)
        self.assertEqual(restored_doc.pages[0].text, "Trang 1 nội dung")
        self.assertEqual(len(restored_doc.pages[0].boxes), 2)

        # Kiểm tra định dạng TXT theo phân cách chuẩn
        txt_str = doc.to_txt()
        self.assertIn("--- [TRANG 1/2] ---", txt_str)
        self.assertIn("Trang 1 nội dung", txt_str)
        self.assertIn("--- [TRANG 2/2] ---", txt_str)
        self.assertIn("Trang 2 nội dung", txt_str)


if __name__ == "__main__":
    unittest.main()
