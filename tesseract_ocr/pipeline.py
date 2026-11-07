"""
Pipeline thực thi Tesseract OCR hoàn chỉnh, độc lập và an toàn.
Hỗ trợ checkpoint theo từng trang, ghi file atomic và resume thông minh.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
try:
    import pymupdf as fitz
except ImportError:
    import fitz

from tesseract_ocr.config import TesseractConfig
from tesseract_ocr.preprocessing import preprocess_image
from tesseract_ocr.scanner import ScannedDocument, scan_input_directory
from tesseract_ocr.schemas import DocumentResult, PageResult
from tesseract_ocr.tesseract_engine import (
    render_pdf_page_to_image,
    run_tesseract_on_image,
    verify_tesseract_environment,
)

logger = logging.getLogger("tesseract_ocr.pipeline")


def atomic_write_text(target_path: Path, content: str, encoding: str = "utf-8") -> None:
    """Ghi file dạng atomic: ghi vào file tạm trên cùng phân vùng rồi rename thay thế."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = target_path.with_name(f"{target_path.name}.tmp_{os.getpid()}_{int(time.time()*1000)}")
    try:
        with open(temp_file, "w", encoding=encoding) as f:
            f.write(content)
        # Thay thế atomic
        os.replace(temp_file, target_path)
    finally:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass


class TesseractPipeline:
    """Pipeline điều phối quét tài liệu, OCR và lưu trữ kết quả."""

    def __init__(self, config: TesseractConfig):
        self.config = config
        self.config.validate()
        self.engine_version = "Tesseract 5.x"

    def initialize(self) -> None:
        """Khởi tạo và kiểm tra môi trường Tesseract."""
        self.engine_version = verify_tesseract_environment(self.config)
        logger.info(f"Đã xác nhận môi trường Tesseract: {self.engine_version}")

    def get_output_paths(self, scanned_doc: ScannedDocument) -> tuple[Path, Path]:
        """
        Xác định đường dẫn JSON checkpoint và TXT đầu ra.

        Khi metadata_dir được cấu hình, JSON được ghi sang thư mục
        metadata riêng; output_dir chỉ còn một file TXT cho mỗi tài liệu.
        """
        stem = scanned_doc.relative_path.stem
        parent_rel = scanned_doc.relative_path.parent
        txt_base = self.config.output_dir / parent_rel / f"{stem}_ocr"
        json_root = self.config.metadata_dir or self.config.output_dir
        json_base = json_root / parent_rel / f"{stem}_ocr"
        # Không dùng Path.with_suffix() ở đây: nếu tên gốc có dấu
        # chấm (ví dụ "HDKT 04.2020-...pdf"), with_suffix sẽ coi phần
        # sau dấu chấm là extension và cắt mất tên, dẫn tới ghi đè.
        return Path(f"{json_base}.json"), Path(f"{txt_base}.txt")

    def should_skip_document(
        self,
        scanned_doc: ScannedDocument,
        json_path: Path,
        txt_path: Path,
    ) -> bool:
        """
        Kiểm tra xem tài liệu đã hoàn thành chưa để resume.
        Bỏ qua nếu JSON và TXT đã tồn tại, source_hash khớp, config_hash khớp và status thành công.
        """
        if self.config.force:
            return False

        if not json_path.exists() or not txt_path.exists():
            return False

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                content = f.read()
            doc_result = DocumentResult.from_json(content)

            # Kiểm tra hash nguồn và hash cấu hình
            current_config_hash = self.config.compute_config_hash()
            if (
                doc_result.source_hash == scanned_doc.file_hash
                and doc_result.config_hash == current_config_hash
                and doc_result.status == "success"
                and doc_result.processed_pages == doc_result.total_pages
                and doc_result.total_pages > 0
            ):
                return True
        except Exception as e:
            logger.warning(f"Không thể đọc kết quả cũ của {json_path}: {e}")

        return False

    def process_document(
        self,
        scanned_doc: ScannedDocument,
        on_page_completed: Optional[Callable[[int, int], None]] = None,
    ) -> DocumentResult:
        """
        Xử lý OCR một tài liệu hoàn chỉnh (PDF nhiều trang hoặc file ảnh đơn).
        Ghi checkpoint JSON và TXT sau mỗi trang.
        """
        json_path, txt_path = self.get_output_paths(scanned_doc)

        # Kiểm tra resume
        if self.should_skip_document(scanned_doc, json_path, txt_path):
            logger.info(f"Bỏ qua (đã hoàn thành trước đó): {scanned_doc.relative_path}")
            with open(json_path, "r", encoding="utf-8") as f:
                return DocumentResult.from_json(f.read())

        config_hash = self.config.compute_config_hash()

        # Tạo khung DocumentResult
        doc_result = DocumentResult(
            schema_version="1.0",
            engine="tesseract",
            engine_version=self.engine_version,
            source_file=str(scanned_doc.relative_path).replace("\\", "/"),
            source_type=scanned_doc.file_type,
            source_hash=scanned_doc.file_hash,
            config_hash=config_hash,
            status="partial",
            total_pages=1,
            processed_pages=0,
            configuration=self.config.to_ocr_dict(),
            pages=[],
        )

        try:
            if scanned_doc.file_type == "pdf":
                self._process_pdf(scanned_doc, doc_result, json_path, txt_path, on_page_completed)
            else:
                self._process_image(scanned_doc, doc_result, json_path, txt_path, on_page_completed)

            # Đánh giá status chung
            has_error = any(p.status == "error" for p in doc_result.pages)
            if has_error:
                doc_result.status = "partial"
            else:
                doc_result.status = "success"

        except Exception as doc_err:
            logger.error(f"Lỗi xử lý tài liệu {scanned_doc.relative_path}: {doc_err}")
            doc_result.status = "error"

        # Ghi file hoàn tất
        atomic_write_text(json_path, doc_result.to_json())
        atomic_write_text(txt_path, doc_result.to_txt())
        return doc_result

    def _process_pdf(
        self,
        scanned_doc: ScannedDocument,
        doc_result: DocumentResult,
        json_path: Path,
        txt_path: Path,
        on_page_completed: Optional[Callable[[int, int], None]],
    ) -> None:
        """Xử lý từng trang của tài liệu PDF."""
        doc = fitz.open(scanned_doc.path)
        total_pages = len(doc)
        doc_result.total_pages = total_pages

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            fitz_page = doc[page_idx]

            # Kiểm tra tùy chọn native text (mặc định tắt theo yêu cầu mục 6.2)
            if self.config.allow_native_text:
                native_text = fitz_page.get_text().strip()
                if native_text:
                    p_res = PageResult(
                        page_number=page_num,
                        status="success",
                        method="native_text",
                        text=native_text,
                        confidence=1.0,
                        image_width=int(fitz_page.rect.width),
                        image_height=int(fitz_page.rect.height),
                        duration_seconds=0.01,
                    )
                    doc_result.pages.append(p_res)
                    doc_result.processed_pages = len(doc_result.pages)
                    atomic_write_text(json_path, doc_result.to_json())
                    atomic_write_text(txt_path, doc_result.to_txt())
                    if on_page_completed:
                        on_page_completed(page_num, total_pages)
                    continue

            # Render trang thành ảnh ở DPI chỉ định
            img = render_pdf_page_to_image(fitz_page, dpi=self.config.dpi)

            # Tiền xử lý ảnh
            processed_img, prep_meta = preprocess_image(img, mode=self.config.preprocessing)

            # Chạy Tesseract OCR
            p_res = run_tesseract_on_image(
                processed_img,
                self.config,
                page_number=page_num,
                prep_meta=prep_meta.to_dict(),
            )
            doc_result.pages.append(p_res)
            doc_result.processed_pages = len(doc_result.pages)

            # Checkpoint ghi sau mỗi trang
            atomic_write_text(json_path, doc_result.to_json())
            atomic_write_text(txt_path, doc_result.to_txt())

            if on_page_completed:
                on_page_completed(page_num, total_pages)

        doc.close()

    def _process_image(
        self,
        scanned_doc: ScannedDocument,
        doc_result: DocumentResult,
        json_path: Path,
        txt_path: Path,
        on_page_completed: Optional[Callable[[int, int], None]],
    ) -> None:
        """Xử lý file ảnh đơn."""
        doc_result.total_pages = 1
        img = cv2.imdecode(np.fromfile(str(scanned_doc.path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Không thể mở file ảnh: {scanned_doc.path}")

        processed_img, prep_meta = preprocess_image(img, mode=self.config.preprocessing)
        p_res = run_tesseract_on_image(
            processed_img,
            self.config,
            page_number=1,
            prep_meta=prep_meta.to_dict(),
        )
        doc_result.pages.append(p_res)
        doc_result.processed_pages = 1

        atomic_write_text(json_path, doc_result.to_json())
        atomic_write_text(txt_path, doc_result.to_txt())

        if on_page_completed:
            on_page_completed(1, 1)

    def run(self) -> List[DocumentResult]:
        """Thực thi pipeline trên toàn bộ tài liệu tìm thấy trong input_dir."""
        self.initialize()
        docs = scan_input_directory(self.config.input_dir, max_documents=self.config.max_documents)
        logger.info(f"Tìm thấy {len(docs)} tài liệu nguồn cần xử lý trong {self.config.input_dir}")

        results: List[DocumentResult] = []
        for i, scanned_doc in enumerate(docs, 1):
            logger.info(f"[{i}/{len(docs)}] Xử lý: {scanned_doc.relative_path}")
            res = self.process_document(scanned_doc)
            results.append(res)

        logger.info("Hoàn thành toàn bộ tiến trình Tesseract OCR.")
        return results
