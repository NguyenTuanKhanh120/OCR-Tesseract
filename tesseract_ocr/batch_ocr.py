"""Tiện ích đầu ra cho chế độ OCR nhiều file của giao diện web."""

from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_text_filename(source_name: str) -> str:
    """Tạo tên TXT an toàn từ tên file tải lên, không giữ thành phần thư mục."""
    leaf_name = Path(str(source_name).replace("\\", "/")).name
    stem = INVALID_FILENAME.sub("_", Path(leaf_name).stem).strip(" .")
    if not stem:
        stem = "document"
    return f"{stem}_ocr.txt"


def unique_text_filenames(source_names: Sequence[str]) -> List[str]:
    """Khử trùng tên TXT theo thứ tự tải lên, không ghi đè file trong ZIP."""
    used = set()
    outputs: List[str] = []
    for source_name in source_names:
        base_name = safe_text_filename(source_name)
        stem = Path(base_name).stem
        suffix = Path(base_name).suffix
        candidate = base_name
        index = 2
        while candidate.casefold() in used:
            candidate = f"{stem}_{index}{suffix}"
            index += 1
        used.add(candidate.casefold())
        outputs.append(candidate)
    return outputs


def combine_page_results(page_records: Sequence[Dict[str, Any]], total_pages: int) -> str:
    """Ghép kết quả của một tài liệu theo đúng thứ tự trang."""
    sections = []
    for record in sorted(page_records, key=lambda item: int(item["page_number"])):
        result = record["result"]
        page_number = int(record["page_number"])
        page_text = (
            result.text
            if result.status != "error"
            else f"[LỖI OCR: {result.error}]"
        )
        page_text = unicodedata.normalize("NFC", page_text or "").strip()
        sections.append(f"--- [TRANG {page_number}/{total_pages}] ---\n{page_text}".rstrip())
    return "\n\n".join(sections).strip()


def build_results_zip(documents: Iterable[Tuple[str, str]]) -> bytes:
    """Đóng gói các TXT UTF-8 vào ZIP trong bộ nhớ."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, text in documents:
            archive.writestr(filename, unicodedata.normalize("NFC", text).encode("utf-8"))
    return buffer.getvalue()
