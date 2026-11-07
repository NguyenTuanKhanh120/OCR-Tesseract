"""
Bộ quét đệ quy các tài liệu PDF và ảnh nguồn.
Bảo toàn cấu trúc thư mục tương đối và tính hash nguồn.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tif",
    ".tiff",
}


@dataclass
class ScannedDocument:
    """Đại diện cho một tài liệu nguồn cần xử lý."""
    path: Path
    relative_path: Path
    file_type: str  # 'pdf' hoặc 'image'
    file_hash: str
    file_size_bytes: int

    @property
    def category(self) -> str:
        """Thư mục phân loại cấp cao nhất (hoặc 'root' nếu nằm ngay ở thư mục gốc)."""
        parts = self.relative_path.parts
        if len(parts) > 1:
            return parts[0]
        return "root"


def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Tính mã SHA256 cho file nguồn."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def scan_input_directory(
    input_dir: Path,
    max_documents: Optional[int] = None,
) -> List[ScannedDocument]:
    """
    Quét toàn bộ tài liệu hợp lệ trong thư mục nguồn theo thứ tự xác định.
    Không ghi đè hoặc can thiệp dữ liệu trong thư mục nguồn.
    """
    input_path = Path(input_dir).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Thư mục nguồn không tồn tại: {input_path}")

    documents: List[ScannedDocument] = []

    # Thu thập tất cả file có đuôi hợp lệ
    for entry in sorted(input_path.rglob("*")):
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
            rel_path = entry.relative_to(input_path)
            ext = entry.suffix.lower()
            file_type = "pdf" if ext == ".pdf" else "image"
            f_hash = compute_file_sha256(entry)
            size = entry.stat().st_size

            documents.append(
                ScannedDocument(
                    path=entry,
                    relative_path=rel_path,
                    file_type=file_type,
                    file_hash=f_hash,
                    file_size_bytes=size,
                )
            )

            if max_documents and len(documents) >= max_documents:
                break

    return documents
