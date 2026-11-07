"""Render toàn bộ trang PDF thành PNG trước các bước phân vùng/OCR.

Module này chỉ thực hiện bước 1 của pipeline OCR theo vùng:
- đọc PDF;
- render đầy đủ mọi trang ở DPI cấu hình được;
- ghi PNG và manifest để kiểm chứng.

Không cắt vùng, không chỉnh phối cảnh và không chạy OCR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from tesseract_ocr.tesseract_engine import render_pdf_page_to_image


if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp_{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_png(path: Path, image) -> None:
    """Ghi PNG an toàn với đường dẫn Windows có ký tự Unicode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"Không thể mã hóa PNG: {path}")
    temporary = path.with_name(f"{path.stem}.tmp_{os.getpid()}.png")
    try:
        encoded.tofile(str(temporary))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def render_pdf_to_pngs(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 400,
) -> Dict[str, Any]:
    """Render mọi trang PDF và trả về manifest mô tả kết quả."""
    pdf_path = Path(pdf_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy PDF: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Đầu vào không phải PDF: {pdf_path}")
    if dpi <= 0:
        raise ValueError(f"DPI phải lớn hơn 0, nhận được: {dpi}")
    if output_dir == pdf_path.parent or output_dir in pdf_path.parents:
        raise ValueError("Thư mục render phải tách khỏi thư mục chứa PDF nguồn")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    page_entries: List[Dict[str, Any]] = []

    try:
        document = fitz.open(pdf_path)
    except Exception as error:
        raise RuntimeError(f"Không thể mở PDF '{pdf_path}': {error}") from error

    try:
        if document.needs_pass:
            raise RuntimeError(f"PDF đang được bảo vệ bằng mật khẩu: {pdf_path}")
        if len(document) <= 0:
            raise RuntimeError(f"PDF không có trang: {pdf_path}")

        for page_index in range(len(document)):
            page_number = page_index + 1
            rendered = render_pdf_page_to_image(document[page_index], dpi=dpi)
            height, width = rendered.shape[:2]
            output_name = f"page_{page_number:04d}.png"
            output_path = output_dir / output_name
            save_png(output_path, rendered)
            page_entries.append(
                {
                    "page_number": page_number,
                    "file": output_name,
                    "width": int(width),
                    "height": int(height),
                    "channels": 1 if len(rendered.shape) == 2 else int(rendered.shape[2]),
                    "png_bytes": output_path.stat().st_size,
                    "png_sha256": sha256_file(output_path),
                }
            )
    finally:
        document.close()

    manifest: Dict[str, Any] = {
        "schema_version": "1.0",
        "operation": "render_full_pdf",
        "source_file": str(pdf_path),
        "source_sha256": sha256_file(pdf_path),
        "dpi": int(dpi),
        "page_count": len(page_entries),
        "render_seconds": round(time.perf_counter() - started, 3),
        "pages": page_entries,
    }
    atomic_write_json(output_dir / "render_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render toàn bộ trang PDF thành PNG, chưa phân vùng và chưa OCR."
    )
    parser.add_argument("--input-pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=400)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = render_pdf_to_pngs(args.input_pdf, args.output_dir, dpi=args.dpi)
    print(
        f"Đã render {manifest['page_count']} trang ở {manifest['dpi']} DPI "
        f"trong {manifest['render_seconds']:.3f}s"
    )
    print(str(Path(args.output_dir).resolve() / "render_manifest.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
