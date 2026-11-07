"""Ghép kết quả OCR theo vùng thành một TXT cho toàn bộ PDF.

Đây là bước 7 của pipeline OCR theo vùng. Module chỉ dùng các file
``selected.txt`` đã được chọn ở bước 6, sắp xếp theo trang rồi theo panel và
tạo đầu ra tương thích định dạng TXT hiện có của hệ thống:
``--- [TRANG X/Y] ---``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tesseract_ocr.document_renderer import atomic_write_json, sha256_file


if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


WINDOWS_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp_{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_manifest(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest JSON không hợp lệ: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Manifest phải là JSON object: {path}")
    return value


def resolve_reference(raw_path: str, owner_manifest: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = owner_manifest.parent / path
    return path.resolve()


def resolve_source_pdf(
    region_manifest_path: Path,
    region_manifest: Dict[str, Any],
    explicit_source_pdf: Optional[Path] = None,
) -> Tuple[Path, List[Dict[str, str]]]:
    """Lần theo chuỗi manifest để tìm đúng PDF nguồn và kiểm tra checksum."""
    if explicit_source_pdf is not None:
        source_pdf = Path(explicit_source_pdf).resolve()
        if not source_pdf.is_file():
            raise FileNotFoundError(f"Không tìm thấy PDF nguồn: {source_pdf}")
        if source_pdf.suffix.lower() != ".pdf":
            raise ValueError(f"Nguồn không phải PDF: {source_pdf}")
        return source_pdf, []

    current_path = Path(region_manifest_path).resolve()
    current_manifest = region_manifest
    visited = set()
    lineage: List[Dict[str, str]] = []

    for _ in range(16):
        if current_path in visited:
            raise ValueError(f"Phát hiện vòng lặp trong chuỗi manifest: {current_path}")
        visited.add(current_path)

        source_file = current_manifest.get("source_file")
        if source_file:
            source_pdf = resolve_reference(str(source_file), current_path)
            if not source_pdf.is_file():
                raise FileNotFoundError(f"Không tìm thấy PDF nguồn: {source_pdf}")
            if source_pdf.suffix.lower() != ".pdf":
                raise ValueError(f"Nguồn cuối không phải PDF: {source_pdf}")
            expected_sha256 = current_manifest.get("source_sha256")
            actual_sha256 = sha256_file(source_pdf)
            if expected_sha256 and str(expected_sha256) != actual_sha256:
                raise ValueError(f"Checksum PDF nguồn không khớp: {source_pdf}")
            return source_pdf, lineage

        reference_keys = sorted(
            key
            for key in current_manifest
            if key.startswith("source_")
            and key.endswith("_manifest")
            and current_manifest.get(key)
        )
        if len(reference_keys) != 1:
            raise ValueError(
                f"Không xác định được manifest nguồn duy nhất từ {current_path}"
            )

        reference_key = reference_keys[0]
        referenced_path = resolve_reference(
            str(current_manifest[reference_key]), current_path
        )
        if not referenced_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy manifest nguồn: {referenced_path}")
        actual_sha256 = sha256_file(referenced_path)
        expected_sha256 = current_manifest.get(f"{reference_key}_sha256")
        if expected_sha256 and str(expected_sha256) != actual_sha256:
            raise ValueError(f"Checksum manifest nguồn không khớp: {referenced_path}")
        lineage.append(
            {
                "reference_key": reference_key,
                "manifest": str(referenced_path),
                "sha256": actual_sha256,
            }
        )
        current_path = referenced_path
        current_manifest = load_manifest(current_path)

    raise ValueError("Chuỗi manifest vượt quá giới hạn 16 cấp")


def safe_default_output_name(source_pdf: Path) -> str:
    stem = WINDOWS_INVALID_FILENAME.sub("_", source_pdf.stem).strip(" .")
    if not stem:
        stem = "document"
    if stem.upper() in WINDOWS_RESERVED_NAMES:
        stem = f"_{stem}"
    return f"{stem}_ocr.txt"


def validate_output_name(output_name: str) -> str:
    if Path(output_name).name != output_name:
        raise ValueError("output_name chỉ được chứa tên file, không chứa thư mục")
    if WINDOWS_INVALID_FILENAME.search(output_name):
        raise ValueError(f"Tên file đầu ra không hợp lệ: {output_name}")
    if Path(output_name).suffix.lower() != ".txt":
        raise ValueError("Tên file đầu ra phải có phần mở rộng .txt")
    if Path(output_name).stem.upper() in WINDOWS_RESERVED_NAMES:
        raise ValueError(f"Tên file đầu ra dành riêng trên Windows: {output_name}")
    return output_name


def normalize_panel_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.strip()


def resolve_selected_text(region_root: Path, relative_file: str) -> Path:
    selected_path = (region_root / relative_file).resolve()
    try:
        selected_path.relative_to(region_root)
    except ValueError as exc:
        raise ValueError(
            f"File selected.txt nằm ngoài thư mục OCR bước 6: {relative_file}"
        ) from exc
    if not selected_path.is_file():
        raise FileNotFoundError(f"Thiếu kết quả OCR đã chọn: {selected_path}")
    return selected_path


def merge_region_ocr_results(
    region_manifest_path: Path,
    output_dir: Path,
    source_pdf: Optional[Path] = None,
    output_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Ghép toàn bộ selected.txt thành một TXT theo trang và panel."""
    region_manifest_path = Path(region_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not region_manifest_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy region manifest: {region_manifest_path}")

    manifest = load_manifest(region_manifest_path)
    if manifest.get("operation") != "ocr_panels_with_psm_selection":
        raise ValueError("Manifest không phải kết quả OCR theo vùng của bước 6")

    pages = manifest.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("Region manifest không có trang OCR")
    declared_page_count = int(manifest.get("page_count", len(pages)))
    if declared_page_count != len(pages):
        raise ValueError(
            f"Số trang trong manifest không khớp: khai báo {declared_page_count}, "
            f"thực tế {len(pages)}"
        )

    page_numbers = [int(page["page_number"]) for page in pages]
    if len(set(page_numbers)) != len(page_numbers):
        raise ValueError("Region manifest có số trang trùng nhau")
    expected_pages = list(range(1, declared_page_count + 1))
    if sorted(page_numbers) != expected_pages:
        raise ValueError(
            f"Region manifest thiếu trang: cần {expected_pages}, có {sorted(page_numbers)}"
        )

    resolved_source_pdf, lineage = resolve_source_pdf(
        region_manifest_path, manifest, source_pdf
    )
    final_output_name = validate_output_name(
        output_name or safe_default_output_name(resolved_source_pdf)
    )
    region_root = region_manifest_path.parent.resolve()
    page_blocks: List[str] = []
    page_records: List[Dict[str, Any]] = []
    merged_panel_count = 0
    needs_review_panels = 0

    for page in sorted(pages, key=lambda item: int(item["page_number"])):
        page_number = int(page["page_number"])
        panels = page.get("panels")
        if not isinstance(panels, list) or not panels:
            raise ValueError(f"Trang {page_number} không có panel OCR")
        declared_panel_count = int(page.get("panel_count", len(panels)))
        if declared_panel_count != len(panels):
            raise ValueError(
                f"Trang {page_number}: số panel khai báo {declared_panel_count}, "
                f"thực tế {len(panels)}"
            )
        panel_indices = [int(panel["panel_index"]) for panel in panels]
        if len(set(panel_indices)) != len(panel_indices):
            raise ValueError(f"Trang {page_number} có panel trùng chỉ số")
        expected_panels = list(range(1, declared_panel_count + 1))
        if sorted(panel_indices) != expected_panels:
            raise ValueError(
                f"Trang {page_number} thiếu panel: cần {expected_panels}, "
                f"có {sorted(panel_indices)}"
            )

        panel_texts: List[str] = []
        panel_records: List[Dict[str, Any]] = []
        for panel in sorted(panels, key=lambda item: int(item["panel_index"])):
            panel_index = int(panel["panel_index"])
            relative_file = str(panel.get("selected_text_file", ""))
            if not relative_file:
                raise ValueError(
                    f"Trang {page_number}, panel {panel_index} thiếu selected_text_file"
                )
            selected_path = resolve_selected_text(region_root, relative_file)
            panel_text = normalize_panel_text(
                selected_path.read_text(encoding="utf-8")
            )
            panel_status = str(panel.get("status", "needs_review"))
            if not panel_text or panel_status != "success":
                needs_review_panels += 1
            panel_texts.append(panel_text)
            panel_records.append(
                {
                    "panel_index": panel_index,
                    "status": panel_status,
                    "selected_candidate_id": panel.get("selected_candidate_id"),
                    "source_text_file": str(selected_path),
                    "source_text_sha256": sha256_file(selected_path),
                    "character_count": len(panel_text),
                }
            )
            merged_panel_count += 1

        page_text = "\n\n".join(panel_texts)
        page_blocks.append(
            f"--- [TRANG {page_number}/{declared_page_count}] ---\n{page_text}".rstrip()
        )
        page_records.append(
            {
                "page_number": page_number,
                "panel_count": len(panel_records),
                "character_count": len(page_text),
                "panels": panel_records,
            }
        )

    declared_panel_count = int(manifest.get("panel_count", merged_panel_count))
    if declared_panel_count != merged_panel_count:
        raise ValueError(
            f"Tổng panel không khớp: khai báo {declared_panel_count}, "
            f"đã ghép {merged_panel_count}"
        )

    document_text = "\n\n".join(page_blocks).rstrip() + "\n"
    output_path = output_dir / final_output_name
    atomic_write_text(output_path, document_text)

    result: Dict[str, Any] = {
        "schema_version": "1.0",
        "operation": "merge_region_ocr_to_document_text",
        "status": "success" if needs_review_panels == 0 else "needs_review",
        "source_pdf": str(resolved_source_pdf),
        "source_pdf_sha256": sha256_file(resolved_source_pdf),
        "source_region_ocr_manifest": str(region_manifest_path),
        "source_region_ocr_manifest_sha256": sha256_file(region_manifest_path),
        "manifest_lineage": lineage,
        "output_text_file": str(output_path),
        "output_text_sha256": sha256_file(output_path),
        "encoding": "utf-8",
        "page_separator_format": "--- [TRANG X/Y] ---",
        "panel_separator": "blank_line",
        "page_count": declared_page_count,
        "panel_count": merged_panel_count,
        "needs_review_panels": needs_review_panels,
        "character_count": len(document_text),
        "word_count": len(re.findall(r"\S+", document_text, flags=re.UNICODE)),
        "pages": page_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "merge_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ghép selected.txt theo trang/panel thành một TXT cho toàn PDF."
    )
    parser.add_argument("--region-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-pdf", type=Path, default=None)
    parser.add_argument("--output-name", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = merge_region_ocr_results(
        args.region_manifest,
        args.output_dir,
        source_pdf=args.source_pdf,
        output_name=args.output_name,
    )
    print(
        f"Đã ghép {result['page_count']} trang, {result['panel_count']} panel "
        f"thành 1 TXT; {result['needs_review_panels']} panel cần kiểm tra"
    )
    print(result["output_text_file"])
    print(str(Path(args.output_dir).resolve() / "merge_manifest.json"))
    return 0 if result["status"] == "success" else 3


if __name__ == "__main__":
    sys.exit(main())
