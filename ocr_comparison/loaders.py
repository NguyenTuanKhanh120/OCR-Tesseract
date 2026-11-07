"""
Bộ nạp dữ liệu kết quả OCR từ hai thư mục và Ground Truth.
Chỉ mở file ở chế độ đọc (read-only), không sửa đổi bất kỳ file nguồn nào.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PAGE_HEADER_REGEX = re.compile(r"^---\s*\[TRANG\s+(\d+)\s*/\s*(\d+)\]\s*---", re.IGNORECASE | re.MULTILINE)


@dataclass
class LoadedPage:
    """Thông tin một trang đã OCR."""
    page_number: int
    text: str = ""
    confidence: float = 0.0
    status: str = "success"  # success, error, empty, missing
    duration_seconds: float = 0.0
    boxes: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class LoadedDocument:
    """Tài liệu được nạp từ thư mục kết quả."""
    relative_path: str  # Đường dẫn tương đối đã chuẩn hóa
    category: str
    stem_name: str  # Tên gốc đã bỏ _ocr
    engine: str
    total_pages: int
    pages: Dict[int, LoadedPage] = field(default_factory=dict)
    json_path: Optional[Path] = None
    txt_path: Optional[Path] = None
    status: str = "success"


@dataclass
class GroundTruthPage:
    """Dữ liệu Ground Truth theo từng trang."""
    document_id: str
    relative_path: str
    category: str
    page_number: int
    ground_truth_text: str
    quality_tags: List[str] = field(default_factory=list)
    notes: str = ""


def parse_pages_from_txt(txt_content: str) -> Dict[int, str]:
    """
    Phân tách nội dung text theo từng trang dựa trên định dạng chuẩn:
    --- [TRANG 1/3] ---
    ...
    --- [TRANG 2/3] ---
    """
    matches = list(PAGE_HEADER_REGEX.finditer(txt_content))
    if not matches:
        # Nếu không có header trang, coi như toàn bộ nội dung thuộc trang 1
        return {1: txt_content.strip()}

    pages: Dict[int, str] = {}
    for i, m in enumerate(matches):
        p_num = int(m.group(1))
        start_idx = m.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(txt_content)
        page_text = txt_content[start_idx:end_idx].strip()
        pages[p_num] = page_text

    return pages


def normalize_relative_stem(rel_path: Path) -> Tuple[str, str, str]:
    """
    Chuẩn hóa đường dẫn tương đối:
    - Bỏ hậu tố _ocr
    - Trả về (normalized_key, category, stem)
    Ví dụ:
      'PCCC/document_ocr.json' -> ('PCCC/document', 'PCCC', 'document')
    """
    parts = list(rel_path.parts)
    raw_stem = parts[-1]
    # Bỏ extension
    if "." in raw_stem:
        raw_stem = raw_stem.rsplit(".", 1)[0]
    # Bỏ _ocr suffix
    if raw_stem.lower().endswith("_ocr"):
        raw_stem = raw_stem[:-4]

    parts[-1] = raw_stem
    norm_key = "/".join(parts).lower()
    category = parts[0] if len(parts) > 1 else "root"
    return norm_key, category, raw_stem


def load_ocr_directory(dir_path: Path, engine_name: str) -> Dict[str, LoadedDocument]:
    """
    Quét và nạp toàn bộ kết quả OCR (JSON và TXT) trong thư mục chỉ định.
    Chỉ mở ở chế độ đọc (r/rb).
    """
    root_path = Path(dir_path).resolve()
    if not root_path.exists():
        return {}

    documents: Dict[str, LoadedDocument] = {}

    # 1. Tìm tất cả các file JSON
    for json_file in root_path.rglob("*.json"):
        if not json_file.is_file():
            continue
        rel_path = json_file.relative_to(root_path)
        norm_key, category, stem = normalize_relative_stem(rel_path)

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        doc = LoadedDocument(
            relative_path="/".join(rel_path.parts),
            category=category,
            stem_name=stem,
            engine=engine_name,
            total_pages=int(data.get("total_pages", 0)),
            json_path=json_file,
            status=str(data.get("status", "success")),
        )

        raw_pages = data.get("pages", [])
        if isinstance(raw_pages, list):
            for p in raw_pages:
                if not isinstance(p, dict):
                    continue
                p_num = int(p.get("page_number", len(doc.pages) + 1))
                p_text = str(p.get("text", "")).strip()
                p_conf = float(p.get("confidence", 0.0))
                p_stat = str(p.get("status", "success" if p_text else "empty"))
                p_dur = float(p.get("duration_seconds", 0.0))
                p_err = p.get("error")
                p_boxes = p.get("boxes", [])

                doc.pages[p_num] = LoadedPage(
                    page_number=p_num,
                    text=p_text,
                    confidence=p_conf,
                    status=p_stat,
                    duration_seconds=p_dur,
                    boxes=p_boxes,
                    error=p_err,
                )

        if doc.total_pages == 0 and doc.pages:
            doc.total_pages = len(doc.pages)

        documents[norm_key] = doc

    # 2. Tìm tất cả các file TXT
    for txt_file in root_path.rglob("*.txt"):
        if not txt_file.is_file():
            continue
        rel_path = txt_file.relative_to(root_path)
        norm_key, category, stem = normalize_relative_stem(rel_path)

        try:
            # Thử utf-8, nếu lỗi thì fallback utf-8 với ignore
            with open(txt_file, "r", encoding="utf-8", errors="replace") as f:
                txt_content = f.read()
        except Exception:
            continue

        pages_dict = parse_pages_from_txt(txt_content)

        if norm_key in documents:
            doc = documents[norm_key]
            doc.txt_path = txt_file
            # Nếu JSON chưa có text hoặc ít trang hơn TXT, bổ sung từ TXT
            for p_num, p_text in pages_dict.items():
                if p_num not in doc.pages or not doc.pages[p_num].text:
                    if p_num not in doc.pages:
                        doc.pages[p_num] = LoadedPage(
                            page_number=p_num,
                            text=p_text,
                            status="success" if p_text else "empty",
                        )
                    else:
                        doc.pages[p_num].text = p_text
            if doc.total_pages < len(pages_dict):
                doc.total_pages = len(pages_dict)
        else:
            # Tài liệu chỉ có TXT, không có JSON
            doc = LoadedDocument(
                relative_path="/".join(rel_path.parts),
                category=category,
                stem_name=stem,
                engine=engine_name,
                total_pages=len(pages_dict),
                txt_path=txt_file,
                status="success",
            )
            for p_num, p_text in pages_dict.items():
                doc.pages[p_num] = LoadedPage(
                    page_number=p_num,
                    text=p_text,
                    status="success" if p_text else "empty",
                )
            documents[norm_key] = doc

    return documents


def load_ground_truth(gt_path: Path) -> Dict[Tuple[str, int], GroundTruthPage]:
    """
    Nạp Ground Truth từ file CSV.
    Key trả về là tuple (normalized_doc_key, page_number).
    Hỗ trợ schema chuẩn mục 11:
      document_id, relative_path, category, page_number, ground_truth_text, quality_tags, notes
    cũng như tương thích với evaluation/manual_test_set.csv.
    """
    path = Path(gt_path).resolve()
    if not path.is_file():
        return {}

    gt_data: Dict[Tuple[str, int], GroundTruthPage] = {}

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Tìm đường dẫn tương đối
            raw_path = row.get("relative_path") or row.get("input_path") or row.get("file_name", "")
            if not raw_path:
                continue

            # Chuẩn hóa key
            path_obj = Path(raw_path)
            norm_key, category, _ = normalize_relative_stem(path_obj)

            p_num = int(row.get("page_number", 1))
            text = (row.get("ground_truth_text") or row.get("reference_text") or "").strip()

            tags_raw = row.get("quality_tags", "")
            tags = [t.strip() for t in tags_raw.split(";") if t.strip()] if tags_raw else []

            doc_id = row.get("document_id") or row.get("doc_id") or row.get("test_id", "")
            notes = row.get("notes") or row.get("human_notes") or ""

            gt_obj = GroundTruthPage(
                document_id=doc_id,
                relative_path=raw_path,
                category=category or row.get("category", "root"),
                page_number=p_num,
                ground_truth_text=text,
                quality_tags=tags,
                notes=notes,
            )
            gt_data[(norm_key, p_num)] = gt_obj

    return gt_data
