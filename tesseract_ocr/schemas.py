"""
Schemas và cấu trúc dữ liệu cho kết quả Tesseract OCR.
Tuân thủ chuẩn đầu ra JSON theo mục 7 của CONTEXT_COMPARE_APPLE_VISION_TESSERACT.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BoundingBox:
    """Bounding box cho từng từ hoặc dòng văn bản."""
    text: str
    confidence: float  # Chuẩn hóa về thang 0..1
    x: int
    y: int
    w: int
    h: int
    block_num: Optional[int] = None
    par_num: Optional[int] = None
    line_num: Optional[int] = None
    word_num: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "x": int(self.x),
            "y": int(self.y),
            "w": int(self.w),
            "h": int(self.h),
        }
        if self.block_num is not None:
            data["block_num"] = self.block_num
        if self.par_num is not None:
            data["par_num"] = self.par_num
        if self.line_num is not None:
            data["line_num"] = self.line_num
        if self.word_num is not None:
            data["word_num"] = self.word_num
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BoundingBox:
        return cls(
            text=str(data.get("text", "")),
            confidence=float(data.get("confidence", 0.0)),
            x=int(data.get("x", 0)),
            y=int(data.get("y", 0)),
            w=int(data.get("w", 0)),
            h=int(data.get("h", 0)),
            block_num=data.get("block_num"),
            par_num=data.get("par_num"),
            line_num=data.get("line_num"),
            word_num=data.get("word_num"),
        )


@dataclass
class PreprocessingMeta:
    """Metadata kết quả tiền xử lý ảnh trước khi OCR."""
    orientation_rotation: int = 0
    deskew_angle: float = 0.0
    threshold_applied: bool = False
    contrast_applied: bool = False
    processing_seconds: float = 0.0
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "orientation_rotation": self.orientation_rotation,
            "deskew_angle": round(self.deskew_angle, 2),
            "threshold_applied": self.threshold_applied,
            "contrast_applied": self.contrast_applied,
            "processing_seconds": round(self.processing_seconds, 3),
        }
        if self.notes:
            data["notes"] = self.notes
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PreprocessingMeta:
        return cls(
            orientation_rotation=int(data.get("orientation_rotation", 0)),
            deskew_angle=float(data.get("deskew_angle", 0.0)),
            threshold_applied=bool(data.get("threshold_applied", False)),
            contrast_applied=bool(data.get("contrast_applied", False)),
            processing_seconds=float(data.get("processing_seconds", 0.0)),
            notes=data.get("notes"),
        )


@dataclass
class PageResult:
    """Kết quả OCR cho một trang tài liệu."""
    page_number: int
    status: str = "success"  # success | error | empty | skipped
    method: str = "tesseract"
    text: str = ""
    confidence: float = 0.0  # Chuẩn hóa về 0..1
    image_width: int = 0
    image_height: int = 0
    duration_seconds: float = 0.0
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    boxes: List[BoundingBox] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "status": self.status,
            "method": self.method,
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "duration_seconds": round(self.duration_seconds, 3),
            "preprocessing": self.preprocessing,
            "boxes": [box.to_dict() for box in self.boxes],
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PageResult:
        boxes_raw = data.get("boxes", [])
        boxes = [BoundingBox.from_dict(b) for b in boxes_raw] if isinstance(boxes_raw, list) else []
        return cls(
            page_number=int(data.get("page_number", 1)),
            status=str(data.get("status", "success")),
            method=str(data.get("method", "tesseract")),
            text=str(data.get("text", "")),
            confidence=float(data.get("confidence", 0.0)),
            image_width=int(data.get("image_width", 0)),
            image_height=int(data.get("image_height", 0)),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            preprocessing=data.get("preprocessing", {}) or {},
            boxes=boxes,
            error=data.get("error"),
        )


@dataclass
class DocumentResult:
    """Toàn bộ kết quả OCR cho một tài liệu."""
    schema_version: str = "1.0"
    engine: str = "tesseract"
    engine_version: str = "5.x"
    source_file: str = ""
    source_type: str = "pdf"  # pdf | image
    source_hash: str = ""
    config_hash: str = ""
    status: str = "success"  # success | partial | error
    total_pages: int = 0
    processed_pages: int = 0
    configuration: Dict[str, Any] = field(default_factory=dict)
    pages: List[PageResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "source_file": self.source_file,
            "source_type": self.source_type,
            "source_hash": self.source_hash,
            "config_hash": self.config_hash,
            "status": self.status,
            "total_pages": self.total_pages,
            "processed_pages": self.processed_pages,
            "configuration": self.configuration,
            "pages": [page.to_dict() for page in self.pages],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_txt(self) -> str:
        """Định dạng phân cách trang theo chuẩn: --- [TRANG X/Y] ---"""
        parts = []
        tot = self.total_pages or len(self.pages)
        for page in sorted(self.pages, key=lambda p: p.page_number):
            parts.append(f"--- [TRANG {page.page_number}/{tot}] ---")
            if page.text:
                parts.append(page.text.strip())
            else:
                parts.append("")
            parts.append("")
        return "\n".join(parts).strip() + "\n"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentResult:
        pages_raw = data.get("pages", [])
        pages = [PageResult.from_dict(p) for p in pages_raw] if isinstance(pages_raw, list) else []
        return cls(
            schema_version=str(data.get("schema_version", "1.0")),
            engine=str(data.get("engine", "tesseract")),
            engine_version=str(data.get("engine_version", "5.x")),
            source_file=str(data.get("source_file", "")),
            source_type=str(data.get("source_type", "pdf")),
            source_hash=str(data.get("source_hash", "")),
            config_hash=str(data.get("config_hash", "")),
            status=str(data.get("status", "success")),
            total_pages=int(data.get("total_pages", len(pages))),
            processed_pages=int(data.get("processed_pages", len(pages))),
            configuration=data.get("configuration", {}) or {},
            pages=pages,
        )

    @classmethod
    def from_json(cls, json_str: str) -> DocumentResult:
        data = json.loads(json_str)
        return cls.from_dict(data)
