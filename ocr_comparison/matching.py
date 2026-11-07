"""
Khớp nối tài liệu và trang giữa hai hệ thống Apple Vision và Tesseract OCR.
Tuân thủ nghiêm ngặt quy tắc ghép tài liệu tại mục 9 của tài liệu thiết kế.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ocr_comparison.loaders import LoadedDocument, LoadedPage


@dataclass
class MatchedPagePair:
    """Cặp trang tương ứng giữa hai engine."""
    page_number: int
    apple_page: Optional[LoadedPage] = None
    tesseract_page: Optional[LoadedPage] = None

    @property
    def is_complete(self) -> bool:
        """Cả hai engine đều có trang này."""
        return self.apple_page is not None and self.tesseract_page is not None

    @property
    def apple_status(self) -> str:
        if self.apple_page is None:
            return "missing"
        return self.apple_page.status

    @property
    def tesseract_status(self) -> str:
        if self.tesseract_page is None:
            return "missing"
        return self.tesseract_page.status


@dataclass
class MatchedDocumentPair:
    """Cặp tài liệu đã khớp nối thành công giữa hai engine."""
    normalized_key: str
    category: str
    stem_name: str
    apple_doc: LoadedDocument
    tesseract_doc: LoadedDocument
    page_pairs: List[MatchedPagePair] = field(default_factory=list)

    @property
    def total_pages(self) -> int:
        return len(self.page_pairs)


@dataclass
class MatchingResult:
    """Tổng thể kết quả khớp nối."""
    matched_pairs: List[MatchedDocumentPair] = field(default_factory=list)
    unmatched_apple: List[LoadedDocument] = field(default_factory=list)
    unmatched_tesseract: List[LoadedDocument] = field(default_factory=list)

    @property
    def total_matched(self) -> int:
        return len(self.matched_pairs)


def match_documents(
    apple_docs: Dict[str, LoadedDocument],
    tesseract_docs: Dict[str, LoadedDocument],
) -> MatchingResult:
    """
    Khớp các tài liệu giữa hai engine dựa trên normalized_key (đường dẫn tương đối chuẩn hóa).
    Đảm bảo:
    - Không ghép chéo giữa các category khác nhau.
    - Không tự đoán nếu không khớp đường dẫn.
    - Ghép trang theo page_number, ghi nhận trang missing nếu một bên thiếu.
    """
    result = MatchingResult()

    all_apple_keys: Set[str] = set(apple_docs.keys())
    all_tess_keys: Set[str] = set(tesseract_docs.keys())

    common_keys = sorted(all_apple_keys.intersection(all_tess_keys))

    # Xử lý các tài liệu khớp
    for key in common_keys:
        a_doc = apple_docs[key]
        t_doc = tesseract_docs[key]

        # Kiểm tra an toàn: category phải khớp
        if a_doc.category.lower() != t_doc.category.lower():
            # Nếu category không trùng, không coi là khớp
            result.unmatched_apple.append(a_doc)
            result.unmatched_tesseract.append(t_doc)
            continue

        # Tập hợp tất cả các số trang có ở cả hai bên
        all_page_nums = sorted(set(a_doc.pages.keys()).union(set(t_doc.pages.keys())))
        if not all_page_nums:
            all_page_nums = [1]

        page_pairs: List[MatchedPagePair] = []
        for p_num in all_page_nums:
            a_page = a_doc.pages.get(p_num)
            t_page = t_doc.pages.get(p_num)
            page_pairs.append(
                MatchedPagePair(
                    page_number=p_num,
                    apple_page=a_page,
                    tesseract_page=t_page,
                )
            )

        pair = MatchedDocumentPair(
            normalized_key=key,
            category=a_doc.category,
            stem_name=a_doc.stem_name,
            apple_doc=a_doc,
            tesseract_doc=t_doc,
            page_pairs=page_pairs,
        )
        result.matched_pairs.append(pair)

    # Tài liệu chỉ có ở Apple
    apple_only_keys = sorted(all_apple_keys - all_tess_keys)
    for k in apple_only_keys:
        result.unmatched_apple.append(apple_docs[k])

    # Tài liệu chỉ có ở Tesseract
    tess_only_keys = sorted(all_tess_keys - all_apple_keys)
    for k in tess_only_keys:
        result.unmatched_tesseract.append(tesseract_docs[k])

    return result
