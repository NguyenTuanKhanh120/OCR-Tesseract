"""
Cấu hình cho công cụ So sánh Apple Vision và Tesseract OCR.
Bảo đảm tính toàn vẹn và nguyên tắc chỉ đọc dữ liệu gốc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ComparisonConfig:
    """Cấu hình cho quá trình so sánh hai thư mục kết quả."""
    apple_dir: Path
    tesseract_dir: Path
    report_dir: Path
    ground_truth_path: Optional[Path] = None
    input_dir: Optional[Path] = None
    cer_threshold: float = 0.05  # Ngưỡng chênh lệch CER (5%) để coi là hòa
    run_id: Optional[str] = None

    def __post_init__(self):
        self.apple_dir = Path(self.apple_dir).resolve()
        self.tesseract_dir = Path(self.tesseract_dir).resolve()
        self.report_dir = Path(self.report_dir).resolve()
        if self.ground_truth_path:
            self.ground_truth_path = Path(self.ground_truth_path).resolve()
        if self.input_dir:
            self.input_dir = Path(self.input_dir).resolve()

    def validate(self) -> None:
        """Kiểm tra đường dẫn và nguyên tắc an toàn dữ liệu."""
        if not self.apple_dir.exists():
            raise FileNotFoundError(f"Thư mục kết quả Apple Vision không tồn tại: {self.apple_dir}")
        if not self.apple_dir.is_dir():
            raise NotADirectoryError(f"apple_dir phải là thư mục: {self.apple_dir}")

        if not self.tesseract_dir.exists():
            raise FileNotFoundError(f"Thư mục kết quả Tesseract không tồn tại: {self.tesseract_dir}")
        if not self.tesseract_dir.is_dir():
            raise NotADirectoryError(f"tesseract_dir phải là thư mục: {self.tesseract_dir}")

        if self.ground_truth_path and not self.ground_truth_path.exists():
            raise FileNotFoundError(f"File ground truth không tồn tại: {self.ground_truth_path}")

        # Kiểm tra an toàn: report_dir không được trùng apple_dir hoặc tesseract_dir
        if self.report_dir in (self.apple_dir, self.tesseract_dir):
            raise ValueError("report_dir không được trùng với apple_dir hoặc tesseract_dir")

        if self.apple_dir == self.tesseract_dir:
            raise ValueError("apple_dir và tesseract_dir phải là hai thư mục tách biệt")
