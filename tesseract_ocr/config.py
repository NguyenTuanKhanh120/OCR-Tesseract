"""
Cấu hình cho Tesseract OCR Pipeline.
Kiểm tra an toàn đường dẫn và sinh hash cấu hình.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TesseractConfig:
    """Cấu hình tham số cho Tesseract OCR Pipeline."""
    input_dir: Path
    output_dir: Path
    dpi: int = 300
    lang: str = "vie"
    oem: int = 1
    psm: int = 3
    preprocessing: str = "auto"  # none | grayscale | auto
    thresholding_method: int = 0  # 0: Otsu, 1: LeptonicaOtsu, 2: Sauvola
    tesseract_cmd: Optional[str] = None
    tessdata_dir: Optional[Path] = None
    metadata_dir: Optional[Path] = None
    allow_native_text: bool = False
    adaptive_background: bool = False
    max_documents: Optional[int] = None
    force: bool = False

    def __post_init__(self):
        self.input_dir = Path(self.input_dir).resolve()
        self.output_dir = Path(self.output_dir).resolve()
        if self.metadata_dir:
            self.metadata_dir = Path(self.metadata_dir).resolve()
        if not self.tesseract_cmd:
            self.tesseract_cmd = os.getenv("TESSERACT_CMD")
        if self.tessdata_dir:
            self.tessdata_dir = Path(self.tessdata_dir).resolve()
        else:
            env_tessdata = os.getenv("TESSDATA_PREFIX")
            project_tessdata = Path(__file__).resolve().parents[1] / "tessdata"
            if env_tessdata:
                self.tessdata_dir = Path(env_tessdata).resolve()
            elif project_tessdata.is_dir():
                self.tessdata_dir = project_tessdata.resolve()

    def validate(self) -> None:
        """Kiểm tra tính hợp lệ và an toàn của cấu hình."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Thư mục nguồn không tồn tại: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"input_dir phải là một thư mục: {self.input_dir}")

        # Bảo vệ: Thư mục nguồn và đầu ra không được trùng nhau
        if self.input_dir == self.output_dir:
            raise ValueError(
                f"LỖI AN TOÀN: input_dir và output_dir không được trùng nhau ({self.input_dir})"
            )

        if self.metadata_dir:
            if self.metadata_dir == self.input_dir:
                raise ValueError(
                    f"metadata_dir không được trùng input_dir ({self.input_dir})"
                )
            if self.metadata_dir == self.output_dir:
                raise ValueError(
                    "metadata_dir phải tách khỏi output_dir để output chỉ chứa file TXT"
                )

        # Kiểm tra nếu output_dir là cha của input_dir hoặc ngược lại (cảnh báo hoặc cấm)
        try:
            if self.input_dir in self.output_dir.parents:
                raise ValueError(
                    f"output_dir ({self.output_dir}) không được nằm bên trong input_dir ({self.input_dir})"
                )
        except Exception:
            pass

        if self.dpi <= 0:
            raise ValueError(f"DPI phải lớn hơn 0, nhận được: {self.dpi}")

        if self.tessdata_dir and not self.tessdata_dir.is_dir():
            raise FileNotFoundError(f"Thư mục tessdata không tồn tại: {self.tessdata_dir}")

        if self.preprocessing not in ("none", "grayscale", "auto"):
            raise ValueError(
                f"Chế độ tiền xử lý không hợp lệ: '{self.preprocessing}'. "
                "Chỉ chấp nhận 'none', 'grayscale' hoặc 'auto'."
            )

        if self.thresholding_method not in (0, 1, 2):
            raise ValueError(
                "thresholding_method phải là 0 (Otsu), 1 (LeptonicaOtsu) "
                f"hoặc 2 (Sauvola), nhận được: {self.thresholding_method}"
            )

        if self.oem not in (0, 1, 2, 3):
            raise ValueError(f"OEM phải thuộc [0, 1, 2, 3], nhận được: {self.oem}")

        if self.psm not in range(0, 14):
            raise ValueError(f"PSM phải thuộc [0..13], nhận được: {self.psm}")

    def to_ocr_dict(self) -> Dict[str, Any]:
        """Các thuộc tính cốt lõi cấu hình OCR để lưu vào DocumentResult."""
        return {
            "dpi": self.dpi,
            "language": self.lang,
            "oem": self.oem,
            "psm": self.psm,
            "preprocessing": self.preprocessing,
            "thresholding_method": self.thresholding_method,
            "tessdata_dir": str(self.tessdata_dir) if self.tessdata_dir else None,
            "allow_native_text": self.allow_native_text,
            "adaptive_background": self.adaptive_background,
        }

    def compute_config_hash(self) -> str:
        """
        Sinh mã băm đại diện cho cấu hình OCR.
        Dùng để phát hiện nếu lần chạy sau khác cấu hình thì cần re-OCR.
        """
        core = self.to_ocr_dict()
        canonical_str = json.dumps(core, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()[:16]
