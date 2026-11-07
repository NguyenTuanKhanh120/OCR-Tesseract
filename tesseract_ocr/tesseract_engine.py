"""
Engine điều khiển Tesseract OCR và render trang tài liệu bằng PyMuPDF.
Kiểm tra nghiêm ngặt môi trường thực thi và trích xuất text/box/confidence.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pymupdf as fitz
except ImportError:
    import fitz
import numpy as np
import pytesseract

from tesseract_ocr.config import TesseractConfig
from tesseract_ocr.schemas import BoundingBox, PageResult


class TesseractEnvironmentError(RuntimeError):
    """Lỗi khi môi trường thiếu executable Tesseract hoặc language pack."""
    pass


def get_project_tessdata_profiles(
    required_langs: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Trả về các profile tessdata trong dự án có đủ ngôn ngữ yêu cầu."""
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        ("best", project_root / "tessdata_best"),
        ("fast", project_root / "tessdata_fast"),
        ("current", project_root / "tessdata"),
    ]
    languages = [lang for lang in (required_langs or []) if lang]
    profiles: Dict[str, Path] = {}
    seen: set[Path] = set()
    for profile_name, candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        if all((resolved / f"{lang}.traineddata").is_file() for lang in languages):
            profiles[profile_name] = resolved
            seen.add(resolved)
    return profiles


def find_tesseract_executable(custom_cmd: Optional[str] = None) -> Optional[str]:
    """Tìm đường dẫn tesseract.exe trên hệ thống Windows/Linux."""
    # 1. Đường dẫn do người dùng truyền vào
    if custom_cmd and Path(custom_cmd).is_file():
        return custom_cmd

    # 2. Biến môi trường TESSERACT_CMD
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).is_file():
        return env_cmd

    # 3. Trong PATH
    which_path = shutil.which("tesseract")
    if which_path:
        return which_path

    # 4. Các vị trí cài đặt mặc định phổ biến trên Windows
    win_common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for p in win_common_paths:
        if Path(p).is_file():
            return p

    return None


def find_tessdata_directory(
    custom_dir: Optional[Path | str] = None,
    executable: Optional[str] = None,
    required_langs: Optional[List[str]] = None,
) -> Optional[Path]:
    """Tìm tessdata thực sự chứa đủ mọi ngôn ngữ được yêu cầu."""
    project_tessdata = Path(__file__).resolve().parents[1] / "tessdata"
    env_tessdata = os.getenv("TESSDATA_PREFIX")
    candidates = [
        Path(custom_dir) if custom_dir else None,
        Path(env_tessdata) if env_tessdata else None,
        project_tessdata,
        Path(executable).parent / "tessdata" if executable else None,
    ]
    languages = [lang for lang in (required_langs or []) if lang]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        if all((resolved / f"{lang}.traineddata").is_file() for lang in languages):
            return resolved
    return None


def verify_tesseract_environment(config: TesseractConfig) -> str:
    """
    Kiểm tra tính sẵn sàng của Tesseract:
    - Executable tồn tại.
    - Phiên bản Tesseract.
    - Dữ liệu ngôn ngữ (ví dụ: 'vie', 'eng').
    Nếu thiếu, ném TesseractEnvironmentError với hướng dẫn chi tiết.
    """
    exe_path = find_tesseract_executable(config.tesseract_cmd)
    if not exe_path:
        raise TesseractEnvironmentError(
            "\n" + "=" * 70 + "\n"
            "KHÔNG TÌM THẤY TESSERACT OCR EXECUTABLE TRÊN HỆ THỐNG!\n"
            "=" * 70 + "\n"
            "Để chạy hệ thống Tesseract độc lập, vui lòng thực hiện các bước sau:\n"
            "1. Tải và cài đặt Tesseract 5 cho Windows từ UB-Mannheim:\n"
            "   https://github.com/UB-Mannheim/tesseract/wiki\n"
            "2. Trong quá trình cài đặt, tích chọn thêm gói ngôn ngữ 'Vietnamese' (vie).\n"
            "3. Thiết lập biến môi trường TESSERACT_CMD trỏ đến file tesseract.exe, ví dụ:\n"
            '   setx TESSERACT_CMD "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"\n'
            "   hoặc truyền qua cờ CLI: --tesseract-cmd <đường_dẫn>\n"
            "=" * 70
        )

    # Cấu hình pytesseract
    pytesseract.pytesseract.tesseract_cmd = exe_path

    # Kiểm tra phiên bản
    try:
        ver_output = subprocess.check_output(
            [exe_path, "--version"], stderr=subprocess.STDOUT, text=True, timeout=10
        )
        version_line = ver_output.splitlines()[0] if ver_output else "unknown"
    except Exception as e:
        raise TesseractEnvironmentError(
            f"Không thể kiểm tra phiên bản Tesseract tại '{exe_path}': {e}"
        )

    required_langs = [l.strip() for l in config.lang.split("+") if l.strip()]
    tessdata_dir = find_tessdata_directory(
        config.tessdata_dir, exe_path, required_langs=required_langs
    )
    if not tessdata_dir:
        raise TesseractEnvironmentError(
            "\n" + "=" * 70 + "\n"
            f"KHÔNG TÌM THẤY TESSDATA CÓ ĐỦ NGÔN NGỮ: {required_langs}\n"
            "Tesseract sẽ không được phép âm thầm fallback sang tiếng Anh.\n"
            f"Thư mục tessdata đã cấu hình: {config.tessdata_dir}\n"
            "=" * 70
        )

    # Kiểm tra language packs trong đúng thư mục đã chọn.
    try:
        langs_output = subprocess.check_output(
            [exe_path, "--tessdata-dir", str(tessdata_dir), "--list-langs"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
        available_langs = set(langs_output.strip().splitlines()[1:])
    except Exception as e:
        available_langs = set()

    missing_langs = [l for l in required_langs if l not in available_langs]

    if missing_langs:
        raise TesseractEnvironmentError(
            "\n" + "=" * 70 + "\n"
            f"THIẾU GÓI DỮ LIỆU NGÔN NGỮ TESSERACT: {missing_langs}\n"
            "=" * 70 + "\n"
            f"Các ngôn ngữ hiện có: {list(available_langs)}\n"
            "Vui lòng tải file *.traineddata tương ứng (ví dụ: vie.traineddata, eng.traineddata)\n"
            "từ https://github.com/tesseract-ocr/tessdata hoặc tessdata_fast\n"
            "và sao chép vào thư mục tessdata của Tesseract OCR (thường là C:\\Program Files\\Tesseract-OCR\\tessdata).\n"
            "=" * 70
        )

    return version_line


def render_pdf_page_to_image(page: fitz.Page, dpi: int = 300) -> np.ndarray:
    """
    Render một trang PDF thành ảnh numpy RGB/BGR bằng PyMuPDF với DPI cấu hình được.
    """
    # 72 points = 1 inch
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)

    # Chuyển đổi pixmap sang numpy array BGR dùng cho OpenCV / PyTesseract
    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
    if pix.n == 3:
        # PyMuPDF samples là RGB, chuyển sang BGR cho đồng nhất OpenCV
        import cv2
        img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    elif pix.n == 1:
        img = img_data
    else:
        img = img_data[:, :, :3]

    return img


def run_tesseract_on_image(
    image: np.ndarray,
    config: TesseractConfig,
    page_number: int = 1,
    prep_meta: Optional[Dict] = None,
) -> PageResult:
    """
    Chạy pytesseract.image_to_data để trích xuất text, bounding box và độ tin cậy.
    Chuẩn hóa confidence về khoảng 0..1.
    Ghép các từ theo block, paragraph và line.
    """
    start_time = time.time()
    h, w = image.shape[:2]

    exe_path = find_tesseract_executable(config.tesseract_cmd)
    if not exe_path:
        raise TesseractEnvironmentError("Không tìm thấy tesseract.exe")
    pytesseract.pytesseract.tesseract_cmd = exe_path
    required_langs = [lang.strip() for lang in config.lang.split("+") if lang.strip()]
    tessdata_dir = find_tessdata_directory(
        config.tessdata_dir, exe_path, required_langs=required_langs
    )
    if not tessdata_dir:
        raise TesseractEnvironmentError(
            f"Không tìm thấy tessdata có đủ ngôn ngữ {required_langs}. "
            "Không chạy fallback sang tiếng Anh."
        )

    tess_custom_config = (
        f"--tessdata-dir {tessdata_dir} "
        f"--oem {config.oem} --psm {config.psm} "
        f"-c thresholding_method={config.thresholding_method} "
        "-c preserve_interword_spaces=1"
    )

    try:
        data = pytesseract.image_to_data(
            image,
            lang=config.lang,
            config=tess_custom_config,
            output_type=pytesseract.Output.DICT,
        )

        n_boxes = len(data["text"])
        boxes: List[BoundingBox] = []
        lines_dict: Dict[Tuple[int, int, int], List[str]] = {}
        conf_sum = 0.0
        conf_count = 0

        for i in range(n_boxes):
            word_text = data["text"][i].strip()
            conf_val = float(data["conf"][i])

            # Tesseract trả conf = -1 cho block, paragraph, newline headers
            if conf_val < 0 or not word_text:
                continue

            # Chuẩn hóa về [0..1]
            normalized_conf = max(0.0, min(1.0, conf_val / 100.0))
            conf_sum += normalized_conf
            conf_count += 1

            x = int(data["left"][i])
            y = int(data["top"][i])
            bw = int(data["width"][i])
            bh = int(data["height"][i])
            b_num = data.get("block_num", [None])[i]
            p_num = data.get("par_num", [None])[i]
            l_num = data.get("line_num", [None])[i]
            w_num = data.get("word_num", [None])[i]

            box = BoundingBox(
                text=word_text,
                confidence=normalized_conf,
                x=x,
                y=y,
                w=bw,
                h=bh,
                block_num=b_num,
                par_num=p_num,
                line_num=l_num,
                word_num=w_num,
            )
            boxes.append(box)

            # Gom từ vào dòng dựa trên (block_num, par_num, line_num)
            line_key = (b_num or 0, p_num or 0, l_num or 0)
            if line_key not in lines_dict:
                lines_dict[line_key] = []
            lines_dict[line_key].append(word_text)

        # Ghép văn bản hoàn chỉnh
        assembled_lines = [" ".join(words) for words in lines_dict.values()]
        full_text = unicodedata.normalize("NFC", "\n".join(assembled_lines).strip())

        avg_conf = (conf_sum / conf_count) if conf_count > 0 else 0.0
        status = "empty" if not full_text else "success"
        duration = time.time() - start_time

        return PageResult(
            page_number=page_number,
            status=status,
            method="tesseract",
            text=full_text,
            confidence=avg_conf,
            image_width=w,
            image_height=h,
            duration_seconds=duration,
            preprocessing=prep_meta or {},
            boxes=boxes,
            error=None,
        )

    except Exception as e:
        duration = time.time() - start_time
        return PageResult(
            page_number=page_number,
            status="error",
            method="tesseract",
            text="",
            confidence=0.0,
            image_width=w,
            image_height=h,
            duration_seconds=duration,
            preprocessing=prep_meta or {},
            boxes=[],
            error=str(e),
        )
