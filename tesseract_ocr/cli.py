"""
Giao diện dòng lệnh (CLI) chạy pipeline Tesseract OCR độc lập.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tesseract_ocr.config import TesseractConfig
from tesseract_ocr.pipeline import TesseractPipeline
from tesseract_ocr.tesseract_engine import TesseractEnvironmentError
from tesseract_ocr.tesseract_engine import get_project_tessdata_profiles


if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tesseract_ocr.cli",
        description="Chạy Tesseract OCR độc lập trên kho tài liệu PDF và ảnh.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Đường dẫn thư mục chứa tài liệu nguồn (chỉ đọc).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Đường dẫn thư mục lưu kết quả Tesseract OCR (tách biệt hoàn toàn).",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help=(
            "Thư mục riêng lưu JSON checkpoint/metadata. "
            "Khi dùng, output-dir chỉ chứa một file TXT cho mỗi tài liệu."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI khi render trang PDF thành ảnh (mặc định: 300).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="vie",
        help="Ngôn ngữ nhận dạng (mặc định: vie để ưu tiên dấu tiếng Việt).",
    )
    parser.add_argument(
        "--oem",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="OCR Engine Mode (0: Legacy, 1: Neural LSTM, 2: Cả hai, 3: Mặc định). Mặc định: 1.",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=3,
        choices=list(range(0, 14)),
        help="Page Segmentation Mode (mặc định: 3 - Fully automatic page segmentation).",
    )
    parser.add_argument(
        "--preprocessing",
        type=str,
        default="auto",
        choices=["none", "grayscale", "auto"],
        help="Chế độ tiền xử lý ảnh (mặc định: auto).",
    )
    parser.add_argument(
        "--thresholding-method",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Nhị phân hóa nội bộ: 0=Otsu, 1=LeptonicaOtsu, 2=Sauvola.",
    )
    parser.add_argument(
        "--tessdata-profile",
        type=str,
        default=None,
        choices=["best", "fast", "current"],
        help="Chọn model trong tessdata_best, tessdata_fast hoặc tessdata hiện tại.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        type=str,
        default=None,
        help="Đường dẫn trực tiếp tới tesseract executable nếu chưa có trong PATH.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Giới hạn số lượng tài liệu cần xử lý (hữu ích khi chạy thử).",
    )
    parser.add_argument(
        "--allow-native-text",
        action="store_true",
        default=False,
        help="Cho phép trích xuất native text trong PDF (mặc định tắt để đánh giá đúng OCR engine).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Chạy lại và ghi đè kết quả cũ kể cả khi cấu hình và file nguồn không đổi.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Bật log chi tiết.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        required_langs = [item.strip() for item in args.lang.split("+") if item.strip()]
        profiles = get_project_tessdata_profiles(required_langs)
        tessdata_dir = None
        if args.tessdata_profile:
            tessdata_dir = profiles.get(args.tessdata_profile)
            if tessdata_dir is None:
                parser.error(
                    f"Profile '{args.tessdata_profile}' chưa có đủ ngôn ngữ: {required_langs}"
                )

        config = TesseractConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            dpi=args.dpi,
            lang=args.lang,
            oem=args.oem,
            psm=args.psm,
            preprocessing=args.preprocessing,
            thresholding_method=args.thresholding_method,
            tesseract_cmd=args.tesseract_cmd,
            tessdata_dir=tessdata_dir,
            metadata_dir=args.metadata_dir,
            allow_native_text=args.allow_native_text,
            max_documents=args.max_documents,
            force=args.force,
        )

        pipeline = TesseractPipeline(config)
        pipeline.run()
        return 0

    except TesseractEnvironmentError as env_err:
        print(f"\n{env_err}\n", file=sys.stderr)
        return 2
    except Exception as e:
        logging.error(f"Lỗi thực thi: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
