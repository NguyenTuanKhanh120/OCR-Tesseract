"""Tạo baseline cho kế hoạch chuẩn hóa nền tài liệu.

Module không thay đổi thuật toán OCR. Nó render các trang được chỉ định, chạy
đúng cấu hình web hiện tại (best + vie + PSM 6 + Sauvola), lưu ảnh trước/sau,
overlay, TXT và các chỉ số dùng để so sánh ở những bước tiếp theo.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from tesseract_ocr.config import TesseractConfig
from tesseract_ocr.document_renderer import atomic_write_json, save_png, sha256_file
from tesseract_ocr.document_text_merger import atomic_write_text
from tesseract_ocr.preprocessing import preprocess_image
from tesseract_ocr.region_ocr import make_ocr_overlay
from tesseract_ocr.tesseract_engine import (
    find_tesseract_executable,
    render_pdf_page_to_image,
    run_tesseract_on_image,
    verify_tesseract_environment,
)


SAFE_PUNCTUATION = set(".,:;!?%()+-/='\"&@#_[]")


def load_color_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Không thể đọc ảnh: {path}")
    return image


def render_sample(source: Path, page_number: int, dpi: int) -> np.ndarray:
    if source.suffix.lower() == ".pdf":
        with fitz.open(source) as document:
            if document.needs_pass:
                raise RuntimeError(f"PDF có mật khẩu: {source}")
            if page_number < 1 or page_number > len(document):
                raise ValueError(
                    f"Trang {page_number} ngoài phạm vi 1..{len(document)}: {source}"
                )
            return render_pdf_page_to_image(document[page_number - 1], dpi=dpi)
    if page_number != 1:
        raise ValueError("Nguồn ảnh chỉ có page_number=1")
    return load_color_image(source)


def _block_background_values(gray: np.ndarray, grid_size: int = 8) -> List[float]:
    height, width = gray.shape[:2]
    values: List[float] = []
    for row in range(grid_size):
        y0 = row * height // grid_size
        y1 = (row + 1) * height // grid_size
        for column in range(grid_size):
            x0 = column * width // grid_size
            x1 = (column + 1) * width // grid_size
            block = gray[y0:y1, x0:x1]
            if block.size:
                values.append(float(np.percentile(block, 85)))
    return values


def image_quality_metrics(image: np.ndarray) -> Dict[str, Any]:
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image.copy()
    )
    background_values = _block_background_values(gray)
    background_p05 = float(np.percentile(background_values, 5))
    background_p95 = float(np.percentile(background_values, 95))
    metrics: Dict[str, Any] = {
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "mean_luminance": round(float(np.mean(gray)), 4),
        "luminance_std": round(float(np.std(gray)), 4),
        "background_p05": round(background_p05, 4),
        "background_p95": round(background_p95, 4),
        "background_range": round(background_p95 - background_p05, 4),
        "dark_pixel_ratio": round(float(np.mean(gray < 120)), 6),
    }
    if image.ndim == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        bright = value >= 160
        bright_saturation = saturation[bright]
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        a = lab[:, :, 1].astype(np.float32) - 128.0
        b = lab[:, :, 2].astype(np.float32) - 128.0
        chroma = np.sqrt((a * a) + (b * b))
        row_ratios = np.mean(saturation >= 90, axis=1)
        metrics.update(
            {
                "bright_saturation_median": round(
                    float(np.median(bright_saturation))
                    if bright_saturation.size
                    else 0.0,
                    4,
                ),
                "bright_saturation_p99": round(
                    float(np.percentile(bright_saturation, 99))
                    if bright_saturation.size
                    else 0.0,
                    4,
                ),
                "bright_chroma_median": round(
                    float(np.median(chroma[bright])) if np.any(bright) else 0.0,
                    4,
                ),
                "bright_chroma_p99": round(
                    float(np.percentile(chroma[bright], 99))
                    if np.any(bright)
                    else 0.0,
                    4,
                ),
                "colored_bright_area_ratio": round(
                    float(np.mean(bright & (chroma >= 10.0))), 6
                ),
                "max_saturated_row_ratio": round(float(np.max(row_ratios)), 6),
            }
        )
    else:
        metrics.update(
            {
                "bright_saturation_median": 0.0,
                "bright_saturation_p99": 0.0,
                "bright_chroma_median": 0.0,
                "bright_chroma_p99": 0.0,
                "colored_bright_area_ratio": 0.0,
                "max_saturated_row_ratio": 0.0,
            }
        )
    return metrics


def _is_non_latin_letter(character: str) -> bool:
    if not character.isalpha():
        return False
    return "LATIN" not in unicodedata.name(character, "")


def text_quality_metrics(text: str) -> Dict[str, Any]:
    normalized = unicodedata.normalize("NFC", text or "")
    characters = [character for character in normalized if not character.isspace()]
    tokens = re.findall(r"\S+", normalized, flags=re.UNICODE)
    symbol_characters = sum(
        not character.isalnum()
        and character not in SAFE_PUNCTUATION
        and not unicodedata.category(character).startswith(("L", "N"))
        for character in characters
    )
    non_latin_letters = sum(_is_non_latin_letter(character) for character in characters)
    suspicious_tokens = 0
    for token in tokens:
        has_alphanumeric = any(character.isalnum() for character in token)
        has_non_latin = any(_is_non_latin_letter(character) for character in token)
        if not has_alphanumeric or has_non_latin:
            suspicious_tokens += 1
    nonempty_lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    symbol_only_lines = sum(
        not any(character.isalnum() for character in line) for line in nonempty_lines
    )
    denominator = max(1, len(characters))
    token_denominator = max(1, len(tokens))
    line_denominator = max(1, len(nonempty_lines))
    return {
        "character_count": len(characters),
        "token_count": len(tokens),
        "line_count": len(nonempty_lines),
        "symbol_character_ratio": round(symbol_characters / denominator, 6),
        "non_latin_letter_ratio": round(non_latin_letters / denominator, 6),
        "suspicious_token_ratio": round(suspicious_tokens / token_denominator, 6),
        "symbol_only_line_ratio": round(symbol_only_lines / line_denominator, 6),
    }


def _create_contact_sheet(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    tile_width, tile_height = 760, 500
    sheet = Image.new(
        "RGB", (tile_width * 2, tile_height * len(records)), (225, 225, 225)
    )
    draw = ImageDraw.Draw(sheet)
    for row, record in enumerate(records):
        for column, key in enumerate(("original_file", "processed_file")):
            image = Image.open(record[key]).convert("RGB")
            image.thumbnail((tile_width - 30, tile_height - 70))
            x = column * tile_width + (tile_width - image.width) // 2
            y = row * tile_height + 45
            sheet.paste(image, (x, y))
        header = (
            f"{record['sample_id']} | conf={record['ocr']['confidence']:.3f} | "
            f"words={record['ocr']['word_count']} | "
            f"bg_range={record['original_metrics']['background_range']:.1f}"
        )
        draw.text((10, row * tile_height + 10), header, fill="black")
        draw.text((10, row * tile_height + 28), "GOC", fill="black")
        draw.text((tile_width + 10, row * tile_height + 28), "HIEN TAI", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def _write_csv(records: Sequence[Dict[str, Any]], path: Path) -> None:
    rows = []
    for record in records:
        rows.append(
            {
                "sample_id": record["sample_id"],
                "source": record["source"],
                "page_number": record["page_number"],
                "issue_tags": ",".join(record["issue_tags"]),
                "confidence": record["ocr"]["confidence"],
                "word_count": record["ocr"]["word_count"],
                **record["text_metrics"],
                **{
                    f"original_{key}": value
                    for key, value in record["original_metrics"].items()
                },
                **{
                    f"processed_{key}": value
                    for key, value in record["processed_metrics"].items()
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(records: Sequence[Dict[str, Any]], path: Path) -> None:
    lines = [
        "# Baseline chuẩn hóa nền OCR",
        "",
        "> Đây là baseline của pipeline hiện tại, không phải độ chính xác CER/WER. "
        "Confidence chỉ dùng để so sánh các biến thể trên cùng trang.",
        "",
        "| Mẫu | Nhóm lỗi | Confidence | Từ | Ký tự | Ký tự nghi vấn | Nền lệch sáng | Màu P99 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| {sample} | {tags} | {confidence:.3f} | {words} | {characters} | "
            "{suspicious:.3f} | {background:.1f} | {chroma:.1f} |".format(
                sample=record["sample_id"],
                tags=", ".join(record["issue_tags"]),
                confidence=record["ocr"]["confidence"],
                words=record["ocr"]["word_count"],
                characters=record["text_metrics"]["character_count"],
                suspicious=record["text_metrics"]["suspicious_token_ratio"],
                background=record["original_metrics"]["background_range"],
                chroma=record["original_metrics"]["bright_chroma_p99"],
            )
        )
    lines.extend(
        [
            "",
            "## Tệp kiểm tra",
            "",
            "- `contact_sheet.jpg`: ảnh gốc và ảnh pipeline hiện tại.",
            "- `baseline_metrics.csv`: số liệu đầy đủ.",
            "- `baseline_manifest.json`: nguồn, checksum, cấu hình và metadata.",
            "- Mỗi thư mục mẫu có `original.png`, `processed.png`, `overlay.png` và `ocr.txt`.",
        ]
    )
    atomic_write_text(path, "\n".join(lines).rstrip() + "\n")


def run_baseline(
    config_path: Path,
    output_dir: Path,
    tessdata_dir: Path,
    tesseract_cmd: str | None = None,
    dpi: int = 300,
) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    tessdata_dir = Path(tessdata_dir).resolve()
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    sample_specs = config_data.get("samples", [])
    if not sample_specs:
        raise ValueError("Config baseline không có samples")

    resolved_command = find_tesseract_executable(tesseract_cmd)
    if not resolved_command:
        raise RuntimeError("Không tìm thấy tesseract.exe")
    ocr_config = TesseractConfig(
        input_dir=config_path.parent,
        output_dir=output_dir,
        dpi=dpi,
        lang="vie",
        oem=1,
        psm=6,
        preprocessing="grayscale",
        thresholding_method=2,
        tesseract_cmd=resolved_command,
        tessdata_dir=tessdata_dir,
    )
    engine_version = verify_tesseract_environment(ocr_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.perf_counter()

    for position, spec in enumerate(sample_specs, 1):
        sample_id = str(spec["id"])
        source = Path(str(spec["source"]))
        if not source.is_absolute():
            source = config_path.parent / source
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Không tìm thấy mẫu: {source}")
        page_number = int(spec.get("page_number", 1))
        sample_dir = output_dir / f"{position:02d}_{sample_id}"
        original = render_sample(source, page_number, dpi=dpi)
        processed, prep_meta = preprocess_image(original, mode="grayscale")
        result = run_tesseract_on_image(
            processed,
            ocr_config,
            page_number=page_number,
            prep_meta=prep_meta.to_dict(),
        )

        original_path = sample_dir / "original.png"
        processed_path = sample_dir / "processed.png"
        overlay_path = sample_dir / "overlay.png"
        text_path = sample_dir / "ocr.txt"
        save_png(original_path, original)
        save_png(processed_path, processed)
        save_png(overlay_path, make_ocr_overlay(processed, result))
        atomic_write_text(text_path, unicodedata.normalize("NFC", result.text).strip() + "\n")

        record = {
            "sample_id": sample_id,
            "source": str(source),
            "source_sha256": sha256_file(source),
            "page_number": page_number,
            "issue_tags": list(spec.get("issue_tags", [])),
            "original_file": str(original_path),
            "processed_file": str(processed_path),
            "overlay_file": str(overlay_path),
            "text_file": str(text_path),
            "preprocessing": prep_meta.to_dict(),
            "original_metrics": image_quality_metrics(original),
            "processed_metrics": image_quality_metrics(processed),
            "ocr": {
                "status": result.status,
                "confidence": round(float(result.confidence), 6),
                "word_count": len(result.boxes),
                "duration_seconds": round(float(result.duration_seconds), 3),
                "error": result.error,
            },
            "text_metrics": text_quality_metrics(result.text),
        }
        records.append(record)
        print(
            f"[{position}/{len(sample_specs)}] {sample_id}: "
            f"conf={result.confidence:.3f}, words={len(result.boxes)}",
            flush=True,
        )

    manifest: Dict[str, Any] = {
        "schema_version": "1.0",
        "operation": "background_normalization_baseline",
        "warning": "Confidence is not accuracy; CER/WER require verified ground truth",
        "config_file": str(config_path),
        "config_sha256": sha256_file(config_path),
        "sample_count": len(records),
        "ocr_config": {
            "engine": "tesseract",
            "engine_version": engine_version,
            "tessdata_profile": "best",
            "tessdata_dir": str(tessdata_dir),
            "language": "vie",
            "oem": 1,
            "psm": 6,
            "dpi": dpi,
            "preprocessing": "grayscale_document_cleanup",
            "thresholding_method": "Sauvola (2)",
        },
        "processing_seconds": round(time.perf_counter() - started, 3),
        "samples": records,
    }
    atomic_write_json(output_dir / "baseline_manifest.json", manifest)
    _write_csv(records, output_dir / "baseline_metrics.csv")
    _write_markdown_report(records, output_dir / "baseline_report.md")
    _create_contact_sheet(records, output_dir / "contact_sheet.jpg")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tạo baseline chuẩn hóa nền OCR")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tessdata-dir", type=Path, required=True)
    parser.add_argument("--tesseract-cmd", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_baseline(
        args.config,
        args.output_dir,
        args.tessdata_dir,
        tesseract_cmd=args.tesseract_cmd,
        dpi=args.dpi,
    )
    print(
        f"Đã tạo baseline {result['sample_count']} mẫu trong "
        f"{result['processing_seconds']:.1f} giây"
    )
    print(str(Path(args.output_dir).resolve() / "baseline_report.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
