"""Đánh giá Bước 2: trung hòa nền màu trên kết quả Bước 1."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tesseract_ocr.background_baseline import image_quality_metrics, text_quality_metrics
from tesseract_ocr.color_background_normalizer import (
    neutralize_color_background,
    prepare_color_document_geometry,
)
from tesseract_ocr.config import TesseractConfig
from tesseract_ocr.document_renderer import atomic_write_json, save_png, sha256_file
from tesseract_ocr.document_text_merger import atomic_write_text
from tesseract_ocr.illumination_normalizer import thin_line_retention
from tesseract_ocr.region_ocr import make_ocr_overlay
from tesseract_ocr.tesseract_engine import (
    find_tesseract_executable,
    run_tesseract_on_image,
    verify_tesseract_environment,
)


COLOR_TARGETS = {
    "colored_certificate",
    "identity_card_control",
    "insurance_mixed_background",
    "insurance_yellow_blue",
}


def _load_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if image is None:
        raise RuntimeError(f"Không thể đọc ảnh: {path}")
    return image


def _create_contact_sheet(records: Sequence[Dict[str, Any]], path: Path) -> None:
    tile_width, tile_height = 760, 500
    sheet = Image.new(
        "RGB", (tile_width * 2, tile_height * len(records)), (225, 225, 225)
    )
    draw = ImageDraw.Draw(sheet)
    for row, record in enumerate(records):
        for column, key in enumerate(("step1_file", "step2_file")):
            image = Image.open(record[key]).convert("RGB")
            image.thumbnail((tile_width - 30, tile_height - 75))
            x = column * tile_width + (tile_width - image.width) // 2
            y = row * tile_height + 52
            sheet.paste(image, (x, y))
        comparison = record["comparison"]
        header = (
            f"{record['sample_id']} | applied={record['color_normalization']['applied']} | "
            f"conf {comparison['confidence_before']:.3f} -> "
            f"{comparison['confidence_after']:.3f} | "
            f"sus {comparison['suspicious_token_before']:.3f} -> "
            f"{comparison['suspicious_token_after']:.3f}"
        )
        draw.text((10, row * tile_height + 10), header, fill="black")
        draw.text((10, row * tile_height + 31), "SAU BUOC 1", fill="black")
        draw.text((tile_width + 10, row * tile_height + 31), "TRUNG HOA MAU", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def _write_csv(records: Sequence[Dict[str, Any]], path: Path) -> None:
    rows = []
    for record in records:
        rows.append(
            {
                "sample_id": record["sample_id"],
                "source": record["source"],
                "page_number": record["page_number"],
                "issue_tags": ",".join(record["issue_tags"]),
                "applied": record["color_normalization"]["applied"],
                **record["comparison"],
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _criteria(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        sample_id = record["sample_id"]
        comparison = record["comparison"]
        if sample_id in COLOR_TARGETS:
            applied = bool(record["color_normalization"]["applied"])
            rows.append(
                {
                    "criterion": "colored_background_normalized",
                    "sample_id": sample_id,
                    "value": 1.0 if applied else 0.0,
                    "threshold": 1.0,
                    "passed": applied,
                }
            )
            rows.append(
                {
                    "criterion": "neutral_output_near_white_at_least_0.70",
                    "sample_id": sample_id,
                    "value": comparison["near_white_ratio"],
                    "threshold": 0.70,
                    "passed": comparison["near_white_ratio"] >= 0.70,
                }
            )
            rows.append(
                {
                    "criterion": "ocr_candidate_not_empty",
                    "sample_id": sample_id,
                    "value": float(comparison["word_count_after"]),
                    "threshold": 10.0,
                    "passed": comparison["word_count_after"] >= 10,
                }
            )
            rows.append(
                {
                    "criterion": "edge_retention_at_least_0.75",
                    "sample_id": sample_id,
                    "value": comparison["edge_retention"],
                    "threshold": 0.75,
                    "passed": comparison["edge_retention"] >= 0.75,
                }
            )
        else:
            unchanged = comparison["pixel_difference_ratio"] == 0.0
            rows.append(
                {
                    "criterion": "non_color_sample_unchanged",
                    "sample_id": sample_id,
                    "value": 1.0 if unchanged else 0.0,
                    "threshold": 1.0,
                    "passed": unchanged,
                }
            )
    return rows


def _write_report(manifest: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Bước 2 - Trung hòa nền màu",
        "",
        "> Confidence không phải độ chính xác. Bước 3 mới chọn giữa ảnh hiện tại và biến thể đã xử lý.",
        "",
        f"- Số mẫu: {manifest['sample_count']}",
        f"- Mẫu nền màu đã xử lý: {manifest['applied_count']}",
        f"- Confidence trung bình: {manifest['summary']['average_confidence_before']:.3f} -> {manifest['summary']['average_confidence_after']:.3f}",
        f"- Token nghi vấn trung bình: {manifest['summary']['average_suspicious_before']:.3f} -> {manifest['summary']['average_suspicious_after']:.3f}",
        f"- Kết luận tiêu chí kỹ thuật: {'ĐẠT' if manifest['criteria_passed'] else 'CHƯA ĐẠT'}",
        "",
        "| Mẫu | Xử lý màu | Conf trước | Conf sau | Δ conf | Nghi vấn trước | Nghi vấn sau | Từ sau | Nền trắng | Giữ biên |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in manifest["samples"]:
        item = record["comparison"]
        lines.append(
            "| {sample} | {applied} | {cb:.3f} | {ca:.3f} | {delta:+.3f} | "
            "{sb:.3f} | {sa:.3f} | {words} | {white:.3f} | {edges:.3f} |".format(
                sample=record["sample_id"],
                applied="có" if record["color_normalization"]["applied"] else "không",
                cb=item["confidence_before"],
                ca=item["confidence_after"],
                delta=item["confidence_delta"],
                sb=item["suspicious_token_before"],
                sa=item["suspicious_token_after"],
                words=item["word_count_after"],
                white=item["near_white_ratio"],
                edges=item["edge_retention"],
            )
        )
    lines.extend(["", "## Tiêu chí kỹ thuật", ""])
    for criterion in manifest["criteria"]:
        lines.append(
            "- [{mark}] `{sample}` - {name}: {value:.4f} (ngưỡng {threshold:.4f})".format(
                mark="x" if criterion["passed"] else " ",
                sample=criterion["sample_id"],
                name=criterion["criterion"],
                value=criterion["value"],
                threshold=criterion["threshold"],
            )
        )
    lines.extend(
        [
            "",
            "## Phạm vi",
            "",
            "Bước này tạo biến thể trung hòa màu và ghi lại tín hiệu OCR. Chưa tự động "
            "chọn ảnh tốt hơn và chưa tích hợp vào giao diện web.",
        ]
    )
    atomic_write_text(path, "\n".join(lines).rstrip() + "\n")


def run_step2(
    baseline_manifest: Path,
    step1_manifest: Path,
    output_dir: Path,
    tessdata_dir: Path,
    tesseract_cmd: str | None = None,
) -> Dict[str, Any]:
    baseline_manifest = Path(baseline_manifest).resolve()
    step1_manifest = Path(step1_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    tessdata_dir = Path(tessdata_dir).resolve()
    baseline = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    step1 = json.loads(step1_manifest.read_text(encoding="utf-8"))
    baseline_by_id = {item["sample_id"]: item for item in baseline["samples"]}
    step1_samples = step1.get("samples", [])
    if not step1_samples:
        raise ValueError("Step 1 manifest không có samples")

    resolved_command = find_tesseract_executable(tesseract_cmd)
    if not resolved_command:
        raise RuntimeError("Không tìm thấy tesseract.exe")
    base_config = baseline["ocr_config"]
    ocr_config = TesseractConfig(
        input_dir=baseline_manifest.parent,
        output_dir=output_dir,
        dpi=int(base_config["dpi"]),
        lang=str(base_config["language"]),
        oem=int(base_config["oem"]),
        psm=int(base_config["psm"]),
        preprocessing="grayscale",
        thresholding_method=2,
        tesseract_cmd=resolved_command,
        tessdata_dir=tessdata_dir,
    )
    engine_version = verify_tesseract_environment(ocr_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    started = time.perf_counter()

    for position, step1_record in enumerate(step1_samples, 1):
        sample_id = str(step1_record["sample_id"])
        baseline_record = baseline_by_id[sample_id]
        sample_dir = output_dir / f"{position:02d}_{sample_id}"
        input_gray = _load_image(Path(step1_record["balanced_file"]), cv2.IMREAD_GRAYSCALE)
        geometry_meta: Dict[str, Any] = {}

        if sample_id in COLOR_TARGETS:
            original = _load_image(Path(baseline_record["original_file"]), cv2.IMREAD_COLOR)
            prepared_color, geometry_meta = prepare_color_document_geometry(original)
            candidate, color_meta = neutralize_color_background(prepared_color)
            if candidate.shape != input_gray.shape:
                raise RuntimeError(
                    f"Kích thước Step 2 khác Step 1 ở {sample_id}: "
                    f"{candidate.shape} != {input_gray.shape}"
                )
        else:
            candidate = input_gray.copy()
            color_meta = {
                "applied": False,
                "reason": "not_a_colored_background_target",
                "near_white_ratio": round(float(np.mean(candidate >= 245)), 6),
            }

        result = run_tesseract_on_image(
            candidate,
            ocr_config,
            page_number=int(step1_record["page_number"]),
            prep_meta={
                "color_normalization": color_meta,
                "geometry": geometry_meta,
            },
        )
        input_path = sample_dir / "step1_input.png"
        output_path = sample_dir / "color_neutralized.png"
        overlay_path = sample_dir / "overlay.png"
        text_path = sample_dir / "ocr.txt"
        save_png(input_path, input_gray)
        save_png(output_path, candidate)
        save_png(overlay_path, make_ocr_overlay(candidate, result))
        atomic_write_text(
            text_path, unicodedata.normalize("NFC", result.text).strip() + "\n"
        )

        after_text_metrics = text_quality_metrics(result.text)
        before_text_metrics = step1_record["text_metrics"]
        pixel_difference_ratio = float(np.mean(input_gray != candidate))
        comparison = {
            "confidence_before": round(float(step1_record["ocr"]["confidence"]), 6),
            "confidence_after": round(float(result.confidence), 6),
            "confidence_delta": round(
                float(result.confidence) - float(step1_record["ocr"]["confidence"]), 6
            ),
            "suspicious_token_before": float(before_text_metrics["suspicious_token_ratio"]),
            "suspicious_token_after": float(after_text_metrics["suspicious_token_ratio"]),
            "suspicious_token_delta": round(
                float(after_text_metrics["suspicious_token_ratio"])
                - float(before_text_metrics["suspicious_token_ratio"]),
                6,
            ),
            "word_count_before": int(step1_record["ocr"]["word_count"]),
            "word_count_after": len(result.boxes),
            "near_white_ratio": round(float(np.mean(candidate >= 245)), 6),
            "edge_retention": round(thin_line_retention(input_gray, candidate), 6),
            "pixel_difference_ratio": round(pixel_difference_ratio, 6),
        }
        records.append(
            {
                "sample_id": sample_id,
                "source": step1_record["source"],
                "source_sha256": step1_record["source_sha256"],
                "page_number": step1_record["page_number"],
                "issue_tags": step1_record["issue_tags"],
                "step1_file": str(input_path),
                "step2_file": str(output_path),
                "overlay_file": str(overlay_path),
                "text_file": str(text_path),
                "geometry": geometry_meta,
                "color_normalization": color_meta,
                "output_metrics": image_quality_metrics(candidate),
                "ocr": {
                    "status": result.status,
                    "confidence": round(float(result.confidence), 6),
                    "word_count": len(result.boxes),
                    "duration_seconds": round(float(result.duration_seconds), 3),
                    "error": result.error,
                },
                "text_metrics": after_text_metrics,
                "comparison": comparison,
            }
        )
        print(
            f"[{position}/{len(step1_samples)}] {sample_id}: "
            f"applied={color_meta['applied']}, "
            f"conf={comparison['confidence_before']:.3f}->{comparison['confidence_after']:.3f}, "
            f"words={comparison['word_count_before']}->{comparison['word_count_after']}",
            flush=True,
        )

    criteria = _criteria(records)
    manifest: Dict[str, Any] = {
        "schema_version": "1.0",
        "operation": "color_background_normalization_step2",
        "warning": "Confidence is not accuracy; Step 3 will choose between variants",
        "baseline_manifest": str(baseline_manifest),
        "baseline_manifest_sha256": sha256_file(baseline_manifest),
        "step1_manifest": str(step1_manifest),
        "step1_manifest_sha256": sha256_file(step1_manifest),
        "algorithm": "local_lab_color_difference_v1",
        "engine_version": engine_version,
        "sample_count": len(records),
        "applied_count": sum(bool(record["color_normalization"]["applied"]) for record in records),
        "processing_seconds": round(time.perf_counter() - started, 3),
        "summary": {
            "average_confidence_before": round(
                float(np.mean([r["comparison"]["confidence_before"] for r in records])), 6
            ),
            "average_confidence_after": round(
                float(np.mean([r["comparison"]["confidence_after"] for r in records])), 6
            ),
            "average_suspicious_before": round(
                float(np.mean([r["comparison"]["suspicious_token_before"] for r in records])), 6
            ),
            "average_suspicious_after": round(
                float(np.mean([r["comparison"]["suspicious_token_after"] for r in records])), 6
            ),
        },
        "criteria_passed": all(bool(item["passed"]) for item in criteria),
        "criteria": criteria,
        "samples": records,
    }
    atomic_write_json(output_dir / "step2_manifest.json", manifest)
    _write_csv(records, output_dir / "step2_metrics.csv")
    _write_report(manifest, output_dir / "step2_report.md")
    _create_contact_sheet(records, output_dir / "contact_sheet.jpg")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Đánh giá Bước 2 trung hòa nền màu")
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--step1-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tessdata-dir", type=Path, required=True)
    parser.add_argument("--tesseract-cmd", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_step2(
        args.baseline_manifest,
        args.step1_manifest,
        args.output_dir,
        args.tessdata_dir,
        tesseract_cmd=args.tesseract_cmd,
    )
    print(
        f"Hoàn tất Bước 2: {result['applied_count']}/{result['sample_count']} mẫu được xử lý; "
        f"tiêu chí={'ĐẠT' if result['criteria_passed'] else 'CHƯA ĐẠT'}"
    )
    print(str(Path(args.output_dir).resolve() / "step2_report.md"))
    return 0 if result["criteria_passed"] else 4


if __name__ == "__main__":
    sys.exit(main())
