"""Đánh giá Bước 1: cân bằng sáng-tối trên bộ baseline đã khóa."""

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

from tesseract_ocr.background_baseline import (
    image_quality_metrics,
    text_quality_metrics,
)
from tesseract_ocr.config import TesseractConfig
from tesseract_ocr.document_renderer import atomic_write_json, save_png, sha256_file
from tesseract_ocr.document_text_merger import atomic_write_text
from tesseract_ocr.illumination_normalizer import (
    normalize_illumination,
    thin_line_retention,
)
from tesseract_ocr.region_ocr import make_ocr_overlay
from tesseract_ocr.tesseract_engine import (
    find_tesseract_executable,
    run_tesseract_on_image,
    verify_tesseract_environment,
)


PRIMARY_TARGETS = {"purchase_order_control"}
COLORED_BACKGROUND_SAMPLES = {
    "colored_certificate",
    "identity_card_control",
    "insurance_mixed_background",
    "insurance_yellow_blue",
}
CONTROL_SAMPLES = {
    "composite_scan",
    "mixed_color_shadow_diagram",
    "delivery_form_control",
}


def _load_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
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
        for column, key in enumerate(("baseline_file", "balanced_file")):
            image = Image.open(record[key]).convert("RGB")
            image.thumbnail((tile_width - 30, tile_height - 75))
            x = column * tile_width + (tile_width - image.width) // 2
            y = row * tile_height + 52
            sheet.paste(image, (x, y))
        comparison = record["comparison"]
        header = (
            f"{record['sample_id']} | applied={record['illumination']['applied']} | "
            f"bg {comparison['background_range_before']:.1f} -> "
            f"{comparison['background_range_after']:.1f} | "
            f"conf {comparison['confidence_before']:.3f} -> "
            f"{comparison['confidence_after']:.3f}"
        )
        draw.text((10, row * tile_height + 10), header, fill="black")
        draw.text((10, row * tile_height + 31), "BASELINE", fill="black")
        draw.text((tile_width + 10, row * tile_height + 31), "CAN BANG SANG", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def _write_csv(records: Sequence[Dict[str, Any]], path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for record in records:
        comparison = record["comparison"]
        rows.append(
            {
                "sample_id": record["sample_id"],
                "source": record["source"],
                "page_number": record["page_number"],
                "issue_tags": ",".join(record["issue_tags"]),
                "applied": record["illumination"]["applied"],
                **comparison,
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _criterion_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    criteria: List[Dict[str, Any]] = []
    by_id = {record["sample_id"]: record for record in records}
    for sample_id in sorted(PRIMARY_TARGETS):
        comparison = by_id[sample_id]["comparison"]
        value = comparison["background_reduction_ratio"]
        criteria.append(
            {
                "criterion": "background_reduction_at_least_30_percent",
                "sample_id": sample_id,
                "value": value,
                "threshold": 0.30,
                "passed": value >= 0.30,
            }
        )
    for sample_id in sorted(COLORED_BACKGROUND_SAMPLES):
        record = by_id[sample_id]
        deferred = (
            not record["illumination"]["applied"]
            and record["illumination"]["reason"]
            == "deferred_colored_background_to_step2"
        )
        criteria.append(
            {
                "criterion": "colored_background_deferred_to_step2",
                "sample_id": sample_id,
                "value": 1.0 if deferred else 0.0,
                "threshold": 1.0,
                "passed": deferred,
            }
        )
    for sample_id in sorted(CONTROL_SAMPLES):
        comparison = by_id[sample_id]["comparison"]
        value = comparison["confidence_delta"]
        criteria.append(
            {
                "criterion": "control_confidence_drop_at_most_0.05",
                "sample_id": sample_id,
                "value": value,
                "threshold": -0.05,
                "passed": value >= -0.05,
            }
        )
    for record in records:
        comparison = record["comparison"]
        value = comparison["suspicious_token_delta"]
        criteria.append(
            {
                "criterion": "suspicious_token_increase_at_most_0.02",
                "sample_id": record["sample_id"],
                "value": value,
                "threshold": 0.02,
                "passed": value <= 0.02,
            }
        )
        if "thin_lines" in record["issue_tags"] or "diagram" in record["issue_tags"]:
            retention = comparison["thin_line_retention"]
            criteria.append(
                {
                    "criterion": "thin_line_retention_at_least_0.95",
                    "sample_id": record["sample_id"],
                    "value": retention,
                    "threshold": 0.95,
                    "passed": retention >= 0.95,
                }
            )
    return criteria


def _write_report(manifest: Dict[str, Any], path: Path) -> None:
    records = manifest["samples"]
    lines = [
        "# Bước 1 - Cân bằng sáng-tối",
        "",
        "> Confidence không phải độ chính xác. CER/WER vẫn cần ground truth đã kiểm chứng.",
        "",
        f"- Số mẫu: {len(records)}",
        f"- Mẫu được cân bằng sáng: {manifest['applied_count']}",
        f"- Confidence trung bình: {manifest['summary']['average_confidence_before']:.3f} -> {manifest['summary']['average_confidence_after']:.3f}",
        f"- Tỷ lệ token nghi vấn trung bình: {manifest['summary']['average_suspicious_before']:.3f} -> {manifest['summary']['average_suspicious_after']:.3f}",
        f"- Kết luận tiêu chí: {'ĐẠT' if manifest['criteria_passed'] else 'CHƯA ĐẠT'}",
        "",
        "| Mẫu | Áp dụng | Nền trước | Nền sau | Giảm nền | Conf trước | Conf sau | Δ nghi vấn | Giữ nét |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        item = record["comparison"]
        lines.append(
            "| {sample} | {applied} | {before:.1f} | {after:.1f} | {reduction:.1%} | "
            "{conf_before:.3f} | {conf_after:.3f} | {suspicious:+.3f} | {retention:.3f} |".format(
                sample=record["sample_id"],
                applied="có" if record["illumination"]["applied"] else "không",
                before=item["background_range_before"],
                after=item["background_range_after"],
                reduction=item["background_reduction_ratio"],
                conf_before=item["confidence_before"],
                conf_after=item["confidence_after"],
                suspicious=item["suspicious_token_delta"],
                retention=item["thin_line_retention"],
            )
        )
    lines.extend(["", "## Tiêu chí", ""])
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
            "Bước này chỉ tạo và đánh giá biến thể cân bằng sáng. Chưa trung hòa nền màu, "
            "chưa chọn tự động giữa ảnh baseline và ảnh xử lý, và chưa tích hợp vào giao diện web.",
        ]
    )
    atomic_write_text(path, "\n".join(lines).rstrip() + "\n")


def run_step1(
    baseline_manifest: Path,
    output_dir: Path,
    tessdata_dir: Path,
    tesseract_cmd: str | None = None,
    activation_range: float = 18.0,
) -> Dict[str, Any]:
    baseline_manifest = Path(baseline_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    tessdata_dir = Path(tessdata_dir).resolve()
    baseline = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    baseline_samples = baseline.get("samples", [])
    if not baseline_samples:
        raise ValueError("Baseline manifest không có samples")

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
    started = time.perf_counter()
    records: List[Dict[str, Any]] = []

    for position, baseline_record in enumerate(baseline_samples, 1):
        sample_id = str(baseline_record["sample_id"])
        sample_dir = output_dir / f"{position:02d}_{sample_id}"
        baseline_image = _load_gray(Path(baseline_record["processed_file"]))
        issue_tags = list(baseline_record["issue_tags"])
        has_colored_background = any(
            tag in {
                "blue_background",
                "color_background",
                "colored_background",
                "identity_card",
                "yellow_background",
            }
            for tag in issue_tags
        )
        color_chroma_p99 = (
            float(baseline_record["original_metrics"]["bright_chroma_p99"])
            if has_colored_background
            else None
        )
        balanced, illumination = normalize_illumination(
            baseline_image,
            activation_range=activation_range,
            color_chroma_p99=color_chroma_p99,
        )
        result = run_tesseract_on_image(
            balanced,
            ocr_config,
            page_number=int(baseline_record["page_number"]),
            prep_meta={"illumination": illumination},
        )

        baseline_path = sample_dir / "baseline.png"
        balanced_path = sample_dir / "illumination_balanced.png"
        overlay_path = sample_dir / "overlay.png"
        text_path = sample_dir / "ocr.txt"
        save_png(baseline_path, baseline_image)
        save_png(balanced_path, balanced)
        save_png(overlay_path, make_ocr_overlay(balanced, result))
        atomic_write_text(
            text_path, unicodedata.normalize("NFC", result.text).strip() + "\n"
        )

        baseline_text_metrics = baseline_record["text_metrics"]
        balanced_text_metrics = text_quality_metrics(result.text)
        before_range = float(illumination["background_range_before"])
        after_range = float(illumination["background_range_after"])
        comparison = {
            "background_range_before": round(before_range, 4),
            "background_range_after": round(after_range, 4),
            "background_reduction_ratio": round(
                (before_range - after_range) / max(before_range, 1e-6), 6
            ),
            "confidence_before": round(float(baseline_record["ocr"]["confidence"]), 6),
            "confidence_after": round(float(result.confidence), 6),
            "confidence_delta": round(
                float(result.confidence) - float(baseline_record["ocr"]["confidence"]),
                6,
            ),
            "word_count_before": int(baseline_record["ocr"]["word_count"]),
            "word_count_after": len(result.boxes),
            "suspicious_token_before": float(
                baseline_text_metrics["suspicious_token_ratio"]
            ),
            "suspicious_token_after": float(
                balanced_text_metrics["suspicious_token_ratio"]
            ),
            "suspicious_token_delta": round(
                float(balanced_text_metrics["suspicious_token_ratio"])
                - float(baseline_text_metrics["suspicious_token_ratio"]),
                6,
            ),
            "thin_line_retention": round(
                thin_line_retention(baseline_image, balanced), 6
            ),
        }
        records.append(
            {
                "sample_id": sample_id,
                "source": baseline_record["source"],
                "source_sha256": baseline_record["source_sha256"],
                "page_number": baseline_record["page_number"],
                "issue_tags": baseline_record["issue_tags"],
                "baseline_file": str(baseline_path),
                "balanced_file": str(balanced_path),
                "overlay_file": str(overlay_path),
                "text_file": str(text_path),
                "illumination": illumination,
                "balanced_metrics": image_quality_metrics(balanced),
                "ocr": {
                    "status": result.status,
                    "confidence": round(float(result.confidence), 6),
                    "word_count": len(result.boxes),
                    "duration_seconds": round(float(result.duration_seconds), 3),
                    "error": result.error,
                },
                "text_metrics": balanced_text_metrics,
                "comparison": comparison,
            }
        )
        print(
            f"[{position}/{len(baseline_samples)}] {sample_id}: "
            f"applied={illumination['applied']}, bg={before_range:.1f}->{after_range:.1f}, "
            f"conf={comparison['confidence_before']:.3f}->{comparison['confidence_after']:.3f}",
            flush=True,
        )

    criteria = _criterion_rows(records)
    manifest: Dict[str, Any] = {
        "schema_version": "1.0",
        "operation": "illumination_normalization_step1",
        "warning": "Confidence is not accuracy; CER/WER require verified ground truth",
        "baseline_manifest": str(baseline_manifest),
        "baseline_manifest_sha256": sha256_file(baseline_manifest),
        "algorithm": "conditional_local_illumination_recenter_v1",
        "activation_range": activation_range,
        "engine_version": engine_version,
        "sample_count": len(records),
        "applied_count": sum(bool(record["illumination"]["applied"]) for record in records),
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
    atomic_write_json(output_dir / "step1_manifest.json", manifest)
    _write_csv(records, output_dir / "step1_metrics.csv")
    _write_report(manifest, output_dir / "step1_report.md")
    _create_contact_sheet(records, output_dir / "contact_sheet.jpg")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Đánh giá Bước 1 cân bằng sáng-tối")
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tessdata-dir", type=Path, required=True)
    parser.add_argument("--tesseract-cmd", default=None)
    parser.add_argument("--activation-range", type=float, default=18.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_step1(
        args.baseline_manifest,
        args.output_dir,
        args.tessdata_dir,
        tesseract_cmd=args.tesseract_cmd,
        activation_range=args.activation_range,
    )
    print(
        f"Hoàn tất Bước 1: {result['applied_count']}/{result['sample_count']} mẫu được xử lý; "
        f"tiêu chí={'ĐẠT' if result['criteria_passed'] else 'CHƯA ĐẠT'}"
    )
    print(str(Path(args.output_dir).resolve() / "step1_report.md"))
    return 0 if result["criteria_passed"] else 4


if __name__ == "__main__":
    sys.exit(main())
