"""Đánh giá Bước 3: chọn tự động giữa OCR hiện tại và ảnh đã xử lý."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tesseract_ocr.document_renderer import atomic_write_json, save_png, sha256_file
from tesseract_ocr.document_text_merger import atomic_write_text
from tesseract_ocr.ocr_variant_selector import select_ocr_variant


def _load_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if image is None:
        raise RuntimeError(f"Không thể đọc ảnh: {path}")
    return image


def _variant(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "confidence": float(record["ocr"]["confidence"]),
        "word_count": int(record["ocr"]["word_count"]),
        "text_metrics": dict(record["text_metrics"]),
    }


def _create_contact_sheet(records: Sequence[Dict[str, Any]], path: Path) -> None:
    tile_width, tile_height = 600, 420
    sheet = Image.new(
        "RGB", (tile_width * 3, tile_height * len(records)), (225, 225, 225)
    )
    draw = ImageDraw.Draw(sheet)
    for row, record in enumerate(records):
        for column, key in enumerate(
            ("current_file", "candidate_file", "selected_file")
        ):
            image = Image.open(record[key]).convert("RGB")
            image.thumbnail((tile_width - 24, tile_height - 72))
            x = column * tile_width + (tile_width - image.width) // 2
            y = row * tile_height + 57
            sheet.paste(image, (x, y))
        decision = record["decision"]
        header = (
            f"{record['sample_id']} | selected={decision['selected']} | "
            f"score {decision['current_score']:.3f}->{decision['candidate_score']:.3f} | "
            f"reason={decision['reason']}"
        )
        draw.text((10, row * tile_height + 10), header, fill="black")
        draw.text((10, row * tile_height + 33), "HIEN TAI", fill="black")
        draw.text((tile_width + 10, row * tile_height + 33), "DA XU LY", fill="black")
        draw.text(
            (2 * tile_width + 10, row * tile_height + 33),
            "DUOC CHON",
            fill="black",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def _write_csv(records: Sequence[Dict[str, Any]], path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "sample_id": record["sample_id"],
                "source": record["source"],
                "page_number": record["page_number"],
                "selected": record["decision"]["selected"],
                "reason": record["decision"]["reason"],
                "current_confidence": record["current"]["confidence"],
                "candidate_confidence": record["candidate"]["confidence"],
                "selected_confidence": record["selected"]["confidence"],
                "current_suspicious": record["current"]["text_metrics"][
                    "suspicious_token_ratio"
                ],
                "candidate_suspicious": record["candidate"]["text_metrics"][
                    "suspicious_token_ratio"
                ],
                "selected_suspicious": record["selected"]["text_metrics"][
                    "suspicious_token_ratio"
                ],
                **record["decision"],
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _criteria(records: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    harmful_selected = sum(
        record["decision"]["selected"] == "candidate"
        and not record["decision"]["guardrail_passed"]
        for record in records
    )
    selected_candidates = sum(
        record["decision"]["selected"] == "candidate" for record in records
    )
    rejected_changed = sum(
        record["decision"]["selected"] == "current"
        and record["decision"]["pixel_difference_ratio"] > 1e-6
        for record in records
    )
    return [
        {
            "criterion": "no_guardrail_violation_selected",
            "value": float(harmful_selected),
            "threshold": 0.0,
            "passed": harmful_selected == 0,
        },
        {
            "criterion": "at_least_one_improved_candidate_selected",
            "value": float(selected_candidates),
            "threshold": 1.0,
            "passed": selected_candidates >= 1,
        },
        {
            "criterion": "at_least_one_changed_candidate_rejected",
            "value": float(rejected_changed),
            "threshold": 1.0,
            "passed": rejected_changed >= 1,
        },
        {
            "criterion": "selected_average_confidence_not_lower",
            "value": float(summary["average_confidence_delta"]),
            "threshold": 0.0,
            "passed": summary["average_confidence_delta"] >= 0.0,
        },
        {
            "criterion": "selected_average_suspicious_not_higher",
            "value": float(summary["average_suspicious_delta"]),
            "threshold": 0.0,
            "passed": summary["average_suspicious_delta"] <= 0.0,
        },
    ]


def _write_report(manifest: Dict[str, Any], path: Path) -> None:
    summary = manifest["summary"]
    lines = [
        "# Bước 3 - Chọn tự động biến thể OCR",
        "",
        "> Điểm chọn là chỉ số tương đối trên cùng trang, không phải độ chính xác CER/WER.",
        "",
        f"- Số mẫu: {manifest['sample_count']}",
        f"- Chọn ảnh đã xử lý: {summary['candidate_selected_count']}",
        f"- Giữ ảnh hiện tại: {summary['current_selected_count']}",
        f"- Confidence trung bình: {summary['average_confidence_current']:.3f} -> {summary['average_confidence_selected']:.3f}",
        f"- Token nghi vấn trung bình: {summary['average_suspicious_current']:.3f} -> {summary['average_suspicious_selected']:.3f}",
        f"- Kết luận tiêu chí: {'ĐẠT' if manifest['criteria_passed'] else 'CHƯA ĐẠT'}",
        "",
        "| Mẫu | Chọn | Lý do | Điểm hiện tại | Điểm xử lý | Conf chọn | Nghi vấn chọn |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for record in manifest["samples"]:
        decision = record["decision"]
        selected = record["selected"]
        lines.append(
            "| {sample} | {selected_name} | {reason} | {current:.3f} | {candidate:.3f} | "
            "{confidence:.3f} | {suspicious:.3f} |".format(
                sample=record["sample_id"],
                selected_name=(
                    "đã xử lý" if decision["selected"] == "candidate" else "hiện tại"
                ),
                reason=decision["reason"],
                current=decision["current_score"],
                candidate=decision["candidate_score"],
                confidence=selected["confidence"],
                suspicious=selected["text_metrics"]["suspicious_token_ratio"],
            )
        )
    lines.extend(["", "## Tiêu chí", ""])
    for criterion in manifest["criteria"]:
        lines.append(
            "- [{mark}] {name}: {value:.4f} (ngưỡng {threshold:.4f})".format(
                mark="x" if criterion["passed"] else " ",
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
            "Bước này đã chọn và lưu artifact cuối cho bộ thử nghiệm. Chưa tích hợp logic "
            "chọn biến thể vào giao diện web/pipeline OCR sản xuất.",
        ]
    )
    atomic_write_text(path, "\n".join(lines).rstrip() + "\n")


def run_step3(
    baseline_manifest: Path,
    step2_manifest: Path,
    output_dir: Path,
    minimum_score_gain: float = 0.015,
) -> Dict[str, Any]:
    baseline_manifest = Path(baseline_manifest).resolve()
    step2_manifest = Path(step2_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    baseline = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    step2 = json.loads(step2_manifest.read_text(encoding="utf-8"))
    baseline_by_id = {item["sample_id"]: item for item in baseline["samples"]}
    step2_samples = step2.get("samples", [])
    if not step2_samples:
        raise ValueError("Step 2 manifest không có samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    for position, candidate_record in enumerate(step2_samples, 1):
        sample_id = str(candidate_record["sample_id"])
        current_record = baseline_by_id[sample_id]
        current_image = _load_image(
            Path(current_record["processed_file"]), cv2.IMREAD_GRAYSCALE
        )
        candidate_image = _load_image(
            Path(candidate_record["step2_file"]), cv2.IMREAD_GRAYSCALE
        )
        if current_image.shape != candidate_image.shape:
            raise RuntimeError(
                f"Hai biến thể khác kích thước ở {sample_id}: "
                f"{current_image.shape} != {candidate_image.shape}"
            )
        pixel_difference_ratio = float(np.mean(current_image != candidate_image))
        current_variant = _variant(current_record)
        candidate_variant = _variant(candidate_record)
        decision = select_ocr_variant(
            current_variant,
            candidate_variant,
            pixel_difference_ratio=pixel_difference_ratio,
            minimum_score_gain=minimum_score_gain,
        )
        choose_candidate = decision["selected"] == "candidate"
        selected_record = candidate_record if choose_candidate else current_record
        selected_variant = candidate_variant if choose_candidate else current_variant
        selected_image = candidate_image if choose_candidate else current_image
        selected_overlay_source = Path(
            candidate_record["overlay_file"]
            if choose_candidate
            else current_record["overlay_file"]
        )
        selected_text_source = Path(
            candidate_record["text_file"]
            if choose_candidate
            else current_record["text_file"]
        )

        sample_dir = output_dir / f"{position:02d}_{sample_id}"
        current_path = sample_dir / "current.png"
        candidate_path = sample_dir / "candidate.png"
        selected_path = sample_dir / "selected.png"
        selected_overlay_path = sample_dir / "selected_overlay.png"
        selected_text_path = sample_dir / "selected_ocr.txt"
        save_png(current_path, current_image)
        save_png(candidate_path, candidate_image)
        save_png(selected_path, selected_image)
        selected_overlay = _load_image(selected_overlay_source, cv2.IMREAD_COLOR)
        save_png(selected_overlay_path, selected_overlay)
        atomic_write_text(
            selected_text_path,
            selected_text_source.read_text(encoding="utf-8").rstrip() + "\n",
        )

        records.append(
            {
                "sample_id": sample_id,
                "source": current_record["source"],
                "source_sha256": current_record["source_sha256"],
                "page_number": current_record["page_number"],
                "issue_tags": current_record["issue_tags"],
                "current_file": str(current_path),
                "candidate_file": str(candidate_path),
                "selected_file": str(selected_path),
                "selected_overlay_file": str(selected_overlay_path),
                "selected_text_file": str(selected_text_path),
                "current": current_variant,
                "candidate": candidate_variant,
                "selected": selected_variant,
                "decision": decision,
                "selected_source_stage": "step2" if choose_candidate else "baseline",
                "selected_ocr_status": selected_record["ocr"]["status"],
            }
        )
        print(
            f"[{position}/{len(step2_samples)}] {sample_id}: "
            f"{decision['selected']} ({decision['reason']}), "
            f"score={decision['current_score']:.3f}->{decision['candidate_score']:.3f}",
            flush=True,
        )

    average_confidence_current = float(
        np.mean([record["current"]["confidence"] for record in records])
    )
    average_confidence_selected = float(
        np.mean([record["selected"]["confidence"] for record in records])
    )
    average_suspicious_current = float(
        np.mean(
            [
                record["current"]["text_metrics"]["suspicious_token_ratio"]
                for record in records
            ]
        )
    )
    average_suspicious_selected = float(
        np.mean(
            [
                record["selected"]["text_metrics"]["suspicious_token_ratio"]
                for record in records
            ]
        )
    )
    summary = {
        "candidate_selected_count": sum(
            record["decision"]["selected"] == "candidate" for record in records
        ),
        "current_selected_count": sum(
            record["decision"]["selected"] == "current" for record in records
        ),
        "average_confidence_current": round(average_confidence_current, 6),
        "average_confidence_selected": round(average_confidence_selected, 6),
        "average_confidence_delta": round(
            average_confidence_selected - average_confidence_current, 6
        ),
        "average_suspicious_current": round(average_suspicious_current, 6),
        "average_suspicious_selected": round(average_suspicious_selected, 6),
        "average_suspicious_delta": round(
            average_suspicious_selected - average_suspicious_current, 6
        ),
    }
    criteria = _criteria(records, summary)
    manifest: Dict[str, Any] = {
        "schema_version": "1.0",
        "operation": "select_ocr_variant_step3",
        "warning": "Selection score is relative; it is not CER/WER accuracy",
        "baseline_manifest": str(baseline_manifest),
        "baseline_manifest_sha256": sha256_file(baseline_manifest),
        "step2_manifest": str(step2_manifest),
        "step2_manifest_sha256": sha256_file(step2_manifest),
        "algorithm": "guardrailed_ocr_quality_score_v1",
        "minimum_score_gain": minimum_score_gain,
        "sample_count": len(records),
        "summary": summary,
        "criteria_passed": all(bool(item["passed"]) for item in criteria),
        "criteria": criteria,
        "samples": records,
    }
    atomic_write_json(output_dir / "step3_manifest.json", manifest)
    _write_csv(records, output_dir / "step3_metrics.csv")
    _write_report(manifest, output_dir / "step3_report.md")
    _create_contact_sheet(records, output_dir / "contact_sheet.jpg")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Đánh giá Bước 3 chọn biến thể OCR")
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--step2-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-score-gain", type=float, default=0.015)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_step3(
        args.baseline_manifest,
        args.step2_manifest,
        args.output_dir,
        minimum_score_gain=args.minimum_score_gain,
    )
    summary = result["summary"]
    print(
        f"Hoàn tất Bước 3: chọn processed={summary['candidate_selected_count']}, "
        f"giữ current={summary['current_selected_count']}; "
        f"tiêu chí={'ĐẠT' if result['criteria_passed'] else 'CHƯA ĐẠT'}"
    )
    print(str(Path(args.output_dir).resolve() / "step3_report.md"))
    return 0 if result["criteria_passed"] else 4


if __name__ == "__main__":
    sys.exit(main())
