"""Điều phối đầy đủ pipeline OCR bảo hiểm từ PDF đến một TXT.

Module nối bảy bước đã kiểm thử độc lập thành một API duy nhất để CLI và
giao diện Streamlit dùng chung. Mỗi bước vẫn ghi ảnh, manifest và checksum
riêng để có thể kiểm tra lại khi OCR sai.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from tesseract_ocr.artifact_masker import mask_panel_artifacts
from tesseract_ocr.background_preprocessor import process_masked_panels
from tesseract_ocr.document_renderer import (
    atomic_write_json,
    render_pdf_to_pngs,
    sha256_file,
)
from tesseract_ocr.document_text_merger import merge_region_ocr_results
from tesseract_ocr.panel_splitter import split_rendered_pages
from tesseract_ocr.perspective_corrector import rectify_panels
from tesseract_ocr.region_ocr import ocr_preprocessed_panels


ProgressCallback = Callable[[int, str], None]


@dataclass(frozen=True)
class InsurancePipelineConfig:
    dpi: int = 400
    expected_panels: Optional[int] = None
    panel_padding: int = 20
    perspective_min_confidence: float = 0.65
    perspective_output_margin: int = 8
    artifact_padding: int = 12
    channel_mode: str = "auto"
    yellow_threshold: float = 0.08
    clahe_clip_limit: float = 2.2
    clahe_tile_size: int = 8
    sauvola_window: int = 41
    sauvola_k: float = 0.20
    language: str = "vie+eng"
    psm_values: Tuple[int, ...] = (3, 6, 11)
    min_screening_score: float = 0.45

    def validate(self) -> None:
        if self.dpi <= 0:
            raise ValueError("DPI phải lớn hơn 0")
        if self.expected_panels not in (None, 2, 3):
            raise ValueError("expected_panels chỉ chấp nhận Auto, 2 hoặc 3")
        if self.panel_padding < 0 or self.artifact_padding < 0:
            raise ValueError("Padding không được âm")
        if not self.psm_values:
            raise ValueError("Phải có ít nhất một PSM")

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["psm_values"] = list(self.psm_values)
        return value


STAGE_NAMES = {
    1: "Render toàn bộ PDF",
    2: "Chia trang thành các vùng",
    3: "Chỉnh nghiêng và phối cảnh",
    4: "Che logo, QR và barcode",
    5: "Xử lý nền màu và tăng tương phản",
    6: "OCR từng vùng và chọn PSM",
    7: "Ghép một TXT cho toàn PDF",
}


def _notify(callback: Optional[ProgressCallback], step: int) -> None:
    if callback is not None:
        callback(step, STAGE_NAMES[step])


def _stage_record(
    step: int,
    manifest_file: Path,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    status = result.get("status")
    if status is None:
        if int(result.get("needs_review_panels", 0)) > 0:
            status = "needs_review"
        elif any(
            page.get("status") not in (None, "success")
            for page in result.get("pages", [])
        ):
            status = "needs_review"
        else:
            status = "success"
    record: Dict[str, Any] = {
        "step": step,
        "name": STAGE_NAMES[step],
        "status": str(status),
        "manifest_file": str(manifest_file.resolve()),
    }
    for key in (
        "page_count",
        "panel_count",
        "successful_panels",
        "needs_review_panels",
        "candidate_run_count",
        "processing_seconds",
    ):
        if key in result:
            record[key] = result[key]
    return record


def _cached_result(
    pipeline_manifest_path: Path,
    source_pdf: Path,
    source_sha256: str,
    config: InsurancePipelineConfig,
) -> Optional[Dict[str, Any]]:
    if not pipeline_manifest_path.is_file():
        return None
    try:
        cached = json.loads(pipeline_manifest_path.read_text(encoding="utf-8"))
        output_path = Path(str(cached["output_text_file"]))
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        cached.get("source_pdf") != str(source_pdf)
        or cached.get("source_pdf_sha256") != source_sha256
        or cached.get("config") != config.to_dict()
        or not output_path.is_file()
        or cached.get("output_text_sha256") != sha256_file(output_path)
    ):
        return None
    cached["cache_hit"] = True
    return cached


def run_insurance_region_pipeline(
    pdf_path: Path,
    run_dir: Path,
    tessdata_dir: Path,
    tesseract_cmd: Optional[str] = None,
    config: Optional[InsurancePipelineConfig] = None,
    progress_callback: Optional[ProgressCallback] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Chạy tuần tự bảy bước OCR bảo hiểm và trả về manifest tổng hợp."""
    pdf_path = Path(pdf_path).resolve()
    run_dir = Path(run_dir).resolve()
    tessdata_dir = Path(tessdata_dir).resolve()
    config = config or InsurancePipelineConfig()
    config.validate()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy PDF: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Đầu vào không phải PDF: {pdf_path}")
    if not tessdata_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy tessdata: {tessdata_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    pipeline_manifest_path = run_dir / "pipeline_manifest.json"
    source_sha256 = sha256_file(pdf_path)
    if not force:
        cached = _cached_result(
            pipeline_manifest_path, pdf_path, source_sha256, config
        )
        if cached is not None:
            return cached

    step_dirs = {
        1: run_dir / "step1_rendered",
        2: run_dir / "step2_panels",
        3: run_dir / "step3_rectified",
        4: run_dir / "step4_masked",
        5: run_dir / "step5_preprocessed",
        6: run_dir / "step6_ocr",
        7: run_dir / "step7_merged",
    }
    stages = []
    started = time.perf_counter()

    _notify(progress_callback, 1)
    render_result = render_pdf_to_pngs(pdf_path, step_dirs[1], dpi=config.dpi)
    stages.append(
        _stage_record(1, step_dirs[1] / "render_manifest.json", render_result)
    )

    _notify(progress_callback, 2)
    split_result = split_rendered_pages(
        step_dirs[1],
        step_dirs[2],
        expected_panels=config.expected_panels,
        padding=config.panel_padding,
    )
    stages.append(
        _stage_record(2, step_dirs[2] / "panels_manifest.json", split_result)
    )

    _notify(progress_callback, 3)
    perspective_result = rectify_panels(
        step_dirs[2],
        step_dirs[3],
        min_confidence=config.perspective_min_confidence,
        output_margin=config.perspective_output_margin,
    )
    stages.append(
        _stage_record(
            3, step_dirs[3] / "perspective_manifest.json", perspective_result
        )
    )

    _notify(progress_callback, 4)
    mask_result = mask_panel_artifacts(
        step_dirs[3], step_dirs[4], padding=config.artifact_padding
    )
    stages.append(
        _stage_record(
            4, step_dirs[4] / "artifact_mask_manifest.json", mask_result
        )
    )

    _notify(progress_callback, 5)
    background_result = process_masked_panels(
        step_dirs[4],
        step_dirs[5],
        channel_mode=config.channel_mode,
        yellow_threshold=config.yellow_threshold,
        clahe_clip_limit=config.clahe_clip_limit,
        clahe_tile_size=config.clahe_tile_size,
        sauvola_window=config.sauvola_window,
        sauvola_k=config.sauvola_k,
    )
    stages.append(
        _stage_record(
            5, step_dirs[5] / "background_manifest.json", background_result
        )
    )

    _notify(progress_callback, 6)
    ocr_result = ocr_preprocessed_panels(
        step_dirs[5],
        step_dirs[6],
        tessdata_dir=tessdata_dir,
        psm_values=config.psm_values,
        language=config.language,
        tesseract_cmd=tesseract_cmd,
        dpi=config.dpi,
        min_screening_score=config.min_screening_score,
    )
    stages.append(
        _stage_record(
            6, step_dirs[6] / "region_ocr_manifest.json", ocr_result
        )
    )

    _notify(progress_callback, 7)
    merge_result = merge_region_ocr_results(
        step_dirs[6] / "region_ocr_manifest.json",
        step_dirs[7],
        source_pdf=pdf_path,
    )
    stages.append(
        _stage_record(7, step_dirs[7] / "merge_manifest.json", merge_result)
    )

    output_text_path = Path(str(merge_result["output_text_file"])).resolve()
    result: Dict[str, Any] = {
        "schema_version": "1.0",
        "operation": "insurance_region_ocr_pipeline_7_steps",
        "status": merge_result["status"],
        "cache_hit": False,
        "source_pdf": str(pdf_path),
        "source_pdf_sha256": source_sha256,
        "run_dir": str(run_dir),
        "config": config.to_dict(),
        "page_count": merge_result["page_count"],
        "panel_count": merge_result["panel_count"],
        "candidate_run_count": ocr_result["candidate_run_count"],
        "needs_review_panels": merge_result["needs_review_panels"],
        "output_text_file": str(output_text_path),
        "output_text_sha256": sha256_file(output_text_path),
        "region_ocr_manifest": str(
            (step_dirs[6] / "region_ocr_manifest.json").resolve()
        ),
        "merge_manifest": str((step_dirs[7] / "merge_manifest.json").resolve()),
        "processing_seconds": round(time.perf_counter() - started, 3),
        "stages": stages,
    }
    atomic_write_json(pipeline_manifest_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chạy đầy đủ 7 bước OCR theo vùng cho một PDF bảo hiểm."
    )
    parser.add_argument("--input-pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tessdata-dir", type=Path, required=True)
    parser.add_argument("--tesseract-cmd", default=None)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--expected-panels", type=int, choices=[2, 3], default=None)
    parser.add_argument("--min-screening-score", type=float, default=0.45)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = InsurancePipelineConfig(
        dpi=args.dpi,
        expected_panels=args.expected_panels,
        min_screening_score=args.min_screening_score,
    )

    def print_progress(step: int, name: str) -> None:
        print(f"Bước {step}/7: {name}", flush=True)

    result = run_insurance_region_pipeline(
        args.input_pdf,
        args.output_dir,
        tessdata_dir=args.tessdata_dir,
        tesseract_cmd=args.tesseract_cmd,
        config=config,
        progress_callback=print_progress,
        force=args.force,
    )
    print(
        f"Hoàn thành {result['page_count']} trang, {result['panel_count']} vùng, "
        f"{result['candidate_run_count']} lượt OCR trong "
        f"{result['processing_seconds']:.1f} giây"
    )
    print(result["output_text_file"])
    return 0 if result["status"] == "success" else 3


if __name__ == "__main__":
    sys.exit(main())
