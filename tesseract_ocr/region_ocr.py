"""OCR từng panel với nhiều PSM và chọn ứng viên phù hợp.

Đây chỉ là bước 6 của pipeline OCR theo vùng. Module chạy Tesseract
``tessdata_best`` trên cả ảnh CLAHE và Sauvola với các PSM 3/6/11, lưu mọi
ứng viên rồi chọn một kết quả theo điểm sàng lọc.

Điểm sàng lọc không phải độ chính xác. CER/WER chỉ có ý nghĩa khi có ground
truth được xác nhận. Module không ghép TXT toàn PDF; việc đó thuộc bước 7.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from tesseract_ocr.config import TesseractConfig
from tesseract_ocr.document_renderer import atomic_write_json, save_png, sha256_file
from tesseract_ocr.schemas import PageResult
from tesseract_ocr.tesseract_engine import (
    find_tesseract_executable,
    run_tesseract_on_image,
    verify_tesseract_environment,
)


if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


DEFAULT_PSMS = (3, 6, 11)
INPUT_VARIANTS = ("enhanced", "binary")
SAFE_PUNCTUATION = set(".,:;!?%()+-/='\"&@#_")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp_{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_psm_values(values: Iterable[int]) -> Tuple[int, ...]:
    parsed = tuple(dict.fromkeys(int(value) for value in values))
    if not parsed:
        raise ValueError("Phải có ít nhất một PSM")
    invalid = [value for value in parsed if value not in range(0, 14)]
    if invalid:
        raise ValueError(f"PSM không hợp lệ: {invalid}")
    return parsed


def normalized_lines(text: str) -> List[str]:
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def text_quality_ratio(text: str) -> float:
    characters = [character for character in text if not character.isspace()]
    if not characters:
        return 0.0
    valid = 0
    for character in characters:
        if character.isalnum() or character in SAFE_PUNCTUATION:
            valid += 1
            continue
        category = unicodedata.category(character)
        if category.startswith("L") or category.startswith("N"):
            valid += 1
    return valid / len(characters)


def candidate_metrics(result: PageResult) -> Dict[str, Any]:
    text = unicodedata.normalize("NFC", result.text or "")
    words = re.findall(r"\S+", text, flags=re.UNICODE)
    nonspace_characters = sum(not character.isspace() for character in text)
    diacritic_characters = sum(
        bool(unicodedata.normalize("NFD", character)[1:])
        for character in text
        if character.isalpha()
    )
    line_count = len(normalized_lines(text))
    words_per_line = len(words) / max(1, line_count)
    return {
        "status": result.status,
        "confidence": round(float(result.confidence), 6),
        "word_count": len(words),
        "box_count": len(result.boxes),
        "line_count": line_count,
        "words_per_line": round(words_per_line, 6),
        "char_count": nonspace_characters,
        "quality_ratio": round(text_quality_ratio(text), 6),
        "vietnamese_diacritic_characters": int(diacritic_characters),
        "duration_seconds": round(float(result.duration_seconds), 3),
        "error": result.error,
    }


def score_candidates(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tính điểm sàng lọc tương đối, không đại diện cho CER/WER."""
    if not candidates:
        return []
    max_words = max(int(item["metrics"]["word_count"]) for item in candidates)
    max_characters = max(int(item["metrics"]["char_count"]) for item in candidates)
    scored: List[Dict[str, Any]] = []
    for candidate in candidates:
        record = dict(candidate)
        metrics = dict(candidate["metrics"])
        word_coverage = (
            metrics["word_count"] / max_words if max_words > 0 else 0.0
        )
        character_coverage = (
            metrics["char_count"] / max_characters if max_characters > 0 else 0.0
        )
        quality = float(metrics["quality_ratio"])
        confidence = float(metrics["confidence"])
        # PSM 11 thường tạo nhiều dòng ngắn và có thể tăng độ phủ bằng nhiễu.
        # Điểm mạch dòng giảm ưu tiên cho kết quả bị phân mảnh, nhưng vẫn cho
        # phép PSM 11 thắng nếu confidence và nội dung thực sự tốt hơn.
        line_coherence = min(1.0, float(metrics["words_per_line"]) / 6.0)
        score = (
            (0.50 * confidence)
            + (0.12 * word_coverage)
            + (0.08 * character_coverage)
            + (0.15 * quality)
            + (0.15 * line_coherence)
        )
        if metrics["status"] != "success" or metrics["char_count"] == 0:
            score = 0.0
        metrics["word_coverage"] = round(word_coverage, 6)
        metrics["character_coverage"] = round(character_coverage, 6)
        metrics["line_coherence"] = round(line_coherence, 6)
        metrics["screening_score"] = round(score, 6)
        record["metrics"] = metrics
        scored.append(record)
    return sorted(
        scored,
        key=lambda item: (
            item["metrics"]["screening_score"],
            item["metrics"]["confidence"],
            -int(item["psm"] == 11),
        ),
        reverse=True,
    )


def candidate_id(input_variant: str, psm: int) -> str:
    return f"{input_variant}_psm{psm}"


def load_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(
        np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE
    )
    if image is None:
        raise RuntimeError(f"Không thể đọc ảnh OCR: {path}")
    return image


def write_candidate_outputs(
    panel_dir: Path,
    variant_id: str,
    result: PageResult,
) -> Tuple[Path, Path]:
    text_path = panel_dir / "candidates" / f"{variant_id}.txt"
    json_path = panel_dir / "candidates" / f"{variant_id}.json"
    atomic_write_text(text_path, unicodedata.normalize("NFC", result.text).strip() + "\n")
    atomic_write_json(json_path, result.to_dict())
    return text_path, json_path


def make_ocr_overlay(image: np.ndarray, result: PageResult) -> np.ndarray:
    """Vẽ hộp từ theo confidence để kiểm tra vùng Tesseract đã đọc."""
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for box in result.boxes:
        if box.confidence >= 0.70:
            color = (0, 170, 0)
        elif box.confidence >= 0.40:
            color = (0, 165, 255)
        else:
            color = (0, 0, 220)
        cv2.rectangle(
            overlay,
            (int(box.x), int(box.y)),
            (int(box.x + box.w), int(box.y + box.h)),
            color,
            2,
        )
    return overlay


def relative_path(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def ocr_preprocessed_panels(
    preprocessed_dir: Path,
    output_dir: Path,
    tessdata_dir: Path,
    psm_values: Sequence[int] = DEFAULT_PSMS,
    language: str = "vie+eng",
    tesseract_cmd: Optional[str] = None,
    dpi: int = 400,
    min_screening_score: float = 0.45,
) -> Dict[str, Any]:
    """OCR tất cả panel và chọn một ứng viên cho mỗi panel."""
    preprocessed_dir = Path(preprocessed_dir).resolve()
    output_dir = Path(output_dir).resolve()
    tessdata_dir = Path(tessdata_dir).resolve()
    manifest_path = preprocessed_dir / "background_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy background manifest: {manifest_path}")
    if output_dir == preprocessed_dir or output_dir in preprocessed_dir.parents:
        raise ValueError("Thư mục OCR vùng phải tách khỏi thư mục tiền xử lý")
    if not tessdata_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy tessdata: {tessdata_dir}")
    required_languages = [item for item in language.split("+") if item]
    missing_languages = [
        item
        for item in required_languages
        if not (tessdata_dir / f"{item}.traineddata").is_file()
    ]
    if missing_languages:
        raise FileNotFoundError(
            f"Tessdata thiếu ngôn ngữ: {missing_languages} tại {tessdata_dir}"
        )
    psms = parse_psm_values(psm_values)
    if not 0.0 <= min_screening_score <= 1.0:
        raise ValueError("min_screening_score phải nằm trong khoảng 0..1")

    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    page_specs = source_manifest.get("pages", [])
    if not page_specs:
        raise RuntimeError("Background manifest không có trang")

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_command = find_tesseract_executable(tesseract_cmd)
    if not resolved_command:
        raise RuntimeError("Không tìm thấy tesseract.exe")
    environment_config = TesseractConfig(
        input_dir=preprocessed_dir,
        output_dir=output_dir,
        dpi=dpi,
        lang=language,
        oem=1,
        psm=psms[0],
        preprocessing="none",
        thresholding_method=0,
        tesseract_cmd=resolved_command,
        tessdata_dir=tessdata_dir,
    )
    tesseract_version = verify_tesseract_environment(environment_config)

    started = time.perf_counter()
    page_results: List[Dict[str, Any]] = []
    successful_panels = 0
    needs_review_panels = 0
    selected_counts = {
        candidate_id(input_variant, psm): 0
        for input_variant in INPUT_VARIANTS
        for psm in psms
    }

    for page_spec in page_specs:
        page_number = int(page_spec["page_number"])
        panel_results: List[Dict[str, Any]] = []
        for panel_spec in page_spec.get("panels", []):
            panel_index = int(panel_spec["panel_index"])
            panel_dir = output_dir / f"page_{page_number:04d}" / f"panel_{panel_index:02d}"
            images = {
                "enhanced": load_image(
                    preprocessed_dir / str(panel_spec["enhanced_file"])
                ),
                "binary": load_image(
                    preprocessed_dir / str(panel_spec["binary_file"])
                ),
            }
            candidates: List[Dict[str, Any]] = []
            candidate_results: Dict[str, PageResult] = {}
            for input_variant, image in images.items():
                for psm in psms:
                    variant_id = candidate_id(input_variant, psm)
                    config = TesseractConfig(
                        input_dir=preprocessed_dir,
                        output_dir=output_dir,
                        dpi=dpi,
                        lang=language,
                        oem=1,
                        psm=psm,
                        preprocessing="none",
                        thresholding_method=0,
                        tesseract_cmd=resolved_command,
                        tessdata_dir=tessdata_dir,
                    )
                    result = run_tesseract_on_image(
                        image,
                        config,
                        page_number=page_number,
                        prep_meta={
                            "input_variant": input_variant,
                            "region_panel_index": panel_index,
                            "source_file": str(
                                panel_spec[f"{input_variant}_file"]
                            ),
                        },
                    )
                    text_path, json_path = write_candidate_outputs(
                        panel_dir, variant_id, result
                    )
                    candidates.append(
                        {
                            "candidate_id": variant_id,
                            "input_variant": input_variant,
                            "psm": psm,
                            "text_file": relative_path(text_path, output_dir),
                            "json_file": relative_path(json_path, output_dir),
                            "metrics": candidate_metrics(result),
                        }
                    )
                    candidate_results[variant_id] = result

            ranked = score_candidates(candidates)
            selected = ranked[0]
            selected_text_source = output_dir / selected["text_file"]
            selected_text = selected_text_source.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
            selected_path = panel_dir / "selected.txt"
            atomic_write_text(selected_path, selected_text + "\n")
            selected_result = candidate_results[selected["candidate_id"]]
            selected_overlay = make_ocr_overlay(
                images[selected["input_variant"]], selected_result
            )
            selected_overlay_path = panel_dir / "selected_overlay.png"
            save_png(selected_overlay_path, selected_overlay)
            selected_metadata_path = panel_dir / "selected.json"
            selected_metadata = {
                "page_number": page_number,
                "panel_index": panel_index,
                "selected_candidate_id": selected["candidate_id"],
                "selection_basis": "screening_score_without_ground_truth",
                "selected_metrics": selected["metrics"],
                "ranked_candidates": ranked,
            }
            atomic_write_json(selected_metadata_path, selected_metadata)

            score = float(selected["metrics"]["screening_score"])
            status = (
                "success"
                if selected_text and score >= min_screening_score
                else "needs_review"
            )
            review_reason = None
            if not selected_text:
                review_reason = "Tất cả ứng viên OCR đều rỗng"
            elif score < min_screening_score:
                review_reason = f"Điểm sàng lọc thấp: {score:.3f}"
            if status == "success":
                successful_panels += 1
            else:
                needs_review_panels += 1
            selected_counts[selected["candidate_id"]] += 1
            panel_results.append(
                {
                    "panel_index": panel_index,
                    "status": status,
                    "review_reason": review_reason,
                    "selected_candidate_id": selected["candidate_id"],
                    "selected_input_variant": selected["input_variant"],
                    "selected_psm": selected["psm"],
                    "selected_text_file": relative_path(selected_path, output_dir),
                    "selected_overlay_file": relative_path(
                        selected_overlay_path, output_dir
                    ),
                    "selected_json_file": relative_path(
                        selected_metadata_path, output_dir
                    ),
                    "selected_metrics": selected["metrics"],
                    "candidate_count": len(ranked),
                    "candidates": ranked,
                }
            )

        page_results.append(
            {
                "page_number": page_number,
                "status": (
                    "success"
                    if panel_results
                    and all(panel["status"] == "success" for panel in panel_results)
                    else "needs_review"
                ),
                "panel_count": len(panel_results),
                "panels": panel_results,
            }
        )

    language_models = {
        item: {
            "file": str(tessdata_dir / f"{item}.traineddata"),
            "bytes": (tessdata_dir / f"{item}.traineddata").stat().st_size,
            "sha256": sha256_file(tessdata_dir / f"{item}.traineddata"),
        }
        for item in required_languages
    }
    result: Dict[str, Any] = {
        "schema_version": "1.0",
        "operation": "ocr_panels_with_psm_selection",
        "selection_warning": (
            "screening_score is not accuracy; CER/WER require verified ground truth"
        ),
        "source_background_manifest": str(manifest_path),
        "source_background_manifest_sha256": sha256_file(manifest_path),
        "engine": "tesseract",
        "engine_version": tesseract_version,
        "tesseract_command": str(resolved_command),
        "tessdata_profile": "best",
        "tessdata_dir": str(tessdata_dir),
        "language": language,
        "language_models": language_models,
        "oem": 1,
        "psm_candidates": list(psms),
        "input_variants": list(INPUT_VARIANTS),
        "candidates_per_panel": len(psms) * len(INPUT_VARIANTS),
        "minimum_screening_score": min_screening_score,
        "page_count": len(page_results),
        "panel_count": successful_panels + needs_review_panels,
        "candidate_run_count": (
            (successful_panels + needs_review_panels)
            * len(psms)
            * len(INPUT_VARIANTS)
        ),
        "successful_panels": successful_panels,
        "needs_review_panels": needs_review_panels,
        "selected_candidate_counts": selected_counts,
        "processing_seconds": round(time.perf_counter() - started, 3),
        "pages": page_results,
    }
    atomic_write_json(output_dir / "region_ocr_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OCR từng panel với CLAHE/Sauvola và nhiều PSM; chưa ghép PDF."
    )
    parser.add_argument("--preprocessed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tessdata-dir", type=Path, required=True)
    parser.add_argument("--tesseract-cmd", default=None)
    parser.add_argument("--lang", default="vie+eng")
    parser.add_argument("--psm", type=int, nargs="+", default=list(DEFAULT_PSMS))
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--min-screening-score", type=float, default=0.45)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = ocr_preprocessed_panels(
        args.preprocessed_dir,
        args.output_dir,
        tessdata_dir=args.tessdata_dir,
        psm_values=args.psm,
        language=args.lang,
        tesseract_cmd=args.tesseract_cmd,
        dpi=args.dpi,
        min_screening_score=args.min_screening_score,
    )
    print(
        f"Đã OCR {result['panel_count']} panel với "
        f"{result['candidate_run_count']} lượt: "
        f"{result['successful_panels']} thành công, "
        f"{result['needs_review_panels']} cần kiểm tra"
    )
    for page in result["pages"]:
        details = ", ".join(
            f"panel {panel['panel_index']}={panel['selected_candidate_id']} "
            f"(score {panel['selected_metrics']['screening_score']:.3f}, "
            f"conf {panel['selected_metrics']['confidence']:.3f})"
            for panel in page["panels"]
        )
        print(f"Trang {page['page_number']}: {details}")
    print(str(Path(args.output_dir).resolve() / "region_ocr_manifest.json"))
    return 0 if result["needs_review_panels"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
