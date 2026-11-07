"""Benchmark có kiểm soát cho các profile Tesseract trên cùng một bộ PDF.

Mặc định chạy ma trận:
  fast/best x PSM 3/6/11 x grayscale/Sauvola.

CER/WER chỉ được tính khi có ground truth do người dùng xác nhận. Khi chưa có,
báo cáo chỉ dùng confidence và độ phủ tương đối để sàng lọc, không gọi đó là
độ chính xác.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from ocr_comparison.metrics import (
    compute_cer,
    compute_multiset_word_f1,
    compute_similarity,
    compute_wer,
)
from tesseract_ocr.config import TesseractConfig
from tesseract_ocr.preprocessing import preprocess_image
from tesseract_ocr.tesseract_engine import (
    get_project_tessdata_profiles,
    render_pdf_page_to_image,
    run_tesseract_on_image,
    verify_tesseract_environment,
)


MODEL_PROFILES = ("fast", "best")
PSM_VALUES = (3, 6, 11)
PREPROCESSING_PROFILES = ("grayscale", "sauvola")


def build_variant_id(model_profile: str, psm: int, preprocessing: str) -> str:
    return f"{model_profile}_psm{psm}_{preprocessing}"


def iter_variants() -> Iterable[Dict[str, Any]]:
    for model_profile in MODEL_PROFILES:
        for psm in PSM_VALUES:
            for preprocessing in PREPROCESSING_PROFILES:
                yield {
                    "model_profile": model_profile,
                    "psm": psm,
                    "preprocessing_profile": preprocessing,
                    "preprocessing": "grayscale",
                    "thresholding_method": 2 if preprocessing == "sauvola" else 0,
                    "variant_id": build_variant_id(model_profile, psm, preprocessing),
                }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_nonempty_lines(text: str) -> List[str]:
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def count_missed_lines(reference: str, hypothesis: str, threshold: float = 60.0) -> tuple[int, int]:
    """Đếm dòng tham chiếu không có dòng OCR tương đồng tối thiểu threshold%."""
    ref_lines = normalized_nonempty_lines(reference)
    hyp_lines = normalized_nonempty_lines(hypothesis)
    if not ref_lines:
        return 0, 0
    if not hyp_lines:
        return len(ref_lines), len(ref_lines)

    missed = 0
    for ref_line in ref_lines:
        best_similarity = max(compute_similarity(ref_line, hyp_line) for hyp_line in hyp_lines)
        if best_similarity < threshold:
            missed += 1
    return missed, len(ref_lines)


def ground_truth_path(root: Path, relative_pdf: Path, page_number: int) -> Path:
    base = root / relative_pdf.parent
    return base / f"{relative_pdf.stem}_page{page_number}.txt"


def text_output_path(root: Path, variant_id: str, relative_pdf: Path, page_number: int) -> Path:
    base = root / "texts" / variant_id / relative_pdf.parent
    return base / f"{relative_pdf.stem}_page{page_number}.txt"


def record_output_path(root: Path, variant_id: str, relative_pdf: Path, page_number: int) -> Path:
    base = root / "records" / variant_id / relative_pdf.parent
    return base / f"{relative_pdf.stem}_page{page_number}.json"


def load_ground_truth(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8-sig").strip()
    return text or None


def load_reference_text(reference_root: Optional[Path], relative_pdf: Path) -> Optional[str]:
    """Nạp OCR tham chiếu (ví dụ iOS); đây không phải ground truth."""
    if reference_root is None:
        return None
    expected_name = f"{relative_pdf.stem}_ocr.txt"
    candidates = [
        reference_root / relative_pdf.parent / expected_name,
        reference_root / expected_name,
    ]
    candidates.extend(reference_root.rglob(expected_name))
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        text = resolved.read_text(encoding="utf-8-sig", errors="replace")
        # Loại dòng đánh dấu trang của output iOS để so phần nội dung.
        content_lines = [
            line for line in text.splitlines() if not line.strip().startswith("--- [TRANG ")
        ]
        content = "\n".join(content_lines).strip()
        if content:
            return content
    return None


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_rows(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()

    page_keys = ["relative_pdf", "page_number"]
    detail = detail.copy()
    detail["max_words_on_page"] = detail.groupby(page_keys)["word_count"].transform("max")
    detail["max_chars_on_page"] = detail.groupby(page_keys)["char_count"].transform("max")
    detail["word_coverage"] = detail.apply(
        lambda row: row["word_count"] / row["max_words_on_page"]
        if row["max_words_on_page"]
        else 1.0,
        axis=1,
    )
    detail["char_coverage"] = detail.apply(
        lambda row: row["char_count"] / row["max_chars_on_page"]
        if row["max_chars_on_page"]
        else 1.0,
        axis=1,
    )
    detail["screening_score"] = (
        detail["confidence"] * 0.55
        + detail["word_coverage"] * 0.25
        + detail["char_coverage"] * 0.20
    )

    grouped = detail.groupby(
        ["variant_id", "model_profile", "psm", "preprocessing_profile"],
        as_index=False,
    )
    summary = grouped.agg(
        pages=("status", "size"),
        successful_pages=("status", lambda values: int((values == "success").sum())),
        mean_confidence=("confidence", "mean"),
        mean_word_coverage=("word_coverage", "mean"),
        mean_char_coverage=("char_coverage", "mean"),
        mean_screening_score=("screening_score", "mean"),
        total_words=("word_count", "sum"),
        total_lines=("line_count", "sum"),
        mean_seconds=("duration_seconds", "mean"),
        total_seconds=("duration_seconds", "sum"),
        labeled_pages=("ground_truth_available", "sum"),
        reference_pages=("reference_available", "sum"),
        mean_reference_similarity=("reference_similarity", "mean"),
        mean_cer=("cer", "mean"),
        mean_wer=("wer", "mean"),
        mean_word_f1=("word_f1", "mean"),
        missed_reference_lines=("missed_reference_lines", "sum"),
        reference_lines=("reference_lines", "sum"),
    )
    summary["reference_line_recall"] = summary.apply(
        lambda row: 1.0 - (row["missed_reference_lines"] / row["reference_lines"])
        if row["reference_lines"]
        else float("nan"),
        axis=1,
    )

    if int(summary["labeled_pages"].max()) > 0:
        summary = summary.sort_values(
            ["mean_cer", "mean_wer", "total_seconds"], ascending=[True, True, True]
        )
    elif int(summary["reference_pages"].max()) > 0:
        summary = summary.sort_values(
            ["mean_reference_similarity", "mean_confidence", "total_seconds"],
            ascending=[False, False, True],
        )
    else:
        summary = summary.sort_values(
            ["mean_screening_score", "total_seconds"], ascending=[False, True]
        )
    return summary.reset_index(drop=True), detail


def write_markdown_report(
    path: Path,
    summary: pd.DataFrame,
    selected_pdfs: List[Path],
    model_metadata: Dict[str, Dict[str, Any]],
) -> None:
    labeled_pages = int(summary["labeled_pages"].max()) if not summary.empty else 0
    reference_pages = int(summary["reference_pages"].max()) if not summary.empty else 0
    lines = [
        "# Báo cáo benchmark Tesseract",
        "",
        f"- PDF đã chọn: {len(selected_pdfs)}",
        f"- Cấu hình: {len(summary)}",
        f"- Trang có ground truth: {labeled_pages}",
        "- Ma trận: fast/best × PSM 3/6/11 × grayscale/Sauvola",
        "",
        "## Model",
        "",
        "| Profile | Kích thước vie | SHA256 vie |",
        "|---|---:|---|",
    ]
    for profile_name, metadata in model_metadata.items():
        lines.append(
            f"| {profile_name} | {metadata['vie_size']} | `{metadata['vie_sha256']}` |"
        )

    lines.extend(["", "## Kết quả tổng hợp", ""])
    if labeled_pages:
        lines.extend(
            [
                "CER/WER dưới đây được tính từ ground truth đã có.",
                "",
                "| Hạng | Cấu hình | CER | WER | F1 từ | Dòng tham chiếu thu hồi | Giây/trang |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for index, row in summary.iterrows():
            lines.append(
                f"| {index + 1} | {row['variant_id']} | {row['mean_cer']:.2%} | "
                f"{row['mean_wer']:.2%} | {row['mean_word_f1']:.2%} | "
                f"{row['reference_line_recall']:.2%} | {row['mean_seconds']:.2f} |"
            )
    elif reference_pages:
        lines.extend(
            [
                "**Chưa có ground truth được xác nhận nên chưa thể tính độ chính xác CER/WER.**",
                "Similarity dưới đây chỉ là mức giống với OCR iOS dùng để loại cấu hình nhiều nhiễu; iOS không phải đáp án chuẩn.",
                "",
                "| Hạng sàng lọc | Cấu hình | Giống iOS | Confidence | Độ phủ từ | Giây/trang |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for index, row in summary.iterrows():
            lines.append(
                f"| {index + 1} | {row['variant_id']} | {row['mean_reference_similarity']:.2f}% | "
                f"{row['mean_confidence']:.2%} | {row['mean_word_coverage']:.2%} | "
                f"{row['mean_seconds']:.2f} |"
            )
        lines.extend(
            [
                "",
                "## Bước để có CER/WER thật",
                "",
                "1. Mở các file trong `ground_truth_drafts/` và sửa lại đúng theo ảnh PDF.",
                "2. Lưu bản đã xác nhận vào đường dẫn tương ứng trong `ground_truth/`.",
                "3. Chạy lại cùng lệnh benchmark; các record OCR được dùng lại nên chỉ mất thời gian tính báo cáo.",
            ]
        )
    else:
        lines.extend(
            [
                "**Chưa có ground truth được xác nhận, vì vậy chưa thể gọi bất kỳ số liệu nào là độ chính xác.**",
                "Bảng này chỉ dùng confidence và độ phủ tương đối để chọn cấu hình cần kiểm tra thủ công trước.",
                "",
                "| Hạng sàng lọc | Cấu hình | Confidence | Độ phủ từ | Độ phủ ký tự | Giây/trang |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for index, row in summary.iterrows():
            lines.append(
                f"| {index + 1} | {row['variant_id']} | {row['mean_confidence']:.2%} | "
                f"{row['mean_word_coverage']:.2%} | {row['mean_char_coverage']:.2%} | "
                f"{row['mean_seconds']:.2f} |"
            )
        lines.extend(
            [
                "",
                "## Bước để có CER/WER thật",
                "",
                "1. Mở các file trong `ground_truth_drafts/` và sửa lại đúng theo ảnh PDF.",
                "2. Lưu bản đã xác nhận vào đường dẫn tương ứng trong `ground_truth/`.",
                "3. Chạy lại cùng lệnh benchmark; các record OCR được dùng lại nên chỉ mất thời gian tính báo cáo.",
            ]
        )

    lines.extend(["", "## PDF đã dùng", ""])
    lines.extend(f"- `{pdf_path.as_posix()}`" for pdf_path in selected_pdfs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark 12 cấu hình Tesseract trên cùng bộ PDF.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-documents", type=int, default=10)
    parser.add_argument("--max-pages-per-document", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--lang", type=str, default="vie")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Thư mục OCR tham chiếu (ví dụ iOS). Chỉ tính similarity, không tính accuracy.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    reference_dir = args.reference_dir.resolve() if args.reference_dir else None
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục PDF: {input_dir}")
    if input_dir == output_dir or input_dir in output_dir.parents:
        raise ValueError("output-dir phải nằm ngoài input-dir")

    output_dir.mkdir(parents=True, exist_ok=True)
    gt_root = output_dir / "ground_truth"
    draft_root = output_dir / "ground_truth_drafts"
    gt_root.mkdir(parents=True, exist_ok=True)
    draft_root.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.rglob("*.pdf"), key=lambda path: path.as_posix().casefold())
    selected = pdfs[: args.max_documents]
    if not selected:
        raise RuntimeError("Không tìm thấy PDF để benchmark")

    required_langs = [item.strip() for item in args.lang.split("+") if item.strip()]
    profiles = get_project_tessdata_profiles(required_langs)
    missing_profiles = [name for name in MODEL_PROFILES if name not in profiles]
    if missing_profiles:
        raise RuntimeError(
            f"Thiếu profile {missing_profiles}. Hãy chạy scripts/setup_tessdata_profiles.ps1 trước."
        )

    model_metadata: Dict[str, Dict[str, Any]] = {}
    for profile_name in MODEL_PROFILES:
        vie_model = profiles[profile_name] / "vie.traineddata"
        model_metadata[profile_name] = {
            "directory": str(profiles[profile_name]),
            "vie_size": vie_model.stat().st_size,
            "vie_sha256": sha256_file(vie_model),
        }

    # Kiểm tra môi trường một lần cho mỗi model trước khi bắt đầu lượt dài.
    for profile_name in MODEL_PROFILES:
        config = TesseractConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            dpi=args.dpi,
            lang=args.lang,
            oem=1,
            psm=3,
            preprocessing="grayscale",
            thresholding_method=0,
            tessdata_dir=profiles[profile_name],
        )
        verify_tesseract_environment(config)

    variants = list(iter_variants())
    total_planned_pages = 0
    for pdf_path in selected:
        with fitz.open(pdf_path) as document:
            total_planned_pages += min(len(document), args.max_pages_per_document)
    total_runs = total_planned_pages * len(variants)
    completed_runs = 0
    rows: List[Dict[str, Any]] = []
    start_all = time.time()

    for document_index, pdf_path in enumerate(selected, 1):
        relative_pdf = pdf_path.relative_to(input_dir)
        with fitz.open(pdf_path) as document:
            page_limit = min(len(document), args.max_pages_per_document)
            for page_index in range(page_limit):
                page_number = page_index + 1
                raw_image = render_pdf_page_to_image(document[page_index], dpi=args.dpi)
                grayscale_image, prep_meta = preprocess_image(raw_image, mode="grayscale")
                gt_path = ground_truth_path(gt_root, relative_pdf, page_number)
                gt_path.parent.mkdir(parents=True, exist_ok=True)
                if not gt_path.exists():
                    gt_path.touch()
                ground_truth = load_ground_truth(gt_path)
                reference_text = load_reference_text(reference_dir, relative_pdf)

                for variant in variants:
                    completed_runs += 1
                    record_path = record_output_path(
                        output_dir, variant["variant_id"], relative_pdf, page_number
                    )
                    text_path = text_output_path(
                        output_dir, variant["variant_id"], relative_pdf, page_number
                    )

                    if record_path.is_file() and not args.force:
                        row = load_json(record_path)
                        ocr_text = text_path.read_text(encoding="utf-8") if text_path.is_file() else ""
                        run_label = "resume"
                    else:
                        config = TesseractConfig(
                            input_dir=input_dir,
                            output_dir=output_dir,
                            dpi=args.dpi,
                            lang=args.lang,
                            oem=1,
                            psm=variant["psm"],
                            preprocessing=variant["preprocessing"],
                            thresholding_method=variant["thresholding_method"],
                            tessdata_dir=profiles[variant["model_profile"]],
                        )
                        result = run_tesseract_on_image(
                            grayscale_image,
                            config,
                            page_number=page_number,
                            prep_meta=prep_meta.to_dict(),
                        )
                        ocr_text = result.text
                        text_path.parent.mkdir(parents=True, exist_ok=True)
                        text_path.write_text(ocr_text, encoding="utf-8")
                        row = {
                            "relative_pdf": relative_pdf.as_posix(),
                            "page_number": page_number,
                            "variant_id": variant["variant_id"],
                            "model_profile": variant["model_profile"],
                            "psm": variant["psm"],
                            "preprocessing_profile": variant["preprocessing_profile"],
                            "thresholding_method": variant["thresholding_method"],
                            "status": result.status,
                            "confidence": result.confidence,
                            "word_count": len(result.boxes),
                            "line_count": len(normalized_nonempty_lines(result.text)),
                            "char_count": len(result.text),
                            "duration_seconds": result.duration_seconds + prep_meta.processing_seconds,
                            "error": result.error,
                        }
                        save_json(record_path, row)
                        run_label = "ocr"

                    row = dict(row)
                    row["ground_truth_available"] = bool(ground_truth)
                    row["reference_available"] = bool(reference_text)
                    row["reference_similarity"] = (
                        compute_similarity(reference_text, ocr_text)
                        if reference_text
                        else float("nan")
                    )
                    row["cer"] = float("nan")
                    row["wer"] = float("nan")
                    row["word_f1"] = float("nan")
                    row["missed_reference_lines"] = 0
                    row["reference_lines"] = 0
                    if ground_truth:
                        row["cer"] = compute_cer(ground_truth, ocr_text)
                        row["wer"] = compute_wer(ground_truth, ocr_text)
                        _, _, row["word_f1"] = compute_multiset_word_f1(ground_truth, ocr_text)
                        missed, reference_lines = count_missed_lines(ground_truth, ocr_text)
                        row["missed_reference_lines"] = missed
                        row["reference_lines"] = reference_lines
                    rows.append(row)

                    elapsed = time.time() - start_all
                    print(
                        f"[{completed_runs}/{total_runs}] {run_label} | PDF {document_index}/{len(selected)} "
                        f"| trang {page_number} | {variant['variant_id']} | "
                        f"conf={row['confidence']:.3f} | {row['duration_seconds']:.2f}s | "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )

    detail_frame = pd.DataFrame(rows)
    summary_frame, enriched_detail = summarize_rows(detail_frame)
    enriched_detail.to_csv(output_dir / "benchmark_detail.csv", index=False, encoding="utf-8-sig")
    summary_frame.to_csv(output_dir / "benchmark_summary.csv", index=False, encoding="utf-8-sig")

    # Tạo bản nháp từ cấu hình sàng lọc đứng đầu để việc xác nhận ground truth nhanh hơn.
    best_variant_id = str(summary_frame.iloc[0]["variant_id"])
    for pdf_path in selected:
        relative_pdf = pdf_path.relative_to(input_dir)
        with fitz.open(pdf_path) as document:
            page_limit = min(len(document), args.max_pages_per_document)
            for page_index in range(page_limit):
                page_number = page_index + 1
                source = text_output_path(output_dir, best_variant_id, relative_pdf, page_number)
                draft = ground_truth_path(draft_root, relative_pdf, page_number)
                if source.is_file():
                    draft.parent.mkdir(parents=True, exist_ok=True)
                    draft.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "dpi": args.dpi,
        "language": args.lang,
        "reference_dir": str(reference_dir) if reference_dir else None,
        "selected_pdfs": [str(path.relative_to(input_dir)).replace("\\", "/") for path in selected],
        "models": model_metadata,
        "variants": variants,
        "top_screening_variant": best_variant_id,
        "elapsed_seconds": round(time.time() - start_all, 3),
    }
    save_json(output_dir / "benchmark_manifest.json", manifest)
    write_markdown_report(
        output_dir / "BENCHMARK_REPORT.md",
        summary_frame,
        [path.relative_to(input_dir) for path in selected],
        model_metadata,
    )

    print(f"Hoàn thành. Báo cáo: {output_dir / 'BENCHMARK_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
