"""
CLI thực thi so sánh hai thư mục kết quả Apple Vision và Tesseract OCR.
Chỉ đọc dữ liệu, không chạy lại bất kỳ OCR engine nào, tạo báo cáo đầy đủ.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ocr_comparison.config import ComparisonConfig
from ocr_comparison.loaders import load_ground_truth, load_ocr_directory
from ocr_comparison.matching import match_documents
from ocr_comparison.metrics import PageComparisonMetric, evaluate_page_pair
from ocr_comparison.report import ComparisonReporter


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
        prog="python -m ocr_comparison.compare_outputs",
        description="So sánh kết quả OCR giữa Apple Vision và Tesseract 5.",
    )
    parser.add_argument(
        "--apple-dir",
        type=Path,
        required=True,
        help="Đường dẫn thư mục chứa kết quả Apple Vision (chỉ đọc).",
    )
    parser.add_argument(
        "--tesseract-dir",
        type=Path,
        required=True,
        help="Đường dẫn thư mục chứa kết quả Tesseract OCR (chỉ đọc).",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Đường dẫn file CSV Ground Truth (tùy chọn).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("comparison_results"),
        help="Đường dẫn thư mục lưu báo cáo so sánh (mặc định: comparison_results).",
    )
    parser.add_argument(
        "--cer-threshold",
        type=float,
        default=0.05,
        help="Ngưỡng chênh lệch CER (mặc định 0.05 tức 5%) để phân xử hòa.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Mã định danh riêng cho lần chạy so sánh.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Bật hiển thị log chi tiết.",
    )
    return parser


def run_comparison(config: ComparisonConfig) -> Path:
    """Hàm lõi thực hiện toàn bộ quy trình so sánh và xuất báo cáo."""
    config.validate()

    logging.info(f"Nạp kết quả Apple Vision từ: {config.apple_dir}")
    apple_docs = load_ocr_directory(config.apple_dir, engine_name="apple_vision")
    logging.info(f"Đã nạp {len(apple_docs)} tài liệu Apple Vision.")

    logging.info(f"Nạp kết quả Tesseract OCR từ: {config.tesseract_dir}")
    tess_docs = load_ocr_directory(config.tesseract_dir, engine_name="tesseract")
    logging.info(f"Đã nạp {len(tess_docs)} tài liệu Tesseract OCR.")

    gt_data = {}
    if config.ground_truth_path:
        logging.info(f"Nạp Ground Truth từ: {config.ground_truth_path}")
        gt_data = load_ground_truth(config.ground_truth_path)
        logging.info(f"Đã nạp {len(gt_data)} mục Ground Truth theo trang.")

    logging.info("Tiến hành khớp nối tài liệu giữa hai thư mục...")
    matching_result = match_documents(apple_docs, tess_docs)
    logging.info(
        f"Khớp nối thành công {matching_result.total_matched} tài liệu. "
        f"(Apple chưa khớp: {len(matching_result.unmatched_apple)}, "
        f"Tesseract chưa khớp: {len(matching_result.unmatched_tesseract)})"
    )

    logging.info("Tính toán các chỉ số so sánh (CER, WER, F1, Pairwise Similarity)...")
    page_metrics: list[PageComparisonMetric] = []

    for pair in matching_result.matched_pairs:
        for p_pair in pair.page_pairs:
            p_num = p_pair.page_number
            a_text = p_pair.apple_page.text if p_pair.apple_page else ""
            t_text = p_pair.tesseract_page.text if p_pair.tesseract_page else ""
            a_status = p_pair.apple_status
            t_status = p_pair.tesseract_status

            # Tìm Ground Truth tương ứng nếu có
            gt_item = gt_data.get((pair.normalized_key, p_num))
            gt_text = gt_item.ground_truth_text if gt_item else None
            quality_tags = gt_item.quality_tags if gt_item else []
            doc_id = gt_item.document_id if gt_item else pair.stem_name

            m = evaluate_page_pair(
                doc_id=doc_id,
                relative_path=pair.normalized_key,
                category=pair.category,
                page_number=p_num,
                apple_text=a_text,
                tesseract_text=t_text,
                apple_status=a_status,
                tesseract_status=t_status,
                ground_truth_text=gt_text,
                quality_tags=quality_tags,
                cer_threshold=config.cer_threshold,
            )
            page_metrics.append(m)

    logging.info(f"Xuất các tệp báo cáo vào: {config.report_dir}")
    reporter = ComparisonReporter(
        config=config,
        matching_result=matching_result,
        page_metrics=page_metrics,
        run_id=config.run_id,
    )
    report_path = reporter.generate_all()

    # Hiển thị tóm tắt ra console
    print("\n" + "=" * 65)
    print(" KẾT QUẢ SO SÁNH OCR (APPLE VISION vs TESSERACT 5)")
    print("=" * 65)
    print(f" Thư mục báo cáo: {report_path}")
    print(f" Tài liệu khớp nối: {matching_result.total_matched}")
    print(f" Tổng số trang so sánh: {len(page_metrics)}")

    sim_values = [p.pairwise_similarity_pct for p in page_metrics]
    if sim_values:
        avg_sim = sum(sim_values) / len(sim_values)
        print(f" Độ tương đồng trung bình (Pairwise Similarity): {avg_sim:.2f}%")

    gt_pages = [p for p in page_metrics if p.ground_truth_available]
    if gt_pages:
        avg_a_cer = sum(p.apple_cer_pct for p in gt_pages) / len(gt_pages)
        avg_t_cer = sum(p.tesseract_cer_pct for p in gt_pages) / len(gt_pages)
        a_wins = sum(1 for p in gt_pages if p.winner == "apple")
        t_wins = sum(1 for p in gt_pages if p.winner == "tesseract")
        ties = sum(1 for p in gt_pages if p.winner == "tie")
        poor = sum(1 for p in gt_pages if p.winner == "both_poor")
        print("\n ĐÁNH GIÁ THEO GROUND TRUTH:")
        print(f" - Số trang có Ground Truth: {len(gt_pages)}")
        print(f" - Apple Vision CER trung bình: {avg_a_cer:.2f}%")
        print(f" - Tesseract CER trung bình:   {avg_t_cer:.2f}%")
        print(f" - Phân bổ kết quả: Apple thắng: {a_wins} | Tesseract thắng: {t_wins} | Hòa: {ties} | Cùng kém: {poor}")
    else:
        print("\n LƯU Ý: Không có Ground Truth. Báo cáo đo mức độ chênh lệch giữa hai engine (Pairwise).")
    print("=" * 65 + "\n")

    return report_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        config = ComparisonConfig(
            apple_dir=args.apple_dir,
            tesseract_dir=args.tesseract_dir,
            ground_truth_path=args.ground_truth,
            report_dir=args.report_dir,
            cer_threshold=args.cer_threshold,
            run_id=args.run_id,
        )
        run_comparison(config)
        return 0
    except Exception as e:
        logging.error(f"Lỗi so sánh: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
