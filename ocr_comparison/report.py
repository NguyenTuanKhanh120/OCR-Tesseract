"""
Bộ tạo báo cáo kết quả so sánh OCR (CSV, JSON và HTML report).
Xuất toàn bộ vào comparison_results/<run_id>/ theo mục 13 của tài liệu thiết kế.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ocr_comparison.config import ComparisonConfig
from ocr_comparison.matching import MatchingResult
from ocr_comparison.metrics import PageComparisonMetric, calculate_distribution_stats


class ComparisonReporter:
    """Ghi nhận và xuất toàn bộ 11 artifacts báo cáo so sánh."""

    def __init__(
        self,
        config: ComparisonConfig,
        matching_result: MatchingResult,
        page_metrics: List[PageComparisonMetric],
        run_id: Optional[str] = None,
    ):
        self.config = config
        self.matching = matching_result
        self.page_metrics = page_metrics
        self.run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = self.config.report_dir / self.run_id

    def generate_all(self) -> Path:
        """Tạo toàn bộ các file báo cáo và trả về thư mục run_dir."""
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._export_config_json()
        self._export_matched_documents_csv()
        self._export_unmatched_csvs()
        self._export_page_pairwise_results_csv()
        self._export_page_ground_truth_results_csv()
        self._export_document_results_csv()
        self._export_category_results_csv()
        self._export_quality_tag_results_csv()
        self._export_failures_csv()
        summary_data = self._export_summary_json()
        self._export_report_html(summary_data)

        return self.run_dir

    def _export_config_json(self) -> None:
        cfg_path = self.run_dir / "config.json"
        data = {
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(),
            "apple_dir": str(self.config.apple_dir),
            "tesseract_dir": str(self.config.tesseract_dir),
            "ground_truth_path": str(self.config.ground_truth_path) if self.config.ground_truth_path else None,
            "report_dir": str(self.config.report_dir),
            "cer_threshold": self.config.cer_threshold,
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _export_matched_documents_csv(self) -> None:
        csv_path = self.run_dir / "matched_documents.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["normalized_key", "category", "stem_name", "total_pages", "apple_path", "tesseract_path"])
            for pair in self.matching.matched_pairs:
                a_p = pair.apple_doc.json_path or pair.apple_doc.txt_path
                t_p = pair.tesseract_doc.json_path or pair.tesseract_doc.txt_path
                writer.writerow([
                    pair.normalized_key,
                    pair.category,
                    pair.stem_name,
                    pair.total_pages,
                    str(a_p),
                    str(t_p),
                ])

    def _export_unmatched_csvs(self) -> None:
        # Apple unmatched
        apple_path = self.run_dir / "unmatched_apple_documents.csv"
        with open(apple_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["relative_path", "category", "stem_name", "total_pages"])
            for doc in self.matching.unmatched_apple:
                writer.writerow([doc.relative_path, doc.category, doc.stem_name, doc.total_pages])

        # Tesseract unmatched
        tess_path = self.run_dir / "unmatched_tesseract_documents.csv"
        with open(tess_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["relative_path", "category", "stem_name", "total_pages"])
            for doc in self.matching.unmatched_tesseract:
                writer.writerow([doc.relative_path, doc.category, doc.stem_name, doc.total_pages])

    def _export_page_pairwise_results_csv(self) -> None:
        csv_path = self.run_dir / "page_pairwise_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "document_id",
                "relative_path",
                "category",
                "page_number",
                "apple_status",
                "tesseract_status",
                "apple_chars",
                "tesseract_chars",
                "apple_words",
                "tesseract_words",
                "pairwise_char_distance",
                "pairwise_word_distance",
                "pairwise_similarity_pct",
                "char_count_diff",
                "word_count_diff",
                "ground_truth_available",
                "apple_cer_pct",
                "tesseract_cer_pct",
                "apple_wer_pct",
                "tesseract_wer_pct",
                "winner",
            ])
            for m in self.page_metrics:
                writer.writerow([
                    m.document_id,
                    m.relative_path,
                    m.category,
                    m.page_number,
                    m.apple_status,
                    m.tesseract_status,
                    m.apple_chars,
                    m.tesseract_chars,
                    m.apple_words,
                    m.tesseract_words,
                    m.pairwise_char_distance,
                    m.pairwise_word_distance,
                    m.pairwise_similarity_pct,
                    m.char_count_diff,
                    m.word_count_diff,
                    m.ground_truth_available,
                    m.apple_cer_pct if m.ground_truth_available else "",
                    m.tesseract_cer_pct if m.ground_truth_available else "",
                    m.apple_wer_pct if m.ground_truth_available else "",
                    m.tesseract_wer_pct if m.ground_truth_available else "",
                    m.winner if m.ground_truth_available else "",
                ])

    def _export_page_ground_truth_results_csv(self) -> None:
        csv_path = self.run_dir / "page_ground_truth_results.csv"
        gt_metrics = [m for m in self.page_metrics if m.ground_truth_available]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "document_id",
                "relative_path",
                "category",
                "page_number",
                "ground_truth_chars",
                "apple_cer_pct",
                "tesseract_cer_pct",
                "apple_wer_pct",
                "tesseract_wer_pct",
                "apple_f1_pct",
                "tesseract_f1_pct",
                "winner",
                "quality_tags",
                "digit_accuracy_apple",
                "digit_accuracy_tesseract",
                "date_accuracy_apple",
                "date_accuracy_tesseract",
                "diacritic_error_apple",
                "diacritic_error_tesseract",
            ])
            for m in gt_metrics:
                writer.writerow([
                    m.document_id,
                    m.relative_path,
                    m.category,
                    m.page_number,
                    m.ground_truth_chars,
                    m.apple_cer_pct,
                    m.tesseract_cer_pct,
                    m.apple_wer_pct,
                    m.tesseract_wer_pct,
                    m.apple_f1_pct,
                    m.tesseract_f1_pct,
                    m.winner,
                    ";".join(m.quality_tags),
                    round(m.digit_accuracy_apple, 4) if m.digit_accuracy_apple is not None else "",
                    round(m.digit_accuracy_tesseract, 4) if m.digit_accuracy_tesseract is not None else "",
                    round(m.date_accuracy_apple, 4) if m.date_accuracy_apple is not None else "",
                    round(m.date_accuracy_tesseract, 4) if m.date_accuracy_tesseract is not None else "",
                    round(m.diacritic_error_apple, 4) if m.diacritic_error_apple is not None else "",
                    round(m.diacritic_error_tesseract, 4) if m.diacritic_error_tesseract is not None else "",
                ])

    def _export_document_results_csv(self) -> None:
        csv_path = self.run_dir / "document_results.csv"
        doc_groups = defaultdict(list)
        for m in self.page_metrics:
            doc_groups[m.relative_path].append(m)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "relative_path",
                "category",
                "total_pages",
                "avg_similarity_pct",
                "gt_pages_count",
                "avg_apple_cer_pct",
                "avg_tesseract_cer_pct",
                "avg_apple_wer_pct",
                "avg_tesseract_wer_pct",
            ])
            for rel_path, pages in doc_groups.items():
                cat = pages[0].category
                avg_sim = sum(p.pairwise_similarity_pct for p in pages) / len(pages)
                gt_pages = [p for p in pages if p.ground_truth_available and p.apple_cer_pct is not None]
                if gt_pages:
                    a_cer = sum(p.apple_cer_pct for p in gt_pages) / len(gt_pages)
                    t_cer = sum(p.tesseract_cer_pct for p in gt_pages) / len(gt_pages)
                    a_wer = sum(p.apple_wer_pct for p in gt_pages) / len(gt_pages)
                    t_wer = sum(p.tesseract_wer_pct for p in gt_pages) / len(gt_pages)
                    gt_count = len(gt_pages)
                else:
                    a_cer = t_cer = a_wer = t_wer = ""
                    gt_count = 0

                writer.writerow([
                    rel_path,
                    cat,
                    len(pages),
                    round(avg_sim, 2),
                    gt_count,
                    round(a_cer, 2) if gt_count else "",
                    round(t_cer, 2) if gt_count else "",
                    round(a_wer, 2) if gt_count else "",
                    round(t_wer, 2) if gt_count else "",
                ])

    def _export_category_results_csv(self) -> None:
        csv_path = self.run_dir / "category_results.csv"
        cat_groups = defaultdict(list)
        for m in self.page_metrics:
            cat_groups[m.category].append(m)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "category",
                "total_pages",
                "avg_similarity_pct",
                "apple_wins",
                "tesseract_wins",
                "ties",
                "both_poor",
                "avg_apple_cer_pct",
                "avg_tesseract_cer_pct",
            ])
            for cat, pages in cat_groups.items():
                avg_sim = sum(p.pairwise_similarity_pct for p in pages) / len(pages)
                gt_pages = [p for p in pages if p.ground_truth_available]
                if gt_pages:
                    a_wins = sum(1 for p in gt_pages if p.winner == "apple")
                    t_wins = sum(1 for p in gt_pages if p.winner == "tesseract")
                    ties = sum(1 for p in gt_pages if p.winner == "tie")
                    poor = sum(1 for p in gt_pages if p.winner == "both_poor")
                    a_cer = sum(p.apple_cer_pct for p in gt_pages) / len(gt_pages)
                    t_cer = sum(p.tesseract_cer_pct for p in gt_pages) / len(gt_pages)
                else:
                    a_wins = t_wins = ties = poor = 0
                    a_cer = t_cer = ""

                writer.writerow([
                    cat,
                    len(pages),
                    round(avg_sim, 2),
                    a_wins,
                    t_wins,
                    ties,
                    poor,
                    round(a_cer, 2) if gt_pages else "",
                    round(t_cer, 2) if gt_pages else "",
                ])

    def _export_quality_tag_results_csv(self) -> None:
        csv_path = self.run_dir / "quality_tag_results.csv"
        tag_groups = defaultdict(list)
        for m in self.page_metrics:
            if m.ground_truth_available:
                for tag in m.quality_tags:
                    tag_groups[tag].append(m)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "quality_tag",
                "sample_count",
                "apple_cer_mean",
                "tesseract_cer_mean",
                "apple_wer_mean",
                "tesseract_wer_mean",
                "apple_wins",
                "tesseract_wins",
            ])
            for tag, pages in sorted(tag_groups.items()):
                n = len(pages)
                a_cer = sum(p.apple_cer_pct for p in pages) / n
                t_cer = sum(p.tesseract_cer_pct for p in pages) / n
                a_wer = sum(p.apple_wer_pct for p in pages) / n
                t_wer = sum(p.tesseract_wer_pct for p in pages) / n
                a_w = sum(1 for p in pages if p.winner == "apple")
                t_w = sum(1 for p in pages if p.winner == "tesseract")
                writer.writerow([
                    tag,
                    n,
                    round(a_cer, 2),
                    round(t_cer, 2),
                    round(a_wer, 2),
                    round(t_wer, 2),
                    a_w,
                    t_w,
                ])

    def _export_failures_csv(self) -> None:
        csv_path = self.run_dir / "failures.csv"
        failures = [
            m for m in self.page_metrics
            if m.apple_status in ("error", "missing")
            or m.tesseract_status in ("error", "missing")
            or m.pairwise_similarity_pct < 50.0
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "relative_path",
                "page_number",
                "apple_status",
                "tesseract_status",
                "pairwise_similarity_pct",
                "issue_description",
            ])
            for f_item in failures:
                issues = []
                if f_item.apple_status != "success":
                    issues.append(f"Apple {f_item.apple_status}")
                if f_item.tesseract_status != "success":
                    issues.append(f"Tesseract {f_item.tesseract_status}")
                if f_item.pairwise_similarity_pct < 50.0:
                    issues.append("High divergence (<50% similarity)")
                writer.writerow([
                    f_item.relative_path,
                    f_item.page_number,
                    f_item.apple_status,
                    f_item.tesseract_status,
                    f_item.pairwise_similarity_pct,
                    "; ".join(issues),
                ])

    def _export_summary_json(self) -> Dict[str, Any]:
        sim_scores = [m.pairwise_similarity_pct for m in self.page_metrics]
        sim_stats = calculate_distribution_stats(sim_scores)

        gt_metrics = [m for m in self.page_metrics if m.ground_truth_available]
        gt_summary = {}

        if gt_metrics:
            a_cers = [m.apple_cer_pct for m in gt_metrics if m.apple_cer_pct is not None]
            t_cers = [m.tesseract_cer_pct for m in gt_metrics if m.tesseract_cer_pct is not None]
            a_wers = [m.apple_wer_pct for m in gt_metrics if m.apple_wer_pct is not None]
            t_wers = [m.tesseract_wer_pct for m in gt_metrics if m.tesseract_wer_pct is not None]

            gt_summary = {
                "ground_truth_pages": len(gt_metrics),
                "winner_counts": {
                    "apple": sum(1 for m in gt_metrics if m.winner == "apple"),
                    "tesseract": sum(1 for m in gt_metrics if m.winner == "tesseract"),
                    "tie": sum(1 for m in gt_metrics if m.winner == "tie"),
                    "both_poor": sum(1 for m in gt_metrics if m.winner == "both_poor"),
                },
                "apple_cer": calculate_distribution_stats(a_cers),
                "tesseract_cer": calculate_distribution_stats(t_cers),
                "apple_wer": calculate_distribution_stats(a_wers),
                "tesseract_wer": calculate_distribution_stats(t_wers),
            }

        summary = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "sources": {
                "apple_dir": str(self.config.apple_dir),
                "tesseract_dir": str(self.config.tesseract_dir),
                "ground_truth": str(self.config.ground_truth_path) if self.config.ground_truth_path else None,
            },
            "documents": {
                "matched": len(self.matching.matched_pairs),
                "unmatched_apple": len(self.matching.unmatched_apple),
                "unmatched_tesseract": len(self.matching.unmatched_tesseract),
            },
            "pages": {
                "total_evaluated": len(self.page_metrics),
                "apple_missing": sum(1 for m in self.page_metrics if m.apple_status == "missing"),
                "tesseract_missing": sum(1 for m in self.page_metrics if m.tesseract_status == "missing"),
            },
            "pairwise_similarity": sim_stats,
            "ground_truth_evaluation": gt_summary if gt_metrics else None,
        }

        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary

    def _export_report_html(self, summary: Dict[str, Any]) -> None:
        html_path = self.run_dir / "report.html"

        has_gt = bool(summary.get("ground_truth_evaluation"))
        gt_info = summary.get("ground_truth_evaluation", {})

        html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Báo Cáo So Sánh OCR - {self.run_id}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 24px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  .header {{ background: #ffffff; padding: 24px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; }}
  h1 {{ margin: 0 0 8px 0; font-size: 24px; color: #0f172a; }}
  .meta {{ color: #64748b; font-size: 14px; line-height: 1.6; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .card {{ background: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #3b82f6; }}
  .card.green {{ border-left-color: #10b981; }}
  .card.purple {{ border-left-color: #8b5cf6; }}
  .card.amber {{ border-left-color: #f59e0b; }}
  .card-val {{ font-size: 28px; font-weight: 700; color: #0f172a; margin-top: 4px; }}
  .card-label {{ font-size: 13px; color: #64748b; text-transform: uppercase; font-weight: 600; }}
  .section {{ background: #ffffff; padding: 24px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; }}
  h2 {{ margin: 0 0 16px 0; font-size: 18px; color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; text-align: left; }}
  th, td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; }}
  th {{ background: #f1f5f9; color: #475569; font-weight: 600; }}
  tr:hover {{ background: #f8fafc; }}
  .badge {{ display: inline-block; padding: 3px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; }}
  .badge-apple {{ background: #dbeafe; color: #1e40af; }}
  .badge-tess {{ background: #dcfce7; color: #166534; }}
  .badge-tie {{ background: #fef3c7; color: #92400e; }}
  .badge-poor {{ background: #fee2e2; color: #991b1b; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>Báo Cáo So Sánh OCR: Apple Vision vs Tesseract 5</h1>
    <div class="meta">
      <strong>Mã lần chạy:</strong> {self.run_id} | <strong>Thời điểm:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
      <strong>Apple Vision kết quả:</strong> {self.config.apple_dir}<br>
      <strong>Tesseract kết quả:</strong> {self.config.tesseract_dir}<br>
      <strong>Ground Truth:</strong> {self.config.ground_truth_path or "Không có (Chế độ Pairwise Difference)"}
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <div class="card-label">Tài liệu khớp nối</div>
      <div class="card-val">{summary['documents']['matched']}</div>
    </div>
    <div class="card green">
      <div class="card-label">Tổng số trang so sánh</div>
      <div class="card-val">{summary['pages']['total_evaluated']}</div>
    </div>
    <div class="card purple">
      <div class="card-label">Độ tương đồng TB (Pairwise)</div>
      <div class="card-val">{summary['pairwise_similarity']['mean']}%</div>
    </div>
    <div class="card amber">
      <div class="card-label">Chưa khớp (Apple / Tess)</div>
      <div class="card-val">{summary['documents']['unmatched_apple']} / {summary['documents']['unmatched_tesseract']}</div>
    </div>
  </div>
"""

        if has_gt:
            w_counts = gt_info.get("winner_counts", {})
            html_content += f"""
  <div class="section">
    <h2>Kết quả đánh giá theo Ground Truth ({gt_info.get('ground_truth_pages')} trang)</h2>
    <div class="grid">
      <div class="card">
        <div class="card-label">Apple CER TB</div>
        <div class="card-val">{gt_info.get('apple_cer', {}).get('mean', 0)}%</div>
      </div>
      <div class="card green">
        <div class="card-label">Tesseract CER TB</div>
        <div class="card-val">{gt_info.get('tesseract_cer', {}).get('mean', 0)}%</div>
      </div>
      <div class="card">
        <div class="card-label">Apple Thắng</div>
        <div class="card-val">{w_counts.get('apple', 0)}</div>
      </div>
      <div class="card green">
        <div class="card-label">Tesseract Thắng</div>
        <div class="card-val">{w_counts.get('tesseract', 0)}</div>
      </div>
      <div class="card amber">
        <div class="card-label">Hòa (ngưỡng 5%)</div>
        <div class="card-val">{w_counts.get('tie', 0)}</div>
      </div>
    </div>
  </div>
"""
        else:
            html_content += """
  <div class="section">
    <h2>Lưu ý đánh giá</h2>
    <p style="color: #64748b; font-size: 14px;">
      Lần chạy này không sử dụng Ground Truth. Báo cáo tập trung vào <strong>mức độ chênh lệch (Pairwise Difference)</strong> và độ phủ giữa hai engine OCR, không kết luận engine nào chính xác hơn.
    </p>
  </div>
"""

        # Bảng danh mục
        cat_groups = defaultdict(list)
        for m in self.page_metrics:
            cat_groups[m.category].append(m)

        html_content += """
  <div class="section">
    <h2>Thống kê theo Danh Mục</h2>
    <table>
      <thead>
        <tr>
          <th>Danh mục</th>
          <th>Số trang</th>
          <th>Độ tương đồng TB</th>
"""
        if has_gt:
            html_content += "<th>Apple CER</th><th>Tesseract CER</th><th>Apple Thắng</th><th>Tess Thắng</th>"
        html_content += """
        </tr>
      </thead>
      <tbody>
"""
        for cat, p_list in cat_groups.items():
            avg_sim = sum(p.pairwise_similarity_pct for p in p_list) / len(p_list)
            html_content += f"<tr><td><strong>{cat}</strong></td><td>{len(p_list)}</td><td>{avg_sim:.1f}%</td>"
            if has_gt:
                gt_p = [p for p in p_list if p.ground_truth_available]
                if gt_p:
                    a_c = sum(p.apple_cer_pct for p in gt_p) / len(gt_p)
                    t_c = sum(p.tesseract_cer_pct for p in gt_p) / len(gt_p)
                    a_w = sum(1 for p in gt_p if p.winner == "apple")
                    t_w = sum(1 for p in gt_p if p.winner == "tesseract")
                    html_content += f"<td>{a_c:.2f}%</td><td>{t_c:.2f}%</td><td>{a_w}</td><td>{t_w}</td>"
                else:
                    html_content += "<td>-</td><td>-</td><td>-</td><td>-</td>"
            html_content += "</tr>"

        html_content += """
      </tbody>
    </table>
  </div>
</div>
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
