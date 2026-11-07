"""
Giao diện trực quan So sánh Kết quả OCR (Streamlit Dashboard).
Chỉ đọc dữ liệu kết quả và báo cáo đã tạo, tuyệt đối không gọi OCR engine.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="So Sánh OCR: Apple Vision vs Tesseract",
    page_icon="🔍",
    layout="wide",
)


def highlight_diff(text1: str, text2: str) -> tuple[str, str]:
    """
    Tạo HTML đánh dấu các ký tự hoặc từ khác biệt giữa hai đoạn văn bản.
    text1: Chuỗi nguồn (bị xóa/thay thế được tô đỏ)
    text2: Chuỗi đích (được thêm/thay thế được tô xanh)
    """
    words1 = text1.split()
    words2 = text2.split()

    matcher = difflib.SequenceMatcher(None, words1, words2)
    h1, h2 = [], []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            h1.extend(words1[i1:i2])
            h2.extend(words2[j1:j2])
        elif tag == "delete":
            h1.append(f"<span style='background-color: #fee2e2; color: #991b1b; text-decoration: line-through;'>{' '.join(words1[i1:i2])}</span>")
        elif tag == "insert":
            h2.append(f"<span style='background-color: #dcfce7; color: #166534; font-weight: bold;'>{' '.join(words2[j1:j2])}</span>")
        elif tag == "replace":
            h1.append(f"<span style='background-color: #fee2e2; color: #991b1b;'>{' '.join(words1[i1:i2])}</span>")
            h2.append(f"<span style='background-color: #dcfce7; color: #166534; font-weight: bold;'>{' '.join(words2[j1:j2])}</span>")

    return " ".join(h1), " ".join(h2)


def main():
    st.title("🔍 So Sánh Độc Lập: Apple Vision vs Tesseract 5")
    st.caption("Chỉ đọc dữ liệu kết quả và báo cáo so sánh. Không kích hoạt hay sửa đổi OCR engines.")

    # Sidebar: Lựa chọn thư mục báo cáo hoặc chỉ định run_id
    st.sidebar.header("Cấu hình báo cáo")
    base_report_dir = Path("comparison_results")

    run_dirs = []
    if base_report_dir.exists():
        run_dirs = [d for d in base_report_dir.iterdir() if d.is_dir()]
        run_dirs.sort(key=lambda d: d.name, reverse=True)

    if not run_dirs:
        st.warning(f"Chưa tìm thấy thư mục báo cáo trong '{base_report_dir}'. Vui lòng chạy lệnh so sánh:")
        st.code("python -m ocr_comparison.compare_outputs --apple-dir output --tesseract-dir tesseract_output")
        return

    selected_run = st.sidebar.selectbox(
        "Chọn lần chạy (Run ID):",
        run_dirs,
        format_func=lambda d: d.name,
    )

    summary_file = selected_run / "summary.json"
    pairwise_file = selected_run / "page_pairwise_results.csv"
    gt_file = selected_run / "page_ground_truth_results.csv"
    cat_file = selected_run / "category_results.csv"
    failures_file = selected_run / "failures.csv"

    if not summary_file.exists():
        st.error(f"Tệp summary.json không tồn tại trong {selected_run}")
        return

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Đọc dữ liệu CSV
    df_pages = pd.read_csv(pairwise_file) if pairwise_file.exists() else pd.DataFrame()
    df_gt = pd.read_csv(gt_file) if gt_file.exists() else pd.DataFrame()
    df_cat = pd.read_csv(cat_file) if cat_file.exists() else pd.DataFrame()
    df_fail = pd.read_csv(failures_file) if failures_file.exists() else pd.DataFrame()

    has_gt = summary.get("ground_truth_evaluation") is not None

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Tổng quan & KPI",
        "📄 So sánh chi tiết từng trang",
        "📁 Thống kê danh mục",
        "⚠️ Sự cố & Lệch lớn",
    ])

    with tab1:
        st.subheader("Chỉ số tổng hợp")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tài liệu khớp", summary["documents"]["matched"])
        col2.metric("Tổng số trang", summary["pages"]["total_evaluated"])
        col3.metric("Độ tương đồng TB (Pairwise)", f"{summary['pairwise_similarity']['mean']}%")
        col4.metric(
            "Chưa khớp (Apple / Tess)",
            f"{summary['documents']['unmatched_apple']} / {summary['documents']['unmatched_tesseract']}",
        )

        if has_gt:
            st.divider()
            st.subheader("Đánh giá qua Ground Truth")
            gt_info = summary["ground_truth_evaluation"]
            w_counts = gt_info.get("winner_counts", {})
            gcol1, gcol2, gcol3, gcol4, gcol5 = st.columns(5)
            gcol1.metric("Apple CER TB", f"{gt_info['apple_cer']['mean']}%")
            gcol2.metric("Tesseract CER TB", f"{gt_info['tesseract_cer']['mean']}%")
            gcol3.metric("Apple Thắng", w_counts.get("apple", 0))
            gcol4.metric("Tesseract Thắng", w_counts.get("tesseract", 0))
            gcol5.metric("Hòa (ngưỡng 5%)", w_counts.get("tie", 0))

        st.divider()
        st.subheader("Đường dẫn và Tham số chạy")
        st.json(summary["sources"])

    with tab2:
        st.subheader("Kiểm tra văn bản cạnh nhau (Side-by-Side)")
        if df_pages.empty:
            st.info("Không có dữ liệu trang.")
        else:
            # Bộ lọc
            fcol1, fcol2, fcol3 = st.columns(3)
            categories = ["Tất cả"] + sorted(df_pages["category"].dropna().unique().tolist())
            selected_cat = fcol1.selectbox("Danh mục:", categories)

            filtered_df = df_pages
            if selected_cat != "Tất cả":
                filtered_df = filtered_df[filtered_df["category"] == selected_cat]

            if has_gt and "winner" in filtered_df.columns:
                winners = ["Tất cả"] + sorted(filtered_df["winner"].dropna().unique().tolist())
                selected_winner = fcol2.selectbox("Kết quả thắng:", winners)
                if selected_winner != "Tất cả":
                    filtered_df = filtered_df[filtered_df["winner"] == selected_winner]

            docs = sorted(filtered_df["relative_path"].dropna().unique().tolist())
            if not docs:
                st.warning("Không có tài liệu nào khớp bộ lọc.")
            else:
                selected_doc = fcol3.selectbox("Chọn tài liệu:", docs)
                doc_pages = filtered_df[filtered_df["relative_path"] == selected_doc]

                p_nums = sorted(doc_pages["page_number"].unique().tolist())
                selected_page_num = st.selectbox("Chọn số trang:", p_nums)

                page_row = doc_pages[doc_pages["page_number"] == selected_page_num].iloc[0]

                # Hiển thị thông số trang
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Độ tương đồng", f"{page_row.get('pairwise_similarity_pct', 0)}%")
                mcol2.metric("Ký tự (Apple / Tess)", f"{page_row.get('apple_chars', 0)} / {page_row.get('tesseract_chars', 0)}")
                if has_gt and pd.notna(page_row.get("apple_cer_pct")):
                    mcol3.metric("Apple CER", f"{page_row.get('apple_cer_pct')}%")
                    mcol4.metric("Tesseract CER", f"{page_row.get('tesseract_cer_pct')}%")

                st.markdown("### Nội dung trích xuất")
                # Đọc nội dung thực tế từ file nguồn hoặc hiển thị kết quả
                c_apple, c_tess = st.columns(2)
                with c_apple:
                    st.markdown("**Kết quả Apple Vision**")
                    st.caption(f"Trạng thái: {page_row.get('apple_status')}")

                with c_tess:
                    st.markdown("**Kết quả Tesseract 5**")
                    st.caption(f"Trạng thái: {page_row.get('tesseract_status')}")

    with tab3:
        st.subheader("Báo cáo phân nhóm theo Danh mục")
        if not df_cat.empty:
            st.dataframe(df_cat, use_container_width=True)
        else:
            st.info("Chưa có bảng dữ liệu danh mục.")

    with tab4:
        st.subheader("Các trang có sự cố hoặc độ tương đồng thấp (< 50%)")
        if not df_fail.empty:
            st.dataframe(df_fail, use_container_width=True)
        else:
            st.success("Không có trang nào bị lỗi hoặc có độ tương đồng dưới 50%!")


if __name__ == "__main__":
    main()
