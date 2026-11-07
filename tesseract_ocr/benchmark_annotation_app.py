"""Giao diện xác nhận ground truth cho benchmark 10 PDF."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import streamlit as st

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from tesseract_ocr.benchmark_variants import ground_truth_path
from tesseract_ocr.tesseract_engine import render_pdf_page_to_image


DEFAULT_INPUT_DIR = Path(
    r"D:\Thuc-Tap\OCR\input\GIAO_NHAN_HANG_HOA\BB_GIAO_NHAN_HANG_HOA"
)
DEFAULT_BENCHMARK_DIR = Path(r"D:\Thuc-Tap\OCR\tesseract_benchmark_10pdf")


def read_text_if_exists(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8-sig", errors="replace")


def main() -> None:
    st.set_page_config(page_title="Xác nhận Ground Truth OCR", page_icon="✅", layout="wide")
    st.title("✅ Xác nhận Ground Truth cho benchmark Tesseract")
    st.warning(
        "Chỉ văn bản đã đối chiếu trực tiếp với ảnh mới được lưu vào ground_truth. "
        "Bản nháp OCR và kết quả iOS đều không phải đáp án chuẩn."
    )

    input_dir = Path(st.sidebar.text_input("Thư mục PDF", str(DEFAULT_INPUT_DIR))).resolve()
    benchmark_dir = Path(
        st.sidebar.text_input("Thư mục benchmark", str(DEFAULT_BENCHMARK_DIR))
    ).resolve()
    manifest_path = benchmark_dir / "benchmark_manifest.json"
    if not manifest_path.is_file():
        st.error(f"Không tìm thấy manifest: {manifest_path}")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_pdfs = [Path(item) for item in manifest.get("selected_pdfs", [])]
    if not selected_pdfs:
        st.error("Manifest không có danh sách PDF.")
        return

    gt_root = benchmark_dir / "ground_truth"
    draft_root = benchmark_dir / "ground_truth_drafts"
    completed = sum(
        1
        for relative_pdf in selected_pdfs
        if read_text_if_exists(ground_truth_path(gt_root, relative_pdf, 1)).strip()
    )
    st.sidebar.progress(completed / len(selected_pdfs))
    st.sidebar.caption(f"Đã xác nhận: {completed}/{len(selected_pdfs)} PDF")

    selected_relative = Path(
        st.sidebar.selectbox(
            "Chọn PDF",
            [path.as_posix() for path in selected_pdfs],
        )
    )
    pdf_path = input_dir / selected_relative
    if not pdf_path.is_file():
        st.error(f"Không tìm thấy PDF: {pdf_path}")
        return

    page_number = 1
    gt_path = ground_truth_path(gt_root, selected_relative, page_number)
    draft_path = ground_truth_path(draft_root, selected_relative, page_number)
    confirmed_text = read_text_if_exists(gt_path)
    draft_text = read_text_if_exists(draft_path)
    editor_key = f"ground_truth::{selected_relative.as_posix()}::{page_number}"
    if editor_key not in st.session_state:
        st.session_state[editor_key] = confirmed_text or draft_text

    with fitz.open(pdf_path) as document:
        rendered = render_pdf_page_to_image(document[page_number - 1], dpi=180)
    rendered_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)

    image_col, text_col = st.columns([1.15, 1])
    with image_col:
        st.subheader("Ảnh PDF để đối chiếu")
        st.image(rendered_rgb, width="stretch")
    with text_col:
        st.subheader("Văn bản chuẩn đã kiểm tra")
        st.text_area(
            "Sửa toàn bộ lỗi OCR, dấu tiếng Việt và các dòng bị thiếu trước khi lưu:",
            key=editor_key,
            height=720,
        )
        save_col, reset_col = st.columns(2)
        if save_col.button("Lưu ground truth đã xác nhận", type="primary", width="stretch"):
            value = st.session_state[editor_key].strip()
            if not value:
                st.error("Ground truth không được để trống.")
            else:
                gt_path.parent.mkdir(parents=True, exist_ok=True)
                gt_path.write_text(value + "\n", encoding="utf-8")
                st.success(f"Đã lưu: {gt_path}")
        if reset_col.button("Nạp lại bản nháp OCR", width="stretch"):
            st.session_state[editor_key] = draft_text
            st.rerun()

    with st.expander("Bản nháp OCR dùng để nhập liệu nhanh"):
        st.code(draft_text or "(Không có bản nháp)", language=None)
    st.info(
        "Sau khi xác nhận đủ 10 PDF, chạy lại `run_benchmark.ps1`. "
        "Hệ thống sẽ dùng checkpoint hiện có và tính CER/WER mà không OCR lại."
    )


if __name__ == "__main__":
    main()
