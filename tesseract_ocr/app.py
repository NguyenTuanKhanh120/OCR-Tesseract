"""
Giao diện trực quan để test Tesseract OCR đơn lẻ (Streamlit Interactive Playground).
Cho phép tải lên file PDF/ảnh, điều chỉnh thông số (DPI, OEM, PSM, Preprocessing)
và xem trực quan Bounding Box, ảnh tiền xử lý, độ tin cậy và văn bản trích xuất.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from tesseract_ocr.config import TesseractConfig
from tesseract_ocr.adaptive_background_ocr import run_adaptive_background_ocr
from tesseract_ocr.batch_ocr import (
    build_results_zip,
    combine_page_results,
    unique_text_filenames,
)
from tesseract_ocr.insurance_pipeline import (
    InsurancePipelineConfig,
    STAGE_NAMES,
    run_insurance_region_pipeline,
)
from tesseract_ocr.tesseract_engine import (
    find_tesseract_executable,
    find_tessdata_directory,
    get_project_tessdata_profiles,
    render_pdf_page_to_image,
    verify_tesseract_environment,
)

st.set_page_config(
    page_title="Tesseract OCR Playground",
    page_icon="🔬",
    layout="wide",
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_RUNS_DIR = PROJECT_ROOT / ".web_runs"
INVALID_UPLOAD_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def encode_png_bytes(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Không thể mã hóa ảnh kết quả để hiển thị")
    return encoded.tobytes()


def decode_png_bytes(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("Không thể đọc ảnh kết quả đã lưu")
    return image


def run_page_ocr(
    raw_image: np.ndarray,
    config: TesseractConfig,
    *,
    page_number: int,
    prep_mode: str,
    adaptive_background_enabled: bool,
    include_display_image: bool,
) -> dict:
    outcome = run_adaptive_background_ocr(
        raw_image,
        config,
        page_number=page_number,
        preprocessing_mode=prep_mode,
        enabled=adaptive_background_enabled,
    )
    result = outcome["selected_result"]
    record = {
        "page_number": page_number,
        "prep_meta": dict(result.preprocessing),
        "result": result,
        "selected_variant": outcome["selected_variant"],
        "candidate_kind": outcome["candidate_kind"],
        "decision": outcome["decision"],
        "total_processing_seconds": outcome["processing_seconds"],
    }
    if include_display_image:
        record["selected_image_png"] = encode_png_bytes(outcome["selected_image"])
    return record


def draw_bounding_boxes(image: np.ndarray, boxes: list) -> np.ndarray:
    """Vẽ bounding boxes của các từ lên ảnh kèm độ tin cậy."""
    annotated = image.copy()
    if len(annotated.shape) == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

    for box in boxes:
        x, y, w, h = box.x, box.y, box.w, box.h
        conf = box.confidence
        # Màu xanh lá nếu conf cao, vàng/cam nếu thấp
        color = (0, 200, 0) if conf >= 0.7 else (0, 165, 255)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

    return annotated


def safe_uploaded_pdf_name(name: str) -> str:
    """Giữ tên tài liệu để đặt TXT nhưng loại bỏ thành phần đường dẫn/ký tự cấm."""
    leaf_name = Path(str(name).replace("\\", "/")).name
    stem = INVALID_UPLOAD_FILENAME.sub("_", Path(leaf_name).stem).strip(" .")
    if not stem:
        stem = "document"
    return f"{stem}.pdf"


def load_json_file(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def display_region_pipeline_result(payload: dict) -> None:
    result = payload["result"]
    combined_text = payload["combined_text"]
    region_manifest = load_json_file(Path(result["region_ocr_manifest"]))
    region_root = Path(result["region_ocr_manifest"]).parent

    if result.get("cache_hit"):
        st.info("Đã dùng lại kết quả đã hoàn thành và có checksum trùng khớp.")
    if result["needs_review_panels"]:
        st.warning(
            f"Có {result['needs_review_panels']} vùng cần kiểm tra thủ công. "
            "TXT vẫn chứa kết quả tốt nhất mà Tesseract đã chọn."
        )
    else:
        st.success("Pipeline 7 bước đã hoàn thành đủ trang và đủ vùng.")

    st.warning(
        "Tesseract phù hợp nhất với chữ in. Chữ viết tay mờ hoặc bị con dấu "
        "chồng lên vẫn có thể sai; confidence không phải tỷ lệ chính xác CER/WER."
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Bước hoàn thành", "7/7")
    m2.metric("Trang", result["page_count"])
    m3.metric("Vùng OCR", result["panel_count"])
    m4.metric("Lượt Tesseract", result["candidate_run_count"])
    m5.metric("Thời gian", f"{result['processing_seconds']:.1f} s")

    tab_text, tab_panels, tab_images, tab_steps = st.tabs(
        [
            "📝 Toàn bộ văn bản",
            "📊 Kết quả từng vùng",
            "🖼️ Ảnh kiểm tra OCR",
            "✅ Tiến trình 7 bước",
        ]
    )

    with tab_text:
        st.subheader("Văn bản OCR toàn bộ PDF")
        st.caption(
            "Đã ghép theo thứ tự trang → vùng; không cần chọn từng trang để xem nội dung."
        )
        st.text_area(
            "Nội dung trích xuất:",
            value=combined_text,
            height=650,
            key=f"region_text_{payload['signature']}",
        )
        st.download_button(
            "📥 Tải xuống TXT toàn bộ PDF",
            data=combined_text,
            file_name=Path(result["output_text_file"]).name,
            mime="text/plain; charset=utf-8",
            key=f"region_download_{payload['signature']}",
        )

    panel_rows = []
    panel_options = []
    for page in region_manifest["pages"]:
        for panel in page["panels"]:
            metrics = panel["selected_metrics"]
            label = f"Trang {page['page_number']} - vùng {panel['panel_index']}"
            panel_options.append((label, page, panel))
            panel_rows.append(
                {
                    "Trang": page["page_number"],
                    "Vùng": panel["panel_index"],
                    "Trạng thái": panel["status"],
                    "Ứng viên được chọn": panel["selected_candidate_id"],
                    "PSM": panel["selected_psm"],
                    "Confidence": round(float(metrics["confidence"]) * 100, 1),
                    "Điểm sàng lọc": round(
                        float(metrics["screening_score"]) * 100, 1
                    ),
                    "Số từ": metrics["word_count"],
                }
            )

    with tab_panels:
        st.caption(
            "Điểm sàng lọc dùng để chọn giữa 6 ứng viên của cùng một vùng, "
            "không phải độ chính xác thực tế."
        )
        st.dataframe(pd.DataFrame(panel_rows), width="stretch", hide_index=True)
        for label, _, panel in panel_options:
            with st.expander(label, expanded=False):
                selected_text_path = region_root / panel["selected_text_file"]
                st.text(
                    selected_text_path.read_text(encoding="utf-8")
                    or "(Không nhận dạng được văn bản)"
                )

    with tab_images:
        labels = [item[0] for item in panel_options]
        selected_label = st.selectbox(
            "Chọn trang và vùng để xem bounding boxes",
            labels,
            key=f"region_overlay_select_{payload['signature']}",
        )
        _, selected_page, selected_panel = next(
            item for item in panel_options if item[0] == selected_label
        )
        overlay_path = region_root / selected_panel["selected_overlay_file"]
        st.image(
            str(overlay_path),
            caption=(
                f"Trang {selected_page['page_number']} - vùng "
                f"{selected_panel['panel_index']} - "
                f"{selected_panel['selected_candidate_id']}"
            ),
            width="stretch",
        )
        st.caption("Xanh: confidence ≥ 70%; cam: 40-70%; đỏ: dưới 40%.")

    with tab_steps:
        stage_rows = []
        for stage in result["stages"]:
            stage_rows.append(
                {
                    "Bước": stage["step"],
                    "Xử lý": stage["name"],
                    "Trạng thái": stage["status"],
                    "Số trang": stage.get("page_count", ""),
                    "Số vùng": stage.get("panel_count", ""),
                    "Cần kiểm tra": stage.get("needs_review_panels", ""),
                }
            )
        st.dataframe(pd.DataFrame(stage_rows), width="stretch", hide_index=True)
        st.caption(f"Thư mục kiểm tra chi tiết: {result['run_dir']}")


def insurance_region_pipeline_ui(tess_exe: str) -> None:
    st.title("🧾 OCR bảo hiểm - Pipeline 7 bước")
    st.caption(
        "Tải một PDF để hệ thống tự render toàn bộ trang, chia vùng, chỉnh phối "
        "cảnh, lọc artifact/nền, OCR nhiều PSM và ghép thành một TXT."
    )

    required_langs = ["vie", "eng"]
    profiles = get_project_tessdata_profiles(required_langs)
    best_profile = profiles.get("best")
    if best_profile is None:
        st.error("Không tìm thấy tessdata_best có đủ model vie và eng.")
        return
    tessdata_dir = find_tessdata_directory(best_profile, tess_exe, required_langs)
    if not tessdata_dir:
        st.error("Không xác định được thư mục tessdata_best.")
        return

    st.sidebar.header("Cấu hình pipeline 7 bước")
    dpi = st.sidebar.slider(
        "DPI render PDF",
        min_value=300,
        max_value=500,
        value=400,
        step=50,
        key="region_dpi",
    )
    panel_choice = st.sidebar.selectbox(
        "Số vùng ngang mỗi trang",
        ["Tự động", "2 vùng", "3 vùng"],
        index=0,
        key="region_expected_panels",
        help="Giữ Tự động nếu tài liệu có số khung khác nhau giữa các trang.",
    )
    expected_panels = {"Tự động": None, "2 vùng": 2, "3 vùng": 3}[panel_choice]
    min_screening_score = st.sidebar.slider(
        "Ngưỡng cảnh báo vùng OCR",
        min_value=0.30,
        max_value=0.80,
        value=0.45,
        step=0.05,
        key="region_min_screening_score",
    )
    st.sidebar.success("Model: tessdata_best - vie+eng")
    st.sidebar.caption("Ứng viên OCR: ảnh CLAHE/Sauvola × PSM 3/6/11")

    with st.expander("Xem 7 bước xử lý", expanded=False):
        for step in range(1, 8):
            st.markdown(f"**Bước {step}:** {STAGE_NAMES[step]}")

    uploaded_file = st.file_uploader(
        "Tải lên tài liệu bảo hiểm PDF:",
        type=["pdf"],
        key="region_pdf_uploader",
    )
    if not uploaded_file:
        st.info("Vui lòng tải lên một PDF để chạy toàn bộ pipeline.")
        return

    file_bytes = uploaded_file.getvalue()
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            total_pages = len(document)
            needs_password = bool(document.needs_pass)
    except Exception as error:
        st.error(f"Không thể mở PDF: {error}")
        return
    if needs_password:
        st.error("PDF đang được bảo vệ bằng mật khẩu.")
        return
    if total_pages <= 0:
        st.error("PDF không có trang để OCR.")
        return

    st.info(
        f"Tệp: **{uploaded_file.name}** - {total_pages} trang. "
        "Pipeline luôn quét toàn bộ PDF."
    )
    config = InsurancePipelineConfig(
        dpi=dpi,
        expected_panels=expected_panels,
        min_screening_score=min_screening_score,
    )
    signature_payload = {
        "pipeline_version": "insurance-region-7-steps-v1",
        "source_name": uploaded_file.name,
        "source_sha256": hashlib.sha256(file_bytes).hexdigest(),
        "config": config.to_dict(),
        "tessdata": str(tessdata_dir),
    }
    run_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if st.button(
        "🚀 Chạy pipeline OCR 7 bước",
        type="primary",
        width="stretch",
        key="run_region_pipeline",
    ):
        run_dir = WEB_RUNS_DIR / run_signature[:24]
        input_dir = run_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        source_pdf = input_dir / safe_uploaded_pdf_name(uploaded_file.name)
        source_pdf.write_bytes(file_bytes)

        progress = st.progress(0.0)
        status_text = st.empty()

        def update_progress(step: int, name: str) -> None:
            progress.progress((step - 1) / 7)
            status_text.info(f"Bước {step}/7: {name}...")

        try:
            result = run_insurance_region_pipeline(
                source_pdf,
                run_dir,
                tessdata_dir=Path(tessdata_dir),
                tesseract_cmd=tess_exe,
                config=config,
                progress_callback=update_progress,
            )
            progress.progress(1.0)
            status_text.success(
                f"Đã hoàn thành 7/7 bước cho {result['page_count']} trang."
            )
            combined_text = Path(result["output_text_file"]).read_text(
                encoding="utf-8"
            )
            st.session_state["last_region_pipeline_payload"] = {
                "signature": run_signature,
                "source_name": uploaded_file.name,
                "result": result,
                "combined_text": combined_text,
            }
        except Exception as error:
            status_text.error(f"Pipeline dừng do lỗi: {error}")
            st.exception(error)
            return

    payload = st.session_state.get("last_region_pipeline_payload")
    if not payload or payload.get("signature") != run_signature:
        st.info("Nhấn **Chạy pipeline OCR 7 bước** để bắt đầu.")
        return
    display_region_pipeline_result(payload)


def multiple_file_ocr_ui(
    uploaded_files,
    *,
    tess_exe: str,
    tessdata_dir: Path,
    dpi: int,
    lang: str,
    oem: int,
    psm: int,
    prep_mode: str,
    prep_profile: str,
    thresholding_method: int,
    model_profile: str,
    adaptive_background_enabled: bool,
) -> None:
    """OCR tuần tự nhiều PDF/ảnh và tạo một TXT riêng cho từng file."""
    output_names = unique_text_filenames([item.name for item in uploaded_files])
    specs = []
    validation_errors = []

    for index, (uploaded_file, output_name) in enumerate(
        zip(uploaded_files, output_names), 1
    ):
        file_bytes = uploaded_file.getvalue()
        is_pdf = uploaded_file.name.lower().endswith(".pdf")
        total_pages = 1
        detail = "Ảnh"
        try:
            if is_pdf:
                with fitz.open(stream=file_bytes, filetype="pdf") as document:
                    if document.needs_pass:
                        raise ValueError("PDF được bảo vệ bằng mật khẩu")
                    total_pages = len(document)
                if total_pages <= 0:
                    raise ValueError("PDF không có trang")
                detail = f"PDF - {total_pages} trang"
            else:
                raw_image = cv2.imdecode(
                    np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR
                )
                if raw_image is None:
                    raise ValueError("Không đọc được định dạng ảnh")
                height, width = raw_image.shape[:2]
                detail = f"Ảnh - {width} × {height} px"
        except Exception as error:
            validation_errors.append(f"{uploaded_file.name}: {error}")

        specs.append(
            {
                "index": index,
                "label": f"{index}. {uploaded_file.name}",
                "source_name": uploaded_file.name,
                "file_bytes": file_bytes,
                "is_pdf": is_pdf,
                "total_pages": total_pages,
                "detail": detail,
                "output_name": output_name,
                "sha256": hashlib.sha256(file_bytes).hexdigest(),
            }
        )

    st.info(
        f"Đã chọn **{len(specs)} file**. Hệ thống sẽ OCR tuần tự và luôn quét "
        "toàn bộ các trang PDF."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "STT": spec["index"],
                    "Tên file": spec["source_name"],
                    "Loại": spec["detail"],
                    "TXT đầu ra": spec["output_name"],
                }
                for spec in specs
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    if validation_errors:
        for message in validation_errors:
            st.error(message)
        st.warning("Hãy bỏ các file lỗi khỏi danh sách trước khi chạy OCR.")
        return

    signature_payload = {
        "files": [
            {"name": spec["source_name"], "sha256": spec["sha256"]}
            for spec in specs
        ],
        "dpi": dpi,
        "lang": lang,
        "oem": oem,
        "psm": psm,
        "preprocessing": prep_profile,
        "thresholding_method": thresholding_method,
        "model_profile": model_profile,
        "adaptive_background": adaptive_background_enabled,
    }
    run_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if st.button(
        f"🚀 Chạy OCR {len(specs)} file",
        type="primary",
        width="stretch",
        key="run_multiple_file_ocr",
    ):
        config = TesseractConfig(
            input_dir=Path("."),
            output_dir=Path("."),
            dpi=dpi,
            lang=lang,
            oem=oem,
            psm=psm,
            preprocessing=prep_mode,
            thresholding_method=thresholding_method,
            tessdata_dir=tessdata_dir,
            adaptive_background=adaptive_background_enabled,
        )
        total_work = sum(spec["total_pages"] for spec in specs)
        completed_work = 0
        progress = st.progress(0.0)
        status_text = st.empty()
        document_results = []

        for file_position, spec in enumerate(specs, 1):
            page_records = []
            file_error = None
            try:
                if spec["is_pdf"]:
                    with fitz.open(
                        stream=spec["file_bytes"], filetype="pdf"
                    ) as document:
                        for page_index in range(spec["total_pages"]):
                            page_number = page_index + 1
                            status_text.info(
                                f"File {file_position}/{len(specs)}: "
                                f"{spec['source_name']} - trang "
                                f"{page_number}/{spec['total_pages']}..."
                            )
                            raw_page = render_pdf_page_to_image(
                                document[page_index], dpi=dpi
                            )
                            page_records.append(
                                run_page_ocr(
                                    raw_page,
                                    config,
                                    page_number=page_number,
                                    prep_mode=prep_mode,
                                    adaptive_background_enabled=adaptive_background_enabled,
                                    include_display_image=False,
                                )
                            )
                            completed_work += 1
                            progress.progress(completed_work / total_work)
                else:
                    status_text.info(
                        f"File {file_position}/{len(specs)}: {spec['source_name']}..."
                    )
                    raw_image = cv2.imdecode(
                        np.frombuffer(spec["file_bytes"], np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    page_records.append(
                        run_page_ocr(
                            raw_image,
                            config,
                            page_number=1,
                            prep_mode=prep_mode,
                            adaptive_background_enabled=adaptive_background_enabled,
                            include_display_image=False,
                        )
                    )
                    completed_work += 1
                    progress.progress(completed_work / total_work)
            except Exception as error:
                file_error = str(error)

            combined_text = (
                combine_page_results(page_records, spec["total_pages"])
                if page_records
                else ""
            )
            has_page_error = any(
                record["result"].status == "error" for record in page_records
            )
            document_results.append(
                {
                    "index": spec["index"],
                    "label": spec["label"],
                    "source_name": spec["source_name"],
                    "output_name": spec["output_name"],
                    "total_pages": spec["total_pages"],
                    "records": page_records,
                    "combined_text": combined_text,
                    "status": (
                        "error"
                        if file_error
                        else "needs_review" if has_page_error else "success"
                    ),
                    "error": file_error,
                }
            )

        progress.progress(1.0)
        successful_count = sum(
            item["status"] != "error" for item in document_results
        )
        status_text.success(
            f"Đã xử lý {successful_count}/{len(document_results)} file."
        )
        st.session_state["last_batch_ocr_payload"] = {
            "signature": run_signature,
            "documents": document_results,
        }

    payload = st.session_state.get("last_batch_ocr_payload")
    if not payload or payload.get("signature") != run_signature:
        st.info("Nhấn nút chạy để OCR toàn bộ danh sách file đã chọn.")
        return

    documents = payload["documents"]
    successful_documents = [item for item in documents if item["combined_text"]]
    total_processed_pages = sum(len(item["records"]) for item in documents)
    m1, m2, m3 = st.columns(3)
    m1.metric("File đã chọn", len(documents))
    m2.metric("File có kết quả", len(successful_documents))
    m3.metric("Trang đã OCR", total_processed_pages)

    summary_rows = []
    for item in documents:
        word_count = sum(len(record["result"].boxes) for record in item["records"])
        summary_rows.append(
            {
                "File": item["source_name"],
                "Trạng thái": item["status"],
                "Trang OCR": len(item["records"]),
                "Số từ": word_count,
                "Lỗi": item["error"] or "",
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    if successful_documents:
        zip_payload = build_results_zip(
            (item["output_name"], item["combined_text"])
            for item in successful_documents
        )
        st.download_button(
            "📦 Tải tất cả TXT dưới dạng ZIP",
            data=zip_payload,
            file_name="ocr_results.zip",
            mime="application/zip",
            key=f"batch_zip_{run_signature}",
        )

    selected_label = st.selectbox(
        "Chọn file để xem kết quả",
        [item["label"] for item in documents],
        key=f"batch_result_select_{run_signature}",
    )
    selected = next(item for item in documents if item["label"] == selected_label)
    if selected["error"]:
        st.error(selected["error"])
        return

    st.text_area(
        "Nội dung OCR của file đã chọn:",
        value=selected["combined_text"],
        height=600,
        key=f"batch_text_{run_signature}_{selected['index']}",
    )
    st.download_button(
        "📥 Tải TXT của file này",
        data=selected["combined_text"],
        file_name=selected["output_name"],
        mime="text/plain; charset=utf-8",
        key=f"batch_txt_{run_signature}_{selected['index']}",
    )

    page_rows = []
    for record in selected["records"]:
        result = record["result"]
        page_rows.append(
            {
                "Trang": record["page_number"],
                "Trạng thái": result.status,
                "Confidence": round(result.confidence * 100, 1),
                "Số từ": len(result.boxes),
                "Số ký tự": len(result.text),
                "Biến thể": record.get("selected_variant", "current"),
            }
        )
    st.dataframe(pd.DataFrame(page_rows), width="stretch", hide_index=True)


def main():
    st.title("🔬 Tesseract OCR 5 Playground")
    st.caption(
        "OCR toàn trang cho PDF/ảnh; PDF mặc định được quét toàn bộ và ghép "
        "thành một nội dung TXT."
    )

    # Kiểm tra executable
    tess_exe = find_tesseract_executable()
    if not tess_exe:
        st.error(
            "⚠️ Không tìm thấy tesseract.exe trên máy. "
            "Vui lòng cài đặt Tesseract 5 (kèm ngôn ngữ vie) hoặc cấu hình biến môi trường TESSERACT_CMD."
        )
        st.info("Tải tại: https://github.com/UB-Mannheim/tesseract/wiki")
        return

    # Sidebar: Tham số OCR
    st.sidebar.header("Cấu hình OCR")
    lang = st.sidebar.selectbox(
        "Ngôn ngữ (lang)",
        ["vie", "vie+eng", "eng"],
        index=0,
        help=(
            "Chọn 'vie' để ưu tiên dấu tiếng Việt. Chỉ dùng 'vie+eng' khi tài liệu "
            "thực sự có nhiều nội dung tiếng Anh."
        ),
    )
    required_langs = [item.strip() for item in lang.split("+") if item.strip()]
    profiles = get_project_tessdata_profiles(required_langs)
    profile_labels = {
        "best": "best – chính xác hơn, chậm hơn",
        "fast": "fast – nhanh hơn",
        "current": "current – cấu hình cũ",
    }
    available_profile_names = list(profiles)
    if not available_profile_names:
        st.sidebar.error("Không tìm thấy model có đủ ngôn ngữ đã chọn.")
        return
    model_profile = st.sidebar.selectbox(
        "Model tessdata",
        available_profile_names,
        index=0,
        format_func=lambda value: profile_labels.get(value, value),
    )
    dpi = st.sidebar.slider("DPI (khi render PDF)", min_value=150, max_value=600, value=300, step=50)
    oem = st.sidebar.selectbox("OEM (Engine Mode)", [1, 0, 2, 3], index=0, help="1: Neural LSTM (Khuyên dùng)")
    psm = st.sidebar.selectbox(
        "PSM (Page Segmentation Mode)",
        [6, 3, 4, 11, 1, 12],
        index=0,
        help=(
            "6: biểu mẫu/khối văn bản; 3: tự động toàn trang; 4: một cột; "
            "11: chữ rải rác. Benchmark BBGN ưu tiên PSM 6."
        ),
    )
    prep_profile = st.sidebar.selectbox(
        "Tiền xử lý ảnh",
        ["sauvola", "grayscale", "auto", "none"],
        index=0,
        help=(
            "grayscale: khử đường màu dài, deskew và dùng Otsu nội bộ; "
            "sauvola: cùng bước làm sạch/deskew nhưng dùng Sauvola nội bộ, phù hợp "
            "nền không đều; auto: thêm CLAHE/threshold có điều kiện; none: ảnh gốc."
        ),
    )
    prep_mode = "grayscale" if prep_profile == "sauvola" else prep_profile
    thresholding_method = 2 if prep_profile == "sauvola" else 0
    st.sidebar.caption("Khuyến nghị BBGN: best + PSM 6 + Sauvola; DPI 300; lang vie.")
    adaptive_background_enabled = st.sidebar.checkbox(
        "Chuẩn hóa nền thông minh (Bước 1-3)",
        value=False,
        help=(
            "Chạy thêm một ứng viên cân bằng sáng hoặc trung hòa nền màu, sau đó "
            "chỉ chọn ứng viên khi điểm OCR vượt qua guardrail. Tắt để dùng nguyên "
            "luồng OCR hiện tại."
        ),
    )
    if adaptive_background_enabled:
        st.sidebar.info(
            "Chế độ thử nghiệm: có thể chạy tối đa 2 lượt Tesseract mỗi trang; "
            "ảnh hiện tại được giữ lại nếu ứng viên mới kém hơn."
        )

    tessdata_dir = find_tessdata_directory(profiles[model_profile], tess_exe, required_langs)
    if tessdata_dir:
        st.sidebar.success(f"Tessdata: {tessdata_dir}")
        st.sidebar.caption(
            f"Profile: {model_profile}; ngôn ngữ: {', '.join(required_langs)}; "
            f"threshold: {thresholding_method}"
        )
    else:
        st.sidebar.error(
            "Không tìm thấy tessdata có đủ: " + ", ".join(required_langs)
        )

    # File uploader
    uploaded_files = st.file_uploader(
        "Tải lên một hoặc nhiều tài liệu PDF/ảnh (PNG, JPG, TIFF, WEBP):",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "webp"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Vui lòng tải lên một hoặc nhiều file để bắt đầu kiểm thử.")
        return


    if len(uploaded_files) > 1:
        multiple_file_ocr_ui(
            uploaded_files,
            tess_exe=tess_exe,
            tessdata_dir=Path(tessdata_dir),
            dpi=dpi,
            lang=lang,
            oem=oem,
            psm=psm,
            prep_mode=prep_mode,
            prep_profile=prep_profile,
            thresholding_method=thresholding_method,
            model_profile=model_profile,
            adaptive_background_enabled=adaptive_background_enabled,
        )
        return

    uploaded_file = uploaded_files[0]

    file_bytes = uploaded_file.getvalue()
    is_pdf = uploaded_file.name.lower().endswith(".pdf")

    # Xác định phạm vi OCR. PDF mặc định luôn chạy toàn bộ trang; chế độ một
    # trang chỉ giữ lại để kiểm tra nhanh khi người dùng chủ động chọn.
    page_num = 1
    total_pages = 1
    ocr_scope = "all"

    if is_pdf:
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                total_pages = len(doc)
        except Exception as error:
            st.error(f"Không thể mở PDF: {error}")
            return
        if total_pages <= 0:
            st.error("PDF không có trang để OCR.")
            return
        scope_label = st.sidebar.radio(
            "Phạm vi OCR",
            ["Toàn bộ PDF", "Một trang để kiểm thử"],
            index=0,
            help="Mặc định xử lý lần lượt tất cả trang và ghép vào một file TXT.",
        )
        ocr_scope = "all" if scope_label == "Toàn bộ PDF" else "single"
        if ocr_scope == "single" and total_pages > 1:
            page_num = int(
                st.sidebar.number_input(
                    "Chọn trang PDF:", min_value=1, max_value=total_pages, value=1
                )
            )
        st.sidebar.markdown(f"**Số trang PDF:** {total_pages}")
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if raw_img is None:
            st.error("Không thể đọc định dạng hình ảnh đã tải lên.")
            return
        h, w = raw_img.shape[:2]
        st.sidebar.markdown(f"**Kích thước ảnh:** {w} x {h} px")

    signature_payload = {
        "source_sha256": hashlib.sha256(file_bytes).hexdigest(),
        "scope": ocr_scope,
        "selected_page": page_num if ocr_scope == "single" else None,
        "dpi": dpi,
        "lang": lang,
        "oem": oem,
        "psm": psm,
        "preprocessing": prep_profile,
        "thresholding_method": thresholding_method,
        "model_profile": model_profile,
        "adaptive_background": adaptive_background_enabled,
    }
    run_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    # Nút chạy OCR
    col_run, _ = st.columns([1, 4])
    run_clicked = col_run.button("🚀 Chạy Tesseract OCR", type="primary", width="stretch")

    if run_clicked:
        config = TesseractConfig(
            input_dir=Path("."),
            output_dir=Path("."),
            dpi=dpi,
            lang=lang,
            oem=oem,
            psm=psm,
            preprocessing=prep_mode,
            thresholding_method=thresholding_method,
            tessdata_dir=tessdata_dir,
            adaptive_background=adaptive_background_enabled,
        )

        progress = st.progress(0.0)
        status_text = st.empty()
        page_records = []

        if is_pdf:
            page_indices = (
                list(range(total_pages)) if ocr_scope == "all" else [page_num - 1]
            )
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for position, page_index in enumerate(page_indices, 1):
                    actual_page = page_index + 1
                    status_text.info(
                        f"Đang OCR trang {actual_page}/{total_pages} "
                        f"({position}/{len(page_indices)})..."
                    )
                    raw_page = render_pdf_page_to_image(doc[page_index], dpi=dpi)
                    page_records.append(
                        run_page_ocr(
                            raw_page,
                            config,
                            page_number=actual_page,
                            prep_mode=prep_mode,
                            adaptive_background_enabled=adaptive_background_enabled,
                            include_display_image=True,
                        )
                    )
                    progress.progress(position / len(page_indices))
        else:
            status_text.info("Đang OCR ảnh...")
            page_records.append(
                run_page_ocr(
                    raw_img,
                    config,
                    page_number=1,
                    prep_mode=prep_mode,
                    adaptive_background_enabled=adaptive_background_enabled,
                    include_display_image=True,
                )
            )
            progress.progress(1.0)

        status_text.success(f"Đã OCR xong {len(page_records)} trang.")
        st.session_state["last_ocr_payload"] = {
            "signature": run_signature,
            "records": page_records,
            "source_name": uploaded_file.name,
            "total_pages": total_pages,
            "scope": ocr_scope,
        }

    payload = st.session_state.get("last_ocr_payload")
    if not payload or payload.get("signature") != run_signature:
        st.info("Nhấn **Chạy Tesseract OCR** để bắt đầu quét toàn bộ tài liệu.")
        return

    page_records = payload["records"]
    if not page_records:
        st.error("Không có kết quả OCR.")
        return

    failed_records = [record for record in page_records if record["result"].status == "error"]
    if failed_records:
        failed_pages = ", ".join(str(record["page_number"]) for record in failed_records)
        st.error(f"OCR lỗi tại trang: {failed_pages}. Các trang còn lại vẫn được giữ.")

    st.warning(
        "Tesseract phù hợp nhất với chữ in. Chữ viết tay trong biểu mẫu có thể "
        "mất dấu hoặc nhận sai dù đã nạp đúng model tiếng Việt."
    )

    combined_sections = []
    for record in page_records:
        result = record["result"]
        page_label = record["page_number"]
        page_text = result.text if result.status != "error" else f"[LỖI OCR: {result.error}]"
        combined_sections.append(
            f"--- TRANG {page_label}/{payload['total_pages']} ---\n{page_text}".rstrip()
        )
    combined_text = "\n\n".join(combined_sections).strip()

    total_words = sum(len(record["result"].boxes) for record in page_records)
    total_chars = sum(len(record["result"].text) for record in page_records)
    total_seconds = sum(
        float(
            record.get(
                "total_processing_seconds",
                record["result"].duration_seconds
                + float(record["prep_meta"].get("processing_seconds", 0.0)),
            )
        )
        for record in page_records
    )
    confidence_weight = sum(max(1, len(record["result"].boxes)) for record in page_records)
    weighted_confidence = (
        sum(
            record["result"].confidence * max(1, len(record["result"].boxes))
            for record in page_records
        )
        / confidence_weight
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Trang đã OCR", f"{len(page_records)}/{payload['total_pages']}")
    m2.metric("Thời gian", f"{total_seconds:.2f} s")
    m3.metric("Độ tin cậy TB", f"{weighted_confidence * 100:.1f}%")
    m4.metric("Tổng số từ", total_words)
    m5.metric("Tổng ký tự", total_chars)

    page_summary = pd.DataFrame(
        [
            {
                "Trang": record["page_number"],
                "Trạng thái": record["result"].status,
                "Confidence": round(record["result"].confidence * 100, 1),
                "Số từ": len(record["result"].boxes),
                "Số ký tự": len(record["result"].text),
                "Biến thể": (
                    "đã xử lý"
                    if record.get("selected_variant") == "processed"
                    else "hiện tại"
                ),
                "Lý do chọn": record.get("decision", {}).get("reason", ""),
                "Thời gian (s)": round(
                    float(
                        record.get(
                            "total_processing_seconds",
                            record["result"].duration_seconds
                            + float(
                                record["prep_meta"].get("processing_seconds", 0.0)
                            ),
                        )
                    ),
                    2,
                ),
            }
            for record in page_records
        ]
    )

    tab_text, tab_boxes, tab_compare_img, tab_pages = st.tabs(
        ["📝 Toàn bộ văn bản", "📦 Bounding Boxes", "🖼️ So sánh Ảnh", "📊 Từng trang"]
    )

    with tab_text:
        st.subheader("Văn bản OCR toàn bộ tài liệu")
        st.caption(
            f"Đang hiển thị toàn bộ {len(page_records)} trang đã OCR; "
            "không cần chọn từng trang."
        )
        st.text_area("Nội dung trích xuất toàn bộ PDF:", value=combined_text, height=650)
        output_suffix = "full" if payload["scope"] == "all" else f"page{page_records[0]['page_number']}"
        download_label = (
            "📥 Tải xuống TXT toàn bộ PDF"
            if is_pdf and payload["scope"] == "all"
            else "📥 Tải xuống TXT"
        )
        st.download_button(
            download_label,
            data=combined_text,
            file_name=f"{Path(uploaded_file.name).stem}_{output_suffix}.txt",
            mime="text/plain",
        )

    with tab_boxes:
        st.caption("Chọn trang chỉ để xem ảnh và Bounding Boxes; không ảnh hưởng văn bản tổng hợp.")
        display_options = [record["page_number"] for record in page_records]
        display_page = st.selectbox(
            "Trang dùng để xem ảnh và Bounding Boxes",
            display_options,
            key="result_display_page",
        )
        display_record = next(
            record for record in page_records if record["page_number"] == display_page
        )
        display_result = display_record["result"]
        if is_pdf:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                display_raw = render_pdf_page_to_image(doc[display_page - 1], dpi=dpi)
        else:
            display_raw = raw_img
        display_processed = decode_png_bytes(display_record["selected_image_png"])

        st.subheader(f"Bounding Boxes – trang {display_page}")
        annotated = draw_bounding_boxes(display_processed, display_result.boxes)
        annotated_rgb = (
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            if len(annotated.shape) == 3
            else annotated
        )
        st.image(annotated_rgb, width="stretch")
        if display_result.boxes:
            df_boxes = pd.DataFrame([box.to_dict() for box in display_result.boxes])
            st.dataframe(df_boxes, width="stretch")

    with tab_compare_img:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Ảnh gốc – trang {display_page}**")
            raw_rgb = cv2.cvtColor(display_raw, cv2.COLOR_BGR2RGB)
            st.image(raw_rgb, width="stretch")
        with c2:
            variant_label = (
                "đã xử lý thông minh"
                if display_record.get("selected_variant") == "processed"
                else "pipeline hiện tại"
            )
            st.markdown(f"**Ảnh được chọn – {variant_label}**")
            st.image(display_processed, width="stretch")

    with tab_pages:
        st.subheader("Kết quả từng trang")
        st.dataframe(page_summary, width="stretch", hide_index=True)
        for record in page_records:
            with st.expander(
                f"Trang {record['page_number']} – {record['result'].status}",
                expanded=False,
            ):
                st.text(record["result"].text or "(Không nhận dạng được văn bản)")


if __name__ == "__main__":
    main()
