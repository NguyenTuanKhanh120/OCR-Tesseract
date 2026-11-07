# OCR Comparison Tool (Apple Vision vs Tesseract 5)

Công cụ độc lập so sánh kết quả OCR giữa hệ thống **Apple Vision** (trên iPhone) và **Tesseract OCR 5**.

## 1. Nguyên Tắc An Toàn Dữ Liệu
- **Read-only tuyệt đối**: Mở các thư mục kết quả ở chế độ chỉ đọc.
- **Không can thiệp**: Không gọi, không khởi động lại và không chỉnh sửa bất kỳ file kết quả nào trong `output/` hay `input/`.
- **Báo cáo độc lập**: Mọi kết quả so sánh được ghi vào thư mục riêng: `comparison_results/<run_id>/`.

## 2. Cách chạy CLI

### Chế độ 1: So sánh khi CHƯA có Ground Truth (Pairwise Difference)
Đo lường mức độ tương đồng, độ chênh lệch ký tự, số từ, độ phủ giữa hai engine:
```powershell
python -m ocr_comparison.compare_outputs `
  --apple-dir ..\output `
  --tesseract-dir ..\tesseract_output `
  --report-dir ..\comparison_results
```

### Chế độ 2: So sánh khi CÓ Ground Truth
Tính toán CER, WER, F1 multiset và xác định engine chiến thắng theo từng trang:
```powershell
python -m ocr_comparison.compare_outputs `
  --apple-dir ..\output `
  --tesseract-dir ..\tesseract_output `
  --ground-truth ..\evaluation\manual_test_set.csv `
  --report-dir ..\comparison_results
```

## 3. Khởi chạy Giao Diện Trực Quan (Streamlit)
```powershell
streamlit run ocr_comparison/comparison_app.py
```
Giao diện cung cấp:
- **Dashboard KPI**: Tỷ lệ khớp tài liệu, độ phủ trang, độ tương đồng trung bình, bảng xếp hạng winner khi có ground truth.
- **Side-by-side Page Viewer**: Xem văn bản Apple Vision song song Tesseract 5 và Ground Truth, kèm tô màu highlight các từ khác biệt (diff view).
- **Phân nhóm**: Bảng kết quả theo Category và Quality Tag (ảnh mờ, nghiêng, xoay, bảng biểu...).
- **Báo cáo lỗi**: Danh sách các trang thiếu sót, lỗi hoặc có độ lệch trên 50%.

## 4. Danh mục 11 Báo Cáo Sinh Ra (`comparison_results/<run_id>/`)
1. `config.json`
2. `matched_documents.csv`
3. `unmatched_apple_documents.csv`
4. `unmatched_tesseract_documents.csv`
5. `page_pairwise_results.csv`
6. `page_ground_truth_results.csv`
7. `document_results.csv`
8. `category_results.csv`
9. `quality_tag_results.csv`
10. `failures.csv`
11. `summary.json`
12. `report.html`
