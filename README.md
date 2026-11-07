# Hệ Thống Tesseract OCR Độc Lập & So Sánh Apple Vision vs Tesseract

Kho mã nguồn độc lập phục vụ chạy Tesseract OCR 5 trên tài liệu scan/PDF và so sánh kết quả đa chiều với Apple Vision.

## Cấu Trúc Thư Mục
```text
Tesseract_OCR/
├── tesseract_ocr/               # Module OCR độc lập sử dụng Tesseract 5
│   ├── config.py                # Cấu hình và hash kiểm soát resume
│   ├── scanner.py               # Quét đệ quy tài liệu PDF / ảnh nguồn
│   ├── preprocessing.py         # Tiền xử lý OpenCV (deskew, xoay, tương phản)
│   ├── tesseract_engine.py      # Giao tiếp Tesseract 5 và PyMuPDF renderer
│   ├── pipeline.py              # Điều phối checkpoint, atomic write, resume
│   ├── schemas.py               # Schemas đầu ra JSON/TXT chuẩn
│   ├── cli.py                   # Giao diện dòng lệnh chạy OCR
│   └── tests/                   # Bộ kiểm thử tự động
├── ocr_comparison/              # Module so sánh hai thư mục kết quả
│   ├── config.py                # Cấu hình so sánh và bảo đảm read-only
│   ├── loaders.py               # Nạp kết quả Apple, Tesseract và Ground Truth
│   ├── matching.py              # Khớp nối tài liệu và số trang theo category
│   ├── metrics.py               # Thuật toán CER, WER, F1 multiset, Business metrics
│   ├── report.py                # Tạo 11 tệp báo cáo (CSV, JSON, HTML)
│   ├── compare_outputs.py       # CLI chạy so sánh độc lập
│   ├── comparison_app.py        # Dashboard Streamlit trực quan side-by-side
│   └── tests/                   # Bộ kiểm thử tự động (khớp nối, metric, hash bất biến)
├── requirements.txt             # Danh sách thư viện Python
└── CONTEXT_COMPARE_APPLE_VISION_TESSERACT.md # Đặc tả thiết kế chi tiết
```

## Khởi động Nhanh

### 1. Cài đặt thư viện:
```powershell
pip install -r requirements.txt
```

Tải model tiếng Việt, tiếng Anh và nhận diện hướng trang. Các file model lớn
không lưu trong Git để repository gọn hơn:

```powershell
.\scripts\setup_tessdata_profiles.ps1
```

### 2. Chạy Tesseract OCR độc lập:
```powershell
python -m tesseract_ocr.cli --input-dir ..\input --output-dir ..\tesseract_output
```

### 2.1. Xuất một file TXT cho mỗi PDF (giống iOS output):
```powershell
python -m tesseract_ocr.cli `
  --input-dir ..\input `
  --output-dir ..\tesseract_output `
  --metadata-dir ..\tesseract_output_metadata
```

JSON checkpoint được lưu riêng trong `tesseract_output_metadata`; thư mục
`tesseract_output` chỉ chứa một file `_ocr.txt` cho mỗi PDF.

### 3. Chạy So Sánh Kết Quả:
```powershell
python -m ocr_comparison.compare_outputs --apple-dir ..\output --tesseract-dir ..\tesseract_output
```

### 4. Mở Dashboard Streamlit:
```powershell
streamlit run ocr_comparison/comparison_app.py
```

### 5. Chạy toàn bộ Unit Tests:
```powershell
python -m unittest discover -s tesseract_ocr/tests -p "test_*.py" -v
python -m unittest discover -s ocr_comparison/tests -p "test_*.py" -v
```

## Benchmark tessdata_fast và tessdata_best

Tải riêng hai profile model, không ghi đè model cũ:

```powershell
.\scripts\setup_tessdata_profiles.ps1
```

Chạy ma trận 10 PDF gồm `fast/best × PSM 3/6/11 × grayscale/Sauvola`:

```powershell
.\run_benchmark.ps1
```

Kết quả nằm tại `D:\Thuc-Tap\OCR\tesseract_benchmark_10pdf`. CER/WER chỉ
được tính khi các file trong thư mục `ground_truth` đã được xác nhận thủ công;
confidence, độ phủ và similarity với OCR iOS không được xem là độ chính xác.

Xác nhận ground truth trên giao diện tại `http://127.0.0.1:8503`:

```powershell
.\run_benchmark_annotation.ps1
```

Giao diện OCR chính ưu tiên cấu hình chất lượng cho biểu mẫu BBGN:
`tessdata_best + PSM 6 + Sauvola + vie + 300 DPI`. Các lựa chọn `fast`,
`grayscale`, PSM 3/4/11 vẫn được giữ để thử theo từng loại tài liệu.

Khi tải PDF lên giao diện chính, phạm vi mặc định là **Toàn bộ PDF**. Hệ thống
xử lý lần lượt tất cả trang, hiển thị tiến độ và xuất một TXT có đánh dấu
`--- TRANG x/y ---`. Chế độ một trang chỉ dùng khi chủ động chọn kiểm thử nhanh.

## OCR theo vùng - Bước 1: render toàn bộ PDF

Bước này chỉ render đầy đủ mọi trang thành PNG và ghi `render_manifest.json`;
chưa chia khung, chưa chỉnh phối cảnh và chưa OCR:

```powershell
.\run_insurance_step1_render.ps1
```

CLI dùng cho PDF khác:

```powershell
python -m tesseract_ocr.document_renderer `
  --input-pdf "D:\path\document.pdf" `
  --output-dir "D:\path\rendered" `
  --dpi 400
```

## OCR theo vùng - Bước 2: chia panel

Bước này dùng ảnh của Bước 1 để phát hiện các khung lớn, lưu từng panel và
ảnh overlay kiểm tra. Chưa chỉnh phối cảnh, chưa tiền xử lý OCR và chưa chạy
Tesseract:

```powershell
.\run_insurance_step2_split.ps1
```

CLI hỗ trợ tài liệu có 2 hoặc 3 khung:

```powershell
python -m tesseract_ocr.panel_splitter `
  --render-dir "D:\path\step1_rendered" `
  --output-dir "D:\path\step2_panels" `
  --expected-panels 3 `
  --padding 20
```

## OCR theo vùng - Bước 3: chỉnh nghiêng và phối cảnh

Bước này đọc từng panel của Bước 2, phát hiện bốn cạnh gần mép khung rồi
biến đổi phối cảnh về hình chữ nhật. Mỗi panel có ảnh đã chỉnh, overlay bốn
góc và điểm tin cậy trong `perspective_manifest.json`. Trường hợp không đủ
tin cậy được đánh dấu `needs_review`. Chưa loại logo/QR/barcode và chưa OCR:

```powershell
.\run_insurance_step3_rectify.ps1
```

CLI dùng cho bộ panel khác:

```powershell
python -m tesseract_ocr.perspective_corrector `
  --panels-dir "D:\path\step2_panels" `
  --output-dir "D:\path\step3_rectified" `
  --min-confidence 0.65 `
  --output-margin 8
```

## OCR theo vùng - Bước 4: che logo, QR và barcode

Bước này đọc ảnh đã chỉnh phối cảnh, dò logo màu ở mép trên, QR và barcode
1D rồi che các vùng đó bằng nền trắng trên một bản sao. Ảnh gốc được giữ
nguyên; mỗi panel có overlay và tọa độ vùng che trong
`artifact_mask_manifest.json`. Chưa xử lý con dấu, nền ảnh và chưa OCR:

```powershell
.\run_insurance_step4_mask.ps1
```

CLI dùng cho bộ panel khác:

```powershell
python -m tesseract_ocr.artifact_masker `
  --rectified-dir "D:\path\step3_rectified" `
  --output-dir "D:\path\step4_masked" `
  --padding 12
```

## OCR theo vùng - Bước 5: xử lý nền màu và tăng tương phản

Bước này tự phát hiện nền vàng để chọn kênh xanh lá; tài liệu không vàng dùng
ảnh xám nhằm giữ nét bút màu. Ảnh sau đó được tăng tương phản bằng CLAHE và
tạo thêm bản nhị phân Sauvola. Manifest ghi kênh đã chọn, mức ám vàng, độ
tương phản và tỷ lệ foreground. Chưa chọn PSM và chưa chạy OCR:

```powershell
.\run_insurance_step5_background.ps1
```

CLI dùng cho bộ panel khác:

```powershell
python -m tesseract_ocr.background_preprocessor `
  --masked-dir "D:\path\step4_masked" `
  --output-dir "D:\path\step5_preprocessed" `
  --channel auto `
  --yellow-threshold 0.08 `
  --clahe-clip-limit 2.2 `
  --sauvola-window 41 `
  --sauvola-k 0.20
```

## OCR theo vùng - Bước 6: OCR và chọn PSM theo từng panel

Bước này chạy `tessdata_best + vie+eng` trên cả ảnh CLAHE và Sauvola với
PSM 3/6/11. Mỗi panel có 6 ứng viên; hệ thống lưu toàn bộ TXT/JSON rồi chọn
một ứng viên bằng confidence, độ phủ và tỷ lệ ký tự hợp lệ. Điểm sàng lọc
không phải độ chính xác; CER/WER cần ground truth. Chưa ghép TXT toàn PDF:

```powershell
.\run_insurance_step6_ocr.ps1
```

CLI dùng cho bộ panel khác:

```powershell
python -m tesseract_ocr.region_ocr `
  --preprocessed-dir "D:\path\step5_preprocessed" `
  --output-dir "D:\path\step6_ocr" `
  --tessdata-dir ".\tessdata_best" `
  --lang "vie+eng" `
  --psm 3 6 11 `
  --dpi 400
```

## OCR theo vùng - Bước 7: ghép một TXT cho toàn PDF

Bước này đọc duy nhất các file `selected.txt` của Bước 6, kiểm tra đủ trang
và panel rồi ghép theo thứ tự trang -> panel. Đầu ra có một file `_ocr.txt`
cho mỗi PDF, dùng phân cách `--- [TRANG X/Y] ---` giống đầu ra hiện có của
hệ thống. `merge_manifest.json` lưu checksum và nguồn của từng vùng:

```powershell
.\run_insurance_step7_merge.ps1
```

CLI dùng cho tài liệu khác:

```powershell
python -m tesseract_ocr.document_text_merger `
  --region-manifest "D:\path\step6_ocr\region_ocr_manifest.json" `
  --output-dir "D:\path\step7_merged"
```

## Giao diện web OCR toàn trang

Giao diện web dùng luồng OCR toàn trang. Có thể chọn một hoặc nhiều PDF/ảnh;
các file được xử lý tuần tự và PDF luôn được quét toàn bộ trang. Mỗi tài liệu
có một TXT riêng, đồng thời có thể tải toàn bộ kết quả dưới dạng ZIP. Khi chỉ
chọn một file, giao diện vẫn hiển thị ảnh tiền xử lý và bounding boxes chi tiết.
Pipeline bảo hiểm 7 bước không xuất hiện trên giao diện; mã thử nghiệm vẫn
được giữ riêng để đối chiếu:

```powershell
.\.venv\Scripts\python.exe -m streamlit run tesseract_ocr\app.py `
  --server.port 8501
```
