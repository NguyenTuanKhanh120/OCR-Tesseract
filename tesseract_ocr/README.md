# Tesseract OCR Pipeline Độc Lập

Hệ thống OCR tài liệu tiếng Việt và tiếng Anh sử dụng **Tesseract 5**, hoạt động hoàn toàn độc lập với Apple Vision và iOS OCR Server.

> Với tài liệu chủ yếu bằng tiếng Việt, dùng `--lang vie` để ưu tiên dấu. Engine
> tự động dùng thư mục `tessdata` của dự án và từ chối chạy nếu thiếu model thay vì
> âm thầm fallback sang tiếng Anh. Chỉ dùng `vie+eng` khi tài liệu có nhiều tiếng Anh.

## 1. Yêu cầu hệ thống và Cài đặt

### Cài đặt thư viện Python:
```bash
pip install -r requirements.txt
```

### Cài đặt Tesseract OCR 5:
1. Tải bản cài đặt cho Windows: [UB-Mannheim Tesseract 5](https://github.com/UB-Mannheim/tesseract/wiki)
2. Trong trình cài đặt, chọn bổ sung gói dữ liệu ngôn ngữ **Vietnamese** (`vie`).
3. Đảm bảo file `vie.traineddata` và `eng.traineddata` nằm trong thư mục `tessdata`.
4. Thiết lập biến môi trường (hoặc truyền qua cờ `--tesseract-cmd`):
```powershell
setx TESSERACT_CMD "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## 2. Cách chạy CLI

### Chạy OCR toàn bộ kho tài liệu:
```powershell
python -m tesseract_ocr.cli `
  --input-dir ..\input `
  --output-dir ..\tesseract_output `
  --metadata-dir ..\tesseract_output_metadata `
  --dpi 300 `
  --lang vie+eng `
  --oem 1 `
  --psm 3 `
  --preprocessing auto
```

Khi có `--metadata-dir`, thư mục `--output-dir` chỉ chứa một file
`_ocr.txt` cho mỗi PDF; JSON checkpoint/metadata được lưu ở thư mục riêng.

### Chạy thử nghiệm 3 tài liệu:
```powershell
python -m tesseract_ocr.cli `
  --input-dir ..\input `
  --output-dir ..\tesseract_output `
  --max-documents 3
```

### Chạy lại có chủ đích (bỏ qua cache resume):
```powershell
python -m tesseract_ocr.cli `
  --input-dir ..\input `
  --output-dir ..\tesseract_output `
  --force
```

## 3. Cấu trúc và Tính năng cốt lõi

- **Tiền xử lý (`preprocessing.py`)**: Khử đường màu dài/mảnh trước khi chuyển xám, deskew theo trung vị nhiều dòng chữ, xoay có mở rộng canvas chống cắt chữ, cắt viền đen scan, tăng tương phản CLAHE có điều kiện và Adaptive Threshold khi nền loang/tối. Con dấu hoặc chữ màu cục bộ được giữ lại.
- **Render PDF (`tesseract_engine.py`)**: Render bằng PyMuPDF với DPI cấu hình được (mặc định 300 DPI). Trích xuất text, bounding box và chuẩn hóa confidence về thang `0..1`.
- **Checkpoint & Atomic Writing (`pipeline.py`)**:
  - Ghi file dạng atomic qua file tạm `.tmp` tránh lỗi nửa chừng khi mất điện/dừng đột ngột.
  - Tự động bỏ qua các tài liệu đã hoàn thành nếu mã hash nguồn và cấu hình không đổi.
- **Đầu ra chuẩn**: Tạo đồng thời file `.json` chi tiết và `.txt` với phân cách `--- [TRANG X/Y] ---`.
