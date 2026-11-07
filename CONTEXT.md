# Context xây dựng hệ thống Tesseract OCR độc lập

## 1. Project root bắt buộc

Toàn bộ hệ thống mới phải được xây dựng bên trong:

```text
D:\Thuc-Tap\OCR\Tesseract_OCR
```

Không tạo source code, dependency, file tạm hoặc kết quả Tesseract bên ngoài
thư mục này, ngoại trừ việc đọc dữ liệu từ hệ thống OCR hiện có.

Các đường dẫn cố định:

```text
Project Tesseract:  D:\Thuc-Tap\OCR\Tesseract_OCR
PDF/ảnh nguồn:      D:\Thuc-Tap\OCR\input
Apple Vision output:D:\Thuc-Tap\OCR\output
Ground truth:       D:\Thuc-Tap\OCR\evaluation\manual_test_set.csv
Tesseract output:   D:\Thuc-Tap\OCR\Tesseract_OCR\output
Báo cáo so sánh:    D:\Thuc-Tap\OCR\Tesseract_OCR\comparison_results
```

## 2. Mục tiêu

Xây dựng hai chức năng độc lập trong cùng project:

1. Pipeline Tesseract OCR đọc PDF/ảnh từ `..\input` và ghi kết quả vào
   `Tesseract_OCR\output`.
2. Công cụ so sánh chỉ đọc kết quả Apple Vision từ `..\output` và kết quả
   Tesseract từ `Tesseract_OCR\output`, sau đó ghi báo cáo vào
   `Tesseract_OCR\comparison_results`.

Apple Vision và Tesseract không chạy đồng thời trong cùng pipeline. Người dùng
sẽ tự chạy Apple Vision trước hoặc sau. Comparator chỉ hoạt động khi các file
kết quả đã tồn tại.

## 3. Ràng buộc tách biệt

Source code bên trong `Tesseract_OCR`:

- Không sửa bất kỳ file nào trong `D:\Thuc-Tap\OCR` bên ngoài project root.
- Không sửa `web_app.py`, `ocr_pipeline.py`, `ocr_transport.py` hoặc iOS Server.
- Không import hoặc gọi Apple Vision/iPhone OCR Server.
- Không yêu cầu iPhone đang hoạt động.
- Không ghi vào `D:\Thuc-Tap\OCR\output`.
- Không ghi vào `D:\Thuc-Tap\OCR\input`.
- Không xóa, đổi tên hoặc di chuyển kết quả Apple Vision.
- Chỉ đọc ground truth và các báo cáo cũ khi cần đối chiếu.
- Không dùng chung file checkpoint hoặc file JSON với hệ thống Apple Vision.

Trước mỗi lần chạy, chương trình phải kiểm tra rằng input, Apple output,
Tesseract output và comparison output là bốn đường dẫn khác nhau.

## 4. Cấu trúc project

```text
D:\Thuc-Tap\OCR\Tesseract_OCR\
├── CONTEXT.md
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── tesseract_ocr/
│   ├── __init__.py
│   ├── config.py
│   ├── cli.py
│   ├── scanner.py
│   ├── preprocessing.py
│   ├── engine.py
│   ├── pipeline.py
│   ├── schemas.py
│   └── storage.py
├── comparison/
│   ├── __init__.py
│   ├── config.py
│   ├── cli.py
│   ├── loaders.py
│   ├── matching.py
│   ├── metrics.py
│   ├── report.py
│   └── comparison_app.py
├── tests/
│   ├── test_metrics.py
│   ├── test_matching.py
│   ├── test_preprocessing.py
│   ├── test_schema.py
│   ├── test_resume.py
│   └── test_read_only_apple_output.py
├── output/
├── comparison_results/
├── logs/
└── temp/
```

`output/`, `comparison_results/`, `logs/` và `temp/` phải được đưa vào
`.gitignore`, ngoại trừ file `.gitkeep` nếu cần giữ cấu trúc.

## 5. Quy trình tổng thể

```text
D:\Thuc-Tap\OCR\input
        │
        │ Tesseract OCR độc lập
        ▼
D:\Thuc-Tap\OCR\Tesseract_OCR\output

D:\Thuc-Tap\OCR\output              (Apple Vision, chỉ đọc)
        │
        ├──────────────┐
        │              │ comparator chỉ đọc
        ▼              ▼
Tesseract_OCR\output
        │
        ▼
Tesseract_OCR\comparison_results\<run_id>
```

Comparator không có nút hoặc hàm gọi OCR. Nó chỉ tải các TXT/JSON đã có.

## 6. Pipeline Tesseract OCR

### 6.1 Đầu vào

Quét đệ quy:

```text
..\input\**\*.pdf
..\input\**\*.png
..\input\**\*.jpg
..\input\**\*.jpeg
..\input\**\*.webp
..\input\**\*.tif
..\input\**\*.tiff
```

Giữ nguyên đường dẫn tương đối khi ghi kết quả. Ví dụ:

```text
..\input\PCCC\AT-01.pdf
    ↓
.\output\PCCC\AT-01_ocr.txt
.\output\PCCC\AT-01_ocr.json
```

### 6.2 Xử lý PDF

Mục tiêu là đánh giá Tesseract nên mặc định mọi trang PDF đều được render thành
ảnh rồi OCR, kể cả PDF có text native.

- Render bằng PyMuPDF.
- DPI cấu hình được, mặc định 300.
- Không lấy text native làm kết quả Tesseract.
- Có thể thêm `--allow-native-text`, nhưng phải tắt mặc định và kết quả đó phải
  được đánh dấu riêng, không đưa vào benchmark engine chính.

### 6.3 Tiền xử lý ảnh

Sử dụng OpenCV và NumPy. Hỗ trợ hai chế độ:

```text
--preprocessing none
--preprocessing auto
```

Chế độ `auto` gồm:

1. Phát hiện chiều 0/90/180/270 độ.
2. Deskew góc nghiêng nhỏ.
3. Xoay mở rộng canvas để không cắt nội dung.
4. Cắt viền tối nếu có.
5. Grayscale.
6. Tăng tương phản có điều kiện.
7. Adaptive threshold chỉ khi nền không đều.

Không áp dụng threshold bắt buộc cho mọi trang vì có thể làm mất dấu tiếng
Việt và nét chữ mảnh.

Lưu metadata:

```json
{
  "mode": "auto",
  "orientation_rotation": 90,
  "deskew_angle": -2.7,
  "contrast_applied": true,
  "threshold_applied": false,
  "processing_seconds": 0.18
}
```

### 6.4 Tesseract engine

Sử dụng Tesseract 5 qua `pytesseract`.

Cấu hình mặc định:

```text
lang: vie+eng
oem: 1
psm: 3
dpi: 300
```

Các tham số phải thay đổi được qua CLI. Hỗ trợ biến môi trường
`TESSERACT_CMD` để xác định executable trên Windows.

Trước khi OCR phải kiểm tra:

- Tesseract executable tồn tại.
- Lấy được phiên bản Tesseract.
- Có `vie.traineddata`.
- Có `eng.traineddata`.

Nếu thiếu dependency, dừng với thông báo cụ thể. Không tự động cài phần mềm hệ
thống hoặc language data.

### 6.5 Text, box và confidence

Sử dụng `pytesseract.image_to_data()` để lấy:

- Text.
- Bounding box.
- Confidence.
- Block, paragraph, line và word number.

Ghép từ thành dòng dựa trên `block_num`, `par_num`, `line_num`; không nối toàn
bộ từ thành một dòng duy nhất.

Confidence Tesseract thường ở thang `0..100`:

- Bỏ giá trị âm hoặc không hợp lệ.
- Chuyển về `0..1` khi lưu JSON.
- Confidence trang tính trung bình có trọng số theo số ký tự.

## 7. Đầu ra Tesseract

### 7.1 TXT

```text
--- [TRANG 1/2] ---
Nội dung OCR trang 1

--- [TRANG 2/2] ---
Nội dung OCR trang 2
```

### 7.2 JSON

```json
{
  "schema_version": "1.0",
  "engine": "tesseract",
  "engine_version": "5.x",
  "source_file": "PCCC/AT-01.pdf",
  "source_type": "pdf",
  "status": "success",
  "total_pages": 2,
  "processed_pages": 2,
  "configuration": {
    "dpi": 300,
    "language": "vie+eng",
    "oem": 1,
    "psm": 3,
    "preprocessing": "auto"
  },
  "pages": [
    {
      "page_number": 1,
      "page_count": 2,
      "method": "tesseract",
      "status": "success",
      "text": "Nội dung OCR",
      "confidence": 0.87,
      "image_width": 2480,
      "image_height": 3508,
      "duration_seconds": 2.31,
      "preprocessing": {},
      "boxes": [
        {
          "text": "Nội dung",
          "confidence": 0.91,
          "x": 100,
          "y": 120,
          "w": 420,
          "h": 38
        }
      ],
      "error": null
    }
  ]
}
```

## 8. Độ ổn định của pipeline Tesseract

- Ghi JSON và TXT theo cách atomic.
- Checkpoint sau mỗi trang.
- Một trang lỗi không làm mất các trang thành công.
- Chạy lại bỏ qua tài liệu hoàn chỉnh nếu file nguồn và cấu hình không đổi.
- Lưu SHA-256 của file nguồn và hash cấu hình.
- Nếu DPI, PSM, language hoặc preprocessing thay đổi thì không dùng nhầm cache.
- Hỗ trợ `--force` để chạy lại có chủ đích.
- Không xóa kết quả cũ trong luồng mặc định.
- Log phải có relative path, page number, thời gian, status và error.

## 9. Comparator hai thư mục kết quả

Comparator đọc:

```text
Apple Vision: ..\output
Tesseract:    .\output
```

Không hard-code hai đường dẫn trong logic; cung cấp default như trên nhưng cho
phép truyền qua CLI.

### 9.1 Ghép tài liệu

Ghép bằng đường dẫn tương đối sau khi:

- Chuẩn hóa `\` và `/`.
- Bỏ hậu tố `_ocr`.
- Không phân biệt hoa/thường của phần mở rộng.
- Giữ category/subdirectory trong khóa ghép.

Ví dụ cặp đúng:

```text
..\output\PCCC\AT-01_ocr.json
.\output\PCCC\AT-01_ocr.json
```

Không ghép chỉ theo `AT-01` nếu có thể có hai file cùng tên trong hai category.
Nếu không tìm thấy hoặc có nhiều ứng viên, ghi vào mismatch report, không tự
đoán.

Ghép trang bằng `page_number`, không giả định hai bên luôn có cùng số trang.

### 9.2 Load dữ liệu

- Ưu tiên JSON.
- Nếu thiếu JSON, parse TXT theo header `--- [TRANG x/y] ---`.
- Không sửa hoặc tự tạo JSON trong thư mục Apple Vision.
- Mọi dữ liệu chuyển đổi tạm thời chỉ được ghi vào
  `Tesseract_OCR\comparison_results` hoặc `Tesseract_OCR\temp`.

## 10. Phân biệt so sánh và đánh giá độ chính xác

### 10.1 Không có ground truth

Có thể tính:

- Pairwise character distance giữa Apple Vision và Tesseract.
- Pairwise word distance.
- Similarity giữa hai output.
- Chênh lệch số ký tự, từ và dòng.
- Trang chỉ một bên có kết quả.

Các số này chỉ cho biết **hai kết quả khác nhau bao nhiêu**. Không được gọi là
CER/WER chuẩn và không được kết luận engine nào chính xác hơn.

### 10.2 Có ground truth

Tính độc lập:

```text
Apple Vision ↔ Ground truth
Tesseract    ↔ Ground truth
```

Sau đó mới xác định:

- Apple Vision thắng.
- Tesseract thắng.
- Hòa trong ngưỡng cho phép.
- Cả hai cùng kém.

## 11. Ground truth

Đọc từ:

```text
..\evaluation\manual_test_set.csv
```

Định dạng mong muốn theo từng trang:

```text
document_id
relative_path
category
page_number
ground_truth_text
quality_tags
notes
```

`quality_tags` có thể gồm:

```text
clean
skewed
rotated
blurry
noisy
stamp
table
multi_column
small_text
handwriting
```

Trang chưa có ground truth vẫn xuất hiện trong coverage và pairwise report,
nhưng không được đưa vào bảng xếp hạng độ chính xác.

## 12. Metric

Khi có ground truth, tính theo trang, tài liệu, category và quality tag:

- CER bằng Levenshtein distance ký tự.
- WER bằng Levenshtein distance token.
- Word precision, recall, F1 có tính số lần xuất hiện, không dùng `set`.
- Similarity chỉ là chỉ số phụ.
- Tỷ lệ trang thành công, lỗi và rỗng.
- Số ký tự, từ và dòng.
- Thời gian trung bình, median, P90, P95 nếu JSON có duration.

Chỉ số nghiệp vụ bổ sung:

- Độ chính xác chuỗi số.
- Ngày tháng.
- Số tiền.
- Mã hồ sơ, biển số và mã số thuế.
- Diacritic error rate cho dấu tiếng Việt.

Hai chế độ chuẩn hóa:

- `strict`: Unicode NFC, giữ chữ hoa/thường và dấu câu.
- `normalized`: Unicode NFC, chuẩn hóa khoảng trắng và xuống dòng.

Không bỏ dấu tiếng Việt trong CER chính.

## 13. Báo cáo so sánh

Mỗi lần chạy tạo thư mục mới:

```text
.\comparison_results\<run_id>\
├── config.json
├── environment.json
├── matched_documents.csv
├── unmatched_apple_documents.csv
├── unmatched_tesseract_documents.csv
├── page_pairwise_results.csv
├── page_ground_truth_results.csv
├── document_results.csv
├── category_results.csv
├── quality_tag_results.csv
├── failures.csv
├── summary.json
└── report.html
```

`page_ground_truth_results.csv` phải có:

```text
document_id
relative_path
category
page_number
apple_status
tesseract_status
apple_chars
tesseract_chars
pairwise_similarity_pct
apple_cer_pct
tesseract_cer_pct
apple_wer_pct
tesseract_wer_pct
apple_word_f1_pct
tesseract_word_f1_pct
winner
ground_truth_available
```

Các cột CER/WER/F1 và `winner` để trống nếu không có ground truth.

## 14. Giao diện Streamlit riêng

File:

```text
.\comparison\comparison_app.py
```

Giao diện không được có chức năng gọi Apple Vision hoặc Tesseract. Nó chỉ đọc
output và báo cáo đã có.

### Tổng quan

- Tài liệu ghép thành công.
- Tài liệu chỉ có ở một bên.
- Coverage theo trang.
- CER/WER/F1 của từng engine khi có ground truth.
- Số trang Apple thắng, Tesseract thắng, hòa, cả hai cùng kém.
- Kết quả theo category và quality tag.

### Từng trang

Hiển thị:

```text
Apple Vision result | Tesseract result | Ground truth (nếu có)
```

Kèm diff, metric, confidence riêng, duration, status và error. Cho phép lọc
theo category, quality tag và winner.

Nếu hiển thị ảnh gốc, chỉ đọc PDF/ảnh từ `..\input`; không dùng ảnh đó để OCR
lại trong giao diện.

## 15. CLI yêu cầu

Mọi lệnh được chạy từ:

```powershell
cd D:\Thuc-Tap\OCR\Tesseract_OCR
```

### OCR bằng Tesseract

```powershell
python -m tesseract_ocr.cli `
  --input-dir ..\input `
  --output-dir .\output `
  --dpi 300 `
  --lang vie+eng `
  --oem 1 `
  --psm 3 `
  --preprocessing auto
```

Chạy thử:

```powershell
python -m tesseract_ocr.cli `
  --input-dir ..\input `
  --output-dir .\output `
  --max-documents 3
```

### So sánh hai thư mục

```powershell
python -m comparison.cli `
  --apple-dir ..\output `
  --tesseract-dir .\output `
  --ground-truth ..\evaluation\manual_test_set.csv `
  --report-dir .\comparison_results
```

Không có ground truth:

```powershell
python -m comparison.cli `
  --apple-dir ..\output `
  --tesseract-dir .\output `
  --report-dir .\comparison_results
```

### Mở giao diện

```powershell
streamlit run .\comparison\comparison_app.py
```

## 16. Dependency

`requirements.txt` dự kiến:

```text
pytesseract
opencv-python-headless
numpy
Pillow
PyMuPDF
pandas
streamlit
```

Tesseract executable, `vie.traineddata` và `eng.traineddata` là dependency hệ
thống, không phải package Python.

## 17. Kiểm thử bắt buộc

- Metric với chuỗi rỗng, Unicode tiếng Việt và từ lặp.
- Ghép đúng relative path và không ghép nhầm file trùng tên.
- Trang bị thiếu ở Apple hoặc Tesseract.
- Parse JSON và fallback TXT.
- Confidence `-1`, rỗng hoặc không phải số.
- Deskew không cắt nội dung.
- Checkpoint và resume.
- Atomic write.
- Input/output path không được trùng nhau.
- Hash toàn bộ `..\output` trước và sau khi chạy Tesseract/comparator phải giữ
  nguyên.
- Không có network request trong source code Tesseract và comparator.

## 18. Tiêu chí nghiệm thu

1. Tất cả source code mới nằm trong `D:\Thuc-Tap\OCR\Tesseract_OCR`.
2. Tesseract chạy được khi iPhone Server tắt.
3. Tesseract chỉ đọc `..\input` và chỉ ghi `.\output`.
4. Comparator chỉ đọc `..\output` và `.\output`.
5. Thư mục Apple Vision `D:\Thuc-Tap\OCR\output` không thay đổi.
6. Giữ nguyên cấu trúc category/subdirectory trong output Tesseract.
7. Có TXT và JSON theo từng trang.
8. Có checkpoint và resume.
9. Comparator xử lý được file/trang thiếu ở một phía.
10. Không kết luận engine thắng nếu thiếu ground truth.
11. Khi có ground truth, báo cáo CER/WER/F1 theo trang, tài liệu và category.
12. Có CSV, JSON, HTML và giao diện Streamlit riêng.
13. README mô tả cài đặt Tesseract trên Windows và toàn bộ lệnh chạy.
14. Test suite chạy thành công trước khi bàn giao.

## 19. Yêu cầu triển khai cuối cùng

- Không sửa bất kỳ file production nào ngoài `Tesseract_OCR`.
- Không tái sử dụng module có side effect từ hệ thống Apple Vision.
- Có thể tham khảo thuật toán metric hiện tại, nhưng implementation và test của
  project này phải tự chứa trong `Tesseract_OCR`.
- Không hard-code IP iPhone hoặc URL OCR Server.
- Không thêm dependency cloud.
- Mọi path phải dùng `pathlib.Path` và kiểm tra resolved path trước khi ghi.
- Không thực hiện thao tác xóa đệ quy trên thư mục `..\input` hoặc `..\output`.
- Báo cáo phải ghi rõ cấu hình Tesseract, phiên bản, DPI, PSM, OEM,
  preprocessing và coverage ground truth.
