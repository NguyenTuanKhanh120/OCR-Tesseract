$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$preprocessedDir = "D:\Thuc-Tap\OCR\insurance_region_pilot\step5_preprocessed\1b_BHVC_02645"
$outputDir = "D:\Thuc-Tap\OCR\insurance_region_pilot\step6_ocr\1b_BHVC_02645"
$tessdataDir = Join-Path $projectRoot "tessdata_best"
$tesseract = "C:\Program Files\Tesseract-OCR\tesseract.exe"

& $python -m tesseract_ocr.region_ocr `
    --preprocessed-dir $preprocessedDir `
    --output-dir $outputDir `
    --tessdata-dir $tessdataDir `
    --tesseract-cmd $tesseract `
    --lang "vie+eng" `
    --psm 3 6 11 `
    --dpi 400 `
    --min-screening-score 0.45
