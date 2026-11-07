$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$inputDir = "D:\Thuc-Tap\OCR\input\GIAO_NHAN_HANG_HOA\BB_GIAO_NHAN_HANG_HOA"
$outputDir = "D:\Thuc-Tap\OCR\tesseract_benchmark_10pdf"
$referenceDir = "D:\Thuc-Tap\OCR\output\GIAO_NHAN_HANG_HOA\BB_GIAO_NHAN_HANG_HOA"

& $python -m tesseract_ocr.benchmark_variants `
    --input-dir $inputDir `
    --output-dir $outputDir `
    --max-documents 10 `
    --max-pages-per-document 1 `
    --dpi 300 `
    --lang vie `
    --reference-dir $referenceDir
