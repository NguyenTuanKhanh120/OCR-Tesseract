$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$inputPdf = "D:\Thuc-Tap\OCR\input\BAO_HIEM\GCN_BAO_HIEM\1b. BHVC 02645 (5-2-24).pdf"
$outputDir = "D:\Thuc-Tap\OCR\insurance_region_pilot\step1_rendered\1b_BHVC_02645"

& $python -m tesseract_ocr.document_renderer `
    --input-pdf $inputPdf `
    --output-dir $outputDir `
    --dpi 400
