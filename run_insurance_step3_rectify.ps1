$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$panelsDir = "D:\Thuc-Tap\OCR\insurance_region_pilot\step2_panels\1b_BHVC_02645"
$outputDir = "D:\Thuc-Tap\OCR\insurance_region_pilot\step3_rectified\1b_BHVC_02645"

& $python -m tesseract_ocr.perspective_corrector `
    --panels-dir $panelsDir `
    --output-dir $outputDir `
    --min-confidence 0.65 `
    --output-margin 8
