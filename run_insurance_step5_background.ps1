$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$maskedDir = "D:\Thuc-Tap\OCR\insurance_region_pilot\step4_masked\1b_BHVC_02645"
$outputDir = "D:\Thuc-Tap\OCR\insurance_region_pilot\step5_preprocessed\1b_BHVC_02645"

& $python -m tesseract_ocr.background_preprocessor `
    --masked-dir $maskedDir `
    --output-dir $outputDir `
    --channel auto `
    --yellow-threshold 0.08 `
    --clahe-clip-limit 2.2 `
    --clahe-tile-size 8 `
    --sauvola-window 41 `
    --sauvola-k 0.20
