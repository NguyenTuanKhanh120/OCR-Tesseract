$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$rectifiedDir = "D:\Thuc-Tap\OCR\insurance_region_pilot\step3_rectified\1b_BHVC_02645"
$outputDir = "D:\Thuc-Tap\OCR\insurance_region_pilot\step4_masked\1b_BHVC_02645"

& $python -m tesseract_ocr.artifact_masker `
    --rectified-dir $rectifiedDir `
    --output-dir $outputDir `
    --padding 12
