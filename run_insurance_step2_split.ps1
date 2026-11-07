$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$renderDir = "D:\Thuc-Tap\OCR\insurance_region_pilot\step1_rendered\1b_BHVC_02645"
$outputDir = "D:\Thuc-Tap\OCR\insurance_region_pilot\step2_panels\1b_BHVC_02645"

& $python -m tesseract_ocr.panel_splitter `
    --render-dir $renderDir `
    --output-dir $outputDir `
    --expected-panels 2 `
    --padding 20
