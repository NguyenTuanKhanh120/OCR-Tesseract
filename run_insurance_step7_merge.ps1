$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$regionManifest = "D:\Thuc-Tap\OCR\insurance_region_pilot\step6_ocr\1b_BHVC_02645\region_ocr_manifest.json"
$outputDir = "D:\Thuc-Tap\OCR\insurance_region_pilot\step7_merged\1b_BHVC_02645"

& $python -m tesseract_ocr.document_text_merger `
    --region-manifest $regionManifest `
    --output-dir $outputDir
