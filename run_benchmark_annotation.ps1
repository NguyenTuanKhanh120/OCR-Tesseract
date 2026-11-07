$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

& $python -m streamlit run `
    (Join-Path $projectRoot "tesseract_ocr\benchmark_annotation_app.py") `
    --server.headless true `
    --server.address 127.0.0.1 `
    --server.port 8503 `
    --browser.gatherUsageStats false
