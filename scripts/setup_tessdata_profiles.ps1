param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$profiles = @(
    @{ Name = "fast"; BaseUrl = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main" },
    @{ Name = "best"; BaseUrl = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main" }
)

foreach ($profile in $profiles) {
    $targetDir = Join-Path $projectRoot ("tessdata_" + $profile.Name)
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null

    foreach ($language in @("vie", "eng", "osd")) {
        $targetFile = Join-Path $targetDir ($language + ".traineddata")
        if ($Force -or -not (Test-Path -LiteralPath $targetFile)) {
            $tempFile = $targetFile + ".download"
            Invoke-WebRequest `
                -Uri ($profile.BaseUrl + "/" + $language + ".traineddata") `
                -OutFile $tempFile
            Move-Item -LiteralPath $tempFile -Destination $targetFile -Force
        }
    }
}

Get-ChildItem -LiteralPath $projectRoot -Directory -Filter "tessdata_*" |
    ForEach-Object {
        Get-ChildItem -LiteralPath $_.FullName -File -Filter "*.traineddata" |
            Select-Object @{Name="Profile";Expression={$_.Directory.Name}}, Name, Length
    }
