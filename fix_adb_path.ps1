# Required paths
$sdkPath = "C:\Users\Umar\AppData\Local\Android\sdk\platform-tools"
$flutterPath = "C:\src\flutter\bin"

function Add-To-User-Path {
    param([string]$newFolder)
    $path = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($path -split ';' -notcontains $newFolder) {
        $newPath = $path + ";" + $newFolder
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        return $true
    }
    return $false
}

$fixedSdk = Add-To-User-Path $sdkPath
$fixedFlutter = Add-To-User-Path $flutterPath

if ($fixedSdk -or $fixedFlutter) {
    Write-Host "SUCCESS: Environment paths updated." -ForegroundColor Green
    Write-Host "1. Added Platform Tools: $fixedSdk"
    Write-Host "2. Added Flutter Bin: $fixedFlutter"
    Write-Host "`nIMPORTANT: You MUST restart VS Code (all windows) for these changes to take effect." -ForegroundColor Yellow
}
else {
    Write-Host "Environment paths were already correctly configured." -ForegroundColor Cyan
}

# Also ensure local .env exists (already done, but good to check)
if (-not (Test-Path ".env")) {
    Write-Host "Creating missing .env file..."
    "API_BASE_URL=http://emotion-api-env.eba-xkn3z6bd.us-east-1.elasticbeanstalk.com`nOPENAI_API_KEY=`nOPENAI_MODEL=gpt-4o-mini" | Out-File -FilePath ".env" -Encoding utf8
}
