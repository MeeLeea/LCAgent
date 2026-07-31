# LangChain Agent Launcher
# Usage: .\LCA.ps1 [-build|-run|-all] or .\LCA.ps1 [build|run|all]

param(
    [Parameter(Position=0)]
    [string]$Action = "all",
    
    [switch]$build,
    [switch]$run,
    [switch]$all
)

# 处理参数
if ($build) { $Action = "build" }
if ($run) { $Action = "run" }
if ($all) { $Action = "all" }

# 标准化 Action 值（支持 --build 这样的写法）
$Action = $Action.ToLower().TrimStart('-')

# 验证 Action 值
if ($Action -notin @("all", "build", "run")) {
    Write-Host "X Invalid action: $Action" -ForegroundColor Red
    Write-Host "  Valid actions: all, build, run" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Cyan
    Write-Host "  .\LCA.ps1              # all (default)"
    Write-Host "  .\LCA.ps1 build        # build only"
    Write-Host "  .\LCA.ps1 run          # run only"
    Write-Host "  .\LCA.ps1 -build       # build only"
    Write-Host "  .\LCA.ps1 -run         # run only"
    exit 1
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$WebDir = Join-Path $ProjectRoot "web"
$TauriDir = Join-Path $ProjectRoot "src-tauri"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$CargoPath = Join-Path $env:USERPROFILE "AppData\Local\cargo\bin"

function Write-Step($msg) {
    Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] " -NoNewline -ForegroundColor Gray
    Write-Host $msg -ForegroundColor Cyan
}

function Write-OK($msg) { Write-Host "OK $msg" -ForegroundColor Green }
function Write-Fail($msg) { Write-Host "X $msg" -ForegroundColor Red }
function Write-Hint($msg) { Write-Host "   $msg" -ForegroundColor Gray }

function Build-Frontend {
    Write-Step "Build Web Frontend"
    Push-Location $WebDir
    npm run build
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { Write-Fail "Web build failed"; exit 1 }
    Write-OK "Web build completed"
}

function Build-Tauri {
    Write-Step "Build Tauri Desktop (Debug mode)"
    Write-Hint "Incremental build, usually 10-30 seconds"
    
    if (Test-Path $CargoPath) { $env:PATH = "$CargoPath;$env:PATH" }
    
    Push-Location $TauriDir
    cargo build
    $code = $LASTEXITCODE
    Pop-Location
    
    if ($code -ne 0) { Write-Fail "Tauri build failed"; exit 1 }
    Write-OK "Tauri build completed"
}

function Start-Backend {
    Write-Step "Start Backend Services"
    if (-not (Test-Path $PythonExe)) {
        Write-Fail "Python venv not found: $PythonExe"
        exit 1
    }
    
    wt -w 0 `
        nt -d "$ProjectRoot" --title "API Server" cmd /k "`"$PythonExe`" -m api.server" `; `
        nt -d "$ProjectRoot" --title "Scheduler" cmd /k "`"$PythonExe`" -m scheduler.run" `; `
        nt -d "$ProjectRoot" --title "Feishu" cmd /k "`"$PythonExe`" -m remote.feishu.remote_agent_control"
    
    Write-OK "Backend services started (3 tabs)"
    
    # 等待 API Server 就绪（健康检查）
    Write-Hint "Waiting for API Server to be ready..."
    $maxWait = 30
    $waited = 0
    $ready = $false
    
    while ($waited -lt $maxWait) {
        Start-Sleep -Seconds 1
        $waited++
        try {
            $null = Invoke-RestMethod -Uri "http://127.0.0.1:8001/api/health" -Method Get -TimeoutSec 2 -ErrorAction Stop
            $ready = $true
            break
        } catch {
            # 继续等待
        }
    }
    
    if ($ready) {
        Write-OK "API Server ready (took ${waited}s)"
    } else {
        Write-Fail "API Server not ready after ${maxWait}s"
        Write-Hint "Check the API Server tab for errors"
    }
}

function Start-Desktop {
    Write-Step "Start Desktop App"
    
    # 优先使用 Debug 版本，如果不存在则使用 Release 版本
    $debugExe = Join-Path $TauriDir "target\debug\langchainagent.exe"
    $releaseExe = Join-Path $TauriDir "target\release\langchainagent.exe"
    
    if (Test-Path $debugExe) {
        $exePath = $debugExe
        $version = "Debug"
    } elseif (Test-Path $releaseExe) {
        $exePath = $releaseExe
        $version = "Release"
        Write-Hint "Debug build not found, using Release build"
    } else {
        Write-Fail "Desktop app not found"
        Write-Hint "Debug: $debugExe"
        Write-Hint "Release: $releaseExe"
        Write-Hint "Run: .\LCA.ps1 build"
        exit 1
    }
    
    Start-Process -FilePath $exePath -WorkingDirectory $ProjectRoot
    Write-OK "Desktop app started ($version mode)"
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  LangChain Agent Launcher" -ForegroundColor Cyan
Write-Host "  Mode: $Action" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

switch ($Action) {
    "build" { Build-Frontend; Build-Tauri }
    "run" { Start-Backend; Start-Desktop }
    "all" { Build-Frontend; Build-Tauri; Start-Backend; Start-Desktop }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Done!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

if ($Action -eq "build") {
    Write-Hint "Next: .\LCA.ps1 run"
} elseif ($Action -eq "run") {
    Write-Hint "Rebuild: .\LCA.ps1 build"
}
Write-Host ""
