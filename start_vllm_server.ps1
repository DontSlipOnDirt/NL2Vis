# vLLM Server Launcher for Windows
# Quick start script for running vLLM with recommended settings

Write-Host "================================" -ForegroundColor Cyan
Write-Host "  vLLM Server Launcher" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$MODEL = "meta-llama/Llama-3.1-8B-Instruct"
$HOST = "0.0.0.0"
$PORT = 8000
$MAX_MODEL_LEN = 8192
$GPU_MEMORY_UTIL = 0.90

Write-Host "Starting vLLM server with:" -ForegroundColor Yellow
Write-Host "  Model: $MODEL" -ForegroundColor White
Write-Host "  URL: http://$HOST:$PORT" -ForegroundColor White
Write-Host "  Max Context: $MAX_MODEL_LEN tokens" -ForegroundColor White
Write-Host "  GPU Memory: $($GPU_MEMORY_UTIL * 100)%" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Check if vLLM is installed
try {
    python -c "import vllm" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: vLLM is not installed!" -ForegroundColor Red
        Write-Host "Install it with: uv add vllm" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "ERROR: Could not check vLLM installation" -ForegroundColor Red
    exit 1
}

# Start the server
try {
    python -m vllm.entrypoints.openai.api_server `
        --model $MODEL `
        --host $HOST `
        --port $PORT `
        --dtype auto `
        --gpu-memory-utilization $GPU_MEMORY_UTIL `
        --max-model-len $MAX_MODEL_LEN
} catch {
    Write-Host ""
    Write-Host "Server stopped." -ForegroundColor Yellow
}
