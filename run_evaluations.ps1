# PowerShell Script to Run All Evaluations
# Usage: .\run_evaluations.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AML 2025 - Step 2 Evaluations" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Activate virtual environment
Write-Host "`n[1/5] Activating virtual environment..." -ForegroundColor Yellow
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1

# Find latest checkpoints
Write-Host "`n[2/5] Finding latest checkpoints..." -ForegroundColor Yellow

$mlp_ckpt = Get-ChildItem -Path "checkpoints\error_recognition\MLP\omnivore" -Filter "*.pt" | 
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName

$transformer_ckpt = Get-ChildItem -Path "checkpoints\error_recognition\Transformer\omnivore" -Filter "*.pt" | 
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName

$lstm_ckpt = Get-ChildItem -Path "checkpoints\error_recognition\LSTM\omnivore" -Filter "*.pt" | 
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName

Write-Host "  MLP: $mlp_ckpt" -ForegroundColor Green
Write-Host "  Transformer: $transformer_ckpt" -ForegroundColor Green
Write-Host "  LSTM: $lstm_ckpt" -ForegroundColor Green

# Check if checkpoints exist
if (-not $mlp_ckpt -or -not $transformer_ckpt -or -not $lstm_ckpt) {
    Write-Host "`n❌ ERROR: Some checkpoints are missing!" -ForegroundColor Red
    Write-Host "Please ensure all models are trained first." -ForegroundColor Red
    exit 1
}

# Part 2a: Error Type Analysis
Write-Host "`n[3/5] Running Error Type Analysis (Part 2a)..." -ForegroundColor Yellow

Write-Host "  - Analyzing MLP..." -ForegroundColor Cyan
python -m core.evaluate_error_types --variant MLP --backbone omnivore --split recordings --ckpt "$mlp_ckpt" --threshold 0.4 --save_csv

Write-Host "  - Analyzing Transformer..." -ForegroundColor Cyan
python -m core.evaluate_error_types --variant Transformer --backbone omnivore --split recordings --ckpt "$transformer_ckpt" --threshold 0.4 --save_csv

Write-Host "  - Analyzing LSTM..." -ForegroundColor Cyan
python -m core.evaluate_error_types --variant LSTM --backbone omnivore --split recordings --ckpt "$lstm_ckpt" --threshold 0.4 --save_csv

# Part 2b: Baseline Comparison
Write-Host "`n[4/5] Running Baseline Comparison (Part 2b)..." -ForegroundColor Yellow
python compare_baselines.py --split recordings --backbone omnivore --mlp_ckpt "$mlp_ckpt" --transformer_ckpt "$transformer_ckpt" --lstm_ckpt "$lstm_ckpt" --save_csv

# Summary
Write-Host "`n[5/5] ✅ All evaluations complete!" -ForegroundColor Green
Write-Host "`nResults saved to:" -ForegroundColor Cyan
Write-Host "  - results/error_type_analysis/" -ForegroundColor White
Write-Host "  - results/baseline_comparison.csv" -ForegroundColor White
Write-Host "`n========================================" -ForegroundColor Cyan

