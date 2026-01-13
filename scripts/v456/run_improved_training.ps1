#!/usr/bin/env pwsh
# Week 4 改善版訓練・検証スクリプト (PowerShell)

$projectRoot = "c:\Users\Admin\dev\zaif-trade-bot"
$python = "$projectRoot\.venv\Scripts\python.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Week 4 改善版訓練 (Stage 1)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "改善内容:" -ForegroundColor Yellow
Write-Host "✓ 初期資金: 124円 → 50,000円" -ForegroundColor Yellow
Write-Host "✓ drawdown_limit: 0.1 → 0.3 (30%)" -ForegroundColor Yellow
Write-Host "✓ max_steps: ∞ → 500" -ForegroundColor Yellow
Write-Host "✓ 報酬関数: スケーリング + アクション奨励" -ForegroundColor Yellow
Write-Host "✓ 特徴量: 実OHLCV指標追加" -ForegroundColor Yellow
Write-Host "✓ 千ステップ統計: コールバック実装" -ForegroundColor Yellow
Write-Host ""

# 訓練実行
Write-Host "ステップ1: 改善版モデル訓練中..." -ForegroundColor Green
Write-Host "実行時間: 約1時間 (30,000ステップ)" -ForegroundColor Green
Write-Host ""

& $python scripts/v456/train_mlp_v456_improved.py --timesteps 30000

$trainStatus = $LASTEXITCODE

if ($trainStatus -eq 0) {
    Write-Host ""
    Write-Host "✓ 訓練完了" -ForegroundColor Green
    Write-Host ""
    Write-Host "ステップ2: 改善版検証中..." -ForegroundColor Green
    Write-Host ""
    
    & $python analysis/validate_week4_improved.py
    
    $valStatus = $LASTEXITCODE
    
    if ($valStatus -eq 0 -or $valStatus -eq 1) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "検証完了" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "結果確認" -ForegroundColor Yellow
        Write-Host "✓ models/week4_improved/ に最新モデル保存" -ForegroundColor Yellow
        Write-Host "✓ 統計情報を千ステップごとに記録" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "次のステップ" -ForegroundColor Cyan
        Write-Host "- Issue改善を確認" -ForegroundColor Cyan
        Write-Host "- 必要に応じて段階2へ進む" -ForegroundColor Cyan
    }
} else {
    Write-Host ""
    Write-Host "❌ 訓練エラー" -ForegroundColor Red
}
