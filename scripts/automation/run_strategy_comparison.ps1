
# SAC v444.2 Strategy Comparison Script
# 5つの異なる改善戦略を実行し、結果を比較

$strategies = @(
    @{ name = "strategy_1_aggressive_balance_reduction"; steps = 5000 },
    @{ name = "strategy_2_balanced_moderate"; steps = 5000 },
    @{ name = "strategy_3_reward_emphasis"; steps = 5000 },
    @{ name = "strategy_4_high_entropy_exploration"; steps = 5000 },
    @{ name = "strategy_5_conservative_tuning"; steps = 5000 }
)

Write-Host "Starting SAC v444.2 Strategy Comparison..." -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green

foreach ($strategy in $strategies) {
    $configPath = "config/sac_v444_strategies/$($strategy.name).json"
    $steps = $strategy.steps
    
    Write-Host "`nTraining with: $($strategy.name)" -ForegroundColor Cyan
    Write-Host "Config: $configPath" -ForegroundColor Cyan
    Write-Host "Steps: $steps" -ForegroundColor Cyan
    
    python quick_train_v444_optimized.py --config $configPath --steps $steps --analyze
    
    Write-Host "✓ Completed: $($strategy.name)" -ForegroundColor Green
    Start-Sleep -Seconds 3
}

Write-Host "`nAll strategies trained! Analyzing results..." -ForegroundColor Green
python scripts/analyze_optimization_results.py results

Write-Host "`n✓ Strategy comparison complete!" -ForegroundColor Green
