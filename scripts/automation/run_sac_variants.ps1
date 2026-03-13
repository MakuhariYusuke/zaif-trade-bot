# SAC v444 Variant Training Commands
# Generated for systematic parameter optimization

$configs = @(
    @{ config = "sac_v444_2_balance_penalty_50"; steps = 2000; }
    @{ config = "sac_v444_2_balance_penalty_100"; steps = 2000; }
    @{ config = "sac_v444_2_balance_penalty_150"; steps = 2000; }
    @{ config = "sac_v444_2_balance_penalty_200"; steps = 2000; }
    @{ config = "sac_v444_2_balance_penalty_300"; steps = 2000; }
    @{ config = "sac_v444_2_balance_penalty_500"; steps = 2000; }
)

foreach ($item in $configs) {
    $config = $item.config
    $steps = $item.steps
    Write-Host "Training with config: $config for $steps steps" -ForegroundColor Green
    python quick_train_v444.py --config "config/sac_v444_variants/$config.json" --steps $steps
    Start-Sleep -Seconds 5
}