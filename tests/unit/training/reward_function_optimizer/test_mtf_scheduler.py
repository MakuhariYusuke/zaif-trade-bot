from pathlib import Path

from ztb.trading.environment.components.reward.mtf_weight_manager import (
    MTFWeightManager,
)
from ztb.training.reward_function_optimizer.mtf_scheduler import (
    MTFScheduler,
    MTFSchedulerConfig,
)


def test_mtf_scheduler_apply_dry_run(tmp_path: Path):
    base_cfg = tmp_path / "base_config.json"
    base_cfg.write_text(
        '{"training": {"model_name": "mtf_test"}, "multi_timeframe": {"feature_weights": {"1min":0.3, "5min":0.6, "15min":0.1}}}'
    )
    cfg = MTFSchedulerConfig(
        base_config=str(base_cfg),
        out_dir=str(tmp_path),
        candidates=3,
        per_seed=1,
        timesteps=100,
    )
    mgr = MTFWeightManager(config={})
    scheduler = MTFScheduler(mgr, cfg)
    before = mgr.get_weights()
    best = scheduler.run_once(dry_run=True, apply=True)
    assert best is not None
    after = mgr.get_weights()
    # weights should have changed away from defaults after applying candidate
    assert abs(sum(after.values()) - 1.0) < 1e-9
    assert before != after


def test_scheduler_stage_change_callback(tmp_path: Path):
    base_cfg = tmp_path / "base_config.json"
    base_cfg.write_text(
        '{"training": {"model_name": "mtf_test"}, "multi_timeframe": {"feature_weights": {"1min":0.3, "5min":0.6, "15min":0.1}}}'
    )
    cfg = MTFSchedulerConfig(
        base_config=str(base_cfg),
        out_dir=str(tmp_path),
        candidates=3,
        per_seed=1,
        timesteps=100,
    )
    mgr = MTFWeightManager(config={})
    scheduler = MTFScheduler(mgr, cfg)
    bcm_cfg = type("C", (), {})()
    bcm_cfg.curriculum_stage = "forced_balance"
    from ztb.trading.environment.components.reward.balance_curriculum import (
        BalanceCurriculumManager,
    )

    bcm = BalanceCurriculumManager(bcm_cfg)

    cb = scheduler.create_stage_change_callback(
        stage_filter=["balanced_transition"], dry_run=True
    )
    bcm.add_stage_change_listener(cb)
    # simulate stage change
    bcm._progress_to_stage("balanced_transition", step=200)
    # callback should have applied scheduler once; manager weights should be updated
    after = mgr.get_weights()
    assert abs(sum(after.values()) - 1.0) < 1e-9
