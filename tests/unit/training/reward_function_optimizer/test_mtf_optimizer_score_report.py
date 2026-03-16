import json
from pathlib import Path

from ztb.trading.environment.components.reward.mtf_weight_manager import (
    MTFWeightManager,
)
from ztb.training.reward_function_optimizer.mtf_optimizer import MTFOptimizer
from ztb.training.reward_function_optimizer.mtf_scheduler import (
    MTFScheduler,
    MTFSchedulerConfig,
)


def test_mtf_optimizer_dry_run_scores_include_report_count(tmp_path: Path):
    base_cfg = tmp_path / "base_config.json"
    base_cfg.write_text(
        '{"training": {"model_name": "mtf_test"}, "multi_timeframe": {"feature_weights": {"1min":0.3, "5min":0.6, "15min":0.1}}}'
    )
    opt = MTFOptimizer(str(base_cfg), out_dir=str(tmp_path), candidates=2, per_seed=1)
    candidates = opt.propose_candidates()
    scores = opt.evaluate_candidates(candidates, dry_run=True)
    assert len(scores) == len(candidates)
    for sc in scores:
        assert hasattr(sc, "report_count")
        assert sc.report_count == 0
        assert hasattr(sc, "run_artifacts")
        assert isinstance(sc.run_artifacts, list)


def test_mtf_scheduler_persists_metrics_in_applied_file(tmp_path: Path):
    # Use tmp_path as CWD so reports go here
    import os

    cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
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
        # run once to apply
        best = scheduler.run_once(dry_run=True, apply=True)
        applied_files = list(Path("reports").glob("applied_candidate_*.json"))
        assert len(applied_files) >= 1
        for p in applied_files:
            obj = json.loads(p.read_text(encoding="utf-8"))
            assert "candidate_id" in obj
            assert "applied_at" in obj
            # ensure the extra metrics were persisted
            assert "composite_score" in obj
            assert "mean_sharpe" in obj
            assert "mean_total_return" in obj
    finally:
        os.chdir(cwd)
