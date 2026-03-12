import json
from pathlib import Path

from ztb.training.reward_function_optimizer.mtf_optimizer import MTFOptimizer


def _create_base_config(tmp_path: Path):
    cfg = {
        "training": {"model_name": "mtf_test", "timesteps": 100},
        "multi_timeframe": {
            "feature_weights": {"1min": 0.3, "5min": 0.6, "15min": 0.1}
        },
        "reward_settings": {
            "balance_penalty_targets": {
                "buy_target": 0.52,
                "sell_target": 0.43,
                "hold_target": 0.05,
            }
        },
    }
    cfg_file = tmp_path / "base_config.json"
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
    return str(cfg_file)


def test_propose_candidates_normalization(tmp_path: Path):
    base_cfg_path = _create_base_config(tmp_path)
    opt = MTFOptimizer(
        base_config_path=base_cfg_path, out_dir=tmp_path, candidates=5, seed=1
    )
    candidates = opt.propose_candidates()
    assert len(candidates) == 5
    for c in candidates:
        # read file and check weights normalize to 1
        obj = json.loads(Path(c.config_path).read_text(encoding="utf-8"))
        fw = obj.get("multi_timeframe", {}).get("feature_weights", {})
        s = sum([float(v) for v in fw.values()])
        assert abs(s - 1.0) < 1e-6


def test_evaluate_candidates_dry_run(tmp_path: Path):
    base_cfg_path = _create_base_config(tmp_path)
    opt = MTFOptimizer(base_config_path=base_cfg_path, out_dir=tmp_path, candidates=3)
    candidates = opt.propose_candidates()
    scores = opt.evaluate_candidates(candidates, dry_run=True)
    assert len(scores) == 3
    for sc in scores:
        assert sc.composite_score == 0.0


def test_run_mtf_optimizer_dry_run(tmp_path: Path):
    base_cfg_path = _create_base_config(tmp_path)
    opt = MTFOptimizer(base_config_path=base_cfg_path, out_dir=tmp_path, candidates=4)
    best, score = opt.run(dry_run=True)
    assert best is not None
    assert best.candidate_id.startswith("mtf_candidate_")


def test_apply_candidate_to_manager(tmp_path: Path):
    from ztb.trading.environment.components.reward.mtf_weight_manager import (
        MTFWeightManager,
    )

    base_cfg_path = _create_base_config(tmp_path)
    opt = MTFOptimizer(base_config_path=base_cfg_path, out_dir=tmp_path, candidates=3)
    candidates = opt.propose_candidates()
    # pick first candidate and apply to manager
    mgr = MTFWeightManager(config={})
    opt.apply_candidate_to_manager(candidates[0], mgr)
    w = mgr.get_weights()
    assert abs(sum(w.values()) - 1.0) < 1e-9
    # Check telemetry: last applied candidate is set
    cid, ts = mgr.get_last_applied_info()
    assert cid is not None
    assert isinstance(ts, float)
