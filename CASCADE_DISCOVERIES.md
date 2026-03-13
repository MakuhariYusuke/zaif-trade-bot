# カスケード発見ログ

| # | 発見元タスク | ファイル:行 | 概要 | 重要度 | 対処 |
|---|---|---|---|---|---|
| D1 | T15 | `ztb/trading/environment/heavy_env/mixins/initialization.py:32` | `maybe_collect_garbage` を import しているが `gc_guard.py` 実体が欠落していた | HIGH | 修正済み |
| D2 | T3 | `ztb/data/marketdata_registry.py:17` | `ReplayMarket` の top-level import が live path typo と circular import を引き起こしていた | HIGH | 修正済み |
| D3 | T11 | `scripts/testing/test_simplified_reward_calculator.py:15` | archived 対象 `SimplifiedRewardCalculator` への live testing script 参照が残っていた | MEDIUM | 修正済み |
| D4 | T6 | `ztb/trading/environment/__init__.py:14` | direct relative import では warning/re-raise 挙動を focused で検証しづらかった | LOW | 修正済み |
| D5 | T14 | `ztb/trading/environment/components/rewards/forced_balance.py:86` | forced-balance penalty/bonus の canonical 実装先を `ForcedBalanceReward` に固定しないと将来再分岐しやすい | MEDIUM | 修正済み |
| D6 | full-suite `-x` | `ztb/trading/performance_optimizer.py:499` | Windows で `psutil.Process.open_files()` が access violation を起こし background monitor thread ごと test を落としていた | HIGH | 修正済み |
