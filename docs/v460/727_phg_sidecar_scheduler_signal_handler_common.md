# 727# Sidecar scheduler signal handler commonization

## 目的

725# 後の継続タスクとして、PPO/SAC scheduler の重複をさらに削る。
今回は学習ロジックには触れず、両者で同型だった graceful shutdown signal handler だけを共通 helper へ移す。

## 実装

- `ztb.training.sidecar.scheduler_common.install_shutdown_signal_handlers()` を追加。
- `scripts/v460/ml/sidecar_scheduler_common.py` の互換 shim からも export。
- `ppo_retrain_scheduler.py` と `sac_retrain_scheduler.py` の `_install_signal_handlers()` は既存名を残し、共通 helper へ委譲。
- `ztb.training.sidecar.__init__` からも export。
- 共通 helper の unit test を `tests/unit/v460/test_sidecar_scheduler_common.py` に追加。

## 設計判断

- 既存テストが各schedulerの `_install_signal_handlers` を patch しているため、wrapper名は削除しない。
- signal handler の label は scheduler 側から渡し、ログ文面の文脈を維持する。
- `PPO/SAC` の retrain loop自体はまだ差分が多いので、今回は loop抽象化までは踏み込まない。

## 横展開候補

- neutral fallback は既に `push_neutral_signal_best_effort()` に集約済み。
- 次の候補は、SAC側の fresh signal 判定を typed reader predicate として共通化できるかの検証。
- fill関連は `fill_quality.py` が452行まで縮んでおり、次は `FillRecord.from_dict()` 周辺または test factory の重複削減が比較的安全。

## 検証結果

- `python3 -m py_compile ztb/training/sidecar/scheduler_common.py ztb/training/sidecar/__init__.py scripts/v460/ml/sidecar_scheduler_common.py scripts/v460/ml/ppo_retrain_scheduler.py scripts/v460/ml/sac_retrain_scheduler.py tests/unit/v460/test_sidecar_scheduler_common.py` PASS
- `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_sidecar_scheduler_common.py tests/unit/v460/test_680_ppo_retrain_scheduler.py::TestPPORunScheduler tests/unit/v460/test_sac_retrain_scheduler.py::TestRunScheduler -x --tb=short --no-cov` PASS: `6 passed in 3.09s`
