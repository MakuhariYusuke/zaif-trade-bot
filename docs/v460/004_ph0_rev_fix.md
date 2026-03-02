# 003# Phase 0 Review Implementation — 修正対応記録

日付: 2025-01-XX
レビュー文書: `docs/v460/003_ph0_rev_impl.md`
対応コミット: (本コミット)

## 概要

外部レビューの 20 指摘 (4 CRITICAL, 5 HIGH, 9 MEDIUM, 2 LOW) を全て評価・対応。
全件有効と判断し修正を実施。

## 修正一覧

| # | 重要度 | 指摘 | 対応 | 対象ファイル |
|---|--------|------|------|-------------|
| 1 | CRITICAL | baseline ゼロベクトル | XGB vs Logistic/Ridge ペア化 | evaluator.py, tasks/feature_info.py |
| 2 | CRITICAL | regression target に Classifier | XGBRegressor 分離, 自動切替 | evaluator.py |
| 3 | CRITICAL | XGB パラメータ二重指定 | _RESERVED_XGB_KEYS + base.yaml 整理 | evaluator.py, base.yaml |
| 4 | CRITICAL | Coincheck float(ISO) ValueError | _parse_timestamp 二重パーサー | coincheck/adapter.py |
| 5 | HIGH | _last_trade_id 未使用 | 複合ID dedup 実装 | market_data_collector.py |
| 6 | HIGH | G1 閾値未参照 | min_ic/min_accuracy/min_sig 検証追加 | run_gate_check.py |
| 7 | HIGH | _evaluate_gate 閾値無視 | gate_thresholds.yaml 読込・渡し | run_experiment.py |
| 8 | HIGH | bfill() look-ahead | bfill() 削除, ffill().fillna(0) のみ | microstructure.py |
| 9 | HIGH | async+sync requests | asyncio.to_thread() wrap | coincheck/adapter.py, bitflyer/adapter.py |
| 10 | MEDIUM | bitFlyer rate limit/例外 | _check_rate_limit + NetworkError | bitflyer/adapter.py |
| 11 | MEDIUM | auto-aggregate なし | run_continuous に auto_aggregate 引数 | market_data_collector.py |
| 12 | MEDIUM | direction NaN→0 | NaN 維持 | data_loader.py |
| 13 | MEDIUM | set() 非決定的順序 | sorted(set(...)) | data_loader.py |
| 14 | MEDIUM | Parquet 全列読込 | pd.read_parquet(columns=) | data_loader.py |
| 15 | MEDIUM | manifest 2行書込 | docstring にイベントログ設計意図記載 | manifest.py |
| 16 | MEDIUM | task ロジックが orchestrator 内 | lib/tasks/feature_info.py に分離 | run_experiment.py, tasks/ |
| 17 | MEDIUM | fold 全サンプル保存 | _signal transient + to_dict() 統計量のみ | evaluator.py |
| 18 | MEDIUM | G0 全カラムカウント | feature columns のみ (target_/close 除外) | run_gate_check.py |
| 19 | LOW | 異常系テスト不足 | 14 テスト追加 (26→40) | test_v460_core.py |
| 20 | LOW | seed 未固定 | test_all_pass, _make_sample_df に seed | test_v460_core.py |

## テスト結果

- 40 passed, 0 failed (26→40 テストに拡充)
- 新規テストクラス:
  - `TestTimestampParsing` (5 tests) — #4
  - `TestCollectorDedup` (1 test) — #5
  - `TestEvaluatorFactories` (4 tests) — #2/#3
  - `TestDataLoaderEdgeCases` (3 tests) — #12/#13/#14
  - `TestGateCheckG0FeatureColumns` (1 test) — #18

## 新規ファイル

- `scripts/v460/lib/tasks/__init__.py`
- `scripts/v460/lib/tasks/feature_info.py` — #16 task 分離

## 設計判断

- #9 (async/sync): `asyncio.to_thread()` でラップ (httpx 全面移行は Phase 2 以降)
- #15 (manifest 2行): イベントログ設計として有効。docstring に意図を明文化
- #17: `_signal` は in-memory 参照のみ、`to_dict()` で除外
