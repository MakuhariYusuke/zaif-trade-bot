# 659# balance_checker 重複排除 + MCB HALT ポジション警告

## 概要

balance_checker の sell/buy 間で重複していたロット縮小・復元ロジックを共通ヘルパーに抽出。
MCB HALT 発動時に BTC ポジションが露出している場合の警告ログを追加。
T1-6（skip_rate × toxic_veto 干渉）は調査の結果、変更不要と判断。

## 変更内容

### 1. balance_checker リファクタリング

| メソッド | 責務 |
|----------|------|
| `_apply_lot_shrink(new_lot, log_msg)` | 残高不足時の共通ロット縮小（pre_shrink_lot 保存含む） |
| `_try_lot_restore(can_afford, side)` | 101# §6 残高回復時の共通ロット復元 |

**削減**: `_check_sell` / `_check_buy` のロット縮小コード（各8-10行）と復元コード（各9-11行）を共通化。
両メソッドの差分（sell: dust_sweep、buy: 476# lot expansion）はそのまま維持。

### 2. T1-1: MCB HALT ポジション警告

`orchestrator_pre_cycle.py::_check_circuit_breakers()` の MCB HALT パスに、
`last_btc_free > 0` なら WARNING ログを出力するガードを追加。

```
[659# MCB] HALT with open BTC position: 0.00100000 BTC exposed during cooldown 300s
```

クールダウン中に BTC を保持している状態は市場急変リスクに直結するため、
運用者の注意を喚起する。

### 3. T1-6: skip_rate × toxic_veto 干渉（変更不要）

**調査結論**: 以下の理由により、現行アーキテクチャで既に正しく分離済み。

1. **rule-based hard skip** (toxic_veto hard / velocity_veto) → `_set_early_skip_result()` →
   `early_return_record` で `SkipGate.evaluate()` をバイパス → `_recent_skips` に記録されない
2. **per-side tracking** → buy/sell 独立した skip_rate → cross-side 干渉なし
3. **toxic_veto soft mode** (A-4) → offset boost 後に ML gate が正常評価 →
   ML の意思決定として skip_rate に正しく反映

### 4. テスト修正

- 657# A-4 で追加された `toxic_veto` ステージキーを `test_585_multiplicative_pipeline.py` の expected_keys に追加
- 659# の MCB HALT 警告追加で `_execute_skip` の位置が移動したため `test_276_blocking_policy_dry.py` の検索窓を拡大

## 変更ファイル

| ファイル | 変更種別 |
|----------|----------|
| `scripts/v460/lib/balance_checker.py` | refactor: _apply_lot_shrink / _try_lot_restore 抽出 |
| `scripts/v460/lib/orchestrator_pre_cycle.py` | feat: MCB HALT ポジション警告 |
| `tests/unit/v460/test_659_balance_checker_refactor.py` | 新規: 19テスト |
| `tests/unit/v460/test_585_multiplicative_pipeline.py` | fix: toxic_veto キー追加 |
| `tests/unit/v460/test_276_blocking_policy_dry.py` | fix: 検索窓拡大 |

## D-2 (lot dynamic sizing) の現状分析

1週間の fill_records 分析で `preflight_insufficient` の90%が buy 側（JPY 不足）。
しかし **balance_jpy = 0** のケースが大半 → D-2（動的ロットサイジング）では解決不能。
根本的には B-3（inv_skew）による在庫バランスの維持が鍵。

buy 側の `_check_buy()` は既に affordable_lot 計算 + min_order_btc 以上なら縮小続行のロジックを実装済み（476#）。
追加の動的サイジングは「残高ゼロ問題」には効果がないため、現時点では変更なし。
