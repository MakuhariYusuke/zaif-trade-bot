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

## B-3 実運用観察: inv_skew_factor=0.000 問題

### 事象

657# B-3 実装後の 1 週間 fill_records を分析したところ、
`inv_skew_factor=0.000` が大半のサイクルで記録されていた。

### 原因分析

B-3 の実装自体は正しい（コードレビューで確認済み、テスト13本通過）。
`factor=0.0` になる経路は以下の2つ：

1. **neutral_band フィルタ**: `|decayed_imbalance| <= 0.05` → factor=0
   - lot=0.001 BTC（~10,800 JPY）の取引では、在庫の buy/sell 偏りが
     neutral_band を超えるほど蓄積しにくい
   - 654# で `neutral_band: 0.10 → 0.05` に下げたが、まだ足りない可能性

2. **在庫ゼロ問題**: preflight_insufficient (JPY=0) で buy が連続スキップ
   → sell のみ約定 → BTC 残高ゼロ → 取引自体が停止 → imbalance 更新なし

### 構造的制約

```
lot=0.001 → 在庫偏り蓄積遅い → imbalance < neutral_band → inv_skew 不発動
         → 在庫バランス維持不能 → JPY=0 に偏る → preflight_insufficient 連打
```

inv_skew が効果を発揮するには**在庫偏りが neutral_band を超える**必要があるが、
現行 lot サイズ (0.001 BTC) では十分な偏りが蓄積する前に残高枯渇する。

### 今後の方針

- **Priority 1**: 24h+ の B-3 効果観測を継続（657# 直後のデータ蓄積待ち）
- **neutral_band の追加引き下げ** (0.05 → 0.02) を検討 — ただし noise sensitivity とのトレードオフ
- D-2 は B-3 効果が確認できない場合のフォールバック策として保留

## D-2 (lot dynamic sizing) の現状分析

1週間の fill_records 分析で `preflight_insufficient` の90%が buy 側（JPY 不足）。
しかし **balance_jpy = 0** のケースが大半 → D-2（動的ロットサイジング）では解決不能。
根本的には B-3（inv_skew）による在庫バランスの維持が鍵。

buy 側の `_check_buy()` は既に affordable_lot 計算 + min_order_btc 以上なら縮小続行のロジックを実装済み（476#）。
追加の動的サイジングは「残高ゼロ問題」には効果がないため、現時点では変更なし。

## 残課題一覧 (659# 時点)

### 計測待ち (Priority 1)
- **B-3 効果観測**: inv_skew + toxic_veto の実運用効果 → 24h+ データ蓄積後に判定

### Tier 1 (短期・保留中)
| ID | タスク | 備考 |
|----|--------|------|
| T1-2 | RT 主 KPI 化 | 分析基盤整理 |
| T1-3 | Regime 遷移 AS セクション | section 未実装 |
| T1-4 | AS burst 自己相関 | section 未実装 |
| T1-5 | PnL 計測窓正規化 | 命名・実態確認 |

### Tier 2 (中期・前提条件付き)
| ID | タスク | 前提条件 |
|----|--------|----------|
| T2-1 | eDRC α/β 再推定 | 50+ RT 蓄積後 |
| T2-2 | Sidecar retrain 成功率 | ログ分析 |
| T2-3 | sell ceiling → VG 連動 | 中期改善 |
| T2-4 | 曜日効果 | 4 週分データ必要 |
| T2-5 | Asymmetric RT exit | position tracking |
| T2-6 | Regime-drift exit | regime 遷移検知 |
| T2-8 | sell_dynamic_kill 存廃 | C-4 ARL 計測後 |
| T2-9 | preflight バッファ | D-2/D-6 B-3 効果次第 |

### 656# 提案・保留中
| ID | 提案 | ステータス |
|----|------|-----------|
| C-4 | sell_dynamic_kill ARL 最適化 | 50+ RT 後 |
| D-2 | lot_size 動的縮小 | B-3 効果観測後のフォールバック |
| D-6 | virtual shadow balance | 長期アーキテクチャ |
