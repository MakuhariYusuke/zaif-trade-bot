# 346# テストカバレッジ拡充 + S-7 テール損失分析スクリプト

**日付**: 2026-03-09
**コミット**: `e73222773`, `aac283f91`
**種別**: テスト拡充 + 分析スクリプト新規作成
**テスト**: 118 tests all passed (67 + 18 + 33)

---

## §1 概要

345# の proactive fixes に続き、ボット稼働中に安全に実施可能な改善を実施。

1. **fill_config_validation テスト** — 323# で分離されたバリデーション関数の全ルールカバレッジ
2. **pre_order_adjustments テスト** — 323# で分離された offset 調整 Mixin のカバレッジ
3. **S-7 テール損失分析スクリプト** — 316# S-7 (downside_p10 改善) の独立分析ツール

同セッションでは他に以下も実施 (346# コミットには含まれないが同一作業の一環):
- ztb_benchmarks 削除 (`d71cada95`)
- R-18 except as-less 修正 (`e770a3082`)
- mypy 実行 (analysis/ml/tests clean 確認)
- R-12 order_monitor except narrowing 調査 (実装はリスタート時)

---

## §2 fill_config_validation テスト (67 tests)

**ファイル**: `tests/unit/v460/test_346_fill_config_validation.py`

323# で `fill_config_validation.py` に分離された `validate_fill_config()` の全ルールに対して
境界値テストを実装。

### テスト構成

| カテゴリ | テスト数 | 内容 |
|----------|---------|------|
| 正常系 | 1 | デフォルト config が通過 |
| 数値範囲バリデーション | ~25 | 各パラメータの ≤0 / 負値 / 非正規値 |
| 構造的整合性 | ~10 | halt_cap, timeout vs interval, lock_stale |
| デッドロック防止 | ~5 | per_side_dd + IE 組合せ |
| 警告系 | ~5 | kyle_lambda, imbalance 依存 warning |
| 境界条件 | ~20 | 閾値ちょうど / 1-off / None ケース |

### 対象バリデーションルール (一部)

- `order_quantity > 0`, `cycle_interval_sec > 0`
- `max_cycle_sleep_sec ≥ cycle_interval_sec`
- `spread_offset_ratio ∈ (0, 1)`
- `lock_stale_heartbeat_sec ≥ 3 × lock_heartbeat_period_sec`
- `per_side_dd + imbalance_threshold → デッドロック警告`
- `kyle_lambda_bps ≠ 0.0 警告`

---

## §3 pre_order_adjustments テスト (18 tests)

**ファイル**: `tests/unit/v460/test_346_pre_order_adjustments.py`

323# で `PreOrderAdjustmentsMixin` に分離された 2 関数のテスト。

### §3.1 `_recalc_price_with_new_offset` (6 tests)

| テスト | 内容 |
|--------|------|
| basic_buy | buy: mid 逆推定 → 新 offset 適用 |
| basic_sell | sell: mid 逆推定 → 新 offset 適用 |
| none_spread | spread_at_order=None → 元価格返却 |
| zero_spread | spread=0 → 元価格返却 |
| negative_spread | spread<0 → 元価格返却 |
| mid_estimation | mid = order_price ± spread×old_offset の逆推定精度 |

### §3.2 `_apply_offset_multiplier` (12 tests)

| テスト | 内容 |
|--------|------|
| conservative_buy | mult > 1 → buy 価格が下がる (より保守的) |
| conservative_sell | mult > 1 → sell 価格が上がる (より保守的) |
| aggressive_buy | mult < 1 → buy 価格が上がる (より積極的) |
| aggressive_sell | mult < 1 → sell 価格が下がる (より積極的) |
| noop (×4) | multiplier=1.0, None, spread=None/0 → 元価格 |
| boundary | mult=0 で offset=0 (mid price に張り付く) |

---

## §4 S-7 テール損失分析スクリプト (33 tests)

**ファイル**: `scripts/v460/analysis/tail_loss_analysis.py`
**テスト**: `tests/v460/test_346_tail_loss_analysis.py`

### §4.1 分析機能

316# S-7「downside_p10 改善のためのテール損失分析」を独立スクリプトとして実装。
`analysis/311_observational_rerun.py` 内の `tail_loss_analysis()` を発展させ、
より網羅的な分析軸とアクション可能な skip rule 候補の自動検出を追加。

| 分析軸 | 内容 |
|--------|------|
| Regime over-representation | テールに集中する regime を特定 |
| Hour over-representation | テールに集中する UTC hour を特定 (JST 表示付き) |
| Decision path 分布 | テール内の decision_path 構成 |
| 数値特徴量 (4 軸) | spread_at_order, mid_price_trend_5s, OBI, skip_gate_score の tail vs total 比較 |
| フラグ分析 | AS rate, Early Exit rate, balance_forced_switch rate の tail vs total |
| Actionable filters | 条件付き skip ルール候補の自動検出 (回避テール数 / 犠牲非テール数 / 効率性指標) |

### §4.2 Actionable filter の自動検出

以下の 4 種のルール候補を自動的に列挙し、効率性 (回避テール / 犠牲非テール) で降順ソート:

1. **regime_skip** — over-representation ≥ 1.5x の regime をスキップ
2. **hour_skip** — over-representation ≥ 1.5x の UTC hour をスキップ
3. **spread_skip** — テールの spread_at_order p75 超をスキップ
4. **velocity_skip** — テールの |mid_price_trend_5s| p75 超をスキップ

### §4.3 CLI

```bash
.venv\Scripts\python.exe scripts/v460/analysis/tail_loss_analysis.py
.venv\Scripts\python.exe scripts/v460/analysis/tail_loss_analysis.py --git-sha abc1234
.venv\Scripts\python.exe scripts/v460/analysis/tail_loss_analysis.py --date-from 2026-03-01
.venv\Scripts\python.exe scripts/v460/analysis/tail_loss_analysis.py --percentile 5 --output results.json
```

### §4.4 テスト構成 (33 tests)

| クラス | テスト数 | 対象 |
|--------|---------|------|
| TestExtractFilled | 3 | 約定フィルタ + side フィルタ |
| TestPnlArray | 3 | PnL 配列変換 + None/空 |
| TestAsRate | 2 | AS レート計算 |
| TestFlagRate | 3 | bool フラグレート (None 対応) |
| TestRecordToUtcHour | 4 | epoch/ISO/None/missing |
| TestComputeOverrep | 2 | カテゴリ over-representation |
| TestComputeHourOverrep | 2 | 時間帯 over-representation + ソート |
| TestNumericFieldStats | 2 | 数値統計 + None |
| TestDeriveActionableFilters | 3 | regime skip 検出 / no-overrep / 効率性ソート |
| TestAnalyzeTailLoss | 8 | 基本分析 / insufficient / both sides / AS overrep / custom percentile |
| TestMainCLI | 1 | 空ディレクトリでのスモークテスト + JSON 出力 |

---

## §5 関連ドキュメント

| # | 関係 |
|---|------|
| 316 | S-7 テール損失分析の仕様策定 |
| 319 | S-7 関数実装 (311_observational_rerun.py 内) |
| 323 | fill_config_validation / pre_order_adjustments のモジュール分離 |
| 345 | 同セッションの proactive fixes |
| 347 | 同セッションの 1mBTC 制約分析 |
