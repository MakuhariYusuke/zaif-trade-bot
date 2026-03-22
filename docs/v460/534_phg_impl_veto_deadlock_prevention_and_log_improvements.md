# 534# Veto デッドロック防止 + ログ可観測性改善 実装ドキュメント

- **日付**: 2026-03-22
- **対象ドキュメント**: 532# (盲点検証), 533# (市場微視構造検証)
- **目的**: 532# で発見された veto デッドロック問題 (05:17→08:25, 3h8m, 37連続) の実装修正 + ログ⇔JSONL 突合の join key 追加 + ログ出力改善

---

## §1 変更概要

| # | 変更内容 | 対象ファイル | 優先度 |
|---|---------|-------------|--------|
| 1 | veto 連続回数上限 (max_consecutive) | `maker_risk_guards.py`, `fill_config.py`, `fill_test.yaml` | P0 |
| 2 | BTC=0 時の buy veto 閾値緩和 | `maker_risk_guards.py`, `fill_config.py`, `fill_test.yaml` | P0 |
| 3 | FillRecord `log_cycle_no` 追加 | `fill_quality.py`, `fill_record_builder.py`, `fill_cycle_executor.py` | P0 |
| 4 | FillRecord `cross_venue_lead_lag_veto_consecutive` 追加 | `fill_quality.py`, `fill_record_builder.py` | P1 |
| 5 | ログ出力: BTC 残高コンテキスト追加 | `fill_cycle_executor.py` | P1 |
| 6 | ログ出力: veto 連続回数表示 | `maker_risk_guards.py` | P1 |

---

## §2 P0: Veto デッドロック防止

### §2.1 問題 (532# §3.2)

- 05:17→08:25 に 37 連続 veto が発生し、3 時間 8 分間のデッドロック状態
- この期間の機会損失: post-veto first-fill は 5/6 が positive (avg +1.36bps)
- 既存実装: `abs(hint.spread_bps) >= threshold` のみのチェック → 無限 veto 可能

### §2.2 解決策 A: 連続 veto 上限 (`veto_max_consecutive`)

```python
# fill_config.py
cross_venue_lead_lag_veto_max_consecutive: int = 20  # デフォルト 20 回

# maker_risk_guards.py
if consecutive_veto_count >= max_consecutive:
    # veto 強制解除 → adverse retreat にフォールスルー
    logger.warning("[cross_venue] veto force-released: consecutive %d >= max %d")
    consecutive_veto_count = 0
```

**設定値根拠**: 20 回 × ~120s/cycle = 約 40 分。3h8m のデッドロックを防止しつつ、本当に危険な相場 (数分間の急変) では veto が機能する。

### §2.3 解決策 B: BTC=0 時の buy 側閾値緩和

```python
# BTC=0 かつ buy 側 → 閾値を ×1.5 に緩和
if btc_balance < 1e-8 and side == "buy":
    effective_threshold *= veto_inventory_zero_threshold_mult  # 8.0 → 12.0bps
```

**理論的根拠**: BTC 在庫がゼロの場合、sell veto は不要 (売るものがない)。buy veto は「これから買う」判断への介入だが、在庫ゼロでの buy 阻止は機会損失が大きい。閾値を緩和して、本当に extreme な場合のみ veto する。

---

## §3 P0: FillRecord `log_cycle_no` (ログ⇔JSONL join key)

### §3.1 問題 (532# §3.5)

- ログファイルの fill 数 (93) と JSONL の fill 数 (60) に乖離
- ログの `=== Cycle NNN ===` と JSONL レコードの突合に join key が存在しない
- `cycle_id` は `{timestamp}_{uuid_hex[:8]}` 形式でログには出力されない

### §3.2 解決策

```python
# fill_quality.py FillRecord
log_cycle_no: int | None = None  # "=== Cycle NNN" の NNN

# fill_cycle_executor.py
record = self._build_fill_record(
    ...,
    log_cycle_no=self._cycle_count,  # ← 追加
)
```

これにより、ログファイルの `Cycle 42 result: filled=True` と JSONL の `{"log_cycle_no": 42, ...}` が一意に突合可能になる。

---

## §4 P1: ログ出力改善

### §4.1 BTC 残高コンテキスト

```
# Before
Cycle 42 result: filled=True, wait=3.5s, pnl=+1.23bps

# After
Cycle 42 result: filled=True, wait=3.5s, pnl=+1.23bps, btc=0.001234
```

在庫状態の可視化により、balance forced switch や inventory skew の分析が容易になる。

### §4.2 Veto 連続回数表示

```
# Before
[cross_venue] cross_venue_veto: sell suppressed by bitflyer up lead (spread=+8.50bps, ...)

# After
[cross_venue] cross_venue_veto: sell suppressed by bitflyer up lead (spread=+8.50bps, ..., consecutive=3/20)
```

veto デッドロックの発生・解消をリアルタイムで監視可能にする。

---

## §5 テスト

### 追加テスト (6 件)

| テスト名 | 検証内容 |
|---------|---------|
| `test_veto_max_consecutive_releases_after_limit` | 連続 veto が上限に達すると強制解除 |
| `test_veto_counter_resets_when_no_veto` | veto 非発動時にカウンタリセット |
| `test_inventory_zero_relaxes_buy_veto_threshold` | BTC=0 で buy 閾値が緩和 |
| `test_inventory_zero_does_not_relax_sell_veto` | BTC=0 でも sell は緩和されない |
| `test_fill_record_includes_veto_consecutive` | FillRecord round-trip |
| `test_fill_record_includes_log_cycle_no` | FillRecord round-trip |

### テスト結果

- `test_439_cross_venue_lead_lag.py`: 50 passed (44 既存 + 6 新規)
- `test_fill_quality.py`: 206 passed
- `tests/unit/v460/` 全体: 95 passed, 1 failed (pre-existing: test_093 sa_boost)

---

## §6 設定値サマリ

```yaml
# configs/v460/fill_test.yaml
cross_venue_lead_lag:
  veto_max_consecutive: 20           # 連続 veto 上限 (3h8m デッドロック → 最大 ~40 分に制限)
  veto_inventory_zero_threshold_mult: 1.5  # BTC=0 時に閾値を ×1.5 (8.0 → 12.0bps)
```

---

## §7 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `ztb/metrics/fill_quality.py` | `log_cycle_no`, `cross_venue_lead_lag_veto_consecutive` フィールド追加 |
| `scripts/v460/lib/fill_config.py` | `veto_max_consecutive`, `veto_inventory_zero_threshold_mult` 設定追加 |
| `scripts/v460/lib/maker_risk_guards.py` | 連続 veto 上限 + BTC=0 緩和ロジック + ログ改善 |
| `scripts/v460/lib/maker_price.py` | `_consecutive_veto_count`, `_veto_btc_balance` 初期化 + setter |
| `scripts/v460/lib/fill_record_builder.py` | `log_cycle_no` param + `veto_consecutive` フィールド |
| `scripts/v460/lib/fill_cycle_executor.py` | `log_cycle_no` 渡し + BTC 残高注入 + ログ改善 |
| `configs/v460/fill_test.yaml` | 新設定値 |
| `tests/unit/v460/test_439_cross_venue_lead_lag.py` | 6 テスト追加 + stub 更新 |
