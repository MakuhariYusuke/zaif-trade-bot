# 215# P0 実装: DD 整合性修復 + Hot-reload 補完 + alert_mode.json

- **種別**: fix
- **フェーズ**: ph2
- **親**: 214# §7 アクションアイテム A/B/C
- **コミット**: (本ドキュメントと同時)

---

## §1 概要

214# で検証・確定した 3 件の P0 課題を一括実装:

| ID | 課題 | 影響度 | 修正ファイル |
|---|---|---|---|
| P0-A | DD state 5 フィールド不整合 (per-side/soft_triggered) | HIGH | `daily_drawdown_guard.py`, `fill_loop_orchestrator.py` |
| P0-B | Hot-reload 防御パラメータ 13 件漏れ | HIGH | `config_hot_reload.py` |
| P0-C | alert_mode.json 未実装 (DEFCON スイッチ) | HIGH | `alert_mode.py` (新規), `fill_loop_orchestrator.py`, `fill_cycle_executor.py`, `cancel_reasons.py` |

---

## §2 P0-A: DD state 整合性修復

### §2.1 根本原因

1. **if/elif バグ**: `update_pnl()` で hard limit チェックが `if`、soft limit が `elif` → PnL が一気に hard limit を超えた場合、soft trigger が永久にスキップされる
2. **warmup 条件の厳格さ**: `import_state()` 後の warmup は `daily_fill_count == 0` でのみ発動 → state file に `fill_count=29` がある場合、per-side PnL が 0.0 でも warmup が実行されない
3. **pre-207# fill records**: per-side PnL フィールドが存在しない時代の fill records が state に 0.0 として復元される

### §2.2 修正内容

#### A. `update_pnl()` の if/elif → 独立 if 化

```python
# Before (bug):
if daily_pnl <= hard_limit:  # hard halt
    ...
elif daily_pnl <= soft_limit:  # soft — hard 時にスキップ!
    ...

# After (fix):
if daily_pnl <= soft_limit and not soft_triggered:  # soft 評価を先に
    soft_triggered = True
if daily_pnl <= hard_limit:  # hard halt (独立)
    ...
```

#### B. `needs_warmup_repair()` 追加

`import_state()` 後に呼び出し、以下を検出:
- per-side PnL が両方 0.0 なのに total PnL が有意 (|daily_pnl| >= 1.0)
- `soft_triggered_today=false` なのに `daily_pnl <= soft_limit`

該当時は `_warmup_daily_drawdown_from_records()` を強制発動。

#### C. orchestrator の warmup 条件緩和

```python
# Before:
if fill_count == 0 and records:
    warmup()

# After:
if records and (fill_count == 0 or guard.needs_warmup_repair()):
    warmup()
```

---

## §3 P0-B: Hot-reload 防御パラメータ追加

### 追加フィールド (13 件)

| 優先度 | フィールド | 出典 |
|---|---|---|
| HIGH | `loss_cooldown_threshold_bps` | 202# A |
| HIGH | `loss_cooldown_interval_mult` | 202# A |
| HIGH | `loss_boost_offset_mult` | 207# §3 |
| HIGH | `toxic_fill_veto_threshold_bps` | 207# §1 |
| HIGH | `toxic_fill_veto_cycles` | 207# §1 |
| HIGH | `one_sided_consecutive_limit` | 209# M-3 |
| HIGH | `one_sided_consecutive_interval_mult` | 209# M-3 |
| MEDIUM | `per_side_dd_enabled` | 205# §9.5 |
| MEDIUM | `per_side_dd_hard_limit_bps` | 205# §9.5 |
| MEDIUM | `per_side_dd_halt_cycles` | 205# §9.5 |
| MEDIUM | `hard_skip_utc_hours` | 205# §9.4 |
| MEDIUM | `max_cycle_sleep_sec` | 209# M-4 |

`per_side_dd_*` 変更時は `_rebuild_daily_drawdown_guard` を自動トリガー。

---

## §4 P0-C: alert_mode.json — DEFCON スイッチ

### §4.1 設計

- **新モジュール**: `scripts/v460/lib/alert_mode.py`
- **方式**: ファイルタッチ型 (211# §8 仕様準拠)
- **チェックタイミング**: サイクル先頭 (DD halt チェック直後, hard_skip_utc_hours 直前)

### §4.2 パラメータ

| キー | 型 | デフォルト | 効果 |
|---|---|---|---|
| `halt` | bool | false | 完全停止 (`operator_halt` record 記録) |
| `offset_mult` | float | 1.0 | offset 乗算 (floor=0.1) |
| `lot_mult` | float | 1.0 | lot 乗算 (0.01～1.0 clamp) |
| `interval_mult` | float | 1.0 | interval 乗算 (floor=1.0) |
| `reason` | str | "" | ログ記録用テキスト |

### §4.3 使用例

```powershell
# 即座に halt (地政学リスク等)
echo '{"halt": true, "reason": "geopolitical risk"}' > results/v460/fill_test/alert_mode.json

# 縮小運転
echo '{"offset_mult": 2.0, "lot_mult": 0.5, "interval_mult": 3.0}' > results/v460/fill_test/alert_mode.json

# 解除
del results/v460/fill_test/alert_mode.json
```

### §4.4 実装詳細

- `AlertModeOverride`: frozen dataclass (slots=True)
- ログ重複抑制: `_last_logged_state` キャッシュで同一設定の毎サイクルログを回避
- fail-safe: JSON パースエラー時はデフォルト (無効) を返す
- orchestrator: `_alert_offset_mult`, `_alert_lot_mult`, `_alert_interval_mult` をクラス属性で宣言、サイクル先頭で更新
- fill_cycle_executor: `_apply_offset_multiplier()` でオフセット適用、lot は `_effective_order_lot()` 後に乗算

---

## §5 追加発見事項

### §5.1 `import_state()` の型安全性 (P1)

`dict[str, object]` パラメータから `float()`, `int()` へのキャストで Pylance の型エラーが 14 件発生。
pre-existing だが、`TypedDict` または `cast` で型安全化すべき。

### §5.2 `hard_skip_utc_hours_buy/sell` の不在 (修正済)

hot-reload に `hard_skip_utc_hours_buy/sell` を追加したが、`FillTestConfig` に該当フィールドが存在しないことをテストが検出。即時除去で解決。

---

## §6 テスト結果

| テストスイート | 結果 |
|---|---|
| `test_168_daily_drawdown_guard.py` | **81 passed** (リグレッションなし) |
| `test_169_config_hot_reload.py` | **16 passed** (フィールド存在検証含む) |
| `test_215_dd_fix_alert_mode.py` | **20 passed** (新規: DD repair 5 + soft fix 2 + alert_mode 13) |

---

## §7 変更ファイル一覧

| ファイル | 変更種別 | 行数 (概算) |
|---|---|---|
| `scripts/v460/lib/daily_drawdown_guard.py` | 修正 | +45 (needs_warmup_repair + soft fix) |
| `scripts/v460/lib/fill_loop_orchestrator.py` | 修正 | +30 (alert_mode + warmup repair) |
| `scripts/v460/lib/fill_cycle_executor.py` | 修正 | +20 (alert offset/lot) |
| `scripts/v460/lib/config_hot_reload.py` | 修正 | +20 (13 fields + prefix) |
| `scripts/v460/lib/alert_mode.py` | **新規** | +105 |
| `scripts/v460/lib/cancel_reasons.py` | 修正 | +5 |
| `tests/unit/v460/test_215_dd_fix_alert_mode.py` | **新規** | +215 |
| `docs/v460/215_ph2_fix_dd_hotreload_alertmode.md` | **新規** | 本文書 |
