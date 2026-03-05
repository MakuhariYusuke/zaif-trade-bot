# 277# BlockingPolicy DRY — skip ceremony 一元化 + halt multiplier config化

## 概要

268# incident の遠因であったブロッキングロジックの複雑性に対し、
**skip ceremony の DRY 化**と**halt sleep multiplier の config 化**を実施。

## 変更内容

### 1. `_execute_skip()` ヘルパー抽出

run_continuous 内の 22 blocking decision point のうち **14 箇所**に共通する
skip ceremony (record → append → count → flush → heartbeat → state_save →
last_side → sleep) を `_execute_skip()` メソッドに一元化。

**Before** (各箇所 5-7 行):
```python
st.batch.append(self._make_loop_skip_record(
    side="none", cancel_reason=CR.OPERATOR_HALT, order_quantity=0.0,
))
st.total_count += 1
st.batch = self._batch_persistence.maybe_flush(st.batch, "operator_halt")
self._update_lock_heartbeat()
await self._effective_sleep(multiplier=5.0)
continue
```

**After** (各箇所 3-4 行):
```python
await self._execute_skip(
    st, side="none", cancel_reason=CR.OPERATOR_HALT,
    heartbeat=True, multiplier=_halt_mult,
)
continue
```

**変換箇所 (14件)**:
| # | Blocking Point | key params |
|---|----------------|------------|
| 1 | operator_halt | heartbeat, halt_mult |
| 2 | mcb_halt | heartbeat, halt_mult |
| 3 | sad_frozen | heartbeat, halt_mult |
| 4 | mcb_sad_escalation | heartbeat, halt_mult |
| 5 | per_side_dd_both | heartbeat, halt_mult |
| 6 | toxic_veto_both | — |
| 7 | phantom_veto_both | — |
| 8 | balance_forced_halt_block | state_save, update_last_side |
| 9 | one_sided_freeze | update_last_side |
| 10 | one_sided_cooldown | update_last_side |
| 11 | balance_forced_skip | update_last_side |
| 12 | gate_block | update_last_side, sleep=False |
| 13 | toxicity_participation | update_last_side |
| 14 | degraded_liquidation_duty | update_last_side |

**未変換 (特殊ロジック 6件)**:
- dd_halt: 条件付き recording + 専用 state save + MCB/SAD feed
- hard_skip_utc: 初回のみ record + logger
- time_filter_both: psutil heartbeat + no total_count
- time_filter_086_deadlock: 条件付き record
- preflight_insufficient: no total_count + shrink/pause/SAFE_STOP 分岐
- preflight_pause: asyncio.sleep(pause_sec) 固定秒

### 2. `halt_sleep_multiplier` config 化

**Before**: `multiplier=5.0` マジックナンバーが 6 箇所に散在

**After**: `FillTestConfig.halt_sleep_multiplier` (デフォルト 5.0) + YAML 設定

理論的根拠: **Brunnermeier & Pedersen (2009)** — 流動性スパイラル発生時は
取引再開までの待機時間を通常サイクルの N 倍に延長し、価格衝撃減衰を待つ。

### 3. 既存テスト更新

- `test_166_remaining_tasks.py`: gate_block path → `update_last_side=True` パターン対応
- `test_211_mcb_sad_escalation.py`: `multiplier=_halt_mult` パターン対応

## ファイル変更一覧

| File | Action |
|------|--------|
| `scripts/v460/lib/fill_loop_orchestrator.py` | `_execute_skip` 追加 + 14箇所変換 + halt_mult config化 |
| `scripts/v460/lib/fill_config.py` | `halt_sleep_multiplier` フィールド追加 + from_yaml flat_keys |
| `configs/v460/fill_test.yaml` | `halt_sleep_multiplier: 5.0` 追加 |
| `tests/unit/v460/test_276_blocking_policy_dry.py` | 新規 32 テスト |
| `tests/unit/v460/test_166_remaining_tasks.py` | テスト更新 |
| `tests/unit/v460/test_211_mcb_sad_escalation.py` | テスト更新 |

## テスト結果

```
3793 passed, 32 skipped (275#: 3761 passed → +32)
```

## 行数変化

- `fill_loop_orchestrator.py`: 2617 → 2612 行 (-5, ヘルパー +58 / 14箇所 -63)
- 実効的複雑度: skip ceremony 14箇所の一貫性確保が主目的
