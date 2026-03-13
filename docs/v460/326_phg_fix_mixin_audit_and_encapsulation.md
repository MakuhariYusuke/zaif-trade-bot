# 326# Mixin Audit & Encapsulation Fix

325# Orchestrator God Object 分割の品質レビューと改善。

## 監査結果と対応

### High Priority (対応済)

| # | 問題 | ファイル | 対応 |
|---|------|---------|------|
| H-1 | `_build_state_snapshot` の戻り値型が `-> object` | `orchestrator_lifecycle.py` | `-> "FillTestState"` に修正、TYPE_CHECKING import 追加 |
| H-2 | `_restore_common_state` の引数型が `object \| None` | `orchestrator_lifecycle.py` | `"FillTestState \| None"` に修正 |
| H-3 | DD guard warmup が 10+ の private 属性に直接アクセス | `orchestrator_lifecycle.py` | `DailyDrawdownGuard.warmup_from_records()` に委譲 |

### Medium Priority (対応済)

| # | 問題 | ファイル | 対応 |
|---|------|---------|------|
| M-1 | `import asyncio` が未使用 | `orchestrator_guards.py` | 削除 |
| M-2 | `"buy" if next_side == "sell" else "sell"` インライン | `fill_loop_orchestrator.py` L1057 | `self._opposite_side(next_side)` に置換 |

### Low Priority (記録・保留)

| # | 問題 | 対応 |
|---|------|------|
| L-1 | `_opposite_side` shared util 昇格 | 296# C-9 で評価済み。Mixin 内 staticmethod で MRO 経由アクセス可能。他モジュール需要発生時に再検討 |
| L-2 | `side_selector.py` / `hindsight_filter.py` のインライン side flip (×4) | L-1 と同時に対応 |
| L-3 | 空 `config/` ディレクトリ | 削除済 (106# R7) |

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `scripts/v460/lib/orchestrator_lifecycle.py` | 型安全修正 (`-> FillTestState`, `FillTestState \| None`), DD warmup を委譲に変更 (2849→539行相当の責務分離の品質向上) |
| `scripts/v460/lib/orchestrator_guards.py` | 未使用 `import asyncio` 削除 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | インライン opposite side → `_opposite_side()` 呼び出し |
| `scripts/v460/lib/daily_drawdown_guard.py` | `warmup_from_records()` メソッド追加 (encapsulation fix) |
| `tests/unit/v460/test_168_daily_drawdown_guard.py` | `TestWarmupFromRecords326` 追加 (5 tests) |

## テスト結果

```
4069 passed, 33 skipped (4064 既存 + 5 新規)
```

## 設計根拠

### DD guard warmup encapsulation (H-3)

**問題**: `_warmup_daily_drawdown_from_records` が `guard._today()`, `guard._day_reset_tz`,
`guard._per_side_enabled`, `guard._per_side_hard_limit_bps`, `guard._per_side_halt_cycles`,
`guard._soft_limit_bps`, `guard._hard_limit_bps`, `guard._soft_triggered_today` など
10+ の private 属性に直接アクセスしていた。

**解決**: `DailyDrawdownGuard.warmup_from_records(records)` を新設し、
ロジックをガード側に移動。Orchestrator は `(timestamp, pnl_bps, side)` タプルのリストを
渡すだけの薄いアダプタに変換。

**利点**:
- Orchestrator と DD guard の結合度を大幅に低下
- DD guard の内部実装変更が Orchestrator に波及しない
- テスト容易性向上 (DD guard 単体でテスト可能)
