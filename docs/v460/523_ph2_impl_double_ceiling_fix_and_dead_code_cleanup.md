# 523# 二重 ceiling 撤廃 + dead code cleanup

> 作成: 2026-03-21 JST
> 対象: 518# §5 / 522# P0 double-ceiling問題 + 522# dead code残置

## §1 二重 ceiling 問題の修正

### §1.1 問題

maker_price.py の `_apply_final_offset_ceiling()` と offset_pipeline.py の `execution_final_clamp` で
ceiling が二重適用されていた。

```
maker_price pipeline:
  base→inv_skew→regime→spread_adapt→kyle→amihud→vol_guard→cross_venue
  →imb_risk→buy_as_guard→sell_hour→loss_boost→FFD boost
  → CEILING #1 (maker_price) ← ここで 0.274→0.25 にキャップ

offset_pipeline:
  受信 0.25 → EV(×0.96) → 0.24 → velocity → trending → toxicity → ...
  → CEILING #2 (execution_final_clamp) ← 0.24 < 0.25 → no-op
```

**影響**:
- FFD boost (maker_price内) が ceiling #1 に吸収 → FFD が実質無力化 (518# §5)
- EV mult が ceiling 後の低い値に作用 → 情報損失
  - 例: base=0.28, EV=0.8 → 期待: 0.224、実際: 0.25×0.8=0.20

### §1.2 修正内容

maker_price.py `compute()` から `_apply_final_offset_ceiling()` 呼び出しを削除。
offset_pipeline の `execution_final_clamp` (421#) のみで ceiling を制御。

```
修正後フロー:
  maker_price: base→...→FFD boost → max_offset_ratio(0.30)で安全キャップ
  offset_pipeline: 受信 0.274 → EV/velocity/trending/toxicity/...
  → CEILING (execution_final_clamp, ceiling=0.25) ← 単一制御
```

### §1.3 修正されたファイル

| ファイル | 変更 |
|---------|------|
| `maker_price.py` | `_apply_final_offset_ceiling()` 呼び出し削除 (メソッドは残置) |

## §2 Dead code cleanup (522# 残置分)

### §2.1 `_inventory_escape_duty_counter` 削除

522# で inventory_escape を撤廃したが、カウンタの宣言・永続化コードが残っていた。

| ファイル | 変更 |
|---------|------|
| `fill_loop_orchestrator.py` | `_inventory_escape_duty_counter: int = 0` 削除 |
| `orchestrator_lifecycle.py` | export/restore コード削除 (L131, L240-243) |
| `resilience.py` | `FillTestState.inventory_escape_duty_counter` フィールド削除 |

### §2.2 FillRecord docstring 更新

| ファイル | 変更 |
|---------|------|
| `ztb/metrics/fill_quality.py` | `resolved_side_reason` コメントを更新 (balance_switch/route_to_kill_deadlock → 522#撤廃後は常に None) |

### §2.3 テスト更新

| ファイル | 変更 |
|---------|------|
| `test_421_final_clamp_deadlock.py` | `TestSideObservability` テスト値を `None` に更新 (balance_switch は存在しない) |

## §3 影響分析

### 二重 ceiling 撤廃の効果

| シナリオ | 修正前 | 修正後 | 改善 |
|---------|--------|--------|------|
| 通常 (no FFD) | offset=0.274→ceil=0.25→EV(0.96)=**0.24** | offset=0.274→EV(0.96)=0.263→ceil=**0.25** | +0.01 (保守的方向) |
| FFD active | offset=0.40→ceil=0.25→EV(0.96)=**0.24** | offset=0.40→max_ratio=0.30→EV(0.96)=0.288→ceil=**0.25** | FFD 保護が ceiling 以下に漏れない |
| hour_ceiling 2.0x | offset=0.274→ceil=0.50→EV=0.263 | offset=0.274→max_ratio=0.30→EV=0.288→ceil=0.50 | 同等 |

**核心的改善**: FFD boost 発動時、EV tightening が ceiling 以下に引き下げることがなくなった。
修正前は FFD→ceiling=0.25→EV=0.24 (FFD の保護を EV が打ち消し)。
修正後は FFD→0.30→EV=0.288→ceiling=0.25 (ceiling が最終的に保護を保証)。

## §4 残存する低優先度課題

| # | 課題 | 優先度 | 状態 |
|---|------|--------|------|
| P1 | dd_soft_lot_scale が min_lot floor で no-op | LOW | 安全床: 意図的設計 |
| P2 | sell_guard.offset_floor=0.05 が実質不要 | LOW | 安全弁として残置 |
| P3 | Disabled features の config subtree (SAD/MCB等) | LOW | 将来の有効化に備え残置 |
| P4 | `_apply_final_offset_ceiling` メソッド未使用 | LOW | compute() から呼ばれないが互換性のため残置 |
