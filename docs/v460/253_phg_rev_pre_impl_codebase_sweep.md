# 253# Pre-Implementation Codebase Sweep Report

> **日付**: 2026-03-03  
> **対象**: v460 HEAD `1c9a12031` (252# 実装後)  
> **テスト**: 3507 passed  
> **スイープ範囲**: 247#/248# 残 TODO, 252# セルフレビュー, コード品質, 型安全性

---

## 0. エグゼクティブサマリー

252# の実装 (Sell Asymmetric Gate, PhantomGuard 三値化/JPY 照合, getattr 除去) は概ね健全。
ただし **hot_reload 配線漏れ** が1件あり、dead config の除去タイミングも到来している。
残 getattr は Mixin 境界の構造的問題であり、段階的に型安全化すべき。

---

## 1. P1 Items (253# で実装すべき)

### P1-1: `sell_asymmetric_high_vol_enabled` hot_reload 配線漏れ

**252# セルフレビュー発見。**

- `fill_config.py:435` — フィールド定義 ✅
- `fill_config.py:1078` — YAML parse (`_parse_stopgap_section`) ✅
- `config_hot_reload.py` — `_RELOADABLE_KEYS` に**未登録** ❌
- `configs/v460/fill_test.yaml` — **未記載** ❌

**影響**: ライブ稼働中に YAML 変更しても反映されない。sell asymmetric は
安全装置であり、hot_reload で即座に ON/OFF できるべき。

**修正**:
1. `config_hot_reload.py` の `_RELOADABLE_KEYS` に `"sell_asymmetric_high_vol_enabled"` 追加
   (L178 付近の `"skip_sell_trending_up_only"` の直後)
2. `configs/v460/fill_test.yaml` の `止血` セクションに
   `sell_asymmetric_high_vol_enabled: false` を追加

---

### P1-2: `balance_forced_apply_trending_offset` dead config 完全削除

`fill_config.py:443` の `TODO(235#)` が残存。
234# で実コードから参照がなくなったが、以下3箇所にまだ配線が残っている:

| 箇所 | ファイル | 行 |
|------|----------|-----|
| YAML parse | `fill_config.py` | L1089 |
| hot_reload | `config_hot_reload.py` | L183 |
| YAML 定義 | `configs/v460/fill_test.yaml` | L433 |

**修正**:
1. `fill_config.py` からフィールド定義 + YAML parse 行を削除
2. `config_hot_reload.py` から `_RELOADABLE_KEYS` の該当行を削除
3. `configs/v460/fill_test.yaml` から該当行を削除
4. テスト (`test_197_*.py`, `test_234_*.py`, `test_196_*.py`) の参照を更新

---

### P1-3: `fill_cycle_executor.py` 残 getattr 型安全化 (5件)

252# で `_maybe_register_phantom` の getattr は除去済みだが、同ファイルに
5件の Mixin 境界 getattr が残存:

| 行 | 式 | 型安全な代替 |
|----|-----|-------------|
| L1032 | `getattr(self, "_alert_offset_mult", 1.0)` | クラスレベル宣言 `_alert_offset_mult: float = 1.0` |
| L1068 | `getattr(self, "_alert_lot_mult", 1.0)` | 同上 `_alert_lot_mult: float = 1.0` |
| L1078 | `getattr(self, "_halt_recovery_lot_mult", 1.0)` | 同上 `_halt_recovery_lot_mult: float = 1.0` |
| L1088 | `getattr(self, "_daily_drawdown_guard", None)` | 同上 `_daily_drawdown_guard: DailyDrawdownGuard \| None = None` |
| L1305 | `getattr(self.config, "macro_regime_conflict_action", "log")` | 直接参照 `self.config.macro_regime_conflict_action` (フィールドは `fill_config.py:163` に存在) |

**注**: L1199 の `getattr(self, "_postonly_crossing_streak", 0)` はクラスレベル宣言済み
(L55) なので冗長。直接参照 `self._postonly_crossing_streak` に変更。

**修正**: クラスレベル宣言追加 (L55 付近) + getattr → 直接参照に置換

---

### P1-4: event_logger.py `TeeWriter` の bare `except Exception: pass` (2件)

`event_logger.py:97,105` の `except Exception: pass` は完全に silent swallow。
TeeWriter は stderr ミラーリングに使うため、片方の writer が壊れたとき
何の通知もなく書き込みが消失する。

**修正**: `pass` → `logger.debug(f"TeeWriter.write failed: {e}")` (最低限の通知)

---

## 2. P2 Items (253# 以降に Defer)

### P2-1: `fill_loop_orchestrator.py` 残 getattr (8件)

主に `saved_state` (リカバリ時のデシリアライズ結果) からの属性取得。
saved_state の型が `object` のため、getattr は構造的に必要。
TypedDict / dataclass 化すれば除去可能だが、影響範囲が大きく 253# 向きではない。

| 行 | 対象 |
|----|------|
| L392 | `saved_state.soft_drawdown_interval_multiplier` |
| L420 | `saved_state.mcb_state` |
| L428 | `saved_state.sad_state` |
| L437 | `saved_state.degraded_liquidation_duty_counter` |
| L441 | `saved_state.one_sided_cooldown_remaining` |
| L445 | `saved_state.one_sided_freeze_remaining` |
| L449 | `saved_state.consecutive_no_feasible` |
| L533 | `self._recent_records` (属性存在不確実) |

---

### P2-2: `fill_loop_orchestrator.py` God Object (2281行)

247# で 2356行 → 現在 2281行。微減だが依然として巨大。
短期では実害は少ないが、次の大型機能追加前に
`saved_state` 管理と heartbeat を分離モジュール化すべき。

---

### P2-3: 247# §1.10 — one-sided freeze/cooldown の side-bound 化

`fill_loop_orchestrator.py` の one-sided freeze/cooldown は
`_one_sided_frozen_side: str | None` を持たず、凍結時の side が記録されない。
在庫変動で wrong side を凍結する可能性がある。
影響は低いが、正確性を上げるために将来対応。

---

### P2-4: 247# §1.11 — degraded_liquidation パラメータ validation

以下の新設パラメータに値域検証がない:
- `degraded_liquidation_lot_mult` (0 < x ≤ 1.0 であるべき)
- `degraded_liquidation_offset_mult` (> 1.0 であるべき)
- `degraded_liquidation_duty_cycle` (≥ 1 であるべき)

`fill_config.py` の `__post_init__` に追加すべき。

---

### P2-5: bare `except Exception:` 残存 (9件)

| ファイル | 行 | コンテキスト |
|----------|-----|------------|
| `fill_cycle_executor.py` | L1187 | OB re-fetch during retry — 許容可 |
| `fill_test_cli.py` | L351, L424 | CLI cleanup — 許容可 |
| `lock_manager.py` | L155 | Lock release — 許容可 |
| `ob_utils.py` | L124, L133 | OB fallback — 許容可 |
| `resilience.py` | L175 | Circuit state — 許容可 |
| `pnl_measurer.py` | L112 | PnL fallback — 許容可 |
| `fill_loop_orchestrator.py` | L1218 | Error handling — 許容可 |

ほとんどが fallback / cleanup パスで実害は低い。
将来的にデバッグ可観測性向上のために `logger.debug` 追加を検討。

---

### P2-6: 247# §1.5 — DD cooldown release 再武装

246# の cooldown release は一方向解除であり、release 後の再悪化で
2回目の halt が発動しない。`re-arm` ロジック未実装。
重要だが、設計を慎重に検討する必要があるため 253# では defer。

---

## 3. 252# Self-Review 詳細

### 3.1 Sell Asymmetric Gate (`cycle_gate_aggregator.py:460-510`)

✅ **良い点**:
- `_is_high_vol` と `_is_trending` を分離し、条件分岐が明瞭
- `skip_sell_trending_up_only` を trending のみに限定し、high_vol は独立判定
- 安全弁 (inv_bypass, HF4, consecutive) が high_vol にも適用
- 市場理論文書化 (Glosten-Milgrom) が docstring に記載

❌ **問題**: hot_reload 配線漏れ (P1-1), YAML 未記載

### 3.2 PhantomGuard 三値化 (`phantom_position_guard.py`)

✅ **良い点**:
- `ReconcileResult` enum が明瞭 (DETECTED / CLEAN / INCONCLUSIVE)
- INCONCLUSIVE → pending 保持 + `reconcile_attempts` 上限管理
- Phase 2b JPY 照合: `jpy_decrease > expected_cost * 0.5` で false positive 防止
- `_maybe_register_phantom` が `last_jpy_free` property を使用 (getattr 除去済み)

✅ **247# §1.8 / §1.9 完全対応確認**: 三値化 + buy JPY 照合 + BalanceChecker property 公開

### 3.3 getattr 除去 (`fill_cycle_executor.py:172`)

✅ `_maybe_register_phantom` の BTC/JPY snapshot 取得が
`self._balance_checker.last_btc_free` / `last_jpy_free` の直接参照に変更済み。

❌ 同ファイルの他 5件が残存 (P1-3)

---

## 4. 247#/248# 残 TODO 消化状況

| 247# Item | 状態 | 対応 |
|-----------|------|------|
| §1.4 dual_kill vs quiescence | ✅ 249# で `dual_kill_quiescence_enabled` 実装済み | — |
| §1.5 DD cooldown 再武装 | ❌ 未実装 | P2-6 defer |
| §1.8 PhantomGuard 三値化 | ✅ 252# T-1/T-2 | — |
| §1.9 buy 側 JPY 照合 | ✅ 252# T-3 | — |
| §1.10 one-sided side-bound | ❌ 未実装 | P2-3 defer |
| §1.11 degraded params validation | ❌ 未実装 | P2-4 defer |

| 248# Item | 状態 | 対応 |
|-----------|------|------|
| P1 Sell Asymmetric | ✅ 252# (high_vol sell skip) | hot_reload 漏れ(P1-1) |
| P0 Regime-Aware Inventory Skew | ❌ 大型設計、defer | 今後の設計課題 |
| P0 Total Equity PnL | ❌ 計測基盤の変更、defer | 今後の設計課題 |

---

## 5. 253# 推奨スコープ

**4 items, 小〜中規模の品質改善イテレーション:**

1. **P1-1**: `sell_asymmetric_high_vol_enabled` hot_reload + YAML 配線
2. **P1-2**: `balance_forced_apply_trending_offset` dead config 完全削除
3. **P1-3**: `fill_cycle_executor.py` getattr → クラスレベル宣言 + 直接参照 (6件)
4. **P1-4**: `event_logger.py` TeeWriter bare except → debug log

**見積もり**: テスト込みで 1 セッション完了可能。既存テスト 3507 の回帰リスクは低い。
