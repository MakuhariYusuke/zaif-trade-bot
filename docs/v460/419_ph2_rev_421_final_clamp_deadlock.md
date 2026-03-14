# 419# Self-Review: 421# Execution Final Clamp + Route-to-Kill Deadlock

**対象コミット**: `4aa779d27` (421#)
**レビュー日**: 2026-03-14
**レビュアー**: AI (自動レビュー)

---

## Summary: 9 項目の検査結果

| # | 項目 | 深刻度 | ステータス |
|---|------|--------|-----------|
| 1 | Ceiling 解決ロジック重複 | **MEDIUM** | 要改善 |
| 2 | CancelReason Literal 型エイリアス未追加 | **HIGH** | 要修正 |
| 3 | YAML config / fill_config_parser 未配線 | **HIGH** | 要修正 |
| 4 | fill_record_builder 署名整合 | LOW | ✅ 問題なし |
| 5 | 下流分析スクリプト互換性 | **MEDIUM** | 要修正 |
| 6 | Route-to-Kill MRO 安全性 | LOW | ✅ 問題なし |
| 7 | guard_fire 分類マッピング不足 | **MEDIUM** | 要修正 |
| 8 | テストカバレッジのギャップ | **MEDIUM** | 要追加 |
| 9 | spread_at_order=None エッジケース | **MEDIUM** | 要確認 |

---

## 1. Ceiling 解決ロジック重複 — MEDIUM

### 検出箇所 (3 箇所)

**箇所 A** — `scripts/v460/lib/fill_cycle_executor.py` L746-750:
```python
_fc_ceil = self.config.offset_ceiling_ratio
if side == "buy" and self.config.offset_ceiling_ratio_buy is not None:
    _fc_ceil = self.config.offset_ceiling_ratio_buy
elif side == "sell" and self.config.offset_ceiling_ratio_sell is not None:
    _fc_ceil = self.config.offset_ceiling_ratio_sell
```

**箇所 B** — `scripts/v460/lib/maker_price.py` L1015-1019:
```python
_ceil = cfg.offset_ceiling_ratio
if side == "buy" and cfg.offset_ceiling_ratio_buy is not None:
    _ceil = cfg.offset_ceiling_ratio_buy
elif side == "sell" and cfg.offset_ceiling_ratio_sell is not None:
    _ceil = cfg.offset_ceiling_ratio_sell
```

**箇所 C** — `scripts/v460/lib/maker_price.py` L562-567 (`_effective_max_ratio`):
```python
if side == "sell" and cfg.offset_ceiling_ratio_sell is not None:
    return max(base, cfg.offset_ceiling_ratio_sell)
if side == "buy" and cfg.offset_ceiling_ratio_buy is not None:
    return max(base, cfg.offset_ceiling_ratio_buy)
```
**注意**: 箇所 C は semantics が異なる (`max(base, ceiling)`) が、ceiling データの引き方は共通。

### 共有ヘルパーの有無
- **存在しない**。test_421 に `_resolve_ceiling()` テストヘルパーがあるが、本番コードには未抽出。

### 推奨修正
`FillTestConfig` に `resolve_ceiling_ratio(side: str) -> float` メソッドを追加し、3 箇所から呼び出す。
```python
# fill_config.py に追加
def resolve_ceiling_ratio(self, side: str) -> float:
    if side == "buy" and self.offset_ceiling_ratio_buy is not None:
        return self.offset_ceiling_ratio_buy
    if side == "sell" and self.offset_ceiling_ratio_sell is not None:
        return self.offset_ceiling_ratio_sell
    return self.offset_ceiling_ratio
```

---

## 2. CancelReason Literal 型エイリアス未追加 — HIGH

### 検出箇所
`scripts/v460/lib/cancel_reasons.py` L148-206 の `CancelReason = Literal[...]` に
`"final_clamp_hard_skip"` と `"route_to_kill_deadlock"` が **未登録**。

定数自体 (L74, L76) と `AUDIT_CANCEL_REASONS` frozenset (L112-113) には追加済みだが、
`CancelReason` Literal 型エイリアス (L148) のドキュメントにも
> 「すべての有効な cancel_reason 値を列挙。新規追加時はここにも追記。」

と明記されている。

### 影響
- mypy で `CancelReason` 型を引数に取る関数に新定数を渡すと型エラーになる可能性
- 型安全性が損なわれる

### 推奨修正
`cancel_reasons.py` L204 (`"poll_error_limit"` の後) に以下を追加:
```python
    # 421# Execution Final Clamp / Route-to-Kill Deadlock
    "final_clamp_hard_skip",
    "route_to_kill_deadlock",
```

---

## 3. YAML config / fill_config_parser 未配線 — HIGH

### 3a. fill_test.yaml に未追加
`configs/v460/fill_test.yaml` に `execution_final_clamp_enabled` と
`execution_final_clamp_hard_skip_mult` が**存在しない**。

現状 `FillTestConfig` のデフォルト値 (`enabled=True`, `hard_skip_mult=0.0`) でフォールバックするが、
運用上 YAML で明示的にチューニング可能にすべき。

### 3b. fill_config_parser.py に未追加
`scripts/v460/lib/fill_config_parser.py` に `execution_final_clamp_*` フィールドの
パース処理が**存在しない**。

YAML に値を書いても `FillTestConfig` に反映されない。

### 3c. config_hot_reload.py に未追加
`scripts/v460/lib/config_hot_reload.py` のホットリロード対象リストに
`execution_final_clamp_enabled` / `execution_final_clamp_hard_skip_mult` が**存在しない**。

### 推奨修正

**fill_config_parser.py** (L983 付近、offset_ceiling 後に):
```python
# 421# Execution Final Clamp
if "execution_final_clamp_enabled" in yaml_cfg:
    kwargs["execution_final_clamp_enabled"] = bool(yaml_cfg["execution_final_clamp_enabled"])
if "execution_final_clamp_hard_skip_mult" in yaml_cfg:
    kwargs["execution_final_clamp_hard_skip_mult"] = float(yaml_cfg["execution_final_clamp_hard_skip_mult"])
```

**fill_test.yaml** (offset_ceiling 後に):
```yaml
# ---- 421# P0: Execution Final Clamp (post-ceiling multiplier leak 防止) ----
execution_final_clamp_enabled: true            # Final Clamp 有効化
execution_final_clamp_hard_skip_mult: 0.0      # >0 で hard skip 有効 (例: 2.0)
```

**config_hot_reload.py** (offset_ceiling_ratio_sell の後に):
```python
"execution_final_clamp_enabled",
"execution_final_clamp_hard_skip_mult",
```

---

## 4. fill_record_builder _build_fill_record 署名整合 — LOW ✅

### 検証結果
- `fill_record_builder.py` L303: `execution_pre_clamp_offset: float | None = None` パラメータ追加済み
- `fill_cycle_executor.py` L1140: `execution_pre_clamp_offset=_execution_pre_clamp_offset` で正しく渡している
- `FillRecord` (fill_quality.py L177): フィールド追加済み、`to_dict`/`from_dict` にも反映
- **問題なし**

---

## 5. 下流分析スクリプト / テスト互換性 — MEDIUM

### 5a. test_145 expected set が stale
`tests/unit/v460/test_145_structural_fixes.py` L147-184 の
`test_audit_reasons_frozenset_matches_constants` テストが **実際に FAIL する**。

Expected set に `CR.FINAL_CLAMP_HARD_SKIP` と `CR.ROUTE_TO_KILL_DEADLOCK` が未追加。

```
E       Extra items in the left set:
E       'route_to_kill_deadlock'
E       'final_clamp_hard_skip'
```

### 推奨修正
`test_145_structural_fixes.py` L182 (`CR.POLL_ERROR_LIMIT` の後) に追加:
```python
            CR.FINAL_CLAMP_HARD_SKIP,          # 421# Final Clamp
            CR.ROUTE_TO_KILL_DEADLOCK,          # 421# Route-to-Kill
```

### 5b. fill_quality quarantine ロジック
`ztb/metrics/fill_quality.py` L1552 で `AUDIT_CANCEL_REASONS` を参照して quarantine 判定。
新定数は `AUDIT_CANCEL_REASONS` に追加済みのため、quarantine bypass は正しく動作する。**問題なし**。

---

## 6. Route-to-Kill MRO 安全性 — LOW ✅

### 検証結果
`fill_loop_orchestrator.py` L86-93:
```python
class FillLoopOrchestratorMixin(
    OrchestratorBalanceMixin,    # ← _resolve_balance_and_preflight
    OrchestratorGuardsMixin,     # ← _is_side_killed
    ...
)
```

- `OrchestratorBalanceMixin` は独立 mixin (base class なし)
- `OrchestratorGuardsMixin` も独立 mixin
- 両方が `FillLoopOrchestratorMixin` で合流
- `self._is_side_killed()` は MRO で `OrchestratorGuardsMixin` から解決される
- **問題なし** — Python MRO (C3 linearization) で正しく動作

---

## 7. guard_fire 分類マッピング不足 — MEDIUM

### 検出箇所
`scripts/v460/lib/guard_reason_classifier.py` の `_CLASSIFICATION` dict に
`"route_to_kill_deadlock"` が**未登録**。

現状のフォールバック動作:
```python
def classify_guard(guard_name: str) -> GuardCategory:
    return _CLASSIFICATION.get(guard_name, GuardCategory.SYSTEM)
```
→ `SYSTEM` にフォールバック。

### 影響
- guard_category_totals の SYSTEM カウントが過大計上される
- `route_to_kill_deadlock` は「buy 残高不足 × sell kill-gated」のデッドロック回避であり、
  semantics としては **RECOVERY** が適切

### 推奨修正
`guard_reason_classifier.py` L86 付近 (RECOVERY セクション末尾) に追加:
```python
"route_to_kill_deadlock": GuardCategory.RECOVERY,  # 421# buy残高不足×sell kill-gated
```

---

## 8. テストカバレッジのギャップ — MEDIUM

### 既存テスト (25 件、全 PASS)
- CancelReason 定数存在確認 (4 件)
- FillConfig フィールド (3 件)
- FillRecord フィールド (4 件)
- Final Clamp ロジック・シミュレーション (11 件)
- Ceiling 解決 (4 件)

### 不足テストケース

#### 8a. spread_at_order=None + Final Clamp
`_recalc_price_with_new_offset(side, price, None, old, new)` は `order_price` をそのまま返すが、
`effective_offset_ratio` は ceiling に設定される。この組み合わせのテストが存在しない。

#### 8b. Route-to-Kill デッドロック シナリオテスト
`OrchestratorBalanceMixin._resolve_balance_and_preflight` の
Route-to-Kill パスを直接テストするケースがない。
mock ベースで `_is_side_killed(opposite) == True` のシナリオを検証すべき。

#### 8c. degraded_liquidation + final_clamp 相互作用
degraded_liquidation が offset を拡大 → Final Clamp で ceiling に再クランプ。
この連鎖が正しく動作するかのテストがない。

#### 8d. Integration-style テスト
`FillCycleExecutor` の実際の `run_single_cycle` (mock 環境) で
Final Clamp が発火する end-to-end テストがない。

#### 8e. fill_config_parser roundtrip
YAML → `FillTestConfig` → `execution_final_clamp_*` フィールドが正しくパースされるテスト
(§3 の修正後に必要)。

---

## 9. spread_at_order=None エッジケース — MEDIUM

### 検出箇所
`fill_cycle_executor.py` L771-774:
```python
order_price = self._recalc_price_with_new_offset(
    side, order_price, spread_at_order,
    effective_offset_ratio, _fc_ceil,
)
...
effective_offset_ratio = _fc_ceil
```

`_recalc_price_with_new_offset` (pre_order_adjustments.py L40):
```python
if spread_at_order is None or spread_at_order <= 0:
    return order_price
```

### 矛盾
- `spread_at_order=None` の場合、`order_price` は変更されない
- しかし `effective_offset_ratio` は `_fc_ceil` に設定される
- FillRecord には `execution_pre_clamp_offset` が記録される

→ **price は old_ratio のまま、ratio だけ ceiling に書き換わる不整合**

### 影響
- `spread_at_order=None` は稀なケース (OB 取得失敗時) だが、
  FillRecord の `effective_offset_ratio` と `order_price` の関係が壊れる
- 後続の PnL 分析・spread capture 計算に影響する可能性

### 推奨修正案
```python
# spread 不明時は clamp せず、ログ警告のみ
if spread_at_order is None or spread_at_order <= 0:
    logger.warning(
        f"[421# final_clamp] {side}: spread unavailable — "
        f"cannot recalculate price, skipping clamp"
    )
else:
    order_price = self._recalc_price_with_new_offset(...)
    effective_offset_ratio = _fc_ceil
```

または degraded_liquidation (L504-508) と同じパターンで
「offset は変更するが price 再計算不可」のログを出す。

---

## アクションアイテム (優先度順)

| 優先度 | 修正内容 | 影響範囲 |
|--------|----------|----------|
| **P0** | §2: CancelReason Literal に 2 定数追加 | 型安全性 |
| **P0** | §3b: fill_config_parser にパース処理追加 | YAML 設定が効かない |
| **P0** | §5a: test_145 expected set 更新 | CI 破損 (テスト FAIL) |
| **P1** | §3a: fill_test.yaml に明示的設定追加 | 運用可視性 |
| **P1** | §7: guard_reason_classifier に分類追加 | メトリクス精度 |
| **P2** | §1: ceiling 解決ヘルパー共通化 | DRY 原則 |
| **P2** | §3c: config_hot_reload 対象追加 | Hot reload 対応 |
| **P2** | §8: 不足テスト追加 (5 件) | テストカバレッジ |
| **P2** | §9: spread_at_order=None ガード改善 | エッジケース堅牢性 |
