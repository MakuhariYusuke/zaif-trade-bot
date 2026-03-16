# 252# Sell Asymmetric Gate・PhantomGuard 三値化・型安全化

最終更新: 2026-03-03  
前提: 250# (commit `d9974dac7`), 251# (pre-impl review)

## 概要

248# P1 項目 (Sell Asymmetric, PhantomGuard 三値化/buy 照合) を実装し、
型安全化 (getattr 排除) を含む保守性改善を行った。

## 変更一覧

### 1. Sell Asymmetric Gate — high\_vol regime 拡張 (248# P1-1)

**ファイル**:
- `fill_config.py` — `sell_asymmetric_high_vol_enabled` 追加 + YAML wiring
- `cycle_gate_aggregator.py` — `_check_trending_sell` 拡張

**市場理論 (Glosten-Milgrom)**:
情報劣位者 (MM) の sell は trending\_up だけでなく high\_vol でも
逆選択リスクが高い。情報優位者が最も活発な高ボラ環境で無防備に流動性を
提供する行為は、directional alpha 放棄と同義。

**実装**:
- `sell_asymmetric_high_vol_enabled: bool = False` (デフォルト無効 = 安全側)
- `_is_high_vol = regime == "high_vol"` を追加
- entry 条件を `_is_trending or (_is_high_vol and config.sell_asymmetric_high_vol_enabled)` に拡張
- `skip_sell_trending_up_only` は trending regime にのみ影響 (high\_vol は独立判定)
- safety valve (consecutive, HF4, inv\_bypass) は high\_vol でも全て有効

**期待効果**: +2.2 bps/day (248# 見積もり)

### 2. PhantomGuard 三値化 (251# T-1/T-2)

**ファイル**: `phantom_position_guard.py`

**課題**: 従来の PhantomGuard は API 障害時に pending エントリを即座にクリア
していた。「観測不能 ≠ clean」であり、Bayesian 事後確率の安易な 0 更新に
相当する (Gelman et al., BDA3 §3.7)。

**実装**:
- `ReconcileResult` enum: `DETECTED` / `CLEAN` / `INCONCLUSIVE`
- `_reconcile_single()` の戻り値を `tuple[ReconcileResult, PhantomDetection | None]` に変更
- Phase 1 + Phase 2 とも API 障害 → `INCONCLUSIVE` (pending 保持)
- Phase 1 障害 + Phase 2 clean → `CLEAN` (残高で判定可能)
- `PendingReconciliation.reconcile_attempts: int` カウンタ追加
- `_MAX_RECONCILE_ATTEMPTS = 3` — 上限超過で強制パージ (stale と同様)

### 3. PhantomGuard buy 側 JPY 残高照合 (248# P1-3, 251# T-3)

**ファイル**:
- `phantom_position_guard.py` — Phase 2b: JPY 残高差分確認
- `balance_checker.py` — `last_jpy_free` property 追加
- `fill_cycle_executor.py` — `balance_jpy` snapshot 受け渡し

**課題**: `_maybe_register_phantom()` は BTC snapshot のみ渡しており、
buy 側の JPY 残高照合が完全に欠落していた。

**実装**:
- `PhantomDetection.balance_delta_jpy: float | None` フィールド追加
- Phase 2b: buy 側のみ JPY 残高差分を確認 (JPY 減少 > tolerance + > 50% of expected cost)
- `detection_method` に `"balance_delta_jpy"` を追加 (JPY のみで検出時)
- `_BALANCE_TOLERANCE_JPY: float = 50.0` — dust レベルの JPY 差分は無視
- `BalanceChecker.last_jpy_free` property 追加 (既存の `_last_jpy_free` を公開)

### 4. getattr 排除・型安全化

**ファイル**: `fill_cycle_executor.py`

**変更**:
- `getattr(self._balance_checker, 'last_btc_free', None)` を
  `self._balance_checker.last_btc_free` に直接参照化
- `self._balance_checker.last_jpy_free` も同時に渡す

## テスト

**テストファイル**: `test_251_sell_asymmetric_phantom_ternary.py`

| セクション | テスト内容 | テスト数 |
|-----------|----------|---------|
| A. SellAsymmetricHighVol | high_vol sell skip/allow, trending_up_only 非干渉, safety valve, offset | 10 |
| B. ReconcileResult/Inconclusive | enum, API障害→保持, リトライ成功/検出, 上限パージ, カウンタ | 10 |
| C. PhantomGuardJPYReconcile | buy JPY 検出, tolerance, sell 非適用, BTC+JPY 同時, API 障害 | 6 |
| D. BalanceCheckerLastJpyFree | property 存在確認, 初期値 | 2 |
| E. GetAttrRemoval | getattr コード非存在, balance_jpy 渡し確認 | 2 |
| F. ExistingBehaviorUnchanged | 既存動作回帰テスト | 5 |
| **合計** | | **35** |

**既存テスト修正**: `test_237_phantom_position_guard.py` 1件
- `test_order_status_api_error_no_crash`: API 障害時の期待値を INCONCLUSIVE 保持に更新

**テスト結果**: v460 unit 3507 passed (251# +35 tests)

## 変更ファイル一覧

| ファイル | 変更種別 | 概要 |
|---------|---------|------|
| `scripts/v460/lib/fill_config.py` | 修正 | `sell_asymmetric_high_vol_enabled` 追加 + YAML wiring |
| `scripts/v460/lib/cycle_gate_aggregator.py` | 修正 | `_check_trending_sell` high_vol 拡張 + 理論コメント |
| `scripts/v460/lib/phantom_position_guard.py` | 修正 | ReconcileResult enum, 三値化, JPY 照合, retry |
| `scripts/v460/lib/balance_checker.py` | 修正 | `last_jpy_free` property 追加 |
| `scripts/v460/lib/fill_cycle_executor.py` | 修正 | getattr→直接参照, balance_jpy 渡し |
| `tests/unit/v460/test_251_*.py` | 新規 | 35 テスト |
| `tests/unit/v460/test_237_*.py` | 修正 | 1テスト (三値化対応) |

## 残課題 (次回以降)

- **P2-1**: Feasible Quote 完全計算
- **P2-2**: Inventory Target Band 導入 (247# §2.1 — レジーム依存の目標帯)
- **P2-3**: God Object 回帰の抑制 (orchestrator 2,434 lines)
- **Dead config**: `balance_forced_apply_trending_offset` (fill_config.py L438, 234#)
