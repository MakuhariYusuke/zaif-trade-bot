# 157: §20 レジームデッドロック修正 + cancel/spread 副次課題解決

> **セッション**: 158#  
> **コミット**: `29353ee41` (§20 本体), `a7e5d0b82` (§20 セルフレビュー修正)  
> **種別**: fix  
> **フェーズ**: ph2 (G1.1-exec)  
> **日付**: 2026-02-24  
> **テスト**: 24 新規 ALL PASSED / v460 unit 1660 passed / 2 pre-existing

---

## 概要

Phase C 168h dry-run のログ分析で発見された **レジームデッドロック** を修正。
加えて、cancel 400 エラーハンドリングと spread_too_narrow 分類の副次課題を解決。

セルフレビューにて、cancel 修正 (§20-C) に **重大なリグレッション** を発見し、
同セッション内で追加修正を実施。

---

## 問題分析

### 根本原因: レジームデッドロック

`regime_detector.update()` が `run_single_cycle()` 内でのみ呼ばれていた。skip パス
(trending_sell_skip, balance_forced_skip, unknown_buy_skip, dynamic_kill) は
`run_single_cycle()` をバイパスするため、**skip 中はレジームが一切更新されない**。

```
[デッドロック経路]
regime=trending → sell skip (trending protection)
  → run_single_cycle() 未到達
  → regime_detector.update() 未呼出
  → regime="trending" のまま固着
  → sell 永久スキップ
```

Phase C ログでは regime=trending, stability=7 が固定。cycle_count=2500 時点で
cumulative_pnl=-282.69 JPY と損失拡大、sell 機会の大半が消失していた。

### 副次課題

| 課題 | 影響 |
|------|------|
| cancel 400 "Failed to cancel" → ERROR ログ増大 | 約定済み注文 cancel 試行は正常系 |
| spread_too_narrow が orderbook_error に混入 | 異常と正常の区別不能 |

---

## 修正内容

### Fix A (§20-A): メインループ毎レジーム更新 — ROOT CAUSE FIX

**場所**: [scripts/v460/run_fill_test.py](../../scripts/v460/run_fill_test.py) `FillTestRunner.run_continuous()`

```python
# §20-A: main loop regime update (skip パスでもデッドロックしない)
if self._regime_detector:
    fb_price, _fb_time = self._maker_price.get_fallback_price()
    if fb_price is not None:
        old_regime = self._regime_detector.current_regime
        self._regime_detector.update(fb_price)
        new_regime = self._regime_detector.current_regime
        if old_regime != new_regime:
            logger.info("[§20-A] regime transition: %s → %s", old_regime, new_regime)
```

- `run_continuous()` のメインループ先頭 (全 skip 判定よりも前) で毎回呼び出し
- `get_fallback_price()` で直近 OB mid price (キャッシュ) を取得
- `run_single_cycle()` 内の既存呼び出しはそのまま維持 (二重更新は冪等)

### Fix B (§20-B): 連続 trending sell skip 安全弁

**場所**: [scripts/v460/run_fill_test.py](../../scripts/v460/run_fill_test.py), [scripts/v460/lib/fill_config.py](../../scripts/v460/lib/fill_config.py)

- `max_consecutive_trending_sell_skip: int = 30` (YAML: `止血.max_consecutive_trending_sell_skip`)
- `_trending_sell_skip_count` カウンター: skip 時 +1、sell 実行 or regime 遷移時 reset
- N 回超過で sell を **強制許可** (WARNING ログ付き)
- 0=無制限 (安全弁無効化)

```python
# §20-B: 安全弁チェック
if self._trending_sell_skip_count >= self._max_consec and self._max_consec > 0:
    logger.warning("[§20-B] trending sell skip 安全弁: %d 回超過 → sell 強制許可",
                   self._trending_sell_skip_count)
    _should_skip = False
```

### Fix C (§20-C): cancel_failed 400 ハンドリング改善

**場所**: [ztb/trading/live/exchanges/coincheck/adapter.py](../../ztb/trading/live/exchanges/coincheck/adapter.py) `_cancel_order_real()`

3 段階のエラー分類:

| パターン | 処理 | 理由 |
|----------|------|------|
| "not found" / "already cancelled" | `return False` | 注文が確実に不在 |
| "failed to cancel" | WARNING + **re-raise** | 約定済みの可能性 → 呼出元で fill recheck |
| 不明エラー | ERROR + **re-raise** | 安全側 |

### Fix D (§20-D): spread_too_narrow 分類

**場所**: [scripts/v460/run_fill_test.py](../../scripts/v460/run_fill_test.py), [scripts/v460/lib/cancel_reasons.py](../../scripts/v460/lib/cancel_reasons.py)

- `CR.SPREAD_TOO_NARROW` 定数追加
- `orderbook_error` から分離して専用分類
- ログレベル ERROR → INFO (スプレッド縮小は正常な市場状態)

---

## セルフレビューで発見した重大バグ (§20-C)

### 問題

初回コミット `29353ee41` の §20-C では、"Failed to cancel" を含む**全パターン**で
`return False` としていた。これが order_monitor.py の **Bug11 fill recheck** パスを
デッドコード化するリグレッションを引き起こした。

```
[リグレッション経路]
注文が cancel/fill 競合 → adapter: "Failed to cancel" → return False (例外なし)
→ order_monitor: cancel 成功と誤認 → "Cancelled unfilled" ログ
→ 実際は fill 済み → fill 検出ロスト → ポジション不整合
```

### order_monitor.py の関連パス (UNCHANGED)

- **L286–301**: stale_order cancel → 例外 catch → "Failed to cancel" → fill recheck
- **L378–400**: post-timeout cancel → 例外 catch → Bug11 fill recheck

### 修正 (`a7e5d0b82`)

"Failed to cancel" は `return False` ではなく **WARNING + re-raise** に変更。
order_monitor のキャッチブロックが正しく動作し、fill recheck が実行される。

"not found" / "already cancelled" は元々 `return False` だったため変更なし。

---

## 変更ファイル一覧

| ファイル | 変更量 | 内容 |
|----------|--------|------|
| [scripts/v460/run_fill_test.py](../../scripts/v460/run_fill_test.py) | +63 lines | §20-A,B,D 実装 |
| [scripts/v460/lib/fill_config.py](../../scripts/v460/lib/fill_config.py) | +5 lines | `max_consecutive_trending_sell_skip` 設定 |
| [scripts/v460/lib/cancel_reasons.py](../../scripts/v460/lib/cancel_reasons.py) | +1 line | `SPREAD_TOO_NARROW` 定数 |
| [ztb/trading/live/exchanges/coincheck/adapter.py](../../ztb/trading/live/exchanges/coincheck/adapter.py) | +24/-9 lines | §20-C 3段階 cancel エラー分類 |
| [tests/unit/v460/test_158_regime_deadlock_fix.py](../../tests/unit/v460/test_158_regime_deadlock_fix.py) | +388 lines (新規) | 24 テスト |
| CHANGELOG.md | +8 lines | §20 エントリ |

---

## テスト

[test_158_regime_deadlock_fix.py](../../tests/unit/v460/test_158_regime_deadlock_fix.py) — 24 tests ALL PASSED

| クラス | テスト数 | 検証内容 |
|--------|----------|----------|
| TestRegimeUpdateDuringSkip | 4 | §20-A ソース検査、skip 前配置、detector 更新、trending→ranging 遷移 |
| TestRegimeUpdateLogging | 1 | 遷移ログフォーマット |
| TestMaxConsecutiveTrendingSellSkip | 8 | config, YAML, counter, 安全弁, reset, logging |
| TestCancelFailedHandling | 5 | adapter ソース, パターン分岐, unknown raises, "Failed to cancel" re-raises, "not found" returns False |
| TestSpreadTooNarrowClassification | 3 | 定数, 分類, ログレベル |
| TestIntegrationConsistency | 3 | skip パス網羅, デッドコードなし, 安全デフォルト |

v460 全体: **1660 passed / 2 pre-existing failures** (0 regressions)

---

## 再起動後の状態

| 項目 | 値 |
|------|-----|
| PID (main) | 124796 |
| PID (retrain_scheduler) | 107960 |
| git_sha | `a7e5d0b82` |
| 開始時刻 | 2026-02-24 03:57:04 |
| 設定 | 168h, coincheck, interval=120s |
| 状態復元 | regime=trending, stability=7, prices=60, cycle=2723 |
| 残高 | JPY 12,346 + BTC 0.001 (評価額 ≈22,356 JPY) |
| fill records | 2,723件 (clean=2,411, quarantine=312) |
| loss_cap | 動的 1,118 JPY (残高 × 5%) |

### 期待効果

1. **regime 遷移の正常化**: skip パス中も `regime_detector.update()` が毎ループ呼ばれるため、trending 固着が解消
2. **sell 機会の回復**: ranging 遷移時に sell が再開、損失拡大を抑止
3. **安全弁**: 万一 trending が継続しても `max_consecutive_trending_sell_skip=30` で sell を強制許可
4. **fill 検出の健全性**: "Failed to cancel" re-raise により order_monitor の Bug11 fill recheck が正常機能
5. **ログ品質**: spread_too_narrow 分類 + cancel WARNING 降格でノイズ低減

---

## 教訓

1. **デッドロックパターン**: 状態更新が特定実行パスにのみ含まれる場合、バイパスパスで状態が固着する。状態更新はループ最上位で行うべき。
2. **セルフレビューの価値**: §20-C の `return False` 変更が order_monitor の fill recheck を破壊することは、コード修正時の影響範囲分析だけでは見落としやすい。呼び出し元の例外処理パスまで遡る系統的レビューが必要。
3. **例外の意味論**: `return False` (成功裏の不在) と `raise` (呼出元に判断を委ねる) は質的に異なる。"Failed to cancel" は後者 — 約定済みの可能性があり、fill recheck が必要。

---

## 関連ドキュメント

- [156_ph2_rpt_sell_root_cause_and_phase_d_plan.md](156_ph2_rpt_sell_root_cause_and_phase_d_plan.md) — Phase C/D 並行計画、sell 構造分析
- [154_ph2_dryrun_10h_analysis.md](154_ph2_dryrun_10h_analysis.md) — P0-08 deadlock 発見
- [143_ph2_impl_regime_utilization.md](143_ph2_impl_regime_utilization.md) — Regime 活用実装 (trending_sell_skip の導入)
- [047_ph2_fix_cancel_race_and_gate_alignment.md](047_ph2_fix_cancel_race_and_gate_alignment.md) — Cancel Race 修正 (Bug11 の起源)
