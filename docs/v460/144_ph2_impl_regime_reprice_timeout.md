# 144# レジーム活用 R-1c/R-1d + 142#/143# レビュー対応

**作成日**: 2026-02-23  
**前提**: 143# R-1a/R-1b 実装済み、142# §6 / 143# §7 レビュー指摘  
**テスト**: 1244 → 1263 (+19)

---

## §1 レビュー対応 (142# §6)

### §1.1 #1 (HIGH): R-2c RegimeAdaptiveTrainer 表現修正

**問題**: `RegimeAdaptiveTrainer` は SAC 向け Mixin であり、fill_test の XGBoost retrain に直結できない。  
**対策**: 142# doc の R-2c を「直接流用」→「設計資産の再利用 + adapter 層別タスク化」に修正。

### §1.2 #2 (MEDIUM): 検証基準に統計ゲート追加

**問題**: 成功指標が平均値のみで統計的有意性を考慮していない。  
**対策**: 142# doc §4 にブートストラップ 95% CI、最低 50 サンプル、fee 控除後指標を必須化。

### §1.3 #3 (MEDIUM): 段階導入方針の明記

**問題**: R-1a/b/c/d の同時投入で寄与分解が困難。  
**対策**: 142# doc §4.1 に「1 変更ずつ A/B 実施」の段階導入を明記。

### §1.4 #4 (LOW): MarketRegime 値数修正

**問題**: 「40+ 値」は不正確、実際は ~20 enum + alias。  
**対策**: 142# doc を「多値 (20+ alias)」に修正。

### §1.5 #5 (MEDIUM): R-1b preflight 拘束の明記

**対策**: 142# doc R-1b の「提案」セルに「※ preflight は調整後 lot 基準」を追記。

---

## §2 レビュー対応 (143# §7)

### §2.1 #1 (HIGH): preflight-lot 整合

**問題**: `BalanceChecker._check_sell/_check_buy` が `self._current_lot` (base) を参照するが、実発注は `_regime_adjusted_lot()` (trending 増量あり)。→ `insufficient_funds` の危険。  
**対策**: `run_single_cycle` 内で `_regime_adjusted_lot()` を `apply_lot_floor()` **直後** に計算。`_order_lot > _current_lot` の場合に `_current_lot` を一時的に引き上げ。以降の preflight balance check が増量済み lot ベースで動作。

**変更**: `scripts/v460/run_fill_test.py` L806-812

### §2.2 #2 (MEDIUM): min_lot 単一ソース化

**問題**: `_regime_adjusted_lot()` が `min_lot = 0.001` をハードコードし、`config.min_order_btc` と二重管理。  
**対策**: `self.config.min_order_btc` に統一。

**変更**: `scripts/v460/run_fill_test.py` L294

### §2.3 #3 (MEDIUM): quarantine bypass 限定

**問題**: ANY `cancel_reason` で side/price/quantity バリデーションを全バイパス → 壊れたレコードも clean 扱い。  
**対策**: `_AUDIT_CANCEL_REASONS` frozenset (9 種) を定義し、監査系 reason のみバイパス。非監査系は従来通り quarantine。

```python
_AUDIT_CANCEL_REASONS = frozenset({
    "circuit_breaker_open", "preflight_pause", "preflight_insufficient",
    "time_filter_both_sides", "time_filter_086_deadlock",
    "narrow_spread_pause", "balance_forced_skip",
    "unknown_regime_buy_skip", "sell_dynamic_kill",
})
```

**変更**: `ztb/metrics/fill_quality.py` `_quarantine_reason()`

### §2.4 #4 (MEDIUM): 動作テスト強化

**対策**: 19 テスト追加 (ソース文字列検査 → 動作確認テストへ拡充):

| カテゴリ | 件数 | 内容 |
|---|---|---|
| quarantine bypass narrowed | 4 | audit/non-audit × side/price の組み合わせ |
| min_lot unification | 2 | source + custom min_order_btc の動作 |
| preflight-lot alignment | 1 | ソース順序検証 |
| R-1c config + YAML | 2 | default + YAML mapping |
| R-1d config + YAML | 2 | default + YAML mapping |
| R-1c order_monitor | 2 | source inspection (offset logic) |
| R-1d order_monitor | 2 | source inspection (timeout logic) |
| R-1c behavioral | 2 | offset clamp / negative offset |
| R-1d behavioral | 2 | multiplier / no-regime fallback |

### §2.5 #5 (LOW): doc 日付修正

**対策**: 143# doc の作成日を `2025-02-24` → `2026-02-22` に修正。

---

## §3 R-1c: レジーム別 reprice 上限適応

### §3.1 概要

142# 計画の R-1c 施策。`stale_max_reprice` にレジーム別オフセットを加算し、市場状況に応じた reprice 粘り強さを実現。

### §3.2 新規 config フィールド

| フィールド | デフォルト | 説明 |
|---|---|---|
| `regime_reprice_adjustments` | `{}` (空 dict) | regime_name → int offset |

YAML マッピング:
```yaml
regime:
  reprice_adjustments:
    high_vol: 1     # +1回 (ボラ時は粘る)
    trending: 2     # +2回 (トレンド追従で積極的reprice)
    ranging: 0      # デフォルト
```

### §3.3 実装方式

`OrderMonitor.monitor()` 内、side 別 `stale_max_reprice` 解決後に regime offset を加算:

```python
_stale_max_rp = max(0, _stale_max_rp_base + _regime_reprice_offset)
```

`max(0, ...)` で負のオフセットによる reprice 無効化も安全にクランプ。

---

## §4 R-1d: レジーム別 timeout 適応

### §4.1 概要

142# 計画の R-1d 施策。`order_timeout_sec` にレジーム別倍率を適用し、市場状況に応じた待機時間を実現。

### §4.2 新規 config フィールド

| フィールド | デフォルト | 説明 |
|---|---|---|
| `regime_timeout_multipliers` | `{}` (空 dict) | regime_name → float multiplier |

YAML マッピング:
```yaml
regime:
  timeout_multipliers:
    high_vol: 0.7   # 63s (早めに撤退)
    trending: 1.3   # 117s (トレンドに乗る)
    ranging: 1.0    # 90s (デフォルト)
```

### §4.3 実装方式

`OrderMonitor.monitor()` 冒頭で `_effective_timeout = order_timeout_sec × multiplier` を計算。while ループの終了条件を `elapsed < _effective_timeout` に変更。

regime_detector が None またはレジーム未検出の場合は multiplier = 1.0 (base timeout)。

---

## §5 テスト (19 件追加)

**テストファイル**: `tests/unit/v460/test_143_regime_utilization.py` (26 → 45, +19)

追加テストクラス:
- `TestQuarantineBypassNarrowed` (4 件)
- `TestMinLotUnification` (2 件)
- `TestPreflightLotAlignment` (1 件)
- `TestRegimeRepriceConfig` (2 件)
- `TestRegimeTimeoutConfig` (2 件)
- `TestRegimeRepriceInOrderMonitor` (2 件)
- `TestRegimeTimeoutInOrderMonitor` (2 件)
- `TestRegimeRepriceMonitorBehavioral` (2 件)
- `TestRegimeTimeoutMonitorBehavioral` (2 件)

**v460 全体**: 1244 → 1263 (+19), 0 failed

---

## §6 変更ファイル一覧

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `scripts/v460/run_fill_test.py` | バグ修正 | #1 preflight-lot 整合, #2 min_lot 統一 |
| `scripts/v460/lib/fill_config.py` | 機能追加 | R-1c `regime_reprice_adjustments`, R-1d `regime_timeout_multipliers` + YAML mapping |
| `scripts/v460/lib/order_monitor.py` | 機能追加 | R-1c reprice offset, R-1d effective timeout, `min_order_btc` 統一 |
| `ztb/metrics/fill_quality.py` | バグ修正 | #3 quarantine bypass 監査系限定 |
| `tests/unit/v460/test_143_regime_utilization.py` | テスト追加 | +19 テスト |
| `tests/unit/v460/test_113_resilience.py` | 閾値更新 | line count 405 → 410 (preflight-lot 3行追加分) |
| `docs/v460/142_ph2_plan_regime_utilization.md` | レビュー対応 | §6 #1-#5 全件修正 |
| `docs/v460/143_ph2_impl_regime_utilization.md` | レビュー対応 | §7 #5 日付修正 |
| `docs/v460/144_ph2_impl_regime_reprice_timeout.md` | 新規 | 本ドキュメント |

---

## §7 134# ロードマップ位置確認

```
Phase A (Data Infra)      : ✅ 135#
Phase B (Observability)   : ✅ 135#
Phase C (Re-measurement)  : ⬜ Operational (24h run 未実施)
Phase D (Retrain restart) : ✅ 136#
Phase E (P1 group)        : ✅ 137#-141# (全 9 項目完了)
142# Self-check           : ✅ C-1/M-1/M-3 修正
143# R-1a/R-1b            : ✅ offset + lot regime adaptation
144# R-1c/R-1d + review   : ✅ 本セッション (reprice + timeout + review 10件)
```

**R-1 全サブタスク完了**: R-1a (offset), R-1b (lot), R-1c (reprice), R-1d (timeout)

**次ステップ**: P2 グループ (P2-01〜P2-12) / Phase C 24h 実測 / R-2 retrain 重み付け
