# 292# 290#/291# Review 評価 & ev_weighted 可観測性強化

> **目的**: 290# (Codex) と 291# (Gemini 3.1 Pro) の外部レビューを評価し、  
> 妥当な提案を実装する  
> **日付**: 2026-03-07  
> **前提**: 289# v3 buy 側分析完了、290#/291# 外部レビュー受領済み

---

## 1. 両レビューの妥当性評価

### 1.1 290# (Codex) の評価

| 指摘 | 妥当性 | 判定 | 対応 |
|---|---|---|---|
| `model_used` は ev_as_offset の誤プロキシ | ✅ **正確** | コード確認済 (skip_gate_evaluator.py L1203-1207) | P0: FillRecord 3 フィールド追加 |
| FillRecord に ev_score_pretrade 追加 | ✅ 有効 | ev_score が未記録だった | 実装済 |
| FillRecord に offset_mult 追加 | ✅ 有効 | 適用乗数が未記録 | 実装済 |
| FillRecord に decision_path 追加 | ✅ 有効 | パス判別が不可能だった | 実装済 |
| forced buy KPI 分離 | ✅ 既存 | 286# で `forced_buy_kpi_tracking` として実装済 | 不要 |

**総評**: 290# の核心指摘 (`model_used` 誤プロキシ) は **100% 正確**。  
289# v3 の「9.6% 利用率」は完全に誤った計測であり、290# のおかげで判明。

### 1.2 291# (Gemini 3.1 Pro) の評価

| 指摘 | 妥当性 | 判定 | 対応 |
|---|---|---|---|
| Queue Position 放棄メカニズム | ⚠️ **一部誤り** | ev_offset は注文時 1 回のみで「毎秒変動」ではない | 但し reprice ペナルティは実在 (reprice=1 PnL=-4.674) |
| Reprice deadband | ✅ 有効 | 小さな価格差での reprice は queue 犠牲の割にメリットなし | P1: 実装済 (min_delta_jpy=500) |
| Forced buy 毒性 veto | ⚠️ **既存** | 286# `forced_buy_delay` として実装済 | Gemini は 286# を認識していない |
| `is_balance_forced` 追加 | ⚠️ **既存** | `balance_forced_switch` が FillRecord に既存 | 不要 |
| ev_score 記録 | ✅ 有効 | 290# と同内容 | P0 で対応済 |
| Regime-aware forced_buy_delay | ✅ 有効 | ranging/trending_down では閾値緩和 | P1: 実装済 |

**総評**: メカニズムの詳細に誤りがあるが、問題意識の方向性は正しい。  
reprice deadband と regime-aware delay は有効な提案。
既存実装 (286# forced_buy_delay, balance_forced_switch) を知らないために重複提案あり。

---

## 2. 実装内容

### 2.1 P0: FillRecord 可観測性強化 (ev_weighted)

**ファイル**: `ztb/metrics/fill_quality.py`, `scripts/v460/lib/fill_cycle_executor.py`

FillRecord に 3 フィールド追加:

| フィールド | 型 | 説明 |
|---|---|---|
| `ev_score_pretrade` | `float \| None` | ランタイム ev_score (ex-ante 予測値) |
| `ev_offset_mult_applied` | `float \| None` | 実適用 offset 乗数 (1.0=変更なし) |
| `decision_path` | `str \| None` | `"primary_only"` / `"ev_offset"` / `"ev_emergency_skip"` / `"ev_no_change"` |

**decision_path 導出ロジック**:
```python
if ev_score_pretrade is not None:
    if "emergency_skip" in sg_reason → "ev_emergency_skip"
    elif ev_offset_applied           → "ev_offset"
    else                             → "ev_no_change"
else:
    → "primary_only"
```

**影響**: 行動変更なし。記録のみ追加。

### 2.2 P1: Reprice Deadband

**ファイル**: `scripts/v460/lib/order_monitor.py`, `scripts/v460/lib/fill_config.py`

新設定: `stale_reprice_min_delta_jpy` (default=0.0, 本番=500.0)

- Reprice 時に `|new_price - order_price| < min_delta_jpy` の場合、
  queue position 保護のため reprice をスキップ
- スキップ時は同じ価格で再発注して queue 復帰

**根拠**: reprice=1 の PnL=-4.674bps (n=11) vs reprice=0 の PnL=-0.486bps (n=198)。
reprice 自体がコストとなっている場合、微小な価格改善は queue 犠牲に見合わない。

### 2.3 P1: Forced Buy Delay Regime-Aware 強化

**ファイル**: `scripts/v460/lib/fill_loop_orchestrator.py`, `scripts/v460/lib/fill_config.py`

新設定: `forced_buy_delay_velocity_threshold_ranging_bps` (default=None, 本番=-3.0)

- `ranging` / `trending_down` レジーム時に、より緩い velocity 閾値を適用
- 通常: velocity ≤ -5.0bps で delay → ranging 時: velocity ≤ -3.0bps で delay
- これらのレジームでは buy が本質的にリスキーなため、早期にガードを発動

### 2.4 289# v4 更新

`model_used` の誤プロキシ問題を反映:
- §4.5, §4.6: 「9.6%」の主張に修正注記追加
- §8: 利用率調査 → 解消済み
- §10.1, §12.1 F-9, §12.2 V-3: 修正

---

## 3. テスト

16/16 PASSED (`test_292_observability.py`):
- FillRecord 新フィールド: default None、round-trip、build_fill_record 受容
- Config: stale_reprice_min_delta_jpy default/explicit/YAML
- Config: forced_buy_delay_velocity_threshold_ranging_bps default/explicit/YAML
- 本番 YAML 存在確認

既存テスト: 256/256 PASSED (リグレッションなし)

---

## 4. 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `ztb/metrics/fill_quality.py` | FillRecord 3 フィールド追加 |
| `scripts/v460/lib/fill_cycle_executor.py` | ev_score/offset_mult 捕捉、decision_path 導出、配線 |
| `scripts/v460/lib/fill_config.py` | `stale_reprice_min_delta_jpy`, `forced_buy_delay_velocity_threshold_ranging_bps` |
| `scripts/v460/lib/order_monitor.py` | reprice deadband チェック |
| `scripts/v460/lib/fill_loop_orchestrator.py` | regime-aware delay 閾値 |
| `configs/v460/fill_test.yaml` | `reprice_min_delta_jpy: 500`, `velocity_threshold_ranging_bps: -3.0` |
| `docs/v460/289_ph2_analysis_buy_side_improvement.md` | v4 修正 (model_used 誤プロキシ) |
| `tests/unit/v460/test_292_observability.py` | 新テスト 16 件 |
