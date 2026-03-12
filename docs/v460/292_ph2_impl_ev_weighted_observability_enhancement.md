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

## 3. ブラインドスポット修正 (292# v2)

290#/291# レビュアー双方が見落としていた実装バグ 3件を独自分析で発見・修正:

### 3.1 BS-1 (CRITICAL): Reprice Deadband がキャンセル後に評価されていた

**問題**: deadband チェックが `cancel` → `compute_maker_price` → deadband 判定の順序で、
deadband スキップしてもキュー位置は既に失われている。

**修正**: `compute_maker_price` をキャンセル前に実行し、
`|new_price - order_price| < min_delta_jpy` の場合はキャンセル自体をスキップ。  
→ **キュー位置が完全に保護される**

**ファイル**: `scripts/v460/lib/order_monitor.py` (favorable drift セクション)

### 3.2 BS-3 (HIGH): Skip レコードに ev フィールドが欠落

**問題**: `_make_skip_fill_record` で生成される emergency_skip レコードに
`ev_score_pretrade`, `decision_path` が含まれていなかった。  
→ 「ev_emergency_skip は発生しているのか？」の分析が不可能。

**修正**: `_make_skip_fill_record` に `ev_score_pretrade`, `decision_path` の
optional パラメータを追加し、emergency_skip 時に自動で渡す。

**ファイル**: `scripts/v460/lib/skip_gate_evaluator.py`

### 3.3 BS-4 (MEDIUM): mult=1.0 と「ev 未計算」の区別不能

**問題**: `_apply_offset_multiplier` が mult=1.0 で `None` を返すと、
`ev_offset_mult_applied = None` → ev 未計算と同じ状態になる。

**修正**: else 分岐で `_ev_mult` (計算済み乗数) を記録。
`ev_offset_mult_applied=1.0, decision_path="ev_no_change"` で明示的に区別。

**ファイル**: `scripts/v460/lib/fill_cycle_executor.py`

---

## 4. セルフレビュー追加修正 (292# v3)

### 4.1 BUG-1/BUG-2 (P0): Config Hot-Reload 配線漏れ

**問題**: `stale_reprice_min_delta_jpy` と `forced_buy_delay_*` 4フィールドが
`config_hot_reload.py` の `_HOT_RELOADABLE_FIELDS` に未登録。  
→ YAML 変更してもプロセス再起動なしでは反映されない。

**修正**: `_HOT_RELOADABLE_FIELDS` に 5 フィールドを追加。

**ファイル**: `scripts/v460/lib/config_hot_reload.py`

### 4.2 M-3 (P2): Normal skip の `decision_path` 曖昧性

**問題**: ev_score 計算済みだが SkipGate の通常判定で skip された場合、
`decision_path=None` で「ev 未計算」と見分けがつかない。

**修正**: `"ev_normal_skip"` を新たに追加。decision_path 全体系は:
- `primary_only`: ev 未計算 (ev_weighted 無効 or alt model 不在)
- `ev_offset`: ev_score 計算 → offset 乗数で価格調整
- `ev_no_change`: ev_score 計算 → mult ≈ 1.0 で変更なし
- `ev_emergency_skip`: ev_score << threshold → 緊急 skip
- `ev_normal_skip`: ev_score 計算済みだが ML 判定で通常 skip

**ファイル**: `scripts/v460/lib/skip_gate_evaluator.py`, `ztb/metrics/fill_quality.py`

### 4.3 両レビュアー共通の見落とし

| 見落とし | 深刻度 | 対応 |
|---|---|---|
| Config Hot-Reload 配線 | HIGH | 4.1 で修正 |
| 複合 Offset 判別不能 | MEDIUM | 将来課題 (velocity/trending/toxicity_offset をそれぞれ記録) |
| Skip 時の ev_score/path 曖昧 | LOW | 4.2 で修正 |

---

## 5. テスト

18/18 PASSED (`test_292_observability.py`):
- FillRecord 新フィールド: default None、round-trip、build_fill_record 受容
- decision_path: 5 値全て (`primary_only`, `ev_offset`, `ev_emergency_skip`, `ev_no_change`, `ev_normal_skip`)
- Config: stale_reprice_min_delta_jpy default/explicit/YAML
- Config: forced_buy_delay_velocity_threshold_ranging_bps default/explicit/YAML
- Hot-Reload: 新設 5 フィールドが `_HOT_RELOADABLE_FIELDS` に含まれることの検証
- 本番 YAML 存在確認

v460 全体: 3910 passed, 32 skipped (リグレッションなし)

---

## 6. 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `ztb/metrics/fill_quality.py` | FillRecord 3 フィールド追加、decision_path に `ev_no_change`/`ev_normal_skip` 追加 |
| `scripts/v460/lib/fill_cycle_executor.py` | ev_score/offset_mult 捕捉、decision_path 導出、BS-4 else 分岐 |
| `scripts/v460/lib/fill_config.py` | `stale_reprice_min_delta_jpy`, `forced_buy_delay_velocity_threshold_ranging_bps` |
| `scripts/v460/lib/config_hot_reload.py` | 新設フィールド 5 件を Hot-Reload 対象に追加 |
| `scripts/v460/lib/order_monitor.py` | reprice deadband チェック、BS-1 キャンセル前評価 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | regime-aware delay 閾値 |
| `scripts/v460/lib/skip_gate_evaluator.py` | BS-3 skip レコード ev フィールド + ev_normal_skip 追加 |
| `configs/v460/fill_test.yaml` | `reprice_min_delta_jpy: 500`, `velocity_threshold_ranging_bps: -3.0` |
| `docs/v460/289_ph2_analysis_buy_side_improvement.md` | v4 修正 (model_used 誤プロキシ) |
| `tests/unit/v460/test_292_observability.py` | 新テスト 18 件 |
