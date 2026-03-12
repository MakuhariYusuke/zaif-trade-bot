# 305# システム工学 × 市場理論 改善分析 + P0 実装

> **文書番号**: 305#  
> **種別**: `analysis` + `impl` (分析 + 実装)  
> **作成日**: 2026-03-06  
> **テスト**: 3985 passed, 0 failed (30.3s)

---

## §1 概要

304# リファクタリング完了後、コードベース全体をシステム工学・市場理論の両面から
深層調査し、短期収益性向上に直結する改善機会を特定。P0 改善 3 件を即時実装。

---

## §2 P0 実装済み改善 (本番号)

### 2.1 PnL Execution Quality 分解 (Kissell & Glantz 2003)

**問題**: `pnl_measurer.py` は total PnL (`mid_at_fill → mid_30s_after`) のみ計測。
「offset が広すぎて fill rate が低い」のか「AS コストが大きい」のかを区別できない。

**解決**: PnL を以下の 2 コンポーネントに分解:

$$\text{PnL} = \underbrace{(\text{fill\_price} - \text{mid\_at\_fill})}_{\text{spread capture}} + \underbrace{(\text{mid\_at\_fill} - \text{mid\_after})}_{\text{adverse selection cost}}$$

| フィールド | 意味 | side="buy" の場合 |
|---|---|---|
| `spread_capture_bps` | fill_price vs mid_at_fill | mid が fill_price より高い → 正 (MM の付加価値) |
| `adverse_selection_cost_bps` | mid_at_fill vs mid_after | mid が上昇 → 正 (有利), 下落 → 負 (AS 損失) |

**変更ファイル**: `fill_config.py` (PnlMeasurement), `pnl_measurer.py`

**期待効果**: adaptation_engine が「offset 縮小で spread capture を改善すべきか」
「offset 拡大で AS 防御を強化すべきか」を数値で判定可能になる。

### 2.2 OB キャッシュ再利用

**問題**: `maker_price.compute()` が `get_orderbook(symbol, depth=1)` を呼出していたが、
直前の `calculate_imbalance()` で `get_orderbook(symbol, depth=5)` の結果が
`_last_ob_snapshot` にキャッシュ済み。API 二重呼出しで 100-200ms/cycle のレイテンシ浪費。

**解決**: `_last_ob_snapshot` が有効なら cached OB を使用、ない場合のみ fresh fetch。

**変更ファイル**: `maker_price.py` (`compute()`)

**期待効果**: サイクル当たり 100-200ms の API レイテンシ削減 → cycle throughput 向上。

### 2.3 Parkinson σ 推定器 (Parkinson 1980)

**問題**: Roll (1984) proxy `σ = spread/(2·mid)` は BTC/JPY の薄い板ではノイジー。
1-tick のスプレッド変動で σ が倍変し、AS δ*, inventory skewing, VG, Kyle λ の
全段に不安定な入力を与えている。

**解決**: Parkinson High-Low Volatility Estimator を追加:

$$\sigma_P = \frac{\ln(H/L)}{2\sqrt{\ln 2}}$$

- rolling window (`sigma_parkinson_window_sec`, default 300s) 内の max/min mid を追跡
- `sigma_parkinson_enabled=True` で有効化 (default: False, 安全な opt-in)
- H == L (動きなし) の場合は Roll proxy にフォールバック
- hot-reload 対応

**変更ファイル**: `fill_config.py`, `maker_price.py`, `config_hot_reload.py`

**期待効果**: σ が安定化し、全 13 段 offset パイプラインの精度が向上。
特に AS δ* = γσ²τ + (2/γ)ln(1+γ/k) の精度改善が顕著。

---

## §3 P1-P2 改善提案 (未実装、次回以降)

### 3.1 市場理論的改善

| ID | 改善 | 理論根拠 | 優先度 | 期待効果 |
|---|---|---|---|---|
| M2 | AS δ* の k 動的化 | A-S 2008: k は注文到着率 ∝ bid_depth | P1 | offset 下限の適応化 |
| M4 | Amihud ILLIQ baseline 適応化 | rolling median | P1 | 市場構造変化への自動追従 |
| M5 | Inventory skewing に保有コスト統合 | q·(current_mid - avg_entry) | P1 | 含み損ポジションの早期解消 |
| E1 | Offset パイプライン統合 (18段→13段) | executor の 5段 post-hoc multiplier を maker_price に統合 | P1 | 理論的整合性回復 |
| E2 | Lot-Offset 結合最適化 | A-S: lot ∝ 1/(γσ²τ), δ* ∝ γσ²τ | P1 | 在庫リスクの最適バランス |
| A1 | EV-based offset adaptation | EV = fill_rate × avg_pnl - (1-fill_rate) × opp_cost | P1 | 直接的利益最大化 |
| A2 | Regime-conditional parameter sets | per-regime offset/lot | P1 | regime 間の最適化分離 |
| O1 | Queue Position Estimation (QPE) | same_side_depth_ahead × cancel_rate | P0 | fill prob 予測 → 早期 cancel |
| L2 | Microprice side selection | microprice = (Pb·Qa + Pa·Qb)/(Qb+Qa) | P0 | AS ratio 構造的低減 |

### 3.2 システム工学的改善

| ID | 改善 | 優先度 | 期待効果 |
|---|---|---|---|
| S1 | Offset stage 寄与分離 (F6) | P1 | per-cycle で「どのステージが何 bps 変更したか」を可視化 |
| L1 | Dynamic cycle interval (∝ 1/σ) | P0 | 高σ時にサイクル頻度増、低σ時にコスト節約 |
| L3 | Duty cycle observability | P1 | active time vs halt/skip/pause の比率メトリクス |
| O3 | Cancel latency tracking | P1 | cancel-replace RTT を offset に反映 |
| D1 | VaR 条件付き DD 限度 | P1 | 高σ日の不要 halt 削減 |
| D3 | Max position size constraint | P2 | PnL 限度では捕捉できない exposure リスク制約 |

### 3.3 最優先実装候補 (次回 306#)

1. **Queue Position Estimation (O1)**: 発注時の `same_side_depth_ahead` を記録し、
   経過時間 × cancel rate から fill probability を推定。
   低確率注文の早期 cancel で機会コスト削減 + cycle throughput 向上。

2. **Microprice side selection (L2)**: microprice が mid より上なら sell 優先、
   下なら buy 優先。追加 API 呼出し不要 (OB キャッシュから算出可能)。

3. **Dynamic cycle interval (L1)**: `cycle_interval_sec ∝ 1/σ` の動的化。
   高 σ 時は短い interval (素早い在庫解消)、低 σ 時は長い interval (API コスト節約)。

---

## §4 アーキテクチャ概観

```
fill_loop_orchestrator (2814L) ── メインループ / side選択 / halt制御
  └→ fill_cycle_executor (1455L) ── 1サイクル実行
       ├→ maker_price (1484L) ── 13段 offset パイプライン
       │    ├ [305# NEW] Parkinson σ estimator (opt-in)
       │    ├ [305# FIX] OB cache reuse (API call削減)
       │    ├ AS reservation shift (A-S 2008)
       │    ├ regime boosts (5段)
       │    ├ spread_adaptive / kyle_λ / amihud_illiq
       │    ├ volatility_guard (velocity + VPIN)
       │    ├ imbalance_risk / buy_as_guard (G-M 1985)
       │    └ loss_boost / ffd_boost
       ├→ skip_gate_evaluator (1272L) ── ML + ルールベース skip
       ├→ cycle_gate_aggregator (774L) ── Hard/Soft gate 集約
       ├→ order_monitor (635L) ── 約定ポーリング + cancel-replace
       ├→ pnl_measurer (195L) ── [305# NEW] spread_capture + AS 分解
       │                           30s/60s/120s multi-timeframe 計測
       └→ adaptation_engine (554L) ── offset/lot 自動適応
```

---

## §5 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `fill_config.py` | PnlMeasurement に `spread_capture_bps`, `adverse_selection_cost_bps` 追加 + `sigma_parkinson_enabled/window_sec` 追加 |
| `pnl_measurer.py` | `measure()` に spread capture / AS cost 分解ロジック追加 |
| `maker_price.py` | `_estimate_sigma()` Parkinson 拡張 + `compute()` OB cache 再利用 + high/low tracking slots |
| `config_hot_reload.py` | `sigma_parkinson_enabled/window_sec` を hot-reload フィールドに追加 |
| `test_260_compute_extract_regime_split.py` | compute() 行数上限 225→235 (OB cache ロジック追加分) |
| `test_305_p0_improvements.py` (NEW) | 12 テスト: PnL分解, Parkinson σ, OBキャッシュ, hot-reload |

---

## §6 テスト結果

| スイート | 結果 |
|---|---|
| 305# 新規テスト | 12 passed |
| 全 v460 テスト | 3985 passed, 0 failed |
| 実行時間 | 30.30s |

---

## §7 関連文書

| 文書 | 関係 |
|---|---|
| [304#](304_ph2_refactor_bps_ssot_dry_helpers.md) | 直前のリファクタリング |
| [303#](303_ph2_resp_301_302_review_response.md) | レビュー応答 (F6 stage記録 → S1, 盲点2 taker → O2) |
| [302#](302_ph2_gemini_31_pro_review_300_301_hft_blindspots.md) | Gemini レビュー |
| [301#](301_ph2_rev_292_300_multifaceted_review.md) | Codex レビュー |
