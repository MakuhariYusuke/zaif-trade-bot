# 303# 301#/302# レビュー応答 — 妥当性検証と実装方針

> **文書番号**: 303#  
> **種別**: `resp` (レスポンス)  
> **作成日**: 2026-03-06  
> **対象**: [301# Codex レビュー](301_ph2_rev_292_300_multifaceted_review.md), [302# Gemini 3.1 Pro レビュー](302_ph2_gemini_31_pro_review_300_301_hft_blindspots.md)

---

## §1 コード照合による妥当性検証

### 検証方法

6 主張すべてについて、実コード (ab_judgment.py, side_selector.py, fill_loop_orchestrator.py,
daily_drawdown_guard.py, maker_price.py, fill_quality.py, test_292_observability.py) を照合。

### 検証結果

| # | 主張 | 出典 | 判定 | 根拠 |
|---|---|---|---|---|
| C1 | `none` レジーム除外で F-4 が楽観化 | 301# F1 / 302# 盲点1 | ✅ **CONFIRMED** | `exclude_regimes` デフォルト `["none"]`。none でも取引は継続実行。offset パイプラインに none 用バイパスなし |
| C2 | sell vs buy は i.i.d. 違反の擬似 A/B | 301# F2 / 302# P0 | ✅ **CONFIRMED** | SideSelector は決定論的交互実行。balance_forced がオーバーライド。ランダム割当なし |
| C3 | forced_buy_delay 後も maker 注文のみ | 302# 盲点2 | ✅ **CONFIRMED** | delay 消化後は通常の `run_single_cycle` に合流。taker 執行パスは一切なし |
| C4 | DD Guard soft lot 削減が side-agnostic | 302# 盲点3 / 300# #5 | ⚠️ **PARTIALLY** | soft lot reduction は `_current_lot` 単一変数で共有（側面確認済）。ただし per-side halt（完全封鎖 → recovery）は既に実装済み |
| C5 | offset stage-by-stage 記録なし | 301# F6 | ✅ **CONFIRMED** | FillRecord に `effective_offset_used` (最終値) のみ。中間ステージ記録なし |
| C6 | hot-reload テストは登録確認のみ | 301# F3 | ✅ **CONFIRMED** | `_HOT_RELOADABLE_FIELDS` frozenset の membership テストのみ。E2E 行動テストなし |

### 特筆事項: C4 の補足

302# が「DD Guard が side-agnostic」と指摘したのは **soft lot reduction** に限れば正確。
しかし `daily_drawdown_guard.py` には既に **per-side halt** が実装されている:

- `per_side_dd_enabled=True` で `daily_pnl_bps_buy/sell` を分離追跡
- 閾値超過時に `side_halted_buy/sell` フラグで片側を完全封鎖
- `per_side_dd_recovery_cycles` + `per_side_dd_recovery_lot_scale` で段階復帰

302# の「Side-aware DD Guard」提言は per-side halt の拡張版として、
**halt ではなく lot 縮小** のレイヤーを追加する方向で同意する。

---

## §2 各提言への対応方針

### 301# Finding ごとの採否

| # | Finding | 採否 | 理由 | 今回実装 |
|---|---|---|---|---|
| F1 | `none` 含有版 A/B 出力追加 | ✅ **採用** | 楽観バイアス除去は必須 | ✅ |
| F2 | 擬似 A/B 表記の明文化修正 | ✅ **採用** | 統計的誤解を防ぐ | ✅ |
| F3 | hot-reload E2E テスト追加 | ⏸️ **保留** | 有用だが優先度低 (P2) | — |
| F4 | forced buy α/repair 分離評価 | ⏸️ **保留** | 286# で `forced_buy_kpi_tracking` 実装済。Dashboard 側拡張で対応可 | — |
| F5 | 統計強化 (BH/bootstrap) | ⏸️ **保留** | 正しいが現段階はサンプル蓄積が先 (P2) | — |
| F6 | offset stage 記録 | ✅ **採用方針** | 定量確証に必須。ただし今回は config + 記録フィールド追加のみ | — (次回) |

### 302# 盲点ごとの採否

| # | 盲点 | 採否 | 理由 | 今回実装 |
|---|---|---|---|---|
| 盲点1 | none レジーム Passive MM フォールバック | ✅ **採用** | AS 43% は許容外。offset パイプラインバイパス+静的モード | ✅ |
| 盲点2 | Toxic Forced Repair の Taker 執行化 | ⏸️ **保留** | 方向は正しいが Coincheck API の成行注文制約を要調査。Phase A 凍結中 | — |
| 盲点3 | Side-aware DD Guard (lot 縮小の side 分離) | ✅ **採用** | per-side halt は既存。soft lot reduction の side 分離を追加 | ✅ |

---

## §3 今回の実装スコープ

### 実装 A: A/B 表記修正 + none 含有版出力 (301# F1/F2, 302# P0)

**対象**: `ab_judgment.py`, `side_regime_dashboard.py`

1. `[A/B Judgment]` → `[Side Comparison]` に表記変更
2. summary に「※ 観察比較であり、ランダム割当の A/B テストではない」注記追加
3. `evaluate_ab_variant()` に `include_none=True` 版を追加 (二系統出力)
4. Dashboard に `excluding_none` / `including_none` 並列表示

### 実装 B: DD Guard soft lot 削減の side 分離 (302# 盲点3)

**対象**: `fill_loop_orchestrator.py`, `daily_drawdown_guard.py`, `fill_config.py`

1. `DailyDrawdownState` に `soft_lot_side: str | None` フィールド追加
2. `update_pnl()` の `soft_triggered` 結果に `side` 情報を含める
3. `_process_post_cycle` で `soft_triggered` 時に当該 side のみ lot 縮小
4. 新 config: `daily_drawdown_soft_lot_side_aware: bool = False`

### 実装 C: none レジーム Passive MM フォールバック (302# 盲点1)

**対象**: `maker_price.py`, `fill_config.py`, `fill_cycle_executor.py`

1. 新 config: `none_regime_passive_mm_enabled: bool = False`
2. 新 config: `none_regime_fixed_offset_bps: float = 2.0`
3. `MakerPriceCalculator.compute()` で regime=None 時にパイプラインをバイパス
4. 固定 offset で指値配置、reprice を抑制

---

## §4 不採用・保留の理由

### Taker 執行化 (302# 盲点2) — 保留

Coincheck の API は `order_type="market_buy"/"market_sell"` を提供するが、
現在のアダプタ (`ccxt_adapter.py`) は limit 注文のみを前提とした設計。
Taker 執行の追加には:
1. アダプタの注文タイプ拡張
2. Taker 手数料の PnL 計算への反映
3. スリッページ計算 + 板の深さ検証
4. Fill 確認ロジックの分岐 (即約定 vs 部分約定)

が必要であり、Phase A 凍結中の現在は安全に実装できない。
方向としては正しいため、凍結解除後の P1 タスクとして記録する。

### 統計強化 BH/bootstrap (301# F5) — 保留

サンプル数不足 (trending_up n≈85, 必要 n≈680) の状態で
統計手法を高度化しても検出力は改善しない。
3月データ蓄積後に BH 法 + block bootstrap を一括実装する方が効率的。

---

## §5 実装結果

### 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `ab_judgment.py` | `[A/B Judgment]` → `[Side Comparison]`, 観察比較注記追加 |
| `side_regime_dashboard.py` | `_ab_result_to_dict` DRY化, `ab_judgment_incl_none` 二系統出力, 出力ラベル更新 |
| `daily_drawdown_guard.py` | `DrawdownAction` に `soft_triggered_side` フィールド追加 |
| `fill_loop_orchestrator.py` | `_dd_soft_lot_scale_buy/sell` 変数追加, side-aware soft lot 分岐, 日替わりリセット |
| `fill_cycle_executor.py` | `_dd_soft_lot_scale_buy/sell` クラス宣言, lot 算出パイプラインに side-aware 乗数追加 |
| `fill_config.py` | `daily_drawdown_soft_lot_side_aware`, `none_regime_passive_mm_enabled`, `none_regime_fixed_offset_bps` 追加 |
| `maker_price.py` | `compute()` に none regime passive MM バイパスロジック (固定 offset 早期 return) |
| `config_hot_reload.py` | 3 新フィールドを `_HOT_RELOADABLE_FIELDS` に登録 |

### テスト結果

- `test_303_review_implementations.py`: 14 passed (新規)
- `test_160_ab_judgment.py`: 93 passed (回帰なし)
- `test_168_daily_drawdown_guard.py`: 100 passed (回帰なし)
- `test_159_side_regime_dashboard.py`: 6 passed (回帰なし)
- 全 v460: **4006 passed**, 0 failed
