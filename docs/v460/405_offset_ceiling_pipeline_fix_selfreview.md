# 405# Offset Ceiling Pipeline Fix + 403#/404# レビュー対応セルフレビュー

> **作成日**: 2026-03-13
> **前提**: 403# (Codex review), 404# (Gemini second opinion)
> **変更種別**: impl (実装) + selfreview

---

## 0. 概要

403#/404# の指摘に基づき、offset パイプラインの構造的デッドロックを修正。
**パラメータ変更なし**（402# P0 提案は 403#/404# の却下勧告に準拠して保留）。

---

## 1. 403#/404# 指摘の採否判定

| # | 指摘 | 判定 | 理由 |
|---|------|------|------|
| 403#§1 | Mixed-SHA 分析は current SHA 変更根拠として弱い | ✅ 採用 | 402# P0 のパラメータ変更は全て保留 |
| 403#§2 | AS率は post-fill 指標、live leakage 危険 | ✅ 採用 | AS 予測モデルは P1 課題として記録 |
| 403#§3 | sell ceiling ではなく pipeline 全体を監査 | ✅ 採用 | **本修正の核心** |
| 403#§4 | hard_skip[21] 解除は選択バイアス | ✅ 採用 | 解除しない |
| 403#§5 | sell_hour_boost[0] 2.5 は liveness 低下 | ✅ 採用 | 変更しない |
| 403#§6 | confidence ≥0.9 は calibration 問題 | ✅ 採用 | 中期課題として記録 |
| 403#§7 | 「機関投資家」は仮説 | ✅ 採用 | 表現を改めるべき |
| 404#§1.1 | AS_raw leakage リスク | ✅ 採用 | 403#§2 と合致 |
| 404#§1.2 | _scale_offset_ratio 各ステップで 0.30 キャップ | **⚠️ 部分採用** | 下記参照 |
| 404#§1.3 | confidence ≥0.9 はcalibration崩れ | ✅ 採用 | 403#§6 と合致 |
| 404# Action 1 | 中間キャップ全除去、最終段一括クランプ | **⚠️ 修正採用** | 下記参照 |
| 404# Action 2 | 時間帯固定ガード変更凍結 | ✅ 採用 | |
| 404# Action 3 | AS 予測モデル構築 | △ 記録 | P1 中期課題 |

---

## 2. 404# 指摘の検証結果：「デッドロック」は部分的に正しい

### 2.1 404# の主張
> `_scale_offset_ratio` が各中間ステップで `max_ratio=0.30` をキャップ → 全ブーストが無効

### 2.2 実際の検証結果

**正しい点:**
- `sell_floor=0.30` と中間 `max_ratio=0.30` が同値 → フロア適用後の **9 箇所の中間ブースト** が無効化
- 特に 397# `_regime_boost_mid_confidence` (×1.2) が sell 側で完全に無効

**404# が過大表現している点:**
1. `sell_hour_boost` は `max_ratio` なし → 0.30 を超えて通過可能
2. `trending` ブーストも `max_ratio` なし → 通過可能
3. 最終 ceiling = 0.50 (sell) → 完全なデッドロックではなく「部分デッドロック」
4. パイプライン順序: mid_conf (stage 4) は spread_adaptive (stage 5) より**前**

**典型 sell フロー (修正前):**
```
base → sell_floor(0.30) → mid_conf×1.2: min(0.36,0.30)=0.30 DEAD
                        → high_vol×1.5: min(0.45,0.30)=0.30 DEAD
                        → vol_guard×1.8: min(0.54,0.30)=0.30 DEAD
                        → sell_hour_boost×1.5: 0.30→0.45 ALIVE (no cap)
                        → final ceiling: min(0.45,0.50)=0.45
```

### 2.3 採用した修正: 404# Action 1 の安全版

404# は「中間キャップ全除去」を提案したが、暴走防止の安全弁機能を失うリスクがある。
代わりに **side-aware intermediate cap** を導入:

```python
def _effective_max_ratio(self, side: str) -> float:
    base = cfg.max_offset_ratio                         # 0.30
    if side == "sell" and cfg.offset_ceiling_ratio_sell:
        return max(base, cfg.offset_ceiling_ratio_sell)  # max(0.30, 0.50) = 0.50
    if side == "buy" and cfg.offset_ceiling_ratio_buy:
        return max(base, cfg.offset_ceiling_ratio_buy)   # max(0.30, 0.20) = 0.30
    return base
```

**効果:**
- sell: 中間キャップが 0.30 → 0.50 に拡大 → 全ブーストが 0.30-0.50 で有効化
- buy: 中間キャップ 0.30 → 0.30 (変化なし)
- 最終 ceiling は既存のまま維持 (sell=0.50, buy=0.20)

**修正前→修正後の sell フロー:**
```
base → sell_floor(0.30) → mid_conf×1.2: min(0.36,0.50)=0.36 ✅ ALIVE!
                        → high_vol×1.5: min(0.54,0.50)=0.50 ✅ ALIVE!
                        → sell_hour_boost×1.5: 0.50→0.75 (no cap)
                        → final ceiling: min(0.75,0.50)=0.50
```

---

## 3. 変更サマリ

### 3.1 本体コード (4ファイル, 全 14 箇所)

| ファイル | 変更箇所 | 内容 |
|---------|---------|------|
| `maker_price.py` | L550 | `_effective_max_ratio()` メソッド追加 |
| `maker_price.py` | L649, L712, L746 | spread_adaptive, loss_boost, ffd_boost |
| `maker_regime_boost.py` | L57 (Protocol) + L133, L208, L236, L266 | high_vol, low_vol, unknown_buy, mid_conf |
| `maker_risk_guards.py` | L61 (Protocol) + L178, L241 | volatility_guard, imbalance_risk |
| `maker_microstructure.py` | L68 (Protocol) + L210, L244, L296, L351 | as_reservation_shift, delta_star, kyle_lambda, amihud_illiq |

### 3.2 テスト (3ファイル)

| ファイル | 変更内容 |
|---------|---------|
| `test_405_offset_ceiling_pipeline.py` | **新規** 14 テスト (_effective_max_ratio, sell deadlock, buy preserved, _scale_offset_ratio) |
| `test_175_code_review_sweep2.py` | 静的チェック文字列マッチ更新 (`cfg.max_offset_ratio` → `max_ratio=`) |
| `test_258_*.py` | SimpleNamespace stub に `_effective_max_ratio` 追加 |
| `test_266_*.py` | SimpleNamespace stub に `_effective_max_ratio` 追加 |

### 3.3 NOT 変更 (403# 却下準拠)

| 項目 | 理由 |
|------|------|
| `sell_hour_boost[0]` 1.5→2.5 | 403#§5: liveness 低下、Mixed-SHA 根拠 |
| `hard_skip_utc_hours[21]` 除外 | 403#§4: 選択バイアス、shadow mode 不要 |
| sell_offset_floor / offset_ceiling_ratio_sell 変更 | パイプライン修正で不要化 |
| confidence ≥0.9 guard 追加 | 403#§6: calibration 問題として別対応 |

---

## 4. セルフレビュー

### 4.1 盲点検証

| 確認項目 | 結果 |
|---------|------|
| buy 側の動作不変 | ✅ max(0.30, 0.20)=0.30 で既存と同一 |
| sell 最終 ceiling 維持 | ✅ 0.50 クランプは compute() 最終段で変更なし |
| sell_hour_boost の max_ratio なし | ✅ 変更なし（元から上限なし） |
| trending boost の max_ratio なし | ✅ 変更なし（元から上限なし） |
| offset_ceiling_ratio_sell = None 時 | ✅ base (0.30) にフォールバック |
| パフォーマンス影響 | ✅ 1 メソッド呼出し追加のみ (if 2 分岐) |
| mixin Protocol 整合 | ✅ 4 ファイルすべてに stub 追加 |
| SimpleNamespace テスト stub | ✅ 258# と 266# の stub に追加 |

### 4.2 潜在リスク

| リスク | 影響 | 対策 |
|-------|------|------|
| sell offset が 0.30 → 0.50 に上がりうる | fill rate 低下の可能性 | 最終 ceiling=0.50 は既存値、sell_hour_boost では既に到達可能だった |
| 複数ブーストの累積 | sell で high_vol×1.5 + mid_conf×1.2 = 0.30×1.8 = 0.54 → ceil 0.50 | 最終 ceiling がガード |
| buy_as_guard は別上限 (0.50) | 変更なし | `buy_as_guard_max_offset_ratio` は別設定 |

### 4.3 残課題

| 優先度 | 課題 | 備考 |
|--------|------|------|
| P0 | 397# mid_conf_guard の再評価 | 今回の修正で sell 側も有効化 → 次 SHA で効果検証 |
| P1 | confidence ≥0.9 対策 | calibration 問題 (403#§6) → 別チケット |
| P1 | AS 予測モデル (pre-trade proxy) | 404# Action 3 → VPIN/OBI/spread ベース |
| P2 | 時間帯レイヤー統合 | 7 層 → AS probability ベース 1 層 |

---

## 5. テスト結果

```
4684 passed, 33 skipped, 12 warnings (22.09s)
```

14 新規テスト + 既存 4670 テスト全通過。
