# 566# [phg] [plan] 執行エンジンの数理刷新と在庫補正の条件付き復活（計測結果反映版）

> **ステータス**: 設計改訂・ファクトベース (Gemini 担当)  
> **更新日**: 2026-03-23  
> **参照**: 565# (実態検証レビュー), 561# (DRC原案), 249# (Directional Alpha)

---

## 0. 計測結果（I3/I2）による現状否定

直前の計測（I3）により、以下の事実が判明した。
- **パイプライン爆発**: Sell/Buy ともに `execution_pre_clamp_offset` の P90 が **0.47〜0.50** に達している。現行 Ceiling 0.30 は機能しておらず、天井を 0.40 に上げたとしても、乗算連鎖の幾何学的爆発により即座に再飽和することが数学的に予見される。
- **PnL 窓の不一致**: Sell PnL (実質 90s) と Buy PnL (30s) の単純比較は誤りであり、Buy 側の「慢性的な微損」も Sell の「テールリスク」と同等に深刻な構造的問題である。

---

## 1. 修正設計案：幾何学的欠陥の根治

### 1.1 加法的（Log-Linear）パイプラインへの移行
9段の乗算チェーンを廃止し、以下の加重平均スコア型へ刷新する。
$$Offset = Base\_Offset + \text{Clamp}\left( \sum (Weight_i \times RiskScore_i), \text{Dynamic\_Ceiling} \right)$$
- **リスクスコアの結合**: 562# 提案の `max()` または `RMS` を採用し、同一リスクの二重・三重計上を構造的に排除する。

### 1.2 指数的 DRC (eDRC) の定数確定
565# の指摘を反映し、発散防止 floor ($\epsilon = 1.0$) を導入した最終式。
$$Ceiling_{dynamic} = Base\_Ceiling \cdot \exp\left( \alpha \cdot \frac{\sigma_{1min}}{\max(Spread_{bps}, 1.0)} + \beta \cdot Adverse\_OFI \right)$$
- **Adverse_OFI**: Side に応じ、Buy 時は正、Sell 時は負の値を反転して「自分にとって不利な圧力」として正値化する。

---

## 2. 在庫補正（inv_skew）の再定義

249# の **Directional Alpha**（トレンド追従）を損なわず、かつ `preflight_insufficient` を回避する「ハイブリッド在庫管理」を導入する。

### 2.1 条件付き強制中立化 (Threshold-based Inv-Skew)
- **通常時 (Trending)**: 249# 準拠。在庫補正をオフにし、トレンド利益を最大化。
- **臨界時 (Extreme Skew)**: 在庫偏重率（Inventory Ratio）が **±80%** を超えた場合のみ、Regime にかかわらず `inv_skew` を強制発動させる。
- **論理**: トレンド追従は重要だが、資金ショート（不戦敗）はマーケットメイカーとして最大の機会損失であるため。

---

## 3. 分割実行タスク (Revised)

### Task A: [Gemini] 加法的パイプラインの数理仕様策定 (P0)
- `offset_pipeline.py` を乗算から加重平均スコア型へ置換するための定数と結合則の定義。

### Task B: [Codex] CV_favorable_tighten の非対称無効化 (P0)
- Sell 側の無効化に加え、Buy 側の窓を揃えた再評価と、攻め（Tighten）の抑制。

### Task C: [Copilot] fill_recorder の I2 計測有効化 (P0)
- `spread_capture_bps` および `adverse_selection_cost_bps` を確実に記録する修正。

---

## 4. 結論

「天井を上げる」だけでは、爆発する数式には勝てない。
本設計は、数理構造そのものを **「乗算」から「加重平均」** へ、在庫管理を **「全か無か」から「条件付き生存」** へとシフトさせるものである。

---
