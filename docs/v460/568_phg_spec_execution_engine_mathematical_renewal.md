# 568# [phg] [spec] 執行エンジンの数理刷新仕様書: 加法パイプラインと指数的 DRC

> **ステータス**: 数理仕様確定 (Gemini 担当)  
> **作成日**: 2026-03-23  
> **参照**: 566# (設計指針), 567# (Copilot実測データ), 249# (在庫思想)

---

## 0. 基本定数と前提条件

Copilot による I3 実測データ（buy p50 > 0.30）に基づき、以下の基準値を設定する。

- **Base Ceiling ($C_{base}$)**: **0.40** (現行 0.30 から引き上げ、P90=0.50 への耐性を確保)
- **PnL Window**: 窓の不一致を解消し、Buy/Sell ともに **30s / 90s 両系列の同時監視** へ移行。
- **發散防止床 ($\epsilon$)**: **1.0 bps** (スプレッド逆数計算時の安全装置)

---

## 1. M1: 指数的 DRC (Exponential Dynamic Risk Ceiling)

固定の天井を廃止し、市場のボラティリティと板圧力に弾力的に応答する天井を定義する。

$$Ceiling_{dynamic} = C_{base} \cdot \exp\left( \alpha \cdot \frac{\sigma_{1min}}{\max(Spread_{bps}, 1.0)} + \beta \cdot Adverse\_OFI \right)$$

- **$Adverse\_OFI$**: 
    - Buy 時: $\max(0, OFI_{mean})$ (買い圧力が強いほど天井を引き上げる)
    - Sell 時: $\max(0, -OFI_{mean})$ (売り圧力が強いほど天井を引き上げる)
- **$\alpha, \beta$**: 感度パラメータ。初期値は実測データの P99 ($0.80$) を荒天時の上限として逆算して設定。

---

## 2. M2: 加法的パイプライン (Additive Risk Integration)

9 段の乗算チェーン（$\prod m_i$）を、幾何学的爆発を防ぐ加法モデルへ刷新する。

$$Offset_{final} = Offset_{base} + \text{Clamp}\left( \Delta Offset_{total}, Ceiling_{dynamic} \right)$$
$$\Delta Offset_{total} = \text{Combine}\left( w_1 \cdot Z_1, w_2 \cdot Z_2, \dots, w_n \cdot Z_n \right)$$

- **刷新対象**: `OffsetPipelineMixin._apply_offset_pipeline` 内の各ステージ。
- **各ステージの出力**: 従来の Multiplier (1.2等) ではなく、ベースオフセットに対する **加算増分 (bps または ratio)** を算出する。

---

## 3. M3: リスク結合則 (Combination Logic)

同一リスク（VPIN, Velocity 等）の多重計上を防ぐため、単純加算ではなく以下の結合則を導入する。

### 採用候補 (563# 要請に基づく比較実装)
1.  **Max 結合 (Dominant Risk)**: $\max(w_i \cdot Z_i)$ 
    - 最も顕著なリスク指標のみを採用。多重計上を完全に排除するが、複合リスクを過小評価する恐れ。
2.  **RMS 結合 (Orthogonal Risk)**: $\sqrt{\sum (w_i \cdot Z_i)^2}$
    - 各指標が独立（直交）していると仮定。中程度の複合リスクを適切に評価。**（推奨）**
3.  **Capped Sum (Conservative)**: $\min(Cap, \sum w_i \cdot Z_i)$
    - 従来の保守性を維持しつつ上限を設定。

---

## 4. M4: テール防絶 (Exponential Tail Defense)

AS 被弾時の損失を非線形に抑制するための減衰関数。

### 期待値（EV）に基づく注文拒絶条件
$$E[PnL] = Spread_{capture} - P(AS) \cdot E[Loss_{AS}]$$
- **Hard Reject**: $E[PnL] < -Threshold$ の場合、Ceiling を待たずに `cancel_replace` を実行。
- **AS Penalty の指数化**: 報酬関数における負の報酬を $R_{neg} = -\lambda \cdot \exp(|PnL_{AS}|)$ とし、巨大な AS への感度を高める（ただし SNR 低下を防ぐため Clipping と併用）。

---

## 5. 在庫管理のハイブリッド実装 (inv_skew Override)

249# の Directional Alpha を尊重しつつ、`preflight_insufficient` を根絶する。

- **Regime-gate**: 既存の `inv_skew_regime_gate_enabled` は維持。
- **Emergency Override**:
    - 在庫比率 $|q| > 0.8$ (80%) の場合、Regime を無視して `inv_skew` を **強制的・指数的** に適用。
    - $Offset_{inv} = k \cdot q^3$ (偏るほど急激に指値を遠ざけ、清算を促す)

---

## 6. 実装への指示 (to Copilot/Codex)

1.  **Task A**: `scripts/v460/lib/offset_pipeline.py` を本仕様書に基づき Additive 型へリファクタリング。
2.  **Task B**: `resolve_offset_ceiling` に M1 の eDRC 式を実装。
3.  **Task C**: `maker_price.py` の在庫補正ロジックに 80% 閾値の Override を追加。

---
