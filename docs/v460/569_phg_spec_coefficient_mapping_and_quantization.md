# 569# [phg] [spec] 執行エンジンの係数マッピングと量子化パラメータ定義

> **ステータス**: 計数仕様確定 (Gemini 担当)  
> **作成日**: 2026-03-23  
> **参照**: 568# (数理仕様), 567# (I3実測データ)

---

## 1. eDRC 感度パラメータ (M1)

直近の P99 (0.80) を「荒天時の天井」と定義し、$C_{base}=0.40$ からの遷移を逆算した初期値。

| パラメータ | 初期値 | 根拠 |
|-----------|--------|------|
| `drc_alpha` ($\alpha$) | **0.16** | $\sigma/Spread$ 比率 2.5 時に Ceiling を約 1.5 倍へシフト |
| `drc_beta` ($\beta$) | **0.28** | $Adverse\_OFI$ 1.0 時に Ceiling を約 1.3 倍へシフト |
| `drc_epsilon` ($\epsilon$) | **1.0** | スプレッド 1.0 bps 未満での感度発散を防止 |

**適用式**:
$$Ceiling_{dynamic} = 0.40 \cdot \exp\left( 0.16 \cdot \frac{\sigma_{1min}}{\max(Spread, 1.0)} + 0.28 \cdot Adverse\_OFI \right)$$

---

## 2. 加法的パイプラインのマッピングテーブル (M2)

乗算モデル ($m_i$) から加算増分 ($\Delta R_i$) への変換。現行の $R_{base}=0.05$ を基準としつつ、RMS 結合による減衰を考慮して Weight を補正。

| コンポーネント | 旧乗数 ($m$) | 新加算増分 ($\Delta R$) | 設定キー案 (YAML) |
|---------------|-------------|-----------------------|-------------------|
| **EV (Active)** | 1.5x | **+0.10** | `additive_ev_weight` |
| **Velocity** | 1.5x | **+0.12** | `additive_velocity_weight` |
| **Trending Sell** | 1.3x | **+0.08** | `additive_trending_weight` |
| **Toxicity (Warn)**| 1.2x | **+0.05** | `additive_tox_warn_weight` |
| **Toxicity (Crit)**| 1.8x | **+0.20** | `additive_tox_crit_weight` |
| **Macro Boost** | 1.6x | **+0.15** | `additive_macro_weight` |
| **Sidecar (Max)** | +10bps | **+0.10** | `additive_sidecar_weight` |

---

## 3. RMS 結合と要素別 Cap (M3)

RMS 結合 $\sqrt{\sum \Delta R_i^2}$ を採用した場合、単一コンポーネントが全体を支配しすぎないよう、要素別に個別 Cap を設定する。

### 結合則の挙動想定
- **平時 (Single Risk)**: Velocity のみが反応 (+0.12) $\rightarrow$ Total $\Delta R = 0.12$.
- **荒天時 (Multi Risk)**: Velocity(0.12) + Macro(0.15) + Tox(0.20) $\rightarrow$ $\sqrt{0.12^2 + 0.15^2 + 0.20^2} \approx \mathbf{0.27}$.
- **最終オフセット**: $0.05 (Base) + 0.27 (RMS) = \mathbf{0.32}$ (Ceiling 0.40 以内に収まる適切な逃げ).

### 要素別 Cap 指針
- 各 $\Delta R_i$ は **0.30** を絶対上限とする。
- RMS 合計後の $\Delta Offset_{total}$ は **Ceiling** によって最終 Clamp される。

---

## 4. 実装（Copilot）への補足事項

- **トグル**: `use_experimental_additive_pipeline: true` 時のみ上記テーブルを適用。
- **単位系**: $\Delta R$ はオフセット比率（0.0 〜 1.0）として扱う。
- **後方互換**: 既存の Multiplier 設定値は読み込むが、加法モード時は無視し、上記 `additive_*` キーを優先参照すること。

---
