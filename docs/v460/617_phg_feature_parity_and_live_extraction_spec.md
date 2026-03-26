# 617# Feature Parity および Live 同期抽出数理仕様

- **日付**: 2026-03-24
- **目的**: 616# §2 の Train-Serve Skew リスクを解消し、訓練時と同一のロジックを同期ループ内で実行するための「Feature Parity」仕様を定義する。
- **前提**: `fill_test` は 120s 周期の同期ループであり、非同期イベント駆動は採用しない。

---

## §1 Feature Parity アーキテクチャの再定義 (T1)

訓練環境（pandas/numpy ベクトル演算）とライブ環境（単一行推論）の計算乖離をゼロにするため、**「ステートレスなバッチ抽出方式」**を採用する。

### 1.1 `FeatureExtractor` の同期構造
ライブ環境の各サイクル開始時において、以下の手順で Observation を構築する。

1.  **データ収集**: 直近 $N$ 件の 1m OHLCV 履歴を API またはローカルキャッシュから取得する（List of Dicts）。
2.  **DataFrame 構築**: 収集したデータを pandas DataFrame に変換する（この際、カラム名と型を訓練時と完全に一致させる）。
3.  **ベクトル演算適用**: 訓練時と同一の関数（e.g., `ztb.features.market_theory.parkinson_sigma`）を DataFrame 全体に適用する。
4.  **最新行抽出**: 生成された特徴量行列の「最終行（最新の 1 分足）」のみを抽出し、推論用ベクトルとする。

### 1.2 メリット
- **ロジックの再利用**: 訓練コードを 1 行も修正せずにライブへ持ち込める。
- **Skew の消滅**: リサンプリングの境界やローリング窓の端数処理が訓練時と数学的に一致する。
- **保守性**: 訓練側で新特徴量を追加した際、ライブ側の抽出ロジックも自動的に追随する。

---

## §2 Window Size (Lookback) の要件定義 (T2)

1 分足履歴から最新の特徴量を 1 行算出するために必要な過去データ数 $N$ を定義する。

### 2.1 特徴量別要求窓幅
614# §5.1 で定義した 17 特徴量（12 既存 + 5 市場理論）の最大窓幅に基づく。

| 特徴量群 | 主要因 | 最大窓幅 (Period) |
|:---|:---|:---|
| **Scalping (12種)** | `tick_volume_ratio`, `realized_volatility` | 10 |
| **Market Theory (5種)** | `parkinson_sigma`, `vpin_proxy`, `illiq` | 20 |
| **Normalization** | Rolling Z-score (if applied) | 20 |
| **EMA Convergence** | `ema_velocity_bps` (span=5) | ~20 |

### 2.2 推奨 Lookback 設定
- **安全 Lookback 数 ($N$)**: **100 bars** (100分)
- **根拠**: 最も長い 20-bar rolling mean の計算に十分であり、かつ EMA の初期値依存性を 99.9% 以上排除（収束）させるために 4〜5 倍の窓幅を確保する。100 分のデータ取得コスト（Coincheck API `limit=100`）は、現在の 120s サイクルにおいて十分に低負荷である。

---

## §3 正規化 (Normalization) の同期フロー (T3)

訓練時の分布情報をライブ環境へ確実に伝播させるためのデータ契約。

### 3.1 `sac_sidecar.norm.json` データスキーマ
訓練側 (`sac_retrain_scheduler`) は、モデル保存と同時に以下の形式で各特徴量の統計量を出力する。

```json
{
  "feature_stats": {
    "price_velocity": {"mean": 0.0012, "std": 0.045, "min": -5.0, "max": 5.0},
    "vpin_proxy": {"mean": 0.45, "std": 0.12, "min": 0.0, "max": 1.0},
    "ema_velocity_bps": {"mean": 0.0, "std": 10.5, "min": -50.0, "max": 50.0}
  },
  "metadata": {
    "generated_at": "2026-03-24T12:00:00Z",
    "train_end_index": 973544
  }
}
```

### 3.2 ライブ環境での標準化・補完手順
抽出された最新の特徴量ベクトル $\mathbf{x}_{raw}$ に対し、以下の数理処理を適用する。

1.  **欠損補完 (Mean Imputation)**:
    もし $x_i$ が `NaN`（起動直後等）であれば、スキーマの `mean` 値で置換する。
    $$x_i' = x_i \text{ if } x_i \neq \text{NaN else } \text{mean}_i$$

2.  **標準化 (Z-score Transformation)**:
    $$z_i = \frac{x_i' - \text{mean}_i}{\text{std}_i + \epsilon} \quad (\epsilon = 1e-10)$$

3.  **クリッピング (Outlier Capping)**:
    訓練時の観測範囲外の値を抑制する。
    $$\text{obs}_i = \min(\max(z_i, \text{min}_i), \text{max}_i)$$

---

*以上。本文書は 616# §2 を全面的に置き換える「同期・一貫性重視」の設計仕様である。*
