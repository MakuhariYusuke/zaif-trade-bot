# 570# [phg] [spec] 執行エンジンのロバスト入力設計と加法パイプライン本番係数マッピング

> **ステータス**: 仕様確定・実装指示 (Gemini 担当)  
> **作成日**: 2026-03-23  
> **参照**: 568# (数理仕様), 569# (初期マッピング), 560# (Buy悪化事実)

---

## 1. Buy 側悪化の真相分析 (Task A)

3/20-22 の詳細ログ解析に基づき、Buy 側の出血原因を特定した。

### 1.1 JST 夜間における「守護の飽和」
- **事象**: JST 22時-23時 (UTC 13-14) の Buy PnL が平均 **-3.0bps** 以下へ急落。
- **原因**: 海外市場のボラティリティ急増に対し、Cross-Venue (CV) ロジックが `widen`（指値を深くする）を指示しているが、それが現行 Ceiling (0.30) に衝突して **「十分に逃げ切れていない」** ことが判明。
- **結論**: Buy 側の悪化は「攻め」の結果ではなく、**「不十分な防御」** の結果である。eDRC による天井の動的拡大が、Buy 側の救済においても P0 優先度となる。

---

## 2. eDRC のためのロバスト入力設計 (Task C)

天井（Ceiling）が高周波ノイズで振動するのを防ぐため、以下の平滑化アルゴリズムを導入する。

### 2.1 Robust Volatility ($\sigma_{robust}$)
- **算出**: 過去 60 秒の $\text{mid\_price\_return}$ に対する EWMA ($\alpha=0.1$)。
- **外れ値除外**: 直近 5 分間の平均ボラティリティの 3 倍を超える瞬間的なスパイクは、算出から除外（またはクリップ）する。

### 2.2 Robust Adverse OFI ($OFI_{robust}$バランス)
- **算出**: 過去 10 秒間の `orderbook_imbalance` の **Median（中央値）**。
- **理由**: 指値板の瞬間的なフラッシングによる天井の誤作動を抑制するため。

---

## 3. 加法パイプライン本番係数テーブル (Task D)

569# を微調整し、実測データの分布を再現するための最終パラメータ案。

### 3.1 YAML 設定定義 (Additive Pipeline)
```yaml
experimental_additive_pipeline:
  enabled: true
  combination_rule: "rms"  # RMS 結合推奨
  base_offset_ratio: 0.05
  
  # 各ステージの加算重み (W_i * Z_i)
  weights:
    ev_score: 0.12          # p50 押し上げの主因
    velocity_bps: 0.15      # 急変時の退避
    trending_regime: 0.10   # トレンド追従
    toxicity_crit: 0.25     # AS 予兆検知時の緊急退避
    macro_trend: 0.15       # マクロ環境
    sidecar_offset: 0.10    # ML 推論
```

### 3.2 eDRC パラメータ最終案
```yaml
edrc_settings:
  enabled: true
  base_ceiling: 0.40        # 568# 合意値
  alpha: 0.16               # Volatility 感度
  beta: 0.28                # OFI 感度
  epsilon: 1.0              # Spread floor (bps)
  hard_cap: 1.0             # 青天井防止の物理限界 (100bps)
```

---

## 4. 期待される執行プロファイル

1.  **通常時**: 各リスクが 0 近辺であれば、Offset は 0.05 前後で安定。
2.  **警戒時**: 単一のリスク（例: Velocity）発生で Offset 0.15 〜 0.20 へシフト。
3.  **危機時 (複合リスク)**: 複数の指標が反応し、RMS 結合により Offset 0.35 〜 0.50 へ急退避。この時、eDRC が天井を 0.80 まで押し広げているため、**強制約定を回避** できる。

---

## 5. Copilot への実装連携

- **配線要求**: `OFI_robust` および `sigma_robust` の計算モジュールを `ztb/utils/robust_stats.py` 等に新設し、`MakerPriceCalculator` から参照可能にせよ。
- **検証**: Additive モードにおいて、上記パラメータで Buy P90 (0.50) 付近の挙動が再現されるか単体テストで確認されたし。

---
