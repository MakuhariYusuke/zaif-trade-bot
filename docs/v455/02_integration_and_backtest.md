# v455 統合とバックテスト (Integration & Backtest)

## 概要

v455戦略では、強化学習モデル(SAC)の出力に対し、**Calibration Gate (EV Gate)** と **Pseudo HFT Execution Model** を適用することで、高頻度取引における収益性を確保します。
本ドキュメントでは、これらのコンポーネントの統合方法と、バックテスト（Shadow Execution）の仕様について記述します。

## 統合アーキテクチャ

### IntegratedEntrySystem

`IntegratedEntrySystem` は、以下の役割を持つファサードクラスです。

1.  **CalibrationMap (統計収集)**:
    *   レジーム別・アクション強度別の勝率、平均利益、平均損失を記録します。
    *   EWMA (指数加重移動平均) を用いて、直近の市場環境に適応します。
    *   `save_state` / `load_state` により、学習した統計情報を永続化可能です。

2.  **CalibrationGate (EV判定)**:
    *   `CalibrationMap` から取得した統計情報と、現在の市場状況（スプレッド、ボラティリティ）から推定されるコストを用いて、期待値 (EV) を計算します。
    *   `EV > 0` の場合のみエントリーを許可します。

### PseudoHFTExecutionModel

`PseudoHFTExecutionModel` は、HFT特有の執行コストをシミュレーションします。

*   **スプレッド**: `High - Low` または `ATR` ベースのプロキシ。
*   **ボラティリティリスク**: レイテンシ間の価格変動リスク。
*   **マーケットインパクト**: 注文サイズと出来高に基づくインパクト。
*   **Fail-Closed (Gate)**: `CalibrationGate` は、市場データが欠損（NaN/Inf）している場合、無限大のコストを返して取引をブロックします。
    *   `PseudoHFTExecutionModel` 自体は、ATRやVolumeが欠損している場合に保守的なフォールバック値（例: 価格の0.05%）を用いて計算を継続しますが、Gate段階でブロックされるため、実質的に安全性が担保されます。

## バックテスト仕様 (Shadow Execution)

`backtest_v455.py` は、既存の強化学習環境 (`HeavyTradingEnv`) と並行して、v455のロジックを「影（Shadow）」として実行します。

### 1. ウォームアップ期間 (Warm-up Period)

バックテスト開始直後は `CalibrationMap` が空であるため、EVが計算できず（または負になり）、すべての取引がブロックされる可能性があります。
これを防ぐため、`--warmup-steps` で指定した期間（デフォルト: 10,000ステップ）は、**Gateの判定結果にかかわらず（ブロックされても）強制的にエントリー**を行います。
ただし、**Fail-Closed（コスト無限大）の場合は、統計汚染を防ぐためにウォームアップ期間中でもエントリーをブロック**します。
これにより、初期の統計情報を収集し、その後のGate判定の精度を高めます。

### 2. ショート戦略 (Short Selling)

v455はロング・ショート双方に対応しています。

*   **ショートエントリー**: ポジションがない状態で、モデル出力が `negative_threshold` を下回った場合。
*   **ショートカバー（買戻し）**: ショートポジション保有中に、モデル出力が `threshold` を上回った場合。
*   **PnL計算**: `(EntryPrice - ExitPrice) * Size`
    *   `PseudoHFTExecutionModel` は、売り注文に対しては `Bid` 側（価格 - スリッページ）、買い注文に対しては `Ask` 側（価格 + スリッページ）の価格を返します。

### 3. 損益計算と二重計上の防止

*   **Gateのコスト見積もり**: スリッページと手数料をコストとしてEVから差し引きます。
*   **CalibrationMapの更新**: **スリッページ適用前**の市場価格（Close）の変動幅を「Gross PnL」として記録します。
    *   これにより、スリッページが「コスト」と「過去の損益実績」の両方で二重に計上されるのを防ぎます。

## 実行方法

```bash
# 10000ステップのウォームアップを行い、合計50000ステップのバックテストを実行
python backtest_v455.py --steps 50000 --warmup-steps 10000
```
