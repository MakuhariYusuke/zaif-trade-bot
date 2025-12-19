# v453 Improvement Plan: Regime-Based Threshold Optimization

## 課題
- `breakdown_setup` レジームで損失が発生している (-408)。
- `moderate_bear_trend` も微損 (-194)。
- これらは下落基調であるにもかかわらず、モデルが適切にショートできていない（またはロングしてしまっている）可能性がある。

## 解決策: ThresholdManagerの最適化
`ThresholdManager` 内のハードコードされたレジーム分類を更新し、これらのレジームを「売り有利 (Sell Favorable)」として扱うように変更しました。

### 変更内容 (`ztb/trading/environment/components/threshold_manager.py`)

1.  **`sell_favorable_regimes` に以下を追加**:
    - `"BREAKDOWN"`: `breakdown_setup` をカバー。
    - `"MODERATE_BEAR"`: `moderate_bear_trend` をカバー。

2.  **`buy_favorable_regimes` に以下を追加**:
    - `"MODERATE_BULL"`: `moderate_bull_trend` をカバー（念のため）。

### 期待される挙動の変化
これらのレジームにおいて：
- **買い閾値 (Buy Threshold)**: `base * 10.0` に上昇 → **買いエントリーが激減**する。
- **売り閾値 (Sell Threshold)**: `base * 0.5` に低下 → **売りエントリーがしやすく**なる。

これにより、下落予兆（Breakdown Setup）や緩やかな下落（Moderate Bear）において、逆張りロングによる損失を防ぎ、順張りショートによる利益獲得を狙います。

## 検証手順
1.  `hybrid_config_v3_optimized.json` を使用してバックテストを実行。
    - 注: `hybrid_config` でこれらを除外(`excluded_regimes`)してしまうと、このThreshold改善の効果が見えなくなるため、**除外リストからは外す**必要があります。

## 設定ファイルの修正
`config/v453/hybrid_config_v3_optimized.json` を修正し、`breakdown_setup` と `moderate_bear_trend` を除外リストから削除（またはコメントアウト）して、Threshold改善の効果をテストします。
