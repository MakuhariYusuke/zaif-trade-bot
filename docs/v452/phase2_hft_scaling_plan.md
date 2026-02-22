# Phase 2: HFT Scaling & Robustness Strategy (v452)

## 1. 概要と背景
Phase 1（適応型閾値の導入）において、HFTロジックが極めて高い収益性（PnL +720万円、収益率 3600%超）を持つことが実証されました。
Phase 2では、この「勝ちパターン」を最大限に活用しつつ、実運用を見据えた堅牢性を確保するための施策を実行します。

**コアコンセプト**: 「勝てる局面（高ボラティリティ）で攻め、負ける局面（トレンド転換）で粘る」

---

## 2. 詳細戦略

### A. レジーム別ダイナミック・ポジションサイジング (Dynamic Position Sizing)
**理論的背景**:
ケリー基準（Kelly Criterion）に基づき、期待値と勝率が高い局面ではリスク（ポジションサイズ）を取るべきです。バックテスト結果は `extreme_volatility` が圧倒的な収益源であることを示しています。

**実装ロジック**:
`PositionManager` または環境の注文執行ロジックにおいて、現在のレジームに基づき基本ロットサイズ（`max_position_size` に対する割合）を動的に変更します。

| レジーム | 係数 (Multiplier) | 理由 |
| :--- | :--- | :--- |
| **Extreme Volatility** | **1.5x** | 最も期待値が高い。積極的にリスクを取る。 |
| **High Volatility Ranging** | **1.2x** | 次点で期待値が高い。やや強気に。 |
| **Strong Bull Trend** | **1.0x** | 順張りは重要だが、急落リスクもあるため標準維持。 |
| **Others** | **1.0x** | 標準サイズ。 |
| **Weak/Uncertain** | **0.5x** | (オプション) 確信度が低い場合はサイズを落とす。 |

**リスク管理**:
ポジションサイズ増加に伴いドローダウンリスクも増大するため、`stop_loss` の幅は変えずに維持、あるいはボラティリティに合わせて微調整します。

### B. 上昇トレンド特化型トレーリングストップ (Trend-Specific Trailing Stop)
**課題**:
現在のモデルは `strong_bull_trend` において、早すぎる利確（Take Profit）や、逆張りショートによる損失（-2.5万円）を出しています。

**実装ロジック**:
`ActionExecutor` または環境のステップ処理において、レジームが `strong_bull_trend` の場合のみ以下の挙動を強制します。

1.  **固定利確（Take Profit）の無効化**:
    *   「上がりすぎ」判定による早期撤退を防ぎます。
2.  **トレーリングストップの強制適用**:
    *   価格上昇に合わせてストップラインを引き上げ続け、トレンド反転が確定するまでポジションを保有します。
    *   更新幅（Callback Rate）はボラティリティ（ATR）の 1.5倍〜2.0倍程度とし、ノイズによる狩りを防ぎます。

### C. 執行リアリズムの検証 (Execution Realism Verification)
**課題**:
HFTロジックはスリッページや約定遅延に対して非常に脆弱です。現在の「理想的な約定」での+720万円が、現実の摩擦を含めた場合にどれだけ残るかを確認する必要があります。

**検証シナリオ**:
`RealisticExecutionModel` を有効化し、以下のパラメータでバックテストを行います。

*   **遅延 (Latency)**: 50ms 〜 200ms (API通信 + 処理時間)
*   **スリッページ (Slippage)**: ATRベースの動的スリッページ (ボラティリティが高いほど滑る)
*   **約定確率 (Fill Probability)**: 1.0 (今回は指値ではなく成行/ストリーミングを想定するため確実性を重視するが、スリッページでコストを払う)

---

## 3. 実装ロードマップ

### Step 1: コンポーネントの改修
1.  **`HeavyTradingEnv`**: `step` メソッド内で `PositionManager` や `ActionExecutor` に `current_regime` を渡すように変更。
2.  **`PositionManager`**: `get_target_size(action, regime)` メソッドを追加し、レジーム係数を適用。
3.  **`ActionExecutor`**: `strong_bull_trend` 時に `take_profit` を無視し、`trailing_stop` を優先するロジックを追加。

### Step 2: 設定ファイルの更新
`config/v452/threshold_optimized.json` (または新規 `position_sizing.json`) に係数設定を追加。

```json
{
    "sizing_multipliers": {
        "extreme_volatility": 1.5,
        "high_volatility_ranging": 1.2
    },
    "trailing_stop_config": {
        "bull_trend_callback_atr": 2.0
    }
}
```

### Step 3: 検証とチューニング
1.  **サイジング検証**: PnLがさらに伸びるか、ドローダウンが許容範囲内か確認。
2.  **リアリズム検証**: `RealisticExecutionModel` で利益がプラスを維持できるか確認（損益分岐点を探る）。

---

## 4. 期待される成果
*   **PnL**: 1000万円超え（サイジング効果）
*   **Bull Trend PnL**: マイナスからプラスへの転換（トレーリングストップ効果）
*   **信頼性**: 実運用に耐えうるロジックであることの証明
