# v453 Threshold Optimization Results

## 概要
`breakdown_setup` と `moderate_bear_trend` を「売り有利 (Sell Favorable)」なレジームとして定義し、閾値を調整（買い閾値↑、売り閾値↓）した結果の検証。

## 結果比較 (v3 Fixed vs Threshold Test)

### 全体パフォーマンス
| Metric | v3 Fixed (Baseline) | Threshold Test | 変化 |
| :--- | :--- | :--- | :--- |
| **Total Return** | **12.43%** | 5.98% | ▼ 大幅悪化 |
| **Profit Factor** | **1.20** | 1.13 | ▼ 悪化 |
| **Max Drawdown** | -2.40% | **-2.05%** | ▲ 改善 |

### ターゲットレジームの損益 (PnL)
| Regime | v3 Fixed PnL | Threshold Test PnL | 評価 |
| :--- | :--- | :--- | :--- |
| **breakdown_setup** | -408 | **+120** | **◎ 黒字化達成！** |
| **moderate_bear_trend** | -194 | **+777** | **◎ 大幅黒字化！** |

### その他のレジームへの影響
| Regime | v3 Fixed PnL | Threshold Test PnL | 変化 |
| :--- | :--- | :--- | :--- |
| **consolidation** | **+6,051** | +122 | **▼▼ 激減 (致命的)** |
| **low_volatility_ranging** | +1,799 | -1,439 | **▼▼ 赤字転落** |
| **strong_bull_trend** | +6,822 | **+7,752** | ▲ 改善 |

## 考察
1.  **ターゲットレジームの改善**:
    - 狙い通り、`breakdown_setup` と `moderate_bear_trend` は黒字化しました。売り閾値を下げたことで、下落局面で適切に利益を上げられるようになったと考えられます。

2.  **副作用 (Consolidation / Ranging の悪化)**:
    - しかし、`consolidation`（保ち合い）や `low_volatility_ranging` での利益が壊滅しました。
    - **原因**: 今回の変更で `ThresholdManager` のロジック全体に影響が出た可能性があります。特に `MODERATE_BULL` を追加したことや、`HeavyTradingEnv` の修正により、これまで「Unknown」や「Consolidation」として扱われていた期間が別のレジーム（例えば誤検知されたBear系レジームなど）として扱われ、不適切な閾値が適用された可能性があります。
    - あるいは、`consolidation` 自体は変更していませんが、レジーム遷移のタイミングや、`HeavyTradingEnv` の修正（`_get_current_market_regime` の使用）により、これまで偶然うまくいっていたバランスが崩れた可能性があります。

## 結論
- **部分的には大成功**: 課題であった下落予兆・緩やかな下落での収益化には成功しました。
- **全体としては失敗**: 安定収益源であったレンジ相場での利益を失いました。

## 次のステップ
- `breakdown_setup` と `moderate_bear_trend` の改善策（売り有利設定）は維持しつつ、`consolidation` や `low_volatility_ranging` への悪影響を調査・修正する必要があります。
- とりあえずの「改善版」としては、**`breakdown_setup` と `moderate_bear_trend` を除外する**（当初の案）のが、全体パフォーマンスとしては安全かつ確実（Total Return 12%以上を維持しつつ、マイナス要因を排除できるため）かもしれません。
