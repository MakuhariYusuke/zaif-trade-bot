# SAC v397h: BUY行動学習修正版

## 概要
v397gの分析結果に基づき、BUY行動0%問題を解決するための修正版。

## 根本原因 (v397g分析結果)
1. **max_position_size=0.01が厳しすぎる**: 購入可能額2000円 (0.0004 BTC) のみ
2. **threshold=0.10が高すぎる**: モデル最大出力0.0718 < 0.10
3. **学習探索不足**: エントロピー係数-4.37、BUY領域未探索
4. **報酬構造問題**: BUYに即時報酬なし、HOLDが最適

## 修正方針
### 1. 閾値調整
- `continuous_to_discrete_threshold`: 0.10 → 0.05
- BUY領域: > 0.05 (従来の2倍に拡大)

### 2. BUY/SELL平等即時ボーナス追加
- `immediate_bonus_rate`: 0.5 (BUY/SELL両方に即時報酬)
- 片方だけを優遇せず、売買バランスを促進

### 3. 探索増加
- `target_entropy`: -2.0 (デフォルト-4.0から緩和)
- 学習初期の探索を促進

### 4. 現実的な購入サイズ
- `max_position_size`: 0.01 → 0.05
- 購入可能額: 2000円 → 10000円 (0.002 BTC)

## 期待される行動分布
- BUY: 15-25% (threshold=0.05で達成可能)
- HOLD: 50-60% (依然として安全策)
- SELL: 15-25% (バランスを取る)

## 報酬構造
```
reward = pnl_ratio * reward_scale +
         trade_bonus (BUY/SELL実行時) +
         immediate_bonus (BUY/SELL実行時) +
         inactivity_penalty (無行動時)
```

## 学習戦略
1. 初期学習: max_position_size=0.05, threshold=0.05
2. 安定化後: 徐々に制約を厳しくするカリキュラム学習

## 成功基準
- BUY行動率: > 10%
- SELL行動率: > 10%
- 収益: > -5% (v397gの-1.03%から改善)
- 安定性: 標準偏差 < 50%</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\configs\sac_v397h_buy_fix.json