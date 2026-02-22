# v455 Sensitivity Analysis / 長期学習 / 可視化プラン

## 現状整理
- 即死は解消（平均生存 8 時間、即死率 0%）。
- ただし手数料 + スプレッドで資金が緩やかに減少。
- 主要パラメータは `min_edge_mult=1.2`、`vol_floor=0.001` で学習中。

## 1. 感度分析（min_edge_mult / vol_floor）
### 目的
- **手数料負け回避**と**低ボラ抑制**の強度が、収益性と取引回数に与える影響を定量化。

### 推奨グリッド
- `min_edge_mult`: **1.0 / 1.5 / 2.0**
- `vol_floor`: **0.0005 / 0.001 / 0.002**（0.05% / 0.1% / 0.2%）

### 評価指標（最低限）
- **Net PnL / 10k steps**
- **Trade Count / 10k steps**
- **Step Cost 合計 / 10k steps**（`trade_cost`）
- **Edge Shortfall 合計 / 10k steps**（`edge_shortfall`）

### 推奨手順
1. **短期学習**（例: 50k steps）× パラメータ組み合わせ × **3 seeds**。
2. 学習済みモデルで **5 エピソード評価**。
3. 上記指標の中央値で比較（外れ値耐性）。

## 2. 可視化（edge_shortfall / trade_cost）
### 追加ログ
`FastIntradayEnv` から `edge_shortfall` / `trade_cost` / `vol_ratio` を `info` 経由で返し、  
`HFTMetricsCallback` で TensorBoard に記録。

### 観察ポイント
- **edge_shortfall が減る**: 無駄なトレードが減少。
- **trade_cost が減る**: 低ボラ局面での取引回避が成功。
- **vol_ratio の分布**: 低ボラ抑制が過剰なら取引ゼロに近づく。

## 3. 「統計的に十分なステップ数」の目安
### サンプルサイズの考え方
平均リターンの推定誤差を `epsilon`（JPY）に抑えるための必要トレード数は:
```
n >= (z * sigma / epsilon)^2
```
- `z`: 信頼水準（95%なら 1.96）
- `sigma`: 1トレードあたりの標準偏差（JPY）

### ステップ数への換算
```
steps_needed = n / trade_rate
```
（`trade_rate` = 1 step あたりの平均トレード数）

### 例
- `sigma=200 JPY`, `epsilon=20 JPY`, `trade_rate=0.01` の場合  
  `n ~ 384`、`steps ~ 38,400`

### 推奨
- **短期比較**: 50k – 200k steps（感度分析向け）
- **本格学習**: 300k – 600k steps
  - 100万 steps は **trade_rate が低い場合のみ**有効。
  - 学習中に **Rolling Sharpe / PnL トレンド**が鈍化したら早期停止。

## 4. 実務的な追加改善案
- **Dynamic min_edge_mult**  
  ボラが低いほど `min_edge_mult` を上げる（低ボラでの無駄打ちを抑制）。
- **Entry Cooldown Adaptive**  
  コストが高い環境ではクールダウンを延長。
- **No-Trade Bonus**  
  低ボラ時に「何もしない」行動へ微小ボーナス（過剰トレード抑制）。

## 5. 次のアクション
1. 感度分析（3x3 グリッド × 3 seeds）  
2. 最良設定で 300k – 600k steps 本格学習  
3. `edge_shortfall` と `trade_cost` のトレンドを可視化して検証

## 変更点（実装反映）
- `ztb/trading/rewards/fast_intraday.py`: reward_info に `expected_move` / `required_edge` / `position_change` を追加
- `scripts/v455/train_hft.py`: `edge_shortfall` / `trade_cost` / `vol_ratio` を TensorBoard へ出力
