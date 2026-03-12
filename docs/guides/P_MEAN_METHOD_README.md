# SAC Session Comparison Tool

SAC訓練ログの比較と統計的検定を行うツールです。

## 特徴

- TensorBoardログからメトリクスを抽出
- セッション間の統計的比較
- t検定による個別メトリクスの有意差検定
- **p平均法による総合的有意性評価**
- 効果量の計算

## p平均法 (p-mean method)

p平均法は、複数の統計検定のp値を統合して総合的な有意性を評価する手法です。

### 算術平均 (Arithmetic Mean)
複数のp値を単純に平均します。
- 直感的で理解しやすい
- 全てのp値に等しい重み付け
- 極端なp値の影響を受けやすい

### 幾何平均 (Geometric Mean)
p値の対数を平均し、指数関数で戻します。
- 極端なp値の影響を緩和
- 0に近いp値を適切に扱える
- 統計学的によりロバスト

### 使用例

```python
from compare_sac_sessions import p_mean_method

# 3つのメトリクスのp値統合
p_values = [0.03, 0.07, 0.02]  # 個別のt検定結果

# 算術平均
combined_p_arithmetic = p_mean_method(p_values, 'arithmetic')

# 幾何平均
combined_p_geometric = p_mean_method(p_values, 'geometric')

# 有意性判断
significant = combined_p_arithmetic < 0.05
```

### 解釈

- `p_mean < 0.05`: 全体として統計的有意な差がある
- `p_mean >= 0.05`: 全体として統計的有意な差がない
- 個別の検定結果も確認し、総合的な判断を行う

### 注意事項

- p値が完全に独立であることを仮定
- 相関のある検定では結果が保守的になる可能性
- 解釈時は個別の検定結果も確認すること

## 出力結果

```
🔬 train/ent_coef:
  t-statistic: -2.3456
  p-value: 0.0234
  Significant (p<0.05): ✅ Yes
  Effect size: 0.4567

🔬 P-MEAN METHOD:
  Arithmetic mean p-value: 0.0345
  Geometric mean p-value: 0.0312
  Overall significant (p<0.05): ✅ Yes
```

## 使用方法

```bash
python compare_sac_sessions.py
```

## 参考文献

- [統計的仮説検定の多重性問題とp値の統合](https://note.com/a_small_hamster/n/n718dbe6bfe9e)
- [BTC ML: p値の統合手法](https://note.com/btcml/n/n0d9575882640)
