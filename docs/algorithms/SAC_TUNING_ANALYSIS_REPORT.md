# SAC Parameter Tuning Analysis Report
## 2025-10-11

## 📊 Executive Summary

3つのSACバージョンを5000 timestepsで訓練し、比較評価を実施。

| Version | Description | Overall Score |
|---------|-------------|---------------|
| **v395a** | Original (learning_rate=0.0003, batch_size=64, gamma=0.99) | 🥇 6 points |
| **v395b** | Aggressive (learning_rate=0.0001, batch_size=128, gamma=0.95, train_freq=4) | 🥈 6 points |
| **v395c** | Conservative (learning_rate=0.0001, batch_size=128, gamma=0.98) | 🥉 4 points |

## 🔍 Detailed Metrics Comparison

### 1. Entropy Coefficient (ent_coef)
**Target Range**: 1.0 - 2.0 (適度な探索)

| Version | Final Value | Status |
|---------|-------------|--------|
| v395a | 4.03 | ❌ 高すぎる（過度な探索） |
| v395b | 1.53 | ✅ 理想的 |
| v395c | 1.53 | ✅ 理想的 |

**Winner**: v395b / v395c (同点)
- `target_entropy = -0.5` の設定が効果的
- v395aの`target_entropy = "auto"` (-1.0)では探索が強すぎる

### 2. Critic Loss
**Target**: できるだけ低く、安定

| Version | Final Value | Min Value | Max Value |
|---------|-------------|-----------|-----------|
| v395a | 4.34e+07 | 2.60e+04 | 2.64e+08 |
| v395b | 1.61e+08 | 1.61e+08 | 3.15e+09 |
| v395c | 1.49e+10 | 1.66e+08 | 3.03e+10 |

**Winner**: 🥇 v395a (最も低い)

**重要な発見**:
- v395aは最終値が最も低いが、最大値は2.64e+08まで跳ね上がる
- v395bは初期値が高いが、最終値で1.61e+08まで下がる（改善傾向）
- v395cは完全に発散している（1.49e+10）

### 3. Actor Loss
**Target**: できるだけ低く

| Version | Final Value |
|---------|-------------|
| v395a | 4.90e+04 |
| v395b | 1.61e+05 |
| v395c | 9.69e+05 |

**Winner**: 🥇 v395a (圧倒的に低い)

## 💡 Key Insights

### 1. target_entropy の重要性
- **v395a**: `target_entropy = "auto"` → ent_coef 4.03 (探索過剰)
- **v395b/c**: `target_entropy = -0.5` → ent_coef 1.53 (適切)
- ✅ **推奨**: 連続行動空間（dim=1）では `target_entropy = -0.5` を使用

### 2. train_freq / gradient_steps の効果
- **v395b**: train_freq=4, gradient_steps=4
  - Critic Loss初期値が高い（3.15e+09）
  - しかし最終的には1.61e+08まで改善
  - 学習効率は向上するが、初期の不安定性あり

- **v395c**: train_freq=1, gradient_steps=1（v395aと同じ）
  - より頻繁な更新だが、Critic Loss が発散

### 3. learning_rate の影響
- **v395a**: learning_rate=0.0003
  - Actor/Critic Loss が最も低い
  - しかし ent_coef が高すぎる

- **v395b/c**: learning_rate=0.0001
  - ent_coef は改善
  - しかしActor/Critic Lossは悪化

### 4. gamma (割引率) の影響
- **v395b**: gamma=0.95（低め）
  - Critic Loss の初期値は高いが、改善傾向

- **v395c**: gamma=0.98（中間）
  - Critic Loss が完全に発散

## 🎯 Optimal Parameter Set Proposal

### v395d - "Best of Both Worlds"

```json
{
  "sac_hyperparameters": {
    "learning_rate": 0.0003,        // v395aを維持（Actor/Critic Lossが低い）
    "buffer_size": 20000,            // v395bから採用（多様な経験）
    "learning_starts": 500,          // v395bから採用（初期安定性）
    "batch_size": 128,               // v395bから採用（分散削減）
    "tau": 0.005,                    // 全て同じ
    "gamma": 0.99,                   // v395aを維持（Lossが低い）
    "train_freq": 1,                 // v395aを維持（安定性優先）
    "gradient_steps": 1,             // v395aを維持
    "ent_coef": "auto",              // 変更なし（自動調整）
    "target_update_interval": 1,     // 変更なし
    "target_entropy": -0.5,          // ✅ v395b/cから採用（最重要！）
    "policy_kwargs": {
      "net_arch": [256, 256]         // v395aを維持（大きめのネットワーク）
    }
  }
}
```

### 変更の根拠

1. ✅ **target_entropy = -0.5**: ent_coefを1.53に抑える（v395bで実証済み）
2. ✅ **learning_rate = 0.0003**: Actor/Critic Lossを低く保つ（v395aで実証済み）
3. ✅ **batch_size = 128**: 勾配の分散を減らす（v395bで採用）
4. ✅ **buffer_size = 20000**: より多様な経験から学習
5. ✅ **learning_starts = 500**: 初期の不安定性を回避
6. ✅ **gamma = 0.99**: Q値の推定精度を維持（v395aで成功）
7. ✅ **train_freq/gradient_steps = 1**: 頻繁な更新で安定性確保

## 📋 Next Steps

### Phase 1: v395d Validation (5k timesteps)
- 設定ファイル作成
- 5k timesteps訓練
- 目標:
  - ent_coef < 2.0
  - Critic Loss < 1e8
  - Actor Loss < 1e5

### Phase 2: Extended Training (10k timesteps)
- v395dが成功したら10kに拡張
- アクション分布の確認（HOLD比率）
- 報酬推移の分析

### Phase 3: Long-term Training (50k - 100k)
- 最終的な100k timesteps訓練
- PPOとの比較評価:
  - HOLD比率: 目標 40-60%
  - エントロピー: 維持 > 1.0
  - 収益性: total return, Sharpe ratio

## 🚨 Lessons Learned

1. **target_entropyは手動設定が必須**:
   - "auto"だと探索が強すぎる（特に低次元の行動空間）
   - 連続行動dim=1では-0.5が適切

2. **learning_rateは慎重に下げる**:
   - 安定性向上のために下げがちだが、性能が悪化することがある
   - v395aの0.0003が最も良好なLoss値を示した

3. **train_freq/gradient_stepsの調整は諸刃の剣**:
   - 計算効率は向上するが、初期の不安定性が増す
   - 当面はtrain_freq=1, gradient_steps=1で安定性優先

4. **gammaは0.99が最適**:
   - 0.95に下げるとCritic Lossの初期値が跳ね上がる
   - 金融取引では将来の報酬も重要なので0.99が適切

## 📊 Comparison Chart Summary

```
Metric              | v395a    | v395b    | v395c    | Target
--------------------|----------|----------|----------|--------
ent_coef (final)    | 4.03 ❌  | 1.53 ✅  | 1.53 ✅  | 1.0-2.0
Critic Loss (final) | 4.3e7 ✅ | 1.6e8 ⚠️ | 1.5e10❌ | < 1e8
Actor Loss (final)  | 4.9e4 ✅ | 1.6e5 ⚠️ | 9.7e5 ❌ | < 1e5
Overall Score       | 6 pts 🥇 | 6 pts 🥈 | 4 pts 🥉 | -
```

**Conclusion**: v395aが最もバランスが取れているが、ent_coefが高すぎる。
**Solution**: v395dでtarget_entropy=-0.5を追加し、他のパラメータはv395aベースを維持。
