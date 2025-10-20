# v383 設定変更サマリー

**作成日**: 2025年10月10日  
**ベース**: v381_revised_profit_focused  
**戦略**: 慎重な微調整のみ（vf_coef=0.3を絶対維持）

---

## 🎯 v383の変更点

### 重要な原則

1. ✅ **vf_coef=0.3を維持**（最重要）
   - v381_revised: 0.3 → EV 51.7% ✅
   - v382_revised: 0.5 → EV 2.3% ❌
   - **結論**: 0.3が最適値、変更しない

2. ✅ **learning_rate=0.003を維持**
   - v381_revised: 0.003 → 報酬-525 (50k) ✅
   - v382_revised: 0.002 → 報酬-991 (100k) ❌
   - **結論**: 0.003の方が収束が早い

3. 🔧 **一度に少数のパラメータのみ変更**
   - v382で複数変更して原因特定困難に
   - v383では慎重に微調整

---

## 📊 パラメータ比較表

### PPOハイパーパラメータ

| パラメータ | v381_revised | v383_optimized | 変更理由 |
|-----------|-------------|---------------|---------|
| **learning_rate** | 0.003 | **0.003** | ✅ 維持（最適） |
| **n_steps** | 1024 | **1024** | ✅ 維持 |
| **batch_size** | 256 | **256** | ✅ 維持 |
| **n_epochs** | 16 | **16** | ✅ 維持 |
| **gamma** | 0.8475 | **0.8475** | ✅ 維持 |
| **gae_lambda** | 0.8 | **0.8** | ✅ 維持 |
| **clip_range** | 0.1 | **0.15** | 🔧 微増（0.2は大きすぎる） |
| **vf_coef** | 0.3 | **0.3** | ✅ 維持（最重要） |
| **max_grad_norm** | 5.05 | **5.05** | ✅ 維持 |
| **target_kl** | 0.01 | **null** | 🔧 無効化（Early stopping削減） |
| **verbose** | 1 | **1** | ✅ 維持 |

### 報酬関数パラメータ

| パラメータ | v381_revised | v383_optimized | 変更理由 |
|-----------|-------------|---------------|---------|
| **hold_penalty_weight** | 0.05 | **0.055** | 🔧 +10%強化（HOLD削減） |
| **consecutive_hold_threshold** | 3 | **2** | 🔧 早期ペナルティ（HOLD削減） |
| **consecutive_hold_penalty** | 0.025 | **0.025** | ✅ 維持 |
| **hold_opportunity_cost** | 0.015 | **0.015** | ✅ 維持 |
| **buy_action_bonus** | 0.15 | **0.20** | 🔧 +33%増（BUY促進） |
| **buy_multiplier** | なし | **1.2** | 🆕 新規（BUY時報酬x1.2） |
| **profit_reward_multiplier** | 5.0 | **5.0** | ✅ 維持 |
| **successful_trade_bonus** | 1.2 | **1.2** | ✅ 維持 |
| **trading_frequency_bonus** | 0.25 | **0.25** | ✅ 維持 |
| **profit_threshold_bonus** | 0.5 | **0.5** | ✅ 維持 |

### トレーニング設定

| パラメータ | v381_revised | v383_optimized | 変更理由 |
|-----------|-------------|---------------|---------|
| **total_timesteps** | 50,000 | **75,000** | 🔧 +50%増（収束確保） |
| **checkpoint_interval** | 5,000 | **5,000** | ✅ 維持 |
| **progress_bar** | false | **false** | ✅ 維持 |

---

## 🎯 期待される効果

### 変更1: target_kl = null（Early stopping無効化）

**問題**:
- v381_revised: Early stopping 100%発生（approx_kl=0.072 vs target=0.01）
- v382_revised: Early stopping 100%発生（approx_kl=0.087 vs target=0.05）

**期待効果**:
- ✅ 全16エポックを実行可能（現状1エポックのみ）
- ✅ 学習速度16倍向上の可能性
- ✅ より深い学習が可能

**リスク**:
- ⚠️ KL divergenceが大きくなりすぎる可能性
- ⚠️ ポリシー更新が不安定になる可能性

**対策**: 最初の20イテレーションで監視、問題あれば中断

### 変更2: clip_range = 0.15（微調整）

**v381_revised**: 0.1 → clip_fraction 99.6-100%
**v382_revised**: 0.2 → clip_fraction 61-74%（しかし報酬悪化）

**期待効果**:
- ✅ clip_fractionを80-90%程度に（適度な制限）
- ✅ 学習の柔軟性向上

**中間値**: 0.1と0.2の中間で様子見

### 変更3: hold_penalty_weight = 0.055（+10%強化）

**v381_revised実績**:
- 最終HOLD率: 51.2%
- 最良HOLD率: 44.5%（Iter 47）

**期待効果**:
- 🎯 HOLD率: 44.5% → 35-40%
- 🎯 目標達成に近づく

### 変更4: consecutive_hold_threshold = 2（早期化）

**v381_revised**: 3回連続HOLDから追加ペナルティ

**期待効果**:
- ✅ 2回連続HOLDから追加ペナルティ
- ✅ よりアグレッシブなHOLD削減

### 変更5: buy_action_bonus = 0.20, buy_multiplier = 1.2

**v381_revised実績**:
- buy_action_bonus = 0.15
- 最終BUY率: 24.6%
- 最良BUY率: 30.5%（Iter 17）

**期待効果**:
- 🎯 BUY率: 24.6% → 30-35%
- ✅ buy_multiplierでBUY時の利益報酬を1.2倍

### 変更6: total_timesteps = 75,000（+50%）

**理由**:
- v381_revised (50k): 後期に不安定化（EV 51.7% → 1.0%）
- v382_revised (100k): learning_rate=0.002で収束せず

**期待効果**:
- ✅ 50kより長く、100kより短い（最適点探索）
- ✅ 予想時間: 約3.5分（50k=113秒 × 1.5）

---

## 📊 性能予測

### 目標値

| 指標 | v381_revised実績 | v383目標 |
|------|-----------------|---------|
| **平均報酬** | -525 | **-450以上** |
| **HOLD率** | 44.5% (最良) | **35-40%** |
| **BUY率** | 30.5% (最良) | **30-35%** |
| **SELL率** | 29.7% (最良) | **25-30%** |
| **Explained Variance** | 51.7% (最良) | **40-60%安定** |
| **Early stop率** | 100% | **0-20%** |
| **Clip fraction** | 99.6-100% | **80-90%** |

### 成功の判断基準

**大成功** ✅✅✅:
- 平均報酬 > -450
- HOLD率 < 40%
- Explained Variance > 40%で安定

**成功** ✅:
- 平均報酬 > -500
- HOLD率 < 43%
- Explained Variance > 30%

**v381_revised同等** ⚠️:
- 平均報酬 -500 ~ -550
- HOLD率 43-45%

**失敗** ❌:
- 平均報酬 < -550
- vf_coefを変更していないのに悪化

---

## 🔍 監視ポイント

### 初期段階（Iter 1-20）

1. **approx_kl値**: target_kl無効化の影響確認
   - 0.1以下: 正常
   - 0.1-0.2: やや高いが許容
   - 0.2以上: 問題、中断検討

2. **clip_fraction**: clip_range=0.15の効果
   - 70-90%: 理想的
   - 50-70%: 許容
   - >95%: 小さすぎる

3. **Explained Variance**: vf_coef=0.3維持の確認
   - 20%以上: 正常
   - 10-20%: やや低い
   - <10%: 問題

### 中期段階（Iter 20-50）

1. **アクション分布の推移**
   - HOLD率が低下傾向か
   - BUY率が上昇傾向か

2. **平均報酬の改善**
   - -600 → -500台に改善か

### 最終段階（Iter 50-75）

1. **学習の安定性**
   - Explained Varianceが維持されているか
   - 報酬が振動せず安定しているか

2. **Early stoppingの発生頻度**
   - 0-20%が理想

---

## 📁 ファイル情報

- **設定ファイル**: `configs/training/ppo_reward_v383_optimized_stable.json`
- **保存先モデル**: `models/ppo_reward_v383_optimized_stable.zip`
- **ログ**: `checkpoints/ppo_reward_v383_optimized_stable_1/`

---

## 🚀 実行コマンド

```bash
cmd /d /c "cd /d C:\Users\Admin\dev\zaif-trade-bot && call .\.venv\Scripts\activate.bat && python run_training.py --config configs/training/ppo_reward_v383_optimized_stable.json"
```

**予想実行時間**: 約3.5分（75kステップ）

---

## 🎯 次のステップ

1. ✅ v383実行
2. 📊 v381_revised vs v383比較分析
3. 🏆 最良モデルでバックテスト
4. 📈 本番デプロイ検討
