# 強化学習アルゴリズム拡張計画

## 🎯 目的

PPOの限界（HOLD 89%収束問題）を克服するため、金融取引に適した別アルゴリズムを導入。

## 📁 提案するディレクトリ構成

```
ztb/
├── training/
│   ├── algorithms/              # 🆕 アルゴリズム別実装
│   │   ├── __init__.py
│   │   ├── base_algorithm.py   # 基底クラス
│   │   ├── ppo/                 # 既存PPO（整理）
│   │   │   ├── __init__.py
│   │   │   ├── ppo_trainer.py
│   │   │   ├── custom_ppo.py
│   │   │   └── config.py
│   │   ├── sac/                 # 🆕 SAC実装
│   │   │   ├── __init__.py
│   │   │   ├── sac_trainer.py
│   │   │   ├── custom_sac.py
│   │   │   └── config.py
│   │   ├── td3/                 # 🆕 TD3実装（将来）
│   │   │   └── __init__.py
│   │   └── a2c/                 # 🆕 A2C実装（将来）
│   │       └── __init__.py
│   ├── core/                    # 既存のまま
│   │   ├── algorithm_trainer.py # 🔄 更新: 複数アルゴリズム対応
│   │   ├── config_manager.py
│   │   ├── feature_schema_manager.py
│   │   └── ...
│   ├── trainers/                # 既存のまま
│   ├── models/                  # 既存のまま
│   └── unified_trainer.py       # 🔄 更新: アルゴリズム選択機能追加
│
├── configs/
│   ├── ppo/                     # 🆕 PPO設定を整理
│   │   ├── ppo_v394d_aggressive.json
│   │   └── ppo_v394f_ultra_entropy.json
│   ├── sac/                     # 🆕 SAC設定
│   │   ├── sac_v395_baseline.json
│   │   └── sac_v395_high_alpha.json
│   └── archive/                 # 古い設定
│
└── docs/
    ├── algorithms/               # 🆕 アルゴリズム別ドキュメント
    │   ├── ppo_analysis.md       # PPOの学び
    │   ├── sac_design.md         # SAC設計
    │   └── comparison.md         # アルゴリズム比較
    └── training_results/         # 訓練結果まとめ
        ├── v394_series.md
        └── v395_sac_series.md
```

## 🆕 金融取引向けアルゴリズム選定

### 1. SAC (Soft Actor-Critic) ⭐⭐⭐⭐⭐

**特徴**:
- **最大エントロピーRL**: 探索がアルゴリズムに組み込まれている
- **Off-policy**: サンプル効率が高い
- **連続・離散両対応**: 柔軟な行動空間

**金融取引への適性**:
```python
✅ 探索維持: エントロピー最大化が目的関数の一部
✅ 安定性: Twin Q-networkで過大評価を抑制
✅ サンプル効率: Replay bufferでデータ再利用
✅ 実績: 金融取引での成功例多数
```

**PPOとの違い**:
```
PPO: 保守的、HOLD収束しやすい
SAC: 積極的探索、多様な行動維持
```

### 2. TD3 (Twin Delayed DDPG) ⭐⭐⭐⭐

**特徴**:
- **連続行動空間**: 売買タイミング・量を細かく制御
- **Delayed policy updates**: 安定性向上
- **Target policy smoothing**: ノイズに強い

**金融取引への適性**:
```python
✅ 連続制御: ポジションサイズを連続値で最適化
✅ 安定性: Twin Q-network
✅ ノイズ耐性: 市場変動に強い
```

### 3. A2C (Advantage Actor-Critic) ⭐⭐⭐

**特徴**:
- **シンプル**: PPOより単純
- **On-policy**: リアルタイム学習向き
- **同期更新**: 安定した学習

**金融取引への適性**:
```python
✅ シンプル: デバッグしやすい
✅ 高速: PPOより軽量
⚠️ サンプル効率: PPOより劣る
```

## 🎯 推奨実装順序

### Phase 1: SAC実装（最優先）

**理由**:
1. 最大エントロピーRLでHOLD収束問題を解決
2. 金融取引での実績が豊富
3. Stable-Baselines3にSAC実装あり

**実装計画**:
```
Week 1: SAC基本実装
  - custom_sac.py作成
  - sac_trainer.py作成
  - 既存環境との統合

Week 2: SAC最適化
  - ハイパーパラメータ調整
  - alpha（エントロピー係数）自動調整
  - Replay buffer最適化

Week 3: 比較評価
  - SAC vs PPO
  - 収益性テスト
  - 実運用判断
```

### Phase 2: TD3実装（オプション）

**条件**: SACで不十分な場合のみ

### Phase 3: 最適アルゴリズム選定

**評価軸**:
- Return (収益率)
- Action分布（HOLD比率）
- 安定性
- 訓練時間

## 📝 設定ファイル設計

### SAC設定ファイル例

```json
{
  "model_name": "sac_v395_baseline",
  "algorithm": "sac",
  "total_timesteps": 100000,
  
  "sac_hyperparameters": {
    "learning_rate": 0.0003,
    "buffer_size": 50000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_entropy": "auto",
    "use_sde": false
  },
  
  "environment": {
    "initial_balance": 200000,
    "transaction_cost": 0.0005,
    "reward_settings": {
      "hold_penalty_weight": 0.5,
      "successful_trade_bonus": 20.0,
      "profit_reward_multiplier": 50.0
    }
  }
}
```

## 🔄 既存コードの更新

### 1. algorithm_trainer.py

```python
class AlgorithmTrainer:
    def train(self, algorithm: str, config: dict):
        if algorithm == "ppo":
            return self.ppo_trainer.train(config)
        elif algorithm == "sac":
            return self.sac_trainer.train(config)  # 🆕
        elif algorithm == "td3":
            return self.td3_trainer.train(config)  # 🆕
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
```

### 2. unified_trainer.py

```python
class UnifiedTrainer:
    def __init__(self, config: dict):
        self.algorithm = config.get("algorithm", "ppo")  # 🆕
        # ...
```

## 📊 期待される改善

### SAC vs PPO比較

| 指標 | PPO (v394d) | SAC (期待) | 改善 |
|------|-------------|------------|------|
| 最終HOLD% | 89.1% | <70% | ✅ |
| エントロピー | 0.61 | >1.0 | ✅ |
| 探索維持 | ❌ | ✅ | ✅ |
| サンプル効率 | 中 | 高 | ✅ |
| 訓練時間 | 6分 | 8-10分 | ⚠️ |

## 🎬 次のステップ

1. **ディレクトリ作成**
   ```bash
   mkdir ztb/training/algorithms
   mkdir ztb/training/algorithms/ppo
   mkdir ztb/training/algorithms/sac
   mkdir configs/sac
   mkdir docs/algorithms
   ```

2. **既存PPOコード整理**
   - ppo_trainer.py → algorithms/ppo/
   - custom_ppo.py → algorithms/ppo/

3. **SAC実装開始**
   - custom_sac.py作成
   - sac_trainer.py作成
   - 設定ファイル作成

4. **統合テスト**
   - SAC訓練実行
   - PPO vs SAC比較
   - 収益性評価

---

**質問**: この構成で進めてよろしいですか？それとも、何か調整が必要ですか？
