# SAC v444.1 設定ファイル設計検討

## 概要
v444モデルの異常収益（75,963%）とSELLバイアス（BUY:46, HOLD:0, SELL:206）の問題を解決するため、v427以降の設定ファイルを参考にv444.1の設定ファイルを作成する。

## 反省点
- v444はreward設計が不十分で、HOLDペナルティが高すぎるためSELLに過剰適合
- モデル再トレーニングが必要だが、設定ファイルの構造化が重要
- v427-v436の進化を参考に、包括的な設定体系を構築

## v427以降の設定ファイル分析

### v427 (sac_v427_market_adaptive_ensemble.json)
**特徴:**
- `v427_advanced_features`: meta_learning, federated_learning, continual_learning, ensemble_system
- `market_regimes`: 相場状況別パラメータ（bull/bear/sideways × high/low vol）
- `reward_settings`: profit_bonuses, action_bonuses, risk_managementの構造化
- `adaptive_feature_selection`: 相場状況に応じた特徴量選択

**良い項目:**
- market_regimesの相場状況別パラメータ設定
- reward_settingsの構造化（profit_bonuses, action_bonuses, risk_management）
- ensemble_systemのspecializations（bull/bear/sideways/high_vol/low_vol）

### v435.5 (sac_v435_config.json)
**特徴:**
- `reward_settings`: base_profit_bonus_atr_coeff, base_profit_bonus_portfolio_coeff, base_action_penalty
- `long_short_asymmetry`: ロング/ショート非対称報酬
- `features`: technical_indicators, price_features, volatility_features, momentum_featuresのカテゴリ化

**良い項目:**
- reward_settingsの詳細パラメータ（base_profit_bonus_atr_coeffなど）
- featuresのカテゴリ化による整理
- long_short_asymmetryの考慮

### v435.6 (sac_v435_config.json + reward_config.json + environment_config.json)
**特徴:**
- `ensemble_system`: models, voting_mechanism, confidence_threshold
- `action_frequency_penalty: 0.0`: スキャルピング最適化
- `ensemble_consensus_bonus`, `diversity_penalty`
- `environment_config`: feature_engineering, market_regime_detection, risk_management, multi_timeframe_integration
- `reward_clipping`: min_reward, max_reward
- `scalping_optimization`: quick_profit_bonus, frequent_trading_bonus

**良い項目:**
- ensemble_systemの詳細設定（voting_mechanism, confidence_threshold）
- action_frequency_penalty: 0.0による取引頻度最適化
- reward_clippingによる報酬安定化
- scalping_optimizationの設定
- multi_timeframe_integration
- risk_managementのdynamic_position_sizing

## v444.1設定ファイルの推奨構造

### 1. 基本構造
```json
{
  "model_name": "sac_v444.1",
  "version": "4.4.4.1",
  "algorithm": "sac",
  "total_timesteps": 50000,
  "description": "SAC v444.1: Balanced reward design with ensemble system"
}
```

### 2. training設定（v435.5/6参考）
```json
"training": {
  "data_config": {
    "data_path": "data/btc_jpy_yahoo_real_20251021_featured_corrected.csv",
    "validation_split": 0.2,
    "test_split": 0.1
  },
  "total_timesteps": 50000,
  "sac_hyperparameters": {
    "learning_rate": 3e-4,
    "batch_size": 256,
    "buffer_size": 1000000,
    "learning_starts": 1000,
    "tau": 0.005,
    "gamma": 0.99,
    "ent_coef": "auto_1.0",
    "target_entropy": "auto"
  }
}
```

### 3. environment設定（v435.6参考）
```json
"environment": {
  "initial_balance": 10000,
  "transaction_cost": 0.0,
  "max_position_size": 1.0,
  "random_start": true,
  "feature_engineering": {
    "enabled": true,
    "adaptive_selection": true,
    "correlation_reduction": true,
    "correlation_threshold": 0.85,
    "max_features": 100
  },
  "market_regime_detection": {
    "enabled": true,
    "regime_window": 50,
    "trend_threshold": 0.02,
    "volatility_threshold": 0.15
  },
  "risk_management": {
    "dynamic_position_sizing": true,
    "max_drawdown_limit": 0.1
  },
  "multi_timeframe_integration": {
    "enabled": true,
    "timeframes": ["5m", "15m", "1h", "4h", "1d"],
    "primary_timeframe": "1h"
  }
}
```

### 4. reward_settings（v427/435参考）
```json
"reward_settings": {
  "base_profit_bonus_atr_coeff": 5.0,
  "base_profit_bonus_portfolio_coeff": 10.0,
  "base_action_penalty": 0.15,
  "loss_penalty_coeff": -1.0,
  "action_frequency_penalty": 0.0,
  "long_short_asymmetry": true,
  "risk_adjusted_bonus": true,
  "market_regime_penalty": true,
  "profit_bonuses": {
    "profit_multipliers": [1.0, 1.0, 1.0]  // BUY/HOLD/SELL均等
  },
  "action_bonuses": {
    "hold_penalty": 0.001,  // HOLDペナルティを極小に
    "transaction_penalty": -0.1
  },
  "risk_management": {
    "max_drawdown_penalty": -0.1,
    "volatility_penalty": -0.05
  },
  "reward_clipping": {
    "enabled": true,
    "min_reward": -10.0,
    "max_reward": 10.0
  }
}
```

### 5. ensemble_system（v427/435.6参考）
```json
"ensemble_system": {
  "enabled": true,
  "models": ["sac_v444.1_base", "sac_v444.1_bull", "sac_v444.1_bear"],
  "voting_mechanism": "majority_vote",
  "confidence_threshold": 0.6,
  "specializations": ["bull", "bear", "sideways"],
  "consensus_bonus": 0.2,
  "diversity_penalty": -0.1
}
```

### 6. market_regimes（v427参考）
```json
"market_regimes": {
  "bull_high_vol": {"correlation_target": 0.3, "risk_multiplier": 1.2},
  "bull_low_vol": {"correlation_target": 0.2, "risk_multiplier": 0.8},
  "bear_high_vol": {"correlation_target": 0.25, "risk_multiplier": 1.5},
  "bear_low_vol": {"correlation_target": 0.15, "risk_multiplier": 1.0},
  "sideways": {"correlation_target": 0.05, "risk_multiplier": 0.7}
}
```

### 7. features（v435.5参考）
```json
"features": {
  "technical_indicators": ["rsi_14", "macd", "stoch_k", "williams_r"],
  "price_features": ["sma_5", "sma_20", "ema_5", "vwap"],
  "volatility_features": ["volatility_5", "atr_14", "bollinger_volatility"],
  "momentum_features": ["roc_5", "momentum_10", "williams_r_5"]
}
```

### 8. validation & logging（v435.6参考）
```json
"validation": {
  "enabled": true,
  "validation_freq": 1000,
  "n_eval_episodes": 5,
  "deterministic_eval": true,
  "validation_threshold": 0.8
},
"logging": {
  "tensorboard_log": "./tensorboard/sac_v444.1",
  "verbose": 1,
  "log_interval": 100
},
"checkpoint": {
  "save_freq": 1000,
  "save_path": "./checkpoints",
  "save_replay_buffer": false
}
```

## 実装優先順位
1. **reward_settings**: HOLDペナルティ最小化、profit_multipliers均等化
2. **ensemble_system**: 多様性確保によるバイアス低減
3. **market_regime_detection**: 相場状況適応
4. **feature_engineering**: 適応的特徴量選択
5. **multi_timeframe_integration**: 複数時間軸統合

## 期待される改善点
- HOLDアクションの出現率向上
- BUY/SELL/HOLDのバランス改善
- 異常収益の解消
- 安定したバックテスト結果

## 次のステップ
1. 上記構造で`configs/sac_v444.1_config.json`を作成
2. v444.1モデルを再トレーニング
3. バックテストでHOLDアクションの出現を確認
4. 必要に応じてrewardパラメータを調整</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\docs\sac_v444.1_config_design.md
