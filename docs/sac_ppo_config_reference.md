# SAC/PPO設定ファイル解説ドキュメント

## 概要

このドキュメントは、zaif-trade-botプロジェクトの設定JSONファイルの構造とパラメータを体系的に解説したものです。全ての設定ファイルを調査し、カテゴリ別に分類して表形式でまとめています。

## 設定ファイルカテゴリ

### 1. SACテスト設定
短期間のテストや検証を目的とした設定ファイル群。

| ファイル名 | 目的 | ステップ数 | 主な特徴 |
|-----------|------|-----------|----------|
| `sac_test_100steps.json` | 基本機能テスト | 100 | 最小限の設定で動作確認 |
| `sac_test_1ksteps.json` | 短時間学習テスト | 1000 | 基本的な学習確認 |
| `sac_test_config.yaml` | YAML形式テスト | 1000 | 設定形式の検証 |

### 2. SAC基本設定 (v395シリーズ)
SACアルゴリズムの基本的なパラメータチューニングを目的とした設定群。

| ファイル名 | 目的 | ステップ数 | 主な特徴 | 分析結果 |
|-----------|------|-----------|----------|----------|
| `sac_v395a_auto_entropy.json` | エントロピー自動調整 | 5000 | `ent_coef: "auto"` | 損失最小だが不安定 |
| `sac_v395b_stable.json` | 安定性重視 | 5000 | `ent_coef: 0.01` | 安定だが探索不足 |
| `sac_v395c_conservative.json` | 保守的設定 | 5000 | `ent_coef: 0.005` | 非常に保守的 |
| `sac_v395d_optimal.json` | 最適設定 | 5000 | `ent_coef: "auto"` + 安定パラメータ | v395a/b/cの最適バランス |
| `sac_v395e_positive_entropy.json` | 正のエントロピー | 5000 | `ent_coef: 0.1` | 探索重視 |
| `sac_v395f_simple_reward.json` | シンプル報酬 | 5000 | 簡素化された報酬関数 | 学習安定性向上 |
| `sac_v395g_micro_reward.json` | 微小報酬 | 5000 | 微調整された報酬スケール | 微調整最適化 |
| `sac_v395h_normalized.json` | 正規化報酬 | 5000 | 正規化された報酬関数 | 安定性向上 |
| `sac_v395i_complete_fix.json` | 完全修正版 | 5000 | 全ての問題修正 | 最終最適化版 |

### 3. SAC高度設定 (v4xxシリーズ)
高度な機能（市場レジーム適応、ポジション管理など）を追加した設定群。

| ファイル名 | 目的 | ステップ数 | 主な特徴 | 高度機能 |
|-----------|------|-----------|----------|----------|
| `sac_v396_optimized.json` | 最適化版 | 50000 | 長期学習最適化 | なし |
| `sac_v411_trading_focused_config.json` | 取引重視 | 5000 | 取引頻度最適化 | なし |
| `sac_v412_profit_focused_config.json` | 利益重視 | 5000 | 利益最大化 | なし |
| `sac_v413_ultra_profit_config.json` | 超利益重視 | 5000 | 積極的利益追求 | なし |
| `sac_v414_balanced_trading_config.json` | バランス取引 | 5000 | リスク/リターンバランス | なし |
| `sac_v415_balanced_trading_config.json` | バランス取引v2 | 5000 | 改良版バランス | なし |
| `sac_v416_balanced_trading_config.json` | バランス取引v3 | 5000 | さらなる改良 | なし |
| `sac_v417_comprehensive_trading_config.json` | 包括的取引 | 5000 | 多角的アプローチ | なし |
| `sac_v418_balanced_adjusted_config.json` | 調整バランス | 5000 | 微調整版 | なし |
| `sac_v419_equalized_actions_config.json` | アクション均等化 | 5000 | BUY/SELLバランス | なし |
| `sac_v420_forced_balance_config.json` | 強制バランス | 5000 | 強制的なバランス | なし |
| `sac_v421_transfer_learning_example.json` | 転移学習 | 5000 | 学習転移 | なし |
| `sac_v422_balanced_trading_config.json` | バランス取引v4 | 5000 | 最新バランス | なし |
| `sac_v427_default_config.json` | デフォルト設定 | 5000 | 標準設定 | なし |
| `sac_v434_2_environment_config.json` | 環境設定 | 5000 | 環境最適化 | なし |
| `sac_v435_unified_config.json` | 統合設定 | 5000 | 統合アプローチ | マルチタイムフレーム |
| `sac_v436_1_full_guidance_config.json` | 完全ガイダンス | 5000 | 完全ガイド | アクションシグナルガイド |
| `sac_v436_signal_guided_config.json` | シグナルガイド | 5000 | シグナルベース | アクションシグナルガイド |
| `sac_v437_1_config.json` | v437設定 | 5000 | 改良版 | なし |
| `sac_v441_stability_focused_config.json` | 安定性重視 | 5000 | 安定性最適化 | なし |
| `sac_v442_16_balance_mechanism_fixed_config.json` | バランス修正 | 5000 | バランス機構修正 | なし |
| `sac_v443_2_market_regime_adaptation_config.json` | 市場レジーム適応 | 5000 | レジーム適応 | 12レジーム分類 |
| `sac_v444_advanced_regime_adaptation_config.json` | 先進レジーム適応 | 1000 | 高度レジーム適応 | 包括的レジーム分類 |

### 4. SAC市場レジーム適応設定
市場レジーム分類と適応機能を特化した設定群。

| ファイル名 | レジーム数 | 適応機能 | ステップ数 | 主な特徴 |
|-----------|-----------|----------|-----------|----------|
| `sac_v443_2_market_regime_adaptation_config.json` | 12 | ハイパーパラメータ調整 | 5000 | 包括的適応 |
| `sac_v443_2_regime_specialization_config.json` | 12 | レジーム専門化 | 5000 | 専門化適応 |
| `sac_v444_advanced_regime_adaptation_config.json` | 12 | 高度適応 | 1000 | 拡張適応 |

### 5. PPO設定
PPO (Proximal Policy Optimization) アルゴリズムを使用した設定群。

| ファイル名 | 目的 | ステップ数 | 主な特徴 | 制約機能 |
|-----------|------|-----------|----------|----------|
| `ppo_profitable_v392_bugfix.json` | 利益重視 | 100000 | 利益最適化 | Lagrange制約 |
| `ppo_profitable_v393_hyperfix.json` | ハイパー修正 | 100000 | パラメータ修正 | Lagrange制約 |
| `ppo_v394a_hold_penalty.json` | HOLD罰則 | 100000 | HOLD抑制 | なし |
| `ppo_v394b_trade_reward.json` | 取引報酬 | 100000 | 取引促進 | なし |
| `ppo_v394c_balanced.json` | バランス | 100000 | 総合バランス | なし |
| `ppo_v394d_aggressive.json` | 積極的 | 100000 | 積極取引 | なし |
| `ppo_v394e_high_entropy.json` | 高エントロピー | 100000 | 探索重視 | なし |
| `ppo_v394f_ultra_entropy.json` | 超高エントロピー | 100000 | 最大探索 | なし |
| `ppo_test_config.yaml` | テスト設定 | 1000 | 基本テスト | なし |

## パラメータ詳細解説

### 共通パラメータ

| パラメータ | 説明 | SACデフォルト | PPOデフォルト |
|-----------|------|--------------|--------------|
| `model_name` | モデル識別子 | 必須 | 必須 |
| `algorithm` | アルゴリズム種別 | "sac" | "ppo" |
| `total_timesteps` | 総学習ステップ数 | 5000 | 100000 |
| `data_source` | データソース種別 | "csv" | "csv" |
| `data_path` | データファイルパス | 必須 | 必須 |

### SACハイパーパラメータ

| パラメータ | 説明 | 推奨範囲 | 影響 |
|-----------|------|----------|------|
| `learning_rate` | 学習率 | 0.0001-0.001 | 学習速度と安定性のトレードオフ |
| `buffer_size` | リプレイバッファサイズ | 1000-100000 | 経験の多様性 |
| `learning_starts` | 学習開始ステップ | 50-1000 | 初期探索期間 |
| `batch_size` | バッチサイズ | 32-256 | 学習安定性 |
| `tau` | ターゲット更新率 | 0.005-0.01 | 学習安定性 |
| `gamma` | 割引率 | 0.95-0.99 | 長期 vs 短期重視 |
| `ent_coef` | エントロピー係数 | 0.005-1.0 or "auto" | 探索 vs 活用 |
| `target_entropy` | 目標エントロピー | -2.0 - -0.1 | エントロピー制御 |

### PPOハイパーパラメータ

| パラメータ | 説明 | 推奨範囲 | 影響 |
|-----------|------|----------|------|
| `learning_rate` | 学習率 | 0.0001-0.01 | 学習速度 |
| `n_steps` | ステップ数 per 更新 | 512-2048 | 学習頻度 |
| `batch_size` | バッチサイズ | 64-256 | 安定性 |
| `n_epochs` | エポック数 | 4-16 | 学習反復 |
| `gamma` | 割引率 | 0.95-0.99 | 長期重視度 |
| `gae_lambda` | GAEラムダ | 0.8-0.95 | 価値推定 |
| `clip_range` | クリップ範囲 | 0.1-0.3 | 学習安定性 |
| `ent_coef` | エントロピー係数 | 0.001-0.1 | 探索制御 |
| `vf_coef` | 価値関数係数 | 0.3-0.7 | 価値学習 |

### 環境パラメータ

| パラメータ | 説明 | デフォルト値 | 影響 |
|-----------|------|--------------|------|
| `initial_balance` | 初期残高 | 200000.0 | ポートフォリオ規模 |
| `transaction_cost` | 取引コスト | 0.0005 | 取引頻度に影響 |
| `max_position_size` | 最大ポジションサイズ | 0.01 | リスク制御 |
| `enable_action_masking` | アクション masking | false | 無効アクション制御 |
| `use_continuous_actions` | 連続アクション使用 | true | アクション空間 |
| `random_start` | ランダム開始 | true | データ多様性 |

### 報酬設定パラメータ

| パラメータ | 説明 | デフォルト値 | 影響 |
|-----------|------|--------------|------|
| `hold_penalty_weight` | HOLD罰則重み | 0.1 | HOLD抑制 |
| `consecutive_hold_penalty` | 連続HOLD罰則 | 0.05 | 長期HOLD抑制 |
| `trading_frequency_bonus` | 取引頻度ボーナス | 0.3 | 取引促進 |
| `profit_reward_multiplier` | 利益報酬倍率 | 10.0 | 利益重視度 |
| `action_diversity_bonus` | アクション多様性ボーナス | 0.1 | 探索促進 |
| `successful_trade_bonus` | 成功取引ボーナス | 5.0 | 取引成功報酬 |

### 市場レジーム適応パラメータ

| パラメータ | 説明 | 設定例 | 影響 |
|-----------|------|--------|------|
| `regime_scheme` | レジーム分類方式 | "comprehensive" | 分類詳細度 |
| `use_multi_timeframe` | マルチタイムフレーム使用 | true | 時間軸多様性 |
| `confidence_threshold` | 信頼性閾値 | 0.6 | 適応トリガー |
| `adaptation_frequency` | 適応頻度 | 100 | 適応速度 |
| `performance_tracking_window` | パフォーマンス追跡窓 | 1000 | 評価期間 |

## 使用方法

### 設定ファイルの選択基準

1. **テスト目的**: `sac_test_*.json` を使用
2. **基本学習**: `sac_v395d_optimal.json` から開始
3. **高度機能**: 目的に応じて `sac_v4xx` シリーズを選択
4. **レジーム適応**: `sac_v443_2_*` または `sac_v444_*` を使用
5. **PPOアルゴリズム**: `ppo_*` シリーズを使用

### カスタマイズの推奨

1. 小規模テストで基本動作を確認
2. パラメータを段階的に調整
3. 分析結果に基づいて設定を改良
4. 安定した設定をベースに高度機能を追加

## 注意事項

- 全ての設定は実験的であり、実際の取引では十分なバックテストを実施してください
- パラメータの変更は学習結果に大きな影響を与えます
- 市場条件によって最適な設定が異なる可能性があります
- 定期的に新しい設定ファイルをテストし、最適化を継続してください

## 更新履歴

- 2025-11-03: 初回作成、全設定ファイルの体系的整理
- 2025-11-03: v427, v437, v440, v443の詳細パラメータ解説を追加

## 詳細バージョン分析 (v427, v437, v440, v443)

### v427: 市場適応アンサンブル設定

`sac_v427_default_config.json` は市場適応機能を強化した高度な設定です。

#### 主な特徴
- **市場適応アンサンブル**: 複数の市場状態に対応した適応機構
- **キュリキュラム学習**: `curriculum_stage: "strong_penalty_trading"` で段階的学習
- **アテンション層**: 特徴量の重要度を動的に調整
- **相関削減**: `enable_correlation_reduction: true` で特徴量の冗長性を低減

#### 主要パラメータ
```json
{
  "market_adaptive_ensemble": {
    "enabled": true,
    "regime_detection_window": 50,
    "adaptation_frequency": 20,
    "ensemble_size": 3
  },
  "curriculum_learning": {
    "enabled": true,
    "curriculum_stage": "strong_penalty_trading",
    "stage_progression": ["exploration", "balanced", "strong_penalty_trading"]
  },
  "attention_layer": {
    "enabled": true,
    "attention_heads": 8,
    "attention_dim": 64
  },
  "correlation_reduction": {
    "enabled": true,
    "correlation_threshold": 0.8,
    "reduction_method": "pca"
  }
}
```

#### 動作原理
1. **レジーム検出**: 50ステップの窓で市場状態を分類
2. **適応調整**: 20ステップごとにパラメータを適応
3. **アンサンブル学習**: 3つの異なるモデルを統合
4. **特徴最適化**: アテンションで重要な特徴を強調

### v437: 拡張特徴エンジニアリング設定

`sac_v437_enhanced_config.json` は150次元の拡張特徴を使用した設定です。

#### 主な特徴
- **拡張特徴セット**: 150次元の包括的特徴量
- **取引頻度制御**: 過度な取引を抑制
- **高度特徴エンジニアリング**: モメンタム、ボラティリティ、出来高の統合
- **適応特徴選択**: 市場状態に応じた特徴量の動的選択

#### 主要パラメータ
```json
{
  "feature_engineering": {
    "target_dimensions": 150,
    "feature_categories": ["price", "volume", "momentum", "volatility", "trend"],
    "advanced_indicators": ["rsi", "macd", "bollinger", "stochastic", "cci"]
  },
  "trading_frequency_control": {
    "enabled": true,
    "max_trades_per_hour": 10,
    "cooldown_period": 6,
    "frequency_penalty": 0.1
  },
  "adaptive_feature_selection": {
    "enabled": true,
    "selection_window": 100,
    "importance_threshold": 0.05,
    "max_features": 100
  }
}
```

#### 動作原理
1. **特徴生成**: 価格・出来高・テクニカル指標を統合
2. **次元削減**: 150次元に最適化
3. **適応選択**: 市場状態に応じて特徴を動的選択
4. **頻度制御**: 取引頻度を適切に制限

### v440: PnLベース最適化設定

`v440` シリーズは純粋な利益ベースの最適化に特化しています。

#### sac_v440_pnl_config.json の特徴
- **シンプル報酬関数**: PnLのみを基準とした報酬設計
- **詳細取引パラメータ**: 取引成功/失敗の細かい制御
- **リスク管理**: ドローダウン制限とポジションサイズ制御

#### 主要パラメータ
```json
{
  "use_simple_reward": true,
  "reward_function": {
    "pnl_weight": 1.0,
    "transaction_cost_weight": 0.1,
    "holding_penalty": 0.01,
    "success_bonus": 10.0,
    "failure_penalty": -5.0
  },
  "trading_parameters": {
    "min_profit_threshold": 0.001,
    "max_loss_threshold": -0.005,
    "profit_bonus_multiplier": 2.0,
    "loss_penalty_multiplier": 1.5
  }
}
```

#### sac_v440_unified_config.json の特徴
- **相関削減**: 特徴量間の冗長性を低減
- **特徴適応**: 市場状態に応じた特徴量調整
- **利益ボーナス閾値**: 段階的な利益報酬設計

#### 主要パラメータ
```json
{
  "enable_correlation_reduction": true,
  "correlation_threshold": 0.85,
  "feature_adaptation": {
    "enabled": true,
    "adaptation_window": 50,
    "min_feature_weight": 0.1
  },
  "profit_bonus_thresholds": {
    "small_profit": 0.005,
    "medium_profit": 0.01,
    "large_profit": 0.02,
    "small_bonus": 5.0,
    "medium_bonus": 10.0,
    "large_bonus": 20.0
  }
}
```

### v443: 市場レジーム適応設定

`v443` シリーズは高度な市場レジーム適応機能を備えています。

#### sac_v443_1_baseline_verification_config.json の特徴
- **ベースライン検証**: 安定した学習の基盤設定
- **行動最適化**: アクションバランスと一貫性の確保
- **カリキュラム学習**: `stability_optimized` ステージ

#### 主要パラメータ
```json
{
  "behavior_optimization": {
    "action_balance_target": 0.8,
    "entropy_regularization": 0.01,
    "action_smoothing": 0.1,
    "consistency_penalty": 0.05,
    "balance_penalty": 1.0
  },
  "curriculum_stage": "stability_optimized",
  "continuous_to_discrete_threshold": 0.08
}
```

#### sac_v443_2_market_regime_adaptation_config.json の特徴
- **動的報酬シェーピング**: 市場レジームに応じた報酬調整
- **レジーム別行動適応**: 強気/弱気/横ばい/変動市場での異なる戦略
- **ボラティリティ調整報酬**: 市場変動に応じた報酬係数

#### 主要パラメータ
```json
{
  "market_regime": {
    "enabled": true,
    "regime_detection_window": 20,
    "adaptation_frequency": 10,
    "regime_adaptive_behavior": {
      "bull_market": {"action_balance_target": 0.75, "entropy_regularization": 0.005},
      "bear_market": {"action_balance_target": 0.85, "entropy_regularization": 0.015},
      "sideways_market": {"action_balance_target": 0.8, "entropy_regularization": 0.01},
      "volatile_market": {"action_balance_target": 0.7, "entropy_regularization": 0.02}
    }
  },
  "dynamic_reward_shaping": {
    "enabled": true,
    "market_regime_awareness": true,
    "regime_coefficients": {
      "bull_market_bonus_coeff": 1.2,
      "bear_market_penalty_coeff": 0.8,
      "volatile_market_bonus_coeff": 1.1
    }
  }
}
```

#### sac_v443_2_regime_specialization_config.json の特徴
- **リスク管理**: 最大ドローダウン制限とストップロス
- **ポジションサイジング**: ボラティリティ/トレンド調整
- **高度特徴セット**: v443拡張特徴を使用

#### 主要パラメータ
```json
{
  "risk_management": {
    "enabled": true,
    "max_drawdown_limit": 0.15,
    "position_sizing": {
      "volatility_adjusted": true,
      "regime_adjusted": true,
      "base_position_size": 0.5
    },
    "stop_loss": {
      "enabled": true,
      "trailing_stop": true,
      "stop_loss_percentage": 0.05
    }
  },
  "features": {
    "feature_set": "v443_enhanced",
    "advanced_features": {
      "momentum_indicators": ["rsi", "macd", "stochastic", "williams_r"],
      "volatility_indicators": ["bollinger_bands", "atr", "cci"]
    }
  }
}
```

## バージョン比較表

| バージョン | 主な特徴 | 強み | 推奨用途 |
|-----------|----------|------|----------|
| v427 | 市場適応アンサンブル | 多様な市場状態対応 | 安定性重視 |
| v437 | 拡張特徴エンジニアリング | 豊富な特徴量 | 包括的分析 |
| v440 | PnLベース最適化 | 純粋利益最大化 | 利益重視 |
| v443 | 高度レジーム適応 | 動的市場適応 | 適応性重視 |

## 設定選択ガイドライン

### 安定性重視の場合
1. `sac_v443_1_baseline_verification_config.json` で基盤を確立
2. `sac_v427_default_config.json` で市場適応を追加
3. `sac_v443_2_regime_specialization_config.json` でリスク管理を強化

### 利益最大化の場合
1. `sac_v440_pnl_config.json` でPnL最適化を開始
2. `sac_v440_unified_config.json` で特徴適応を追加
3. `sac_v443_2_market_regime_adaptation_config.json` で動的適応を導入

### 包括的アプローチの場合
1. `sac_v437_enhanced_config.json` で特徴量を最大化
2. `sac_v443_2_regime_specialization_config.json` で専門化
3. `sac_v444_advanced_regime_adaptation_config.json` で高度適応</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\docs\sac_ppo_config_reference.md
