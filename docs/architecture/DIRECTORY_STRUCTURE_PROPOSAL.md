# プロジェクト構造改善提案 - 保守性の向上

**日付**: 2025年10月11日
**目的**: 保守性、拡張性、発見性を高めるための深いディレクトリ構造の設計

---

## 🎯 現在の問題点

### 1. ルートディレクトリの混雑
```
zaif-trade-bot/
├── analyze_*.py (10+個の分析スクリプト)
├── *.md (20+個のドキュメント)
├── *.json (設定、結果ファイル)
├── *.py (訓練スクリプト)
└── ... 合計100+ファイル
```

**問題**:
- ファイル検索が困難
- 関連ファイルの特定が難しい
- 新規開発者の学習コストが高い

### 2. 浅い階層構造
```
ztb/
├── training/ (20+ファイル)
├── analysis/ (15+ファイル)
├── environment/ (30+ファイル)
└── ... フラットな構造
```

**問題**:
- モジュールの責任が不明確
- 関連コードの発見が困難
- import文が長く複雑

### 3. 混在する関心事
- 訓練コード、分析コード、設定ファイルが混在
- テストとドキュメントの対応が不明確
- 実験結果の整理が不十分

---

## 🏗️ 提案する新しい構造

### 階層設計の原則

1. **機能による分離** (Separation by Feature)
2. **階層的な整理** (Hierarchical Organization)
3. **明確な責任** (Clear Responsibilities)
4. **予測可能性** (Predictability)
5. **拡張性** (Extensibility)

---

## 📁 詳細なディレクトリ構造

### 1. コア機能 (`ztb/core/`)

```
ztb/core/
├── algorithms/              # アルゴリズム実装
│   ├── base/                # 基底クラス
│   │   ├── __init__.py
│   │   ├── algorithm.py     # 基底アルゴリズム
│   │   └── factory.py       # ファクトリーパターン
│   ├── ppo/                 # PPO実装
│   │   ├── __init__.py
│   │   ├── algorithm.py     # PPOアルゴリズム
│   │   ├── policy.py        # ポリシー
│   │   └── value.py         # 価値関数
│   └── sac/                 # SAC実装
│       ├── __init__.py
│       ├── algorithm.py     # SACアルゴリズム
│       ├── policy.py        # ポリシー
│       ├── critic.py        # Critic
│       └── entropy.py       # エントロピー調整
│
├── environment/             # 環境
│   ├── base/                # 基底環境
│   │   ├── __init__.py
│   │   ├── trading_env.py   # 基底取引環境
│   │   └── gym_wrapper.py   # Gymラッパー
│   ├── components/          # 環境コンポーネント
│   │   ├── observation/     # 観測値構築
│   │   │   ├── __init__.py
│   │   │   ├── builder.py   # ObservationBuilder
│   │   │   ├── scaler.py    # スケーラー
│   │   │   └── features.py  # 特徴量エンジニアリング
│   │   ├── reward/          # 報酬計算
│   │   │   ├── __init__.py
│   │   │   ├── calculator.py # RewardCalculator
│   │   │   ├── simple.py     # シンプル報酬
│   │   │   └── risk_adjusted.py # リスク調整報酬
│   │   ├── action/          # 行動処理
│   │   │   ├── __init__.py
│   │   │   ├── processor.py  # 行動処理
│   │   │   └── masking.py    # アクションマスキング
│   │   └── state/           # 状態管理
│   │       ├── __init__.py
│   │       ├── manager.py    # 状態管理
│   │       └── portfolio.py  # ポートフォリオ状態
│   └── wrappers/            # 環境ラッパー
│       ├── __init__.py
│       ├── normalization.py  # 正規化
│       └── logging.py        # ロギング
│
└── data/                    # データ処理
    ├── loaders/             # データローダー
    │   ├── __init__.py
    │   ├── csv_loader.py    # CSVローダー
    │   └── api_loader.py    # APIローダー
    ├── preprocessors/       # 前処理
    │   ├── __init__.py
    │   ├── cleaner.py       # データクリーニング
    │   └── feature.py       # 特徴量生成
    └── validators/          # バリデーション
        ├── __init__.py
        └── data_validator.py
```

**利点**:
- アルゴリズムと環境が明確に分離
- コンポーネントの責任が明確
- 新しいアルゴリズムの追加が容易

---

### 2. 訓練 (`ztb/training/`)

```
ztb/training/
├── trainers/                # トレーナー実装
│   ├── base/                # 基底トレーナー
│   │   ├── __init__.py
│   │   ├── trainer.py       # 基底Trainer
│   │   └── config.py        # 設定管理
│   ├── sac/                 # SAC専用トレーナー
│   │   ├── __init__.py
│   │   ├── trainer.py       # SACTrainer
│   │   ├── buffer.py        # リプレイバッファ
│   │   └── updater.py       # パラメータ更新
│   └── ppo/                 # PPO専用トレーナー
│       ├── __init__.py
│       ├── trainer.py       # PPOTrainer
│       └── rollout.py       # ロールアウト
│
├── callbacks/               # コールバック
│   ├── __init__.py
│   ├── checkpoint.py        # チェックポイント
│   ├── early_stopping.py    # 早期停止
│   ├── tensorboard.py       # TensorBoard
│   └── metric_logger.py     # メトリクスロギング
│
├── loggers/                 # ロギング
│   ├── __init__.py
│   ├── console.py           # コンソール出力
│   ├── file.py              # ファイル出力
│   └── tensorboard.py       # TensorBoard
│
├── schedulers/              # スケジューラー
│   ├── __init__.py
│   ├── lr_scheduler.py      # 学習率スケジューラー
│   └── curriculum.py        # カリキュラム学習
│
└── scripts/                 # 訓練スクリプト
    ├── sac/                 # SAC訓練
    │   ├── __init__.py
    │   ├── train_v395i.py   # v395i訓練
    │   ├── train_short.py   # 短期訓練
    │   └── train_long.py    # 長期訓練
    └── ppo/                 # PPO訓練
        ├── __init__.py
        └── train.py         # PPO訓練
```

**利点**:
- アルゴリズムごとの訓練ロジックを分離
- コールバック、ロガーの再利用性向上
- 訓練スクリプトの整理

---

### 3. 最適化 (`ztb/optimization/`)

```
ztb/optimization/
├── methods/                 # 最適化手法
│   ├── grid_search/         # Grid Search
│   │   ├── __init__.py
│   │   ├── optimizer.py     # Grid Search実装
│   │   └── config.py        # 設定
│   ├── random_search/       # Random Search
│   │   ├── __init__.py
│   │   ├── optimizer.py     # Random Search実装
│   │   └── sampler.py       # サンプラー
│   ├── bayesian/            # Bayesian Optimization
│   │   ├── __init__.py
│   │   ├── optimizer.py     # BO実装
│   │   ├── acquisition.py   # 獲得関数
│   │   └── gp.py            # ガウス過程
│   ├── binary_search/       # Binary Search
│   │   ├── __init__.py
│   │   └── optimizer.py     # Binary Search実装
│   └── evolutionary/        # 進化的アルゴリズム
│       ├── __init__.py
│       ├── genetic.py       # 遺伝的アルゴリズム
│       └── cma_es.py        # CMA-ES
│
├── objectives/              # 目的関数
│   ├── sac/                 # SAC用
│   │   ├── __init__.py
│   │   ├── critic_loss.py   # Critic Loss最小化
│   │   └── reward.py        # 報酬最大化
│   └── ppo/                 # PPO用
│       ├── __init__.py
│       └── return.py        # リターン最大化
│
├── strategies/              # 最適化戦略
│   ├── __init__.py
│   ├── staged.py            # 段階的最適化
│   ├── multiobjective.py    # 多目的最適化
│   └── ensemble.py          # アンサンブル
│
├── configs/                 # 最適化設定
│   ├── __init__.py
│   ├── sac_presets.py       # SACプリセット
│   └── ppo_presets.py       # PPOプリセット
│
└── results/                 # 結果保存
    ├── __init__.py
    ├── storage.py           # 結果保存
    └── loader.py            # 結果読込
```

**利点**:
- 各最適化手法が独立したモジュール
- 目的関数とアルゴリズムの分離
- 戦略パターンの実装が容易

---

### 4. 分析 (`ztb/analysis/`)

```
ztb/analysis/
├── diagnostics/             # 診断ツール
│   ├── environment/         # 環境診断
│   │   ├── __init__.py
│   │   ├── observation.py   # 観測値診断
│   │   ├── reward.py        # 報酬診断
│   │   └── action.py        # 行動診断
│   ├── training/            # 訓練診断
│   │   ├── __init__.py
│   │   ├── convergence.py   # 収束診断
│   │   ├── stability.py     # 安定性診断
│   │   └── gradient.py      # 勾配診断
│   └── model/               # モデル診断
│       ├── __init__.py
│       ├── prediction.py    # 予測診断
│       └── behavior.py      # 行動診断
│
├── metrics/                 # メトリクス計算
│   ├── training/            # 訓練メトリクス
│   │   ├── __init__.py
│   │   ├── loss.py          # Loss計算
│   │   └── learning.py      # 学習進捗
│   └── trading/             # 取引メトリクス
│       ├── __init__.py
│       ├── sharpe.py        # Sharpe Ratio
│       ├── drawdown.py      # Drawdown
│       └── win_rate.py      # 勝率
│
├── visualization/           # 可視化
│   ├── tensorboard/         # TensorBoard関連
│   │   ├── __init__.py
│   │   ├── parser.py        # イベントパーサー
│   │   └── analyzer.py      # 分析
│   ├── plots/               # プロット生成
│   │   ├── __init__.py
│   │   ├── training.py      # 訓練プロット
│   │   ├── trading.py       # 取引プロット
│   │   └── comparison.py    # 比較プロット
│   └── reports/             # レポート生成
│       ├── __init__.py
│       ├── generator.py     # レポート生成器
│       └── templates/       # テンプレート
│
└── statistical/             # 統計分析
    ├── hypothesis_tests/    # 仮説検定
    │   ├── __init__.py
    │   ├── t_test.py        # t検定
    │   ├── mann_whitney.py  # Mann-Whitney U検定
    │   └── anova.py         # ANOVA
    └── comparisons/         # 比較分析
        ├── __init__.py
        ├── pairwise.py      # ペアワイズ比較
        └── effect_size.py   # 効果量
```

**利点**:
- 診断、メトリクス、可視化が明確に分離
- 再利用可能なコンポーネント
- 統計分析の体系的な整理

---

### 5. 評価 (`ztb/evaluation/`)

```
ztb/evaluation/
├── backtest/                # バックテスト
│   ├── engine/              # バックテストエンジン
│   │   ├── __init__.py
│   │   ├── simulator.py     # シミュレーター
│   │   └── executor.py      # 実行エンジン
│   ├── strategies/          # 戦略評価
│   │   ├── __init__.py
│   │   ├── evaluator.py     # 評価器
│   │   └── comparator.py    # 比較器
│   └── reports/             # レポート
│       ├── __init__.py
│       └── backtest_report.py
│
├── live/                    # ライブ評価
│   ├── __init__.py
│   ├── monitor.py           # モニター
│   └── validator.py         # バリデーター
│
└── metrics/                 # 評価メトリクス
    ├── __init__.py
    ├── performance.py       # パフォーマンス
    ├── risk.py              # リスク
    └── statistical.py       # 統計的指標
```

**利点**:
- バックテストとライブ評価の分離
- 評価メトリクスの再利用
- レポート生成の標準化

---

## 📊 設定とデータの整理

### 設定ファイル (`configs/`)

```
configs/
├── algorithms/              # アルゴリズム設定
│   ├── sac/                 # SAC設定
│   │   ├── v395i.json       # v395i
│   │   ├── default.json     # デフォルト
│   │   └── optimized.json   # 最適化済み
│   └── ppo/                 # PPO設定
│       └── default.json
│
├── environments/            # 環境設定
│   ├── trading_env.json     # 取引環境
│   └── reward.json          # 報酬設定
│
├── optimization/            # 最適化設定
│   ├── grid_search.json     # Grid Search
│   ├── random_search.json   # Random Search
│   └── bayesian.json        # Bayesian
│
└── training/                # 訓練設定
    ├── short.json           # 短期訓練
    ├── medium.json          # 中期訓練
    └── long.json            # 長期訓練
```

### データディレクトリ (`data/`)

```
data/
├── raw/                     # 生データ
│   ├── btc_jpy/             # BTC/JPY
│   └── historical/          # 過去データ
│
├── processed/               # 処理済みデータ
│   ├── normalized/          # 正規化済み
│   └── features/            # 特徴量
│
├── datasets/                # データセット
│   ├── train/               # 訓練用
│   ├── val/                 # 検証用
│   └── test/                # テスト用
│
└── cache/                   # キャッシュ
    └── scalers/             # スケーラー
```

### モデル (`models/`)

```
models/
├── sac/                     # SACモデル
│   ├── v395i/               # バージョン管理
│   │   ├── 5k/              # 5kステップ
│   │   ├── 10k/             # 10kステップ
│   │   └── 50k/             # 50kステップ
│   └── production/          # 本番モデル
│       └── latest.zip
└── ppo/                     # PPOモデル
    └── production/
```

---

## 🎯 移行計画

### Phase 1: ディレクトリ構造の作成（1-2時間）
1. 新しいディレクトリ構造を作成
2. `__init__.py`ファイルを配置
3. READMEファイルを各ディレクトリに作成

### Phase 2: コアモジュールの移行（2-3時間）
1. `ztb/core/algorithms/` - アルゴリズム移動
2. `ztb/core/environment/` - 環境コンポーネント再構成
3. `ztb/core/data/` - データ処理モジュール整理

### Phase 3: 訓練・最適化の移行（2-3時間）
1. `ztb/training/` - 訓練コード再編成
2. `ztb/optimization/` - 最適化モジュール深化

### Phase 4: 分析・評価の移行（2-3時間）
1. `ztb/analysis/` - 分析ツール再編成
2. `ztb/evaluation/` - 評価モジュール整理

### Phase 5: 設定・データ・ドキュメントの移行（1-2時間）
1. `configs/` - 設定ファイル分類
2. `data/` - データディレクトリ整理
3. `docs/` - ドキュメント分類

### Phase 6: テストとCI/CDの更新（1-2時間)
1. import文の更新
2. テストパスの修正
3. CI/CDパイプラインの更新

---

## ✅ 期待される効果

### 1. 保守性の向上
- **責任の明確化**: 各モジュールの役割が明確
- **変更の局所化**: 影響範囲が限定的
- **バグ修正の容易化**: 問題箇所の特定が容易

### 2. 拡張性の向上
- **新機能の追加**: 適切な場所に追加できる
- **プラグイン化**: モジュールの差し替えが容易
- **スケーラビリティ**: プロジェクトの成長に対応

### 3. 発見性の向上
- **ファイルの場所**: 予測可能な構造
- **関連コードの発見**: 階層的な整理
- **ドキュメント**: 構造に沿った整理

### 4. チーム開発の効率化
- **新規参画者の学習**: 構造が理解しやすい
- **並行開発**: 責任範囲が明確
- **コードレビュー**: 変更範囲が明確

---

## 🚀 次のアクション

1. **構造承認**: 提案構造のレビューと承認
2. **優先順位決定**: どのモジュールから移行するか
3. **移行計画の詳細化**: タスク分解とスケジュール
4. **段階的実施**: リスクを最小化しながら移行

**推奨**: まず`ztb/optimization/`を深化させ、効果を確認してから全体移行
