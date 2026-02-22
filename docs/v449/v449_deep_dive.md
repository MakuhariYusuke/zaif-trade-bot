# v449 技術的深堀りと改善提案

## 1. Unified Trainer との統合性分析

### 現状の確認
- `unified_trainer` は `ztb.trading.environment.utils.config.EnvironmentConfig` を使用して環境設定を行っています。
- `EnvironmentConfig` クラス定義を確認したところ、`continuous_to_discrete_threshold` および `continuous_to_discrete_threshold_neg` フィールドは既に存在します。
- `unified_trainer` は設定辞書（`env_config_dict`）を展開して `EnvironmentConfig` を初期化しているため（`EnvironmentConfig(**env_config_dict)`）、設定ファイル（YAML/JSON）にこれらのキーを追加するだけで、学習時にも閾値を制御可能です。

### 課題と対応
- **課題**: `run_ab_test.py` で発生していたLintエラーは、静的解析ツールが動的に生成されるフィールドや複雑なインポート解決に失敗している可能性が高いです。
- **対応**: `unified_trainer` 側のコード修正は不要ですが、学習設定ファイル（例: `configs/training/sac_v449.yaml`）を作成する際には、明示的に `continuous_to_discrete_threshold` を含めることを推奨します。

## 2. 報酬関数の高度化 (Forced Balance から Smart Incentive へ)

現在の "Forced Balance" は、アクション分布の偏差に対して即時かつ線形（または段階的）なペナルティを与えています。これは「矯正」としては機能しますが、エージェントが「ペナルティ回避」を目的化してしまい、本来の目的である「収益最大化」がおろそかになるリスクがあります。

### 提案A: Regime-Adaptive Target Distribution (レジーム適応型目標分布)
固定の目標分布（例: BUY 33%, SELL 33%, HOLD 33%）を強制するのではなく、市場環境（Regime）に応じて目標分布を動的に変化させます。

- **トレンド相場**: BUY/SELL比率を高める（例: BUY 40%, SELL 40%, HOLD 20%）。
- **レンジ相場**: 回転売買を推奨しつつ、無駄なエントリーを控える（例: BUY 30%, SELL 30%, HOLD 40%）。
- **低ボラティリティ**: HOLDを許容する（例: BUY 10%, SELL 10%, HOLD 80%）。

これにより、「動くべき時に動き、待つべき時に待つ」という自然な行動を学習させます。

### 提案B: Soft Constraint と Curriculum Learning
- **Soft Constraint (ソフト制約)**:
  閾値を超えた瞬間にペナルティを与えるハード制約ではなく、シグモイド関数やTanh関数を用いて、偏差が大きくなるにつれて滑らかにペナルティが増加するようにします。
  $$ Penalty = \alpha \times \tanh(\beta \times (Deviation - Threshold)) $$

- **Curriculum Learning (カリキュラム学習)**:
  学習初期（例: 最初の100万ステップ）はペナルティ係数 $\alpha$ を小さくし、学習が進むにつれて徐々に係数を上げていくことで、初期の探索を妨げないようにします。

### 提案C: Trading Frequency Reward (取引頻度報酬)
HOLDを罰する（Negative Reinforcement）のではなく、取引を推奨する（Positive Reinforcement）アプローチです。
- **ロジック**: 直近Nステップの取引回数がM回以下の場合、BUY/SELLアクションに対して小さなボーナスを与える。
- **利点**: 「HOLDしてはいけない」という否定的な学習ではなく、「取引すると良いことがある」という肯定的な学習を促せます。

## 3. アクション空間の最適化 (Adaptive Thresholding)

### 実装完了: Volatility-Adjusted Threshold (VAT)
市場のボラティリティ（ATR）に基づいてアクション閾値を動的に調整する `ThresholdManager` を実装し、`HeavyTradingEnv` に統合しました。

$$ Threshold_t = BaseThreshold \times (1 + \gamma \times \frac{ATR_t}{Price_t}) $$

- **機能概要**:
    - **適応的閾値**: ボラティリティが高い時は閾値を上げ（ノイズ対策）、低い時は下げる（機会損失防止）。
    - **アクションシグナルガイドとの統合**: ユーザーからの要望により、この適応的閾値ロジックはアクションシグナルガイド（Action Signal Guide）の一部としても機能するように調整されています。これにより、エージェントの自律的な判断と、ルールベースのガイドラインの整合性が向上します。

### 検証結果 (A/Bテスト)
`experiments/v449/run_ab_test_threshold.py` による検証結果：
- **Baseline (固定 0.01)**: 平均閾値 0.0100
- **Adaptive (x1.0)**: 平均閾値 0.0491
- **Adaptive (x2.0)**: 平均閾値 0.0498
※ 高ボラティリティ環境下では、適応的閾値がベースラインよりも高く設定され、より慎重なエントリーを促す挙動が確認されました。

## 4. 市場レジーム検知機能 (Market Regime Detection)

### 新機能: 高度な市場環境認識
v449では、単なるボラティリティ測定を超えた、包括的な市場レジーム検知機能を導入しました。

- **検知ロジック**:
    - **トレンド判定**: 移動平均線やADXを用いたトレンドの有無と方向性の判定。
    - **ボラティリティ判定**: ATRやボリンジャーバンド幅を用いた市場の活発さの判定。
    - **レンジ判定**: 価格の振動パターンからのレンジ相場の識別。
- **活用方法**:
    - **報酬関数の切り替え**: レジームに応じて「トレンドフォロー型」や「ミーンリバージョン型」の報酬戦略を動的に切り替える基盤となります。
    - **適応的閾値への入力**: `ThresholdManager` は、単純なATRだけでなく、このレジーム情報を加味して閾値を決定することが可能です（例：明確なトレンド発生時は閾値を下げて順張りをしやすくする等）。

## 5. ディレクトリ構成の刷新案

現在のプロジェクトルート直下には、一時的なスクリプトや実験コードが散乱しており、可読性と保守性を低下させています。v449以降は以下の構成への移行を提案します。

### 推奨ディレクトリ構造

```
zaif-trade-bot/
├── experiments/           # 実験用スクリプトを集約
│   ├── v449/              # v449関連の実験
│   │   ├── run_ab_test.py # 今回作成したABテスト
│   │   ├── reward_tuning/ # 報酬関数調整のログやスクリプト
│   │   └── ...
│   └── archive/           # 終了した実験
├── configs/               # 設定ファイル
│   ├── training/          # 学習用設定 (YAML)
│   │   ├── sac_v449.yaml  # 今回の知見を反映した設定
│   │   └── ...
│   └── environment/       # 環境設定
├── docs/                  # ドキュメント
│   ├── v449/              # v449関連ドキュメント
│   │   ├── progress_report.md
│   │   └── deep_dive.md
│   └── ...
├── scripts/               # ユーティリティスクリプト
│   ├── maintenance/       # DBメンテ、ログ掃除など
│   └── analysis/          # 分析用ツール
└── ztb/                   # コアライブラリ (変更なし)
```

### 移行のメリット
1.  **実験の再現性**: 実験ごとにフォルダを分けることで、どのコードでどの結果が出たかが明確になる。
2.  **ルートディレクトリの清浄化**: 開発者がプロジェクトの全体像を把握しやすくなる。
3.  **設定の管理**: `configs/` に設定を集約することで、パラメータのバージョン管理が容易になる。

## 6. 今後のロードマップ
1.  **[完了] ディレクトリ整理**: `experiments/v449` や `config/v449` の作成と整理が完了。
2.  **[完了] v449 学習設定の作成**: `config/v449/base/config.yaml` を作成し、適応的閾値やSmart Incentiveの設定を統合。
3.  **[完了] 報酬関数の改修**: Strategy Patternへの移行とSmart Incentiveの実装完了。
4.  **[完了] アクション空間の最適化**: Adaptive Thresholdingの実装と検証完了。
5.  **[進行中] 大規模学習の実施**: 統合された新機能（報酬、閾値、レジーム検知）を用いた本番学習の実施とパフォーマンス検証。
6.  **[新規] レジーム検知の精度向上**: 実データを用いたレジーム検知ロジックのチューニングと、報酬関数へのフィードバックループの強化。
