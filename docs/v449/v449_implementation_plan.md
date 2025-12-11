# v449 実装計画とアーキテクチャ設計

## 1. はじめに
本ドキュメントは、v449（報酬設計の高度化とアクション空間の最適化）を効率的かつ安全に実装するための計画書です。
既存のコードベース（特に `RewardCalculator` や `EnvironmentConfig`）は既に大規模化しており、無計画な修正はリグレッション（退行）バグを招くリスクがあります。
そのため、**「既存機能を壊さずに拡張する」** ことを最優先とした実装方針を提案します。

## 2. ディレクトリ構成と開発環境の整備

ユーザー様による `docs/v449` の作成は、プロジェクト構造化の第一歩として非常に重要です。これを踏まえ、以下の構成への段階的移行を推奨します。

### 推奨ディレクトリ構造
`config/v449` を中心に、v448以前の慣習（`experiments`, `templates` 等）を踏襲しつつ整理します。

```
zaif-trade-bot/
├── config/
│   └── v449/              # [New] v449用設定ファイル集約
│       ├── base/          # 基本設定 (継承元)
│       │   ├── env_base.yaml
│       │   └── train_base.yaml
│       ├── experiments/   # 実験用設定 (A/Bテスト等)
│       │   └── ab_test_reward.yaml
│       └── templates/     # Unified Trainer用テンプレート
│           └── sac_v449_template.yaml
├── experiments/           # [New] 実験コード置き場
│   └── v449/
│       ├── run_ab_test.py # 既存のスクリプトを移動
│       └── ...
├── ztb/
│   └── trading/
│       └── environment/
│           └── components/
│               └── rewards/ # [Refactor] 巨大なRewardCalculatorを分割
│                   ├── __init__.py
│                   ├── base.py
│                   ├── forced_balance.py
│                   └── smart_incentive.py
└── docs/
    └── v449/              # [Moved] ドキュメント
```

### 実装の都合が良い点
- **バージョン管理の明確化**: `config/v449/` 配下に閉じることで、過去バージョン（v448等）への影響を完全に排除できます。
- **実験の分離**: `experiments/` を分けることで、メインのソースコード (`ztb/`) を汚さずに試行錯誤が可能になります。
- **設定の階層化**: `base/` と `experiments/` を分けることで、共通設定と実験固有の差分を管理しやすくなります。

### 進捗状況
- [x] `config/v449` ディレクトリ作成
- [x] `experiments/v449` ディレクトリ作成
- [x] `run_ab_test.py` の移動
- [x] `ztb/trading/environment/components/rewards` パッケージ作成

## 3. 報酬関数のリファクタリング (Strategy Patternの導入)

### 現状の課題
`ztb/trading/environment/components/reward_calculator.py` は2000行を超えており、これ以上の機能追加（Smart Incentiveの実装など）は保守性を著しく低下させます。

### 提案する実装方針
**Strategy Pattern** を導入し、報酬計算ロジックをコンポーネント化します。

1.  **インターフェース定義**:
    ```python
    class RewardComponent(ABC):
        @abstractmethod
        def calculate(self, context: RewardContext) -> float: ...
    ```
2.  **コンポーネント化**:
    - `ForcedBalanceReward` (既存ロジック)
    - `SmartIncentiveReward` (新規ロジック: Regime-Adaptive, Soft Constraint)
    - `TradingFrequencyReward` (新規ロジック)
3.  **RewardCalculatorの軽量化**:
    `RewardCalculator` はこれらのコンポーネントをリストとして保持し、合計値を計算する「コンテナ」としての役割に徹します。

### 実装の都合が良い点
- **A/Bテストの容易化**: コンポーネントを差し替えるだけで、異なる報酬設計を比較できます。
- **テスト容易性**: 各コンポーネント単位で単体テストが可能になり、品質が向上します。

### 進捗状況
- [x] `RewardComponent` インターフェース定義 (`base.py`)
- [x] `RewardContext` 定義 (`base.py`)
- [x] `ForcedBalanceReward` 実装 (`forced_balance.py`)
- [x] `SmartIncentiveReward` 実装 (`smart_incentive.py`)
- [x] `PnlFocusedReward` 実装 (`pnl_focused.py`)
- [x] `UltraProfitReward` 実装 (`ultra_profit.py`)
- [x] `TradingFocusedReward` 実装 (`trading_focused.py`)
- [x] `ProfitOptimizedReward` 実装 (`profit_optimized.py`)
- [x] `RewardUtils` 実装 (`utils.py`) - 共通ロジックの集約
- [x] `RewardCalculator` の改修 (コンポーネント利用への切り替え)
- [x] 各コンポーネントの単体テスト作成とパス

## 4. アクション閾値の動的制御 (Adaptive Thresholding)

### 実装方針
`ActionExecutor` 内に直接ロジックを書くのではなく、`ThresholdManager` クラスを新設することを提案します。

```python
class ThresholdManager:
    def __init__(self, config: EnvironmentConfig):
        self.base_threshold = config.continuous_to_discrete_threshold

    def get_threshold(self, volatility: float) -> float:
        # ボラティリティに応じた動的閾値計算
        return self.base_threshold * (1.0 + ...)
```

### 実装の都合が良い点
- **責務の分離**: アクションの「実行」と、アクションの「解釈（閾値判定）」を分離できます。
- **拡張性**: 将来的にAIモデル自体に閾値を決定させる場合も、このクラスを拡張するだけで対応可能です。

### 進捗状況
- [x] `ThresholdManager` クラス作成 (`ztb/trading/environment/components/threshold_manager.py`)
- [x] `HeavyTradingEnv` への統合
- [x] 単体テスト作成 (`tests/test_threshold_manager.py`)

## 5. 市場レジーム検知機能 (Market Regime Detection)

### 実装方針
市場環境（トレンド、レンジ、高ボラティリティ等）を識別し、他のコンポーネント（報酬計算、閾値管理）にコンテキストを提供する `MarketRegimeClassifier` を実装します。

```python
class MarketRegimeClassifier:
    def detect_regime(self, data: pd.DataFrame) -> MarketRegime:
        # トレンド、ボラティリティ、レンジの判定
        ...
```

### 連携
- **ThresholdManager**: レジームに応じて閾値を調整（例：トレンド時は閾値を下げて順張り推奨）。
- **RewardCalculator**: レジームに応じて報酬戦略を切り替え（例：レンジ時はミーンリバージョン報酬を強化）。

### 進捗状況
- [x] `MarketRegimeClassifier` クラス作成
- [x] `HeavyTradingEnv` への統合
- [x] `ThresholdManager` との連携

## 6. Unified Trainer との連携

### 設定ファイルの活用
`unified_trainer` は `EnvironmentConfig` を辞書から生成します。v449用の設定ファイル (`config/v449/templates/sac_v449_template.yaml`) を作成し、そこに以下のパラメータを明記します。

```yaml
environment:
  continuous_to_discrete_threshold: 0.05
  reward_settings:
    use_smart_incentive: true
    smart_incentive_mode: "regime_adaptive"
```

### 気づきと提言: Configの型安全性
現在、`EnvironmentConfig` は巨大な dataclass ですが、`unified_trainer` 側での辞書展開 (`**kwargs`) は型チェックが効きにくい弱点があります。
- **提言**: `pydantic` などのバリデーションライブラリの導入を検討するか、少なくとも `EnvironmentConfig.from_dict()` メソッド内で厳密な型チェックを行うロジックを追加すべきです。これにより、学習開始直後に「設定ミスで落ちる」時間を節約できます。

## 7. 開発ロードマップ (優先順位付き)

1.  **[Done] ディレクトリ構造の整備**: `experiments/` と `config/v449/` の作成、ファイルの移動。
2.  **[Done] RewardCalculatorのリファクタリング**: Strategy Patternへの移行準備（まずは既存ロジックをクラスに切り出す）。
3.  **[Done] Smart Incentiveの実装**: 新しい報酬ロジックの実装と単体テスト。
4.  **[Done] Adaptive Thresholdの実装**: `ThresholdManager` の作成と組み込み。
5.  **[Done] Market Regime Detectionの実装**: レジーム検知機能の実装と統合。
6.  **[Done] Unified Trainerでの学習準備**: 設定ファイル作成と統合テスト完了。
7.  **[Done] ActionSignalGuide Refactoring**: `ActionSignalGuide` のリファクタリングと `SignalIntegrator` との連携修正。
8.  **[In Progress] コンポーネント連携の最適化**:
    - Smart Incentive のレジーム適応調整
    - Adaptive Threshold の感度調整
    - Market Regime Detection の精度向上

## 8. 検証と反省 (Validation & Retrospective)

### Run Short Training (2025-12-03)
`experiments/v449/run_short_training.py` の実行結果に基づく分析。

#### 成功点
- **SAC Training**: 連続値アクション空間での学習が正常に完了。
- **Warning Free**: `ActionSignalGuide` 関連の警告が解消され、シグナル統合が機能していることを確認。
- **Regime Detection**: `RegimeType.CONSOLIDATION` 等のレジーム検知が動作している。

#### 課題と改善案
1.  **BUY Bias in Consolidation**:
    - **現象**: レンジ相場 (`CONSOLIDATION`) と判定されているにもかかわらず、アクション分布が BUY (62.3%) に大きく偏っている。
    - **原因仮説**:
        - 強気相場のデータセットを使用しているため、レンジ判定でも上昇バイアスがかかっている。
        - `SmartIncentive` のレンジ相場向け報酬（ミーンリバージョン等）が十分に機能していない、または BUY を推奨するシグナルが支配的。
    - **対策**: `SmartIncentiveReward` のレジーム別重み付けを見直し、レンジ相場での逆張りインセンティブを強化する。

2.  **High Memory Usage**:
    - **現象**: 短時間の学習でもメモリ使用量警告が出る。
    - **対策**: 不要なデータ保持（特に `mtf_data` 等の一時変数）の削減を徹底する。`gc.collect()` の適切な配置（対応済み）。

3.  **Missing Dependencies**:
    - **現象**: `WalkForwardAnalyzer` 等のインポートエラー警告。
    - **対策**: 依存関係の整理、または該当機能の無効化設定を確認する。

## 9. まとめ
v449の実装は、単なる機能追加ではなく、**「技術的負債の返済（リファクタリング）」と「機能拡張」をセットで行う** 絶好の機会です。
特に `RewardCalculator` の分割は、今後の開発速度を維持するために不可欠です。まずはディレクトリ整理から着手し、きれいな環境で実装を進めることを強く推奨します。
