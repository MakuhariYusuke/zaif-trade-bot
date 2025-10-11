# Unified Trainer Refactoring Plan

## 🎯 目的

1. **AlgorithmFactory統合**: 全ての訓練がAlgorithmFactoryを経由
2. **コード切り分け**: 肥大化している876行を適切に分割
3. **保守性向上**: 新しいアルゴリズム追加が容易

## 📊 現在の問題点

### 現状分析
- **ファイルサイズ**: 876行（肥大化）
- **メソッド数**: 18個
- **アルゴリズム**: PPO, base_ml, iterative, ensemble, curriculum
- **構造**: 全てのアルゴリズムが1つのクラスに混在

### 主要メソッド
```
UnifiedTrainer:
  - __init__()
  - _get_config_value()          # 設定取得ヘルパー
  - get_memory_optimization_config()
  - get_environment_config()
  - get_ppo_core_config()
  - get_feature_config()
  - build_unified_config()       # 設定統合
  - train()                      # メインエントリー
  - _train_impl()                # アルゴリズム振り分け
  - _train_ppo()                 # PPO訓練（200行以上）
  - _train_base_ml()
  - _train_iterative()           # 156行
  - _train_ensemble()
  - _train_curriculum()
```

## 📁 新しい構造設計

```
ztb/training/
├── unified_trainer.py          # 🔄 簡素化（200行以下）
│   └── UnifiedTrainer          # AlgorithmFactory使用の薄いラッパー
│
├── core/
│   ├── config_builder.py       # 🆕 設定構築ロジック
│   └── trainer_helpers.py      # 🆕 共通ヘルパー関数
│
└── algorithms/
    ├── __init__.py
    ├── base_algorithm.py
    ├── algorithm_factory.py
    └── ppo/
        ├── __init__.py
        ├── ppo_algorithm.py    # 🔄 _train_ppoのロジックを移動
        └── ppo_config_builder.py # 🆕 PPO設定構築
```

## 🔧 リファクタリングステップ

### Step 1: 設定構築ロジックの抽出

**新規ファイル**: `ztb/training/core/config_builder.py`

```python
class ConfigBuilder:
    """設定構築の統一インターフェース"""
    
    @staticmethod
    def build_memory_optimization_config(config: dict) -> dict:
        """メモリ最適化設定を構築"""
        
    @staticmethod
    def build_environment_config(config: dict) -> dict:
        """環境設定を構築"""
        
    @staticmethod
    def build_ppo_config(config: dict) -> dict:
        """PPO設定を構築"""
        
    @staticmethod
    def build_unified_config(config: dict) -> dict:
        """統合設定を構築"""
```

**移動元**: `UnifiedTrainer` の以下のメソッド
- `get_memory_optimization_config()`
- `get_environment_config()`
- `get_ppo_core_config()`
- `get_feature_config()`
- `build_unified_config()`

### Step 2: PPO訓練ロジックの移動

**更新ファイル**: `ztb/training/algorithms/ppo/ppo_algorithm.py`

```python
class PPOAlgorithm(BaseRLAlgorithm):
    """PPO実装（完全版）"""
    
    def train_full(
        self,
        config: dict,
        env: VecEnv,
        session_id: str,
        **kwargs
    ) -> BaseAlgorithm:
        """
        完全なPPO訓練パイプライン。
        
        既存の_train_ppo()のロジックを統合。
        - チェックポイント管理
        - TensorBoardログ
        - Lagrangeパラメータ
        - 評価ゲート
        """
```

**移動元**: `UnifiedTrainer._train_ppo()` (200行以上)

### Step 3: unified_trainer.py の簡素化

**新しいUnifiedTrainer**:

```python
class UnifiedTrainer:
    """
    統一訓練インターフェース。
    
    全てのアルゴリズムをAlgorithmFactoryで管理。
    設定構築はConfigBuilderに委譲。
    """
    
    def __init__(self, config: dict, **kwargs):
        self.config = config
        self.config_builder = ConfigBuilder()
        
        # 🆕 アルゴリズムを動的に選択
        algorithm_name = config.get("algorithm", "ppo")
        self.algorithm = AlgorithmFactory.create(algorithm_name)
    
    def train(self) -> TrainingResult:
        """訓練実行（薄いラッパー）"""
        # 設定構築
        unified_config = self.config_builder.build_unified_config(self.config)
        
        # アルゴリズムに委譲
        if self.algorithm.algorithm_name == "ppo":
            return self._train_with_algorithm(unified_config)
        # 他のアルゴリズムも同様
    
    def _train_with_algorithm(self, config: dict) -> TrainingResult:
        """アルゴリズムを使用した訓練"""
        # 環境作成
        env = self._create_environment(config)
        
        # モデル作成
        model = self.algorithm.create_model(env, config)
        
        # 訓練実行
        return self.algorithm.train(model, config["total_timesteps"])
```

**削減見込み**: 876行 → 200行以下

### Step 4: 他のアルゴリズムの整理

**オプション**:
- `_train_base_ml()`, `_train_iterative()`, `_train_ensemble()`, `_train_curriculum()` も
  それぞれのアルゴリズム実装に移動するか、
  または別ファイルに切り出す

**提案**: 現時点ではPPOに集中し、他は後回し

## 📋 実装順序

### Phase 1: ConfigBuilder抽出（今回）

1. ✅ `ztb/training/core/config_builder.py` 作成
2. ✅ 設定構築メソッドを移動
3. ✅ `unified_trainer.py` で使用
4. ✅ テスト

### Phase 2: PPOAlgorithm完全実装（今回）

1. ✅ `ppo_algorithm.py` に `train_full()` メソッド追加
2. ✅ `_train_ppo()` のロジックを移動
3. ✅ チェックポイント、TensorBoard統合
4. ✅ テスト

### Phase 3: unified_trainer.py簡素化（今回）

1. ✅ AlgorithmFactory使用に変更
2. ✅ PPO以外のアルゴリズムは暫定的にそのまま
3. ✅ テスト

### Phase 4: 既存スクリプトの動作確認（次回）

1. ⏭️ `train_v394d.py` 等でテスト
2. ⏭️ 問題があれば修正

### Phase 5: SAC実装（次回）

1. ⏭️ SACAlgorithm実装
2. ⏭️ 設定ファイルで切り替え

## 🎯 目標

### Before（現在）
```
unified_trainer.py: 876行
  - 全アルゴリズムのロジックが混在
  - 設定構築ロジックも混在
  - 保守が困難
```

### After（目標）
```
unified_trainer.py: 200行以下
  - AlgorithmFactory使用
  - 薄いラッパー層
  - 設定構築はConfigBuilderに委譲
  
algorithms/ppo/: PPOロジック集約
core/config_builder.py: 設定構築統一
```

## ✅ 期待される効果

1. **可読性向上**: ファイルサイズ削減
2. **保守性向上**: 関心の分離
3. **拡張性向上**: 新アルゴリズム追加が容易
4. **テスト容易**: コンポーネント単位でテスト
5. **再利用性**: ConfigBuilderを他でも使用可能

---

**次のアクション**: ConfigBuilder作成から開始
