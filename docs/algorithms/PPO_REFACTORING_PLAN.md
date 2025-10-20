# PPO Refactoring Plan - アルゴリズム差し替え可能な設計へ

## 🎯 目的

既存PPOコードを整理し、将来的にSAC/TD3などを簡単に差し替えられる設計にする。

## 📁 現在の構造

```
ztb/training/
├── core/
│   └── ppo_trainer.py          # PPOTrainer, PPOTrainerAutoHalt
├── models/
│   └── custom_ppo.py           # CustomPPO (MaskablePPO拡張)
└── unified_trainer.py          # メインエントリーポイント
```

## 📁 目標構造

```
ztb/training/
├── algorithms/                  # 🆕 アルゴリズム別実装
│   ├── __init__.py
│   ├── base_algorithm.py       # 🆕 共通インターフェース
│   ├── algorithm_factory.py    # 🆕 Factory Pattern
│   └── ppo/                    # 🆕 PPO専用ディレクトリ
│       ├── __init__.py
│       ├── ppo_trainer.py      # 移動: core/ppo_trainer.py
│       ├── custom_ppo.py       # 移動: models/custom_ppo.py
│       └── config.py           # 🆕 PPO設定のデフォルト値
├── core/
│   └── (ppo_trainer.py 削除)
├── models/
│   └── (custom_ppo.py 削除)
└── unified_trainer.py          # 🔄 更新: algorithm_factory使用
```

## 🔧 実装ステップ

### Step 1: ディレクトリ作成

```bash
mkdir ztb\training\algorithms
mkdir ztb\training\algorithms\ppo
```

### Step 2: 共通インターフェース定義

**ファイル**: `ztb/training/algorithms/base_algorithm.py`

```python
"""
強化学習アルゴリズムの共通インターフェース。

全てのアルゴリズム（PPO, SAC, TD3等）はこのインターフェースを実装する。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv


class BaseRLAlgorithm(ABC):
    """強化学習アルゴリズムの基底クラス"""

    @abstractmethod
    def create_model(
        self,
        env: VecEnv,
        config: Dict[str, Any],
        tensorboard_log: Optional[str] = None,
    ) -> BaseAlgorithm:
        """
        モデルを作成する。

        Args:
            env: 訓練環境
            config: アルゴリズム設定
            tensorboard_log: TensorBoardログディレクトリ

        Returns:
            作成されたモデル
        """
        pass

    @abstractmethod
    def train(
        self,
        model: BaseAlgorithm,
        total_timesteps: int,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> BaseAlgorithm:
        """
        モデルを訓練する。

        Args:
            model: 訓練するモデル
            total_timesteps: 総ステップ数
            callback: コールバック関数
            **kwargs: その他のパラメータ

        Returns:
            訓練済みモデル
        """
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        デフォルト設定を取得する。

        Returns:
            デフォルト設定の辞書
        """
        pass

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """アルゴリズム名を返す（例: "ppo", "sac"）"""
        pass
```

### Step 3: Factory Pattern実装

**ファイル**: `ztb/training/algorithms/algorithm_factory.py`

```python
"""
アルゴリズムのファクトリークラス。

設定ファイルの "algorithm" フィールドに基づいて、
適切なアルゴリズム実装を返す。
"""

from typing import Dict, Any
from .base_algorithm import BaseRLAlgorithm


class AlgorithmFactory:
    """アルゴリズムのファクトリー"""

    _algorithms: Dict[str, type] = {}

    @classmethod
    def register(cls, algorithm_name: str, algorithm_class: type):
        """
        アルゴリズムを登録する。

        Args:
            algorithm_name: アルゴリズム名（例: "ppo"）
            algorithm_class: アルゴリズムクラス
        """
        cls._algorithms[algorithm_name] = algorithm_class

    @classmethod
    def create(cls, algorithm_name: str, **kwargs) -> BaseRLAlgorithm:
        """
        アルゴリズムのインスタンスを作成する。

        Args:
            algorithm_name: アルゴリズム名
            **kwargs: アルゴリズムのコンストラクタ引数

        Returns:
            アルゴリズムインスタンス

        Raises:
            ValueError: 未登録のアルゴリズム名
        """
        if algorithm_name not in cls._algorithms:
            available = ", ".join(cls._algorithms.keys())
            raise ValueError(
                f"Unknown algorithm: {algorithm_name}. "
                f"Available algorithms: {available}"
            )

        algorithm_class = cls._algorithms[algorithm_name]
        return algorithm_class(**kwargs)

    @classmethod
    def list_algorithms(cls) -> list[str]:
        """利用可能なアルゴリズムのリストを取得"""
        return list(cls._algorithms.keys())
```

### Step 4: PPO実装の移動と適合

**ファイル**: `ztb/training/algorithms/ppo/__init__.py`

```python
"""PPO (Proximal Policy Optimization) アルゴリズム実装"""

from .ppo_algorithm import PPOAlgorithm

__all__ = ["PPOAlgorithm"]
```

**ファイル**: `ztb/training/algorithms/ppo/ppo_algorithm.py`

```python
"""
PPOアルゴリズムの実装。

既存のPPOTrainerをBaseRLAlgorithmインターフェースに適合させる。
"""

from typing import Any, Dict, Optional, Callable
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from ..base_algorithm import BaseRLAlgorithm
from .ppo_trainer import PPOTrainer  # 既存のPPOTrainer
from .custom_ppo import CustomPPO    # 既存のCustomPPO


class PPOAlgorithm(BaseRLAlgorithm):
    """PPOアルゴリズムの実装"""

    def __init__(self):
        self._trainer = None

    @property
    def algorithm_name(self) -> str:
        return "ppo"

    def create_model(
        self,
        env: VecEnv,
        config: Dict[str, Any],
        tensorboard_log: Optional[str] = None,
    ) -> BaseAlgorithm:
        """PPOモデルを作成"""
        # 既存のPPOTrainerを使用
        self._trainer = PPOTrainer(
            env=env,
            config=config,
            tensorboard_log=tensorboard_log
        )
        return self._trainer.model

    def train(
        self,
        model: BaseAlgorithm,
        total_timesteps: int,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> BaseAlgorithm:
        """PPOモデルを訓練"""
        if self._trainer is None:
            raise RuntimeError("Model not created. Call create_model() first.")

        return self._trainer.train(
            total_timesteps=total_timesteps,
            callback=callback,
            **kwargs
        )

    def get_default_config(self) -> Dict[str, Any]:
        """PPOのデフォルト設定"""
        return {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
```

### Step 5: unified_trainer.py の更新

**変更前**:
```python
from ztb.training.core.ppo_trainer import PPOTrainer

class UnifiedTrainer:
    def __init__(self, config: dict):
        self.ppo_trainer = PPOTrainer(...)

    def train(self):
        self.ppo_trainer.train(...)
```

**変更後**:
```python
from ztb.training.algorithms.algorithm_factory import AlgorithmFactory

class UnifiedTrainer:
    def __init__(self, config: dict):
        algorithm_name = config.get("algorithm", "ppo")  # デフォルトPPO
        self.algorithm = AlgorithmFactory.create(algorithm_name)

    def train(self):
        model = self.algorithm.create_model(...)
        self.algorithm.train(model, ...)
```

### Step 6: アルゴリズム登録

**ファイル**: `ztb/training/algorithms/__init__.py`

```python
"""強化学習アルゴリズムモジュール"""

from .algorithm_factory import AlgorithmFactory
from .ppo import PPOAlgorithm

# PPOを登録
AlgorithmFactory.register("ppo", PPOAlgorithm)

# 将来の拡張（コメントアウト）
# from .sac import SACAlgorithm
# AlgorithmFactory.register("sac", SACAlgorithm)

__all__ = ["AlgorithmFactory", "PPOAlgorithm"]
```

## 📝 設定ファイルの変更

### 既存の設定ファイル

```json
{
  "model_name": "ppo_v394d_aggressive",
  "total_timesteps": 100000,
  "ppo_hyperparameters": {
    "learning_rate": 0.007503,
    ...
  }
}
```

### 新しい設定ファイル（algorithm フィールド追加）

```json
{
  "model_name": "ppo_v394d_aggressive",
  "algorithm": "ppo",  // 🆕 アルゴリズム指定
  "total_timesteps": 100000,
  "ppo_hyperparameters": {
    "learning_rate": 0.007503,
    ...
  }
}
```

## 🧪 動作確認

### テストスクリプト

```python
# test_algorithm_factory.py

from ztb.training.algorithms import AlgorithmFactory

# 利用可能なアルゴリズムを確認
print("Available algorithms:", AlgorithmFactory.list_algorithms())
# 出力: Available algorithms: ['ppo']

# PPOアルゴリズムを作成
ppo = AlgorithmFactory.create("ppo")
print(f"Created: {ppo.algorithm_name}")
# 出力: Created: ppo

# デフォルト設定を確認
config = ppo.get_default_config()
print("Default config:", config)
```

## ✅ 移行チェックリスト

- [ ] Step 1: ディレクトリ作成
- [ ] Step 2: base_algorithm.py 作成
- [ ] Step 3: algorithm_factory.py 作成
- [ ] Step 4: PPO関連ファイルを移動
- [ ] Step 5: unified_trainer.py 更新
- [ ] Step 6: __init__.py でPPO登録
- [ ] Step 7: 動作確認テスト
- [ ] Step 8: 既存訓練スクリプトの動作確認

## 🎯 完了後の利点

1. **簡単なアルゴリズム切り替え**
   ```json
   {"algorithm": "ppo"}  // PPO使用
   {"algorithm": "sac"}  // SAC使用（将来）
   ```

2. **コードの再利用性向上**
   - 共通インターフェースで統一
   - 新しいアルゴリズム追加が容易

3. **保守性向上**
   - アルゴリズム別にディレクトリ分離
   - 関心の分離が明確

4. **テストの簡素化**
   - アルゴリズム単位でテスト可能
   - モックの作成が容易

## 📅 次のステップ（Phase 2）

PPO整理完了後:
1. SAC実装追加
2. TD3実装追加（オプション）
3. アルゴリズム比較ツール作成
4. 最適アルゴリズム選定

---

**開始してよろしいですか？**
