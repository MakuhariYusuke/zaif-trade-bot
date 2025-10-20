"""
強化学習アルゴリズムの共通インターフェース。

全てのアルゴリズム（PPO, SAC, TD3等）はこのインターフェースを実装する。
これにより、アルゴリズムを簡単に差し替えられる設計を実現する。

Example:
    >>> from ztb.training.algorithms import AlgorithmFactory
    >>> algorithm = AlgorithmFactory.create("ppo")
    >>> model = algorithm.create_model(env, config)
    >>> algorithm.train(model, total_timesteps=100000)
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv


class BaseRLAlgorithm(ABC):
    """
    強化学習アルゴリズムの基底クラス。

    全てのRL アルゴリズム実装はこのクラスを継承し、
    必要なメソッドを実装する必要がある。
    """

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
            env: 訓練環境（VecEnv）
            config: アルゴリズム設定（ハイパーパラメータ等）
            tensorboard_log: TensorBoardログディレクトリ（Optional）

        Returns:
            作成されたモデル（BaseAlgorithm）

        Example:
            >>> model = algorithm.create_model(
            ...     env=vec_env,
            ...     config={"learning_rate": 0.0003},
            ...     tensorboard_log="./logs"
            ... )
        """
        pass

    @abstractmethod
    def train(
        self,
        model: BaseAlgorithm,
        total_timesteps: int,
        callback: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ) -> BaseAlgorithm:
        """
        モデルを訓練する。

        Args:
            model: 訓練するモデル
            total_timesteps: 総ステップ数
            callback: コールバック関数（Optional）
            **kwargs: その他のパラメータ

        Returns:
            訓練済みモデル

        Example:
            >>> trained_model = algorithm.train(
            ...     model=model,
            ...     total_timesteps=100000,
            ...     callback=checkpoint_callback
            ... )
        """
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        デフォルト設定を取得する。

        アルゴリズム固有のデフォルトハイパーパラメータを返す。
        ユーザー設定とマージして使用される。

        Returns:
            デフォルト設定の辞書

        Example:
            >>> config = algorithm.get_default_config()
            >>> print(config["learning_rate"])
            0.0003
        """
        pass

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """
        アルゴリズム名を返す。

        Returns:
            アルゴリズム名（例: "ppo", "sac", "td3"）

        Example:
            >>> algorithm.algorithm_name
            'ppo'
        """
        pass

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        設定の妥当性を検証する（オプション）。

        Args:
            config: 検証する設定

        Returns:
            設定が妥当ならTrue

        Note:
            デフォルト実装は常にTrueを返す。
            必要に応じてサブクラスでオーバーライドする。
        """
        return True

    def __repr__(self) -> str:
        """アルゴリズムの文字列表現"""
        return f"{self.__class__.__name__}(algorithm='{self.algorithm_name}')"
