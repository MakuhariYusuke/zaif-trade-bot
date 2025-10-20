"""
アルゴリズムのファクトリークラス。

設定ファイルの "algorithm" フィールドに基づいて、
適切なアルゴリズム実装を動的に生成する。

Example:
    >>> from ztb.training.algorithms import AlgorithmFactory
    >>>
    >>> # 利用可能なアルゴリズムを確認
    >>> print(AlgorithmFactory.list_algorithms())
    ['ppo', 'sac']
    >>>
    >>> # PPOアルゴリズムを作成
    >>> ppo = AlgorithmFactory.create("ppo")
    >>> print(ppo.algorithm_name)
    'ppo'
"""

from typing import Any, Dict, Type

from .base_algorithm import BaseRLAlgorithm


class AlgorithmFactory:
    """
    アルゴリズムのファクトリークラス。

    Registry Patternを使用して、アルゴリズムを動的に登録・生成する。
    新しいアルゴリズムを追加する場合は、register()で登録するだけでよい。
    """

    _algorithms: Dict[str, Type[BaseRLAlgorithm]] = {}

    @classmethod
    def register(
        cls, algorithm_name: str, algorithm_class: Type[BaseRLAlgorithm]
    ) -> None:
        """
        アルゴリズムを登録する。

        Args:
            algorithm_name: アルゴリズム名（例: "ppo", "sac"）
            algorithm_class: アルゴリズムクラス（BaseRLAlgorithmのサブクラス）

        Raises:
            TypeError: algorithm_classがBaseRLAlgorithmのサブクラスでない場合

        Example:
            >>> from .ppo import PPOAlgorithm
            >>> AlgorithmFactory.register("ppo", PPOAlgorithm)
        """
        if not issubclass(algorithm_class, BaseRLAlgorithm):
            raise TypeError(
                f"{algorithm_class.__name__} must be a subclass of BaseRLAlgorithm"
            )

        cls._algorithms[algorithm_name.lower()] = algorithm_class
        print(f"✅ Registered algorithm: {algorithm_name}")

    @classmethod
    def create(cls, algorithm_name: str, **kwargs: Any) -> BaseRLAlgorithm:
        """
        アルゴリズムのインスタンスを作成する。

        Args:
            algorithm_name: アルゴリズム名（大文字小文字を区別しない）
            **kwargs: アルゴリズムのコンストラクタ引数

        Returns:
            アルゴリズムインスタンス

        Raises:
            ValueError: 未登録のアルゴリズム名が指定された場合

        Example:
            >>> algorithm = AlgorithmFactory.create("ppo")
            >>> model = algorithm.create_model(env, config)
        """
        algorithm_name = algorithm_name.lower()

        if algorithm_name not in cls._algorithms:
            available = ", ".join(sorted(cls._algorithms.keys()))
            raise ValueError(
                f"Unknown algorithm: '{algorithm_name}'. "
                f"Available algorithms: [{available}]"
            )

        algorithm_class = cls._algorithms[algorithm_name]
        return algorithm_class(**kwargs)

    @classmethod
    def list_algorithms(cls) -> list[str]:
        """
        利用可能なアルゴリズムのリストを取得。

        Returns:
            登録されているアルゴリズム名のリスト（アルファベット順）

        Example:
            >>> AlgorithmFactory.list_algorithms()
            ['ppo', 'sac', 'td3']
        """
        return sorted(cls._algorithms.keys())

    @classmethod
    def is_registered(cls, algorithm_name: str) -> bool:
        """
        アルゴリズムが登録されているか確認。

        Args:
            algorithm_name: アルゴリズム名

        Returns:
            登録されていればTrue

        Example:
            >>> AlgorithmFactory.is_registered("ppo")
            True
            >>> AlgorithmFactory.is_registered("unknown")
            False
        """
        return algorithm_name.lower() in cls._algorithms

    @classmethod
    def unregister(cls, algorithm_name: str) -> bool:
        """
        アルゴリズムの登録を解除（主にテスト用）。

        Args:
            algorithm_name: アルゴリズム名

        Returns:
            解除に成功したらTrue、未登録ならFalse

        Example:
            >>> AlgorithmFactory.unregister("ppo")
            True
        """
        algorithm_name = algorithm_name.lower()
        if algorithm_name in cls._algorithms:
            del cls._algorithms[algorithm_name]
            print(f"🗑️ Unregistered algorithm: {algorithm_name}")
            return True
        return False

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """
        登録済みアルゴリズムの情報を取得。

        Returns:
            アルゴリズム情報の辞書

        Example:
            >>> info = AlgorithmFactory.get_info()
            >>> print(info["count"])
            2
            >>> print(info["algorithms"])
            ['ppo', 'sac']
        """
        return {
            "count": len(cls._algorithms),
            "algorithms": cls.list_algorithms(),
            "registry": {
                name: cls_type.__name__ for name, cls_type in cls._algorithms.items()
            },
        }
