"""
Advanced Callbacks for SAC Training.

高度な訓練制御機能を提供するコールバック集:
- Early Stopping: 収束検出時の自動停止
- Best Model Selection: 最良モデルの自動保存
- Enhanced Metrics: 拡張メトリクス記録
"""

from collections import deque
from pathlib import Path
from typing import Optional


from stable_baselines3.common.callbacks import BaseCallback
from ztb.utils.logging_utils import get_logger
from ztb.utils.training_utils import get_metric_from_logger

logger = get_logger(__name__)


class EarlyStoppingCallback(BaseCallback):
    """
    Early Stopping Callback for SAC training.

    訓練の収束を検出し、自動的に訓練を停止する。

    収束判定基準:
    1. 改善率が閾値以下（例: 1%）が連続でpatience回
    2. メトリクスの変動が小さい（変動係数 < 閾値）

    Args:
        metric_name: 監視するメトリクス名（例: 'critic_loss', 'episode_reward'）
        min_delta: 改善とみなす最小変化量（絶対値）
        patience: 改善が見られない最大ステップ数
        check_interval: 収束チェックの間隔（ステップ数）
        window_size: メトリクスの移動平均ウィンドウサイズ
        cv_threshold: 変動係数の閾値（これ以下で安定とみなす）
        verbose: ログ出力の詳細度
    """

    def __init__(
        self,
        metric_name: str = "critic_loss",
        min_delta: float = 0.0001,
        patience: int = 5000,
        check_interval: int = 1000,
        window_size: int = 1000,
        cv_threshold: float = 0.05,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.metric_name = metric_name
        self.min_delta = min_delta
        self.patience = patience
        self.check_interval = check_interval
        self.window_size = window_size
        self.cv_threshold = cv_threshold

        self.metric_history: deque = deque(maxlen=window_size)
        self.best_metric: Optional[float] = None
        self.steps_without_improvement = 0
        self.stopped_step = 0

    def _on_step(self) -> bool:
        """
        各ステップで呼ばれるコールバック。

        Returns:
            False: 訓練を停止、True: 訓練を継続
        """
        # check_intervalごとに収束チェック
        if self.n_calls % self.check_interval != 0:
            return True

        # メトリクスを取得
        current_metric = self._get_metric()

        if current_metric is None:
            return True

        # 履歴に追加
        self.metric_history.append(current_metric)

        # 初回は比較対象がないのでスキップ
        if self.best_metric is None:
            self.best_metric = current_metric
            return True

        # 改善判定
        improvement = abs(self.best_metric - current_metric)

        if improvement > self.min_delta:
            # 改善あり
            self.best_metric = current_metric
            self.steps_without_improvement = 0

            if self.verbose > 0:
                logger.info(
                    f"✅ Early Stopping: Improvement detected | "
                    f"{self.metric_name}={current_metric:.6f} | "
                    f"improvement={improvement:.6f}"
                )
        else:
            # 改善なし
            self.steps_without_improvement += self.check_interval

            if self.verbose > 1:
                logger.info(
                    f"⏳ Early Stopping: No improvement | "
                    f"{self.metric_name}={current_metric:.6f} | "
                    f"steps_without_improvement={self.steps_without_improvement}/{self.patience}"
                )

        # Patience超過チェック
        if self.steps_without_improvement >= self.patience:
            # 変動係数もチェック
            if len(self.metric_history) >= self.window_size:
                cv = self._calculate_cv()

                if cv < self.cv_threshold:
                    # 収束と判定
                    self.stopped_step = self.n_calls

                    logger.info("=" * 80)
                    logger.info(
                        "🛑 Early Stopping: Training stopped due to convergence"
                    )
                    logger.info("=" * 80)
                    logger.info(f"  Metric: {self.metric_name}")
                    logger.info(f"  Best Value: {self.best_metric:.6f}")
                    logger.info(f"  Current Value: {current_metric:.6f}")
                    logger.info(
                        f"  Steps without improvement: {self.steps_without_improvement}"
                    )
                    logger.info(
                        f"  Coefficient of Variation: {cv:.4f} (< {self.cv_threshold})"
                    )
                    logger.info(f"  Stopped at step: {self.stopped_step}")
                    logger.info("=" * 80)

                    return False  # 訓練停止
                else:
                    # CVが高い（変動が大きい）のでpatienceをリセット
                    self.steps_without_improvement = 0

                    if self.verbose > 0:
                        logger.info(
                            f"⚠️ Early Stopping: CV too high ({cv:.4f} >= {self.cv_threshold}), "
                            f"patience reset"
                        )

        return True  # 訓練継続

    def _get_metric(self) -> Optional[float]:
        """現在のメトリクス値を取得"""
        return get_metric_from_logger(self.model, self.metric_name)


class BestModelSaveCallback(BaseCallback):
    """訓練中に最良のモデルを自動的に保存する。

    Args:
        save_path: モデル保存先ディレクトリ
        model_name: モデル名（プレフィックス）
        metric_name: 評価メトリクス名
        mode: 'min'（小さいほど良い）または 'max'（大きいほど良い）
        check_interval: チェック間隔（ステップ数）
        verbose: ログ出力の詳細度
    """

    def __init__(
        self,
        save_path: Path,
        model_name: str = "best_model",
        metric_name: str = "critic_loss",
        mode: str = "min",
        check_interval: int = 1000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.model_name = model_name
        self.metric_name = metric_name
        self.mode = mode
        self.check_interval = check_interval

        self.best_metric: Optional[float] = None
        self.best_model_path: Optional[Path] = None

        # ディレクトリ作成
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """各ステップで呼ばれるコールバック"""
        if self.n_calls % self.check_interval != 0:
            return True

        # メトリクスを取得
        current_metric = self._get_metric()

        if current_metric is None:
            return True

        # 初回または改善時にモデルを保存
        is_best = False

        if self.best_metric is None:
            is_best = True
        elif self.mode == "min" and current_metric < self.best_metric:
            is_best = True
        elif self.mode == "max" and current_metric > self.best_metric:
            is_best = True

        if is_best:
            self.best_metric = current_metric

            # モデル保存
            model_path = (
                self.save_path
                / f"{self.model_name}_best_{self.metric_name}_{current_metric:.6f}_step_{self.n_calls}.zip"
            )
            self.model.save(str(model_path))
            self.best_model_path = model_path

            if self.verbose > 0:
                logger.info(
                    f"💾 Best Model Saved: {self.metric_name}={current_metric:.6f} | "
                    f"step={self.n_calls} | path={model_path.name}"
                )

        return True

    def _get_metric(self) -> Optional[float]:
        """現在のメトリクス値を取得"""
        if not hasattr(self.model, "logger") or self.model.logger is None:
            return None

        try:
            name_to_value = self.model.logger.name_to_value

            possible_names = [
                self.metric_name,
                f"train/{self.metric_name}",
                f"rollout/{self.metric_name}",
            ]

            for name in possible_names:
                if name in name_to_value:
                    return float(name_to_value[name])

            return None
        except (AttributeError, KeyError, TypeError):
            return None

    def _on_training_end(self) -> None:
        """訓練終了時のコールバック"""
        if self.best_model_path:
            logger.info("=" * 80)
            logger.info("🏆 Best Model Summary:")
            logger.info(f"  Metric: {self.metric_name} = {self.best_metric:.6f}")
            logger.info(f"  Path: {self.best_model_path}")
            logger.info("=" * 80)
