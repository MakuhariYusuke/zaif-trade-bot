"""
Meta-Learning for SAC v421
MAMLスタイルのメタラーニングによる迅速な市場適応
"""

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class MetaTaskData:
    """タスクデータ"""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    task_id: str

@dataclass
class MetaLearningConfig:
    """メタラーニング設定"""

    inner_lr: float = 0.01
    outer_lr: float = 0.001
    meta_batch_size: int = 4
    num_inner_updates: int = 5
    num_meta_updates: int = 1000
    adaptation_steps: int = 10
    task_batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 50

class MAML(nn.Module):
    """Model-Agnostic Meta-Learning"""

    def __init__(self, model: nn.Module, config: MetaLearningConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=config.outer_lr)

    def forward(self, x):
        return self.model(x)

    def adapt_to_task(self, task_data: MetaTaskData, loss_fn: Callable) -> nn.Module:
        """タスクへの適応"""
        # モデルコピー
        adapted_model = copy.deepcopy(self.model)
        adapted_optimizer = optim.SGD(
            adapted_model.parameters(), lr=self.config.inner_lr
        )

        # 適応ステップ
        adapted_model.train()
        for _ in range(self.config.adaptation_steps):
            adapted_optimizer.zero_grad()

            # タスクデータでの損失計算
            outputs = adapted_model(task_data.states)
            loss = loss_fn(
                outputs,
                task_data.actions,
                task_data.rewards,
                adapted_model(task_data.next_states),
                task_data.dones,
            )

            loss.backward()
            adapted_optimizer.step()

        return adapted_model

    def meta_update(self, task_losses: list[torch.Tensor]):
        """メタ更新"""
        self.meta_optimizer.zero_grad()

        # メタ損失
        meta_loss = torch.stack(task_losses).mean()

        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

class Reptile(nn.Module):
    """Reptileアルゴリズム"""

    def forward(self, x):
        return self.model(x)

    def adapt_to_task(
        self, task_data: MetaTaskData, loss_fn: Callable
    ) -> tuple[nn.Module, dict[str, Any]]:
        """タスクへの適応"""
        adapted_model = copy.deepcopy(self.model)
        optimizer = optim.SGD(adapted_model.parameters(), lr=self.config.inner_lr)

        initial_params = {
            name: param.clone() for name, param in adapted_model.named_parameters()
        }

        # 適応ステップ
        adapted_model.train()
        losses = []

        for step in range(self.config.adaptation_steps):
            optimizer.zero_grad()

            outputs = adapted_model(task_data.states)
            loss = loss_fn(
                outputs,
                task_data.actions,
                task_data.rewards,
                adapted_model(task_data.next_states),
                task_data.dones,
            )

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # パラメータ差分計算
        final_params = {
            name: param.clone() for name, param in adapted_model.named_parameters()
        }
        param_updates = {}

        for name in initial_params:
            param_updates[name] = final_params[name] - initial_params[name]

        return adapted_model, {
            "initial_params": initial_params,
            "final_params": final_params,
            "param_updates": param_updates,
            "adaptation_losses": losses,
        }

    def meta_update(
        self, adapted_models: list[nn.Module], task_infos: list[dict[str, Any]]
    ):
        """メタ更新"""
        self.meta_optimizer.zero_grad()

        # Reptile更新: 各タスクの適応後パラメータの平均に向かって更新
        meta_gradients = defaultdict(list)

        for adapted_model, task_info in zip(adapted_models, task_infos):
            for name, param in adapted_model.named_parameters():
                initial_param = task_info["initial_params"][name]
                final_param = task_info["final_params"][name]

                # 適応方向の勾配
                gradient = final_param - initial_param
                meta_gradients[name].append(gradient)

        # 平均勾配で更新
        for name, param in self.model.named_parameters():
            if name in meta_gradients:
                avg_gradient = torch.stack(meta_gradients[name]).mean(dim=0)
                param.grad = -avg_gradient  # 負の勾配方向に更新

        self.meta_optimizer.step()

class MetaLearner:
    """メタラーニング統合クラス"""

    def __init__(
        self,
        base_model: nn.Module,
        algorithm: str = "maml",
        config: MetaLearningConfig | None = None,
    ):
        self.base_model = base_model
        self.algorithm = algorithm
        self.config = config or MetaLearningConfig()

        if algorithm.lower() == "maml":
            self.meta_model = MAML(base_model, self.config)
        elif algorithm.lower() == "reptile":
            self.meta_model = Reptile(base_model, self.config)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        self.task_buffer = []
        self.performance_history = []
        self.best_model_state = None
        self.best_performance = float("-inf")

    def collect_task_data(self, task_data: MetaTaskData):
        """タスクデータ収集"""
        self.task_buffer.append(task_data)

        # バッファサイズ制限
        if len(self.task_buffer) > 100:
            self.task_buffer = self.task_buffer[-100:]

    def sample_tasks(self, num_tasks: int) -> list[MetaTaskData]:
        """タスクサンプリング"""
        if len(self.task_buffer) < num_tasks:
            logger.warning(
                f"Insufficient tasks in buffer: {len(self.task_buffer)} < {num_tasks}"
            )
            return self.task_buffer.copy()

        # ランダムサンプリング（重複なし）
        indices = np.random.choice(len(self.task_buffer), num_tasks, replace=False)
        return [self.task_buffer[i] for i in indices]

    def train_meta(
        self, loss_fn: Callable, num_epochs: int = 100
    ) -> dict[str, list[float]]:
        """メタ学習実行"""
        training_history = {
            "meta_losses": [],
            "task_losses": [],
            "validation_performance": [],
        }

        patience_counter = 0

        for epoch in range(num_epochs):
            # タスクサンプリング
            tasks = self.sample_tasks(self.config.meta_batch_size)
            if len(tasks) < self.config.meta_batch_size:
                logger.warning(f"Skipping epoch {epoch}: insufficient tasks")
                continue

            task_losses = []

            if self.algorithm.lower() == "maml":
                # MAML学習
                adapted_models = []
                task_losses_epoch = []

                for task in tasks:
                    adapted_model = self.meta_model.adapt_to_task(task, loss_fn)
                    adapted_models.append(adapted_model)

                    # 適応後モデルの損失計算
                    with torch.no_grad():
                        outputs = adapted_model(task.states)
                        task_loss = loss_fn(
                            outputs,
                            task.actions,
                            task.rewards,
                            adapted_model(task.next_states),
                            task.dones,
                        )
                        task_losses_epoch.append(task_loss)

                # メタ更新
                meta_loss = self.meta_model.meta_update(task_losses_epoch)
                task_losses.extend(task_losses_epoch)

            elif self.algorithm.lower() == "reptile":
                # Reptile学習
                adapted_models = []
                task_infos = []

                for task in tasks:
                    adapted_model, task_info = self.meta_model.adapt_to_task(
                        task, loss_fn
                    )
                    adapted_models.append(adapted_model)
                    task_infos.append(task_info)

                    # タスク損失
                    final_loss = task_info["adaptation_losses"][-1]
                    task_losses.append(torch.tensor(final_loss))

                # メタ更新
                self.meta_model.meta_update(adapted_models, task_infos)
                meta_loss = np.mean(
                    [info["adaptation_losses"][-1] for info in task_infos]
                )

            # 履歴記録
            training_history["meta_losses"].append(meta_loss)
            training_history["task_losses"].append(
                np.mean([loss.item() for loss in task_losses])
            )

            # 検証性能評価（簡易版）
            val_performance = self._evaluate_meta_performance(
                tasks[:2], loss_fn
            )  # 最初の2タスクで評価
            training_history["validation_performance"].append(val_performance)

            # Early stopping
            if val_performance > self.best_performance:
                self.best_performance = val_performance
                self.best_model_state = copy.deepcopy(self.meta_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Meta epoch {epoch+1}/{num_epochs}, "
                    f"meta_loss: {meta_loss:.6f}, "
                    f"task_loss: {training_history['task_losses'][-1]:.6f}, "
                    f"val_perf: {val_performance:.6f}"
                )

        # 最適モデル復元
        if self.best_model_state is not None:
            self.meta_model.load_state_dict(self.best_model_state)

        return training_history

    def _evaluate_meta_performance(
        self, tasks: list[MetaTaskData], loss_fn: Callable
    ) -> float:
        """メタ性能評価"""
        total_performance = 0

        for task in tasks:
            adapted_model = self.meta_model.adapt_to_task(task, loss_fn)

            # 適応後性能評価
            with torch.no_grad():
                outputs = adapted_model(task.states)
                loss = loss_fn(
                    outputs,
                    task.actions,
                    task.rewards,
                    adapted_model(task.next_states),
                    task.dones,
                )
                # 負の損失を性能として使用（高いほど良い）
                performance = -loss.item()
                total_performance += performance

        return total_performance / len(tasks) if tasks else 0.0

    def adapt_to_new_market(
        self, market_data: MetaTaskData, adaptation_steps: int | None = None
    ) -> nn.Module:
        """新規市場への適応"""
        if adaptation_steps is None:
            adaptation_steps = self.config.adaptation_steps

        # SACの損失関数（仮定）
        def sac_loss(outputs, actions, rewards, next_outputs, dones):
            # 簡易的なSAC損失（実際の実装では適切なSAC損失を使用）
            return torch.mean((outputs - actions) ** 2)

        adapted_model = self.meta_model.adapt_to_task(market_data, sac_loss)

        logger.info(f"Adapted to new market: {market_data.task_id}")
        return adapted_model

    def get_adapted_model(self) -> nn.Module:
        """適応済みモデル取得"""
        return self.meta_model.model

    def save_meta_model(self, path: str):
        """メタモデル保存"""
        torch.save(
            {
                "model_state_dict": self.meta_model.state_dict(),
                "config": self.config,
                "algorithm": self.algorithm,
                "best_performance": self.best_performance,
            },
            path,
        )
        logger.info(f"Meta model saved to {path}")

    def load_meta_model(self, path: str):
        """メタモデル読み込み"""
        checkpoint = torch.load(path)
        self.meta_model.load_state_dict(checkpoint["model_state_dict"])
        self.config = checkpoint["config"]
        self.algorithm = checkpoint["algorithm"]
        self.best_performance = checkpoint["best_performance"]
        logger.info(f"Meta model loaded from {path}")

class MarketMetaLearner:
    """市場特化メタラーニング"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int] = None):
        if hidden_dims is None:
            hidden_dims = [256, 256]

        # SAC Actorネットワークをベースモデルとして使用
        base_model = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim),
            nn.Tanh(),  # 行動空間を[-1, 1]に制限
        )

        self.meta_learner = MetaLearner(base_model, algorithm="maml")
        self.market_models = {}  # 市場ごとの適応済みモデル

    def add_market_data(
        self,
        market_name: str,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ):
        """市場データ追加"""
        task_data = MetaTaskData(
            states=torch.FloatTensor(states),
            actions=torch.FloatTensor(actions),
            rewards=torch.FloatTensor(rewards),
            next_states=torch.FloatTensor(next_states),
            dones=torch.FloatTensor(dones),
            task_id=market_name,
        )

        self.meta_learner.collect_task_data(task_data)

    def train_on_markets(self, num_epochs: int = 100) -> dict[str, list[float]]:
        """複数市場でのメタ学習"""

        def market_loss(outputs, actions, rewards, next_outputs, dones):
            # SACスタイルの損失（簡易版）
            action_loss = torch.mean((outputs - actions) ** 2)
            return action_loss

        return self.meta_learner.train_meta(market_loss, num_epochs)

    def adapt_to_market(
        self, market_name: str, market_data: dict[str, np.ndarray]
    ) -> nn.Module:
        """特定市場への適応"""
        task_data = MetaTaskData(
            states=torch.FloatTensor(market_data["states"]),
            actions=torch.FloatTensor(market_data["actions"]),
            rewards=torch.FloatTensor(market_data["rewards"]),
            next_states=torch.FloatTensor(market_data["next_states"]),
            dones=torch.FloatTensor(market_data["dones"]),
            task_id=market_name,
        )

        adapted_model = self.meta_learner.adapt_to_new_market(task_data)
        self.market_models[market_name] = adapted_model

        return adapted_model

    def get_market_model(self, market_name: str) -> nn.Module | None:
        """市場モデル取得"""
        return self.market_models.get(market_name)

    def predict_market_action(self, market_name: str, state: np.ndarray) -> np.ndarray:
        """市場行動予測"""
        model = self.get_market_model(market_name)
        if model is None:
            logger.warning(f"No adapted model for market: {market_name}")
            return np.zeros(self.meta_learner.base_model[-2].out_features)  # 行動次元

        model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = model(state_tensor).squeeze(0).numpy()

        return action

    def get_adaptation_stats(self) -> dict[str, Any]:
        """適応統計取得"""
        return {
            "num_markets": len(self.market_models),
            "market_names": list(self.market_models.keys()),
            "meta_performance": self.meta_learner.best_performance,
            "algorithm": self.meta_learner.algorithm,
        }
