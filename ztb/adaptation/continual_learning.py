"""
Continual Learning for SAC v421
長期的な知識蓄積とモデル劣化防止
"""

import copy
import gc
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ztb.utils.logging_utils import get_logger
from ztb.utils.memory_utils import OperationMemoryTracker

logger = get_logger(__name__)


@dataclass
class TaskData:
    """タスクデータ"""

    task_id: str
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    num_samples: int


@dataclass
class ContinualLearningConfig:
    """継続学習設定"""

    method: str = "ewc"  # 'ewc', 'rehearsal', 'progressive'
    ewc_lambda: float = 0.1  # EWC正則化強度
    rehearsal_buffer_size: int = 1000  # リハーサル用バッファサイズ
    progressive_network_growth: float = 0.5  # Progressive NNの成長率
    memory_importance_threshold: float = 0.8  # メモリ重要度の閾値
    max_tasks_in_memory: int = 5  # メモリに保持する最大タスク数
    enable_memory_tracking: bool = True


class ElasticWeightConsolidation:
    """Elastic Weight Consolidation (EWC)"""

    def __init__(self, model: nn.Module, config: ContinualLearningConfig):
        self.model = model
        self.config = config
        self.fisher_information = {}  # フィッシャー情報行列
        self.optimal_params = {}  # 各タスクの最適パラメータ
        self.task_count = 0
        self.memory_tracker = OperationMemoryTracker() if config.enable_memory_tracking else None

    def consolidate_task(
        self, task_data: TaskData, loss_fn: Callable
    ) -> Dict[str, Any]:
        """タスクの統合（重要パラメータの保存）"""
        logger.info(f"Consolidating task: {task_data.task_id}")

        # 現在のモデルを保存
        current_params = {
            name: param.clone() for name, param in self.model.named_parameters()
        }

        # タスクでの学習（簡易版）
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.model.train()

        # フィッシャー情報行列の計算
        fisher_info = self._compute_fisher_information(task_data, loss_fn)

        # 最適パラメータ保存
        self.optimal_params[task_data.task_id] = current_params
        self.fisher_information[task_data.task_id] = fisher_info
        self.task_count += 1

        # メモリ管理
        if self.memory_tracker:
            # MemoryTrackerはコンテキストマネージャーなので、直接使用せずログのみ
            logger.debug(f"Task consolidated: {task_data.task_id}")

        # 古いタスクの削除（メモリ節約）
        if len(self.optimal_params) > self.config.max_tasks_in_memory:
            oldest_task = list(self.optimal_params.keys())[0]
            del self.optimal_params[oldest_task]
            del self.fisher_information[oldest_task]
            logger.info(f"Removed old task from memory: {oldest_task}")

        return {
            "task_id": task_data.task_id,
            "fisher_norm": torch.norm(
                torch.cat([f.flatten() for f in fisher_info.values()])
            ),
            "param_count": len(current_params),
        }

    def _compute_fisher_information(
        self, task_data: TaskData, loss_fn: Callable
    ) -> Dict[str, torch.Tensor]:
        """フィッシャー情報行列の計算"""
        fisher_info = {}

        # パラメータごとの勾配を計算
        self.model.eval()
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param.grad.zero_()

        # 少量のデータで勾配を計算
        sample_size = min(10, task_data.num_samples)  # メモリ節約のため少量使用
        indices = torch.randperm(task_data.num_samples)[:sample_size]

        # テンソルを直接使用（memory_efficient_processingはnumpy用）
        states = task_data.states[indices]
        actions = task_data.actions[indices]
        rewards = task_data.rewards[indices]
        next_states = task_data.next_states[indices]
        dones = task_data.dones[indices]

        outputs = self.model(states)
        loss = loss_fn(outputs, actions, rewards, self.model(next_states), dones)
        loss.backward()

        # 勾配の二乗をフィッシャー情報として保存
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                fisher_info[name] = param.grad.data.clone() ** 2
            else:
                fisher_info[name] = torch.zeros_like(param)

        # 勾配をクリア
        self.model.zero_grad()

        return fisher_info

    def regularization_loss(
        self, current_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """EWC正則化損失の計算"""
        if not self.optimal_params:
            return torch.tensor(0.0)

        total_loss = torch.tensor(0.0)

        for task_id, optimal_params in self.optimal_params.items():
            fisher_info = self.fisher_information[task_id]

            for name, param in current_params.items():
                if name in optimal_params and name in fisher_info:
                    optimal_param = optimal_params[name]
                    fisher_value = fisher_info[name]

                    # EWC損失: λ * F * (θ - θ*)²
                    param_diff = (param - optimal_param) ** 2
                    loss_contribution = fisher_value * param_diff
                    total_loss += loss_contribution.sum()

        return self.config.ewc_lambda * total_loss


class RehearsalBuffer:
    """リハーサル用バッファ（過去データの保存）"""

    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.buffer = deque(maxlen=config.rehearsal_buffer_size)
        self.task_buffers = defaultdict(
            lambda: deque(maxlen=config.rehearsal_buffer_size // 5)
        )  # タスクごとのバッファ
        self.memory_tracker = OperationMemoryTracker() if config.enable_memory_tracking else None

    def add_samples(self, task_data: TaskData):
        """サンプルの追加"""
        # タスク全体のバッファ
        for i in range(
            min(50, task_data.num_samples)
        ):  # メモリ節約のため最大50サンプル
            sample = {
                "states": task_data.states[i].clone(),
                "actions": task_data.actions[i].clone(),
                "rewards": task_data.rewards[i].clone(),
                "next_states": task_data.next_states[i].clone(),
                "dones": task_data.dones[i].clone(),
                "task_id": task_data.task_id,
            }
            self.buffer.append(sample)

        # タスク固有のバッファ
        task_samples = []
        for i in range(min(20, task_data.num_samples)):  # タスクごと最大20サンプル
            sample = {
                "states": task_data.states[i].clone(),
                "actions": task_data.actions[i].clone(),
                "rewards": task_data.rewards[i].clone(),
                "next_states": task_data.next_states[i].clone(),
                "dones": task_data.dones[i].clone(),
            }
            task_samples.append(sample)

        self.task_buffers[task_data.task_id].extend(task_samples)

        # メモリ管理
        if self.memory_tracker:
            logger.debug(f"Added {len(task_samples)} samples to rehearsal buffer")

    def get_rehearsal_batch(
        self, batch_size: int = 32
    ) -> Optional[Dict[str, torch.Tensor]]:
        """リハーサル用バッチの取得"""
        if len(self.buffer) < batch_size:
            return None

        # ランダムサンプリング
        indices = torch.randperm(len(self.buffer))[:batch_size]
        batch_samples = [self.buffer[i] for i in indices]

        # バッチ作成
        batch = {
            "states": torch.stack([s["states"] for s in batch_samples]),
            "actions": torch.stack([s["actions"] for s in batch_samples]),
            "rewards": torch.stack([s["rewards"] for s in batch_samples]),
            "next_states": torch.stack([s["next_states"] for s in batch_samples]),
            "dones": torch.stack([s["dones"] for s in batch_samples]),
        }

        return batch

    def get_task_rehearsal_batch(
        self, task_id: str, batch_size: int = 16
    ) -> Optional[Dict[str, torch.Tensor]]:
        """タスク固有のリハーサルバッチ取得"""
        if (
            task_id not in self.task_buffers
            or len(self.task_buffers[task_id]) < batch_size
        ):
            return None

        task_buffer = self.task_buffers[task_id]
        indices = torch.randperm(len(task_buffer))[:batch_size]
        batch_samples = [task_buffer[i] for i in indices]

        batch = {
            "states": torch.stack([s["states"] for s in batch_samples]),
            "actions": torch.stack([s["actions"] for s in batch_samples]),
            "rewards": torch.stack([s["rewards"] for s in batch_samples]),
            "next_states": torch.stack([s["next_states"] for s in batch_samples]),
            "dones": torch.stack([s["dones"] for s in batch_samples]),
        }

        return batch

    def get_buffer_stats(self) -> Dict[str, Any]:
        """バッファ統計取得"""
        return {
            "total_samples": len(self.buffer),
            "task_buffers": {
                task_id: len(buffer) for task_id, buffer in self.task_buffers.items()
            },
            "max_buffer_size": self.config.rehearsal_buffer_size,
        }


class ProgressiveNetwork:
    """Progressive Neural Networks"""

    def __init__(self, base_model: nn.Module, config: ContinualLearningConfig):
        self.base_model = base_model
        self.config = config
        self.task_networks = {}  # タスクごとのネットワーク
        self.task_count = 0
        self.lateral_connections = {}  # ネットワーク間の横接続
        self.memory_tracker = OperationMemoryTracker() if config.enable_memory_tracking else None

    def add_task_network(self, task_id: str) -> nn.Module:
        """新規タスク用のネットワーク追加"""
        # ベースネットワークのコピー
        task_network = copy.deepcopy(self.base_model)

        # ネットワーク拡張（オプション）
        if self.config.progressive_network_growth > 0:
            growth_factor = 1 + self.config.progressive_network_growth * self.task_count
            task_network = self._expand_network(task_network, growth_factor)

        self.task_networks[task_id] = task_network
        self.task_count += 1

        # 横接続の設定（前のタスクからの知識転移）
        if self.task_count > 1:
            self._setup_lateral_connections(task_id)

        logger.info(f"Added progressive network for task: {task_id}")
        return task_network

    def _expand_network(self, network: nn.Module, growth_factor: float) -> nn.Module:
        """ネットワークの拡張"""
        # 簡易的な拡張：一部の層のサイズを増加
        expanded_network = copy.deepcopy(network)

        for name, module in expanded_network.named_modules():
            if isinstance(module, nn.Linear):
                in_features = int(module.in_features * growth_factor)
                out_features = int(module.out_features * growth_factor)

                # 新しい層を作成
                new_layer = nn.Linear(in_features, out_features)
                # 重みをコピー（可能な範囲で）
                with torch.no_grad():
                    min_in = min(module.in_features, in_features)
                    min_out = min(module.out_features, out_features)
                    new_layer.weight[:min_out, :min_in] = module.weight[
                        :min_out, :min_in
                    ]
                    if module.bias is not None and new_layer.bias is not None:
                        new_layer.bias[:min_out] = module.bias[:min_out]

                # モジュールを置き換え
                if "." in name:
                    parent_name, child_name = name.rsplit(".", 1)
                    parent = expanded_network
                    for part in parent_name.split("."):
                        parent = getattr(parent, part)
                    setattr(parent, child_name, new_layer)
                else:
                    # トップレベルモジュールの場合
                    setattr(expanded_network, name, new_layer)

        return expanded_network

    def _setup_lateral_connections(self, new_task_id: str):
        """横接続の設定"""
        # 簡易実装：前のタスクからの知識転移用接続
        prev_tasks = list(self.task_networks.keys())[:-1]  # 最後のタスク以外

        for prev_task_id in prev_tasks:
            # 横接続の設定（実際の実装ではアダプター層などを使用）
            self.lateral_connections[f"{prev_task_id}_to_{new_task_id}"] = {
                "source_task": prev_task_id,
                "target_task": new_task_id,
                "connection_type": "knowledge_transfer",
            }

    def forward_with_lateral(self, task_id: str, x: torch.Tensor) -> torch.Tensor:
        """横接続を含むフォワードパス"""
        if task_id not in self.task_networks:
            raise ValueError(f"Task network not found: {task_id}")

        task_network = self.task_networks[task_id]
        output = task_network(x)

        # 横接続からの知識統合（簡易版）
        if task_id in self.lateral_connections:
            # 前のタスクからの出力を統合
            lateral_contributions = []
            for connection_key, connection_info in self.lateral_connections.items():
                if connection_info["target_task"] == task_id:
                    source_task = connection_info["source_task"]
                    if source_task in self.task_networks:
                        source_output = self.task_networks[source_task](x)
                        # 簡易的な統合（平均など）
                        lateral_contributions.append(source_output * 0.1)  # 小さな重み

            if lateral_contributions:
                lateral_combined = torch.stack(lateral_contributions).mean(dim=0)
                output = output + lateral_combined

        return output

    def get_network_stats(self) -> Dict[str, Any]:
        """ネットワーク統計取得"""
        stats = {
            "task_count": self.task_count,
            "task_networks": list(self.task_networks.keys()),
            "lateral_connections": len(self.lateral_connections),
        }

        if self.memory_tracker:
            stats["memory_stats"] = {"note": "MemoryTracker context manager used"}

        return stats


class ContinualLearner:
    """継続学習統合クラス"""

    def __init__(self, model: nn.Module, config: ContinualLearningConfig):
        self.model = model
        self.config = config
        self.current_task = None

        # 手法ごとのコンポーネント初期化
        if config.method == "ewc":
            self.ewc = ElasticWeightConsolidation(model, config)
            self.rehearsal_buffer = None
            self.progressive_net = None
        elif config.method == "rehearsal":
            self.ewc = None
            self.rehearsal_buffer = RehearsalBuffer(config)
            self.progressive_net = None
        elif config.method == "progressive":
            self.ewc = None
            self.rehearsal_buffer = None
            self.progressive_net = ProgressiveNetwork(model, config)
        else:
            raise ValueError(f"Unsupported continual learning method: {config.method}")

        self.task_history = []
        self.memory_tracker = OperationMemoryTracker() if config.enable_memory_tracking else None

    def learn_task(
        self,
        task_data: TaskData,
        loss_fn: Callable,
        optimizer: optim.Optimizer,
        num_epochs: int = 10,
    ) -> Dict[str, Any]:
        """新規タスクの学習"""
        logger.info(
            f"Learning task: {task_data.task_id} using method: {self.config.method}"
        )

        # タスク固有のモデル取得/設定
        if self.config.method == "progressive":
            task_model = self.progressive_net.add_task_network(task_data.task_id)
        else:
            task_model = self.model

        # タスクデータの追加（リハーサル用）
        if self.rehearsal_buffer:
            self.rehearsal_buffer.add_samples(task_data)

        # 学習実行
        training_stats = self._train_task_model(
            task_model, task_data, loss_fn, optimizer, num_epochs
        )

        # タスク統合（EWC用）
        if self.ewc:
            consolidation_stats = self.ewc.consolidate_task(task_data, loss_fn)
            training_stats.update(consolidation_stats)

        # 履歴記録
        self.task_history.append(
            {
                "task_id": task_data.task_id,
                "method": self.config.method,
                "training_stats": training_stats,
                "timestamp": torch.cuda.Event(enable_timing=True).elapsed_time(
                    torch.cuda.Event(enable_timing=True)
                )
                if torch.cuda.is_available()
                else 0,
            }
        )

        self.current_task = task_data.task_id

        # メモリクリーンアップ
        if self.memory_tracker:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return training_stats

    def _train_task_model(
        self,
        task_model: nn.Module,
        task_data: TaskData,
        loss_fn: Callable,
        optimizer: optim.Optimizer,
        num_epochs: int,
    ) -> Dict[str, Any]:
        """タスクモデルの学習"""
        task_model.train()
        training_losses = []

        # データローダー作成
        dataset = TensorDataset(
            task_data.states,
            task_data.actions,
            task_data.rewards,
            task_data.next_states,
            task_data.dones,
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(num_epochs):
            epoch_loss = 0
            batch_count = 0

            for batch in dataloader:
                states, actions, rewards, next_states, dones = batch

                # 通常の損失
                optimizer.zero_grad()
                outputs = task_model(states)
                loss = loss_fn(
                    outputs, actions, rewards, task_model(next_states), dones
                )

                # 継続学習の正則化損失追加
                if self.ewc:
                    current_params = {
                        name: param for name, param in task_model.named_parameters()
                    }
                    regularization_loss = self.ewc.regularization_loss(current_params)
                    loss += regularization_loss

                # リハーサル損失追加
                if self.rehearsal_buffer and epoch % 2 == 0:  # 2エポックに1回
                    rehearsal_batch = self.rehearsal_buffer.get_rehearsal_batch(
                        batch_size=16
                    )
                    if rehearsal_batch:
                        rehearsal_outputs = task_model(rehearsal_batch["states"])
                        rehearsal_loss = loss_fn(
                            rehearsal_outputs,
                            rehearsal_batch["actions"],
                            rehearsal_batch["rewards"],
                            task_model(rehearsal_batch["next_states"]),
                            rehearsal_batch["dones"],
                        )
                        loss += 0.1 * rehearsal_loss  # 小さな重み

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            training_losses.append(avg_loss)

            if (epoch + 1) % 5 == 0:
                logger.debug(
                    f"Task {task_data.task_id}, epoch {epoch+1}/{num_epochs}, loss: {avg_loss:.6f}"
                )

        return {
            "final_loss": training_losses[-1] if training_losses else 0,
            "avg_loss": np.mean(training_losses),
            "training_losses": training_losses,
            "epochs_completed": len(training_losses),
        }

    def predict_with_continual(
        self, state: torch.Tensor, task_id: Optional[str] = None
    ) -> torch.Tensor:
        """継続学習を考慮した予測"""
        if task_id is None:
            task_id = self.current_task

        if self.config.method == "progressive" and task_id:
            return self.progressive_net.forward_with_lateral(task_id, state)
        else:
            return self.model(state)

    def get_continual_stats(self) -> Dict[str, Any]:
        """継続学習統計取得"""
        stats = {
            "method": self.config.method,
            "current_task": self.current_task,
            "task_history": len(self.task_history),
            "learned_tasks": [task["task_id"] for task in self.task_history],
        }

        if self.ewc:
            stats["ewc_tasks"] = len(self.ewc.optimal_params)

        if self.rehearsal_buffer:
            stats["rehearsal_stats"] = self.rehearsal_buffer.get_buffer_stats()

        if self.progressive_net:
            stats["progressive_stats"] = self.progressive_net.get_network_stats()

        if self.memory_tracker:
            stats["memory_stats"] = {"note": "MemoryTracker context manager used"}

        return stats

    def save_continual_state(self, path: str):
        """継続学習状態の保存"""
        state = {
            "config": self.config,
            "current_task": self.current_task,
            "task_history": self.task_history,
            "method": self.config.method,
        }

        # 手法固有の状態保存
        if self.ewc:
            state["ewc_params"] = self.ewc.optimal_params
            state["ewc_fisher"] = self.ewc.fisher_information

        if self.rehearsal_buffer:
            # バッファはメモリのため保存しない（再構築が必要）
            state["rehearsal_config"] = self.rehearsal_buffer.config

        torch.save(state, path)
        logger.info(f"Continual learning state saved to {path}")

    def load_continual_state(self, path: str):
        """継続学習状態の読み込み"""
        state = torch.load(path)

        self.config = state["config"]
        self.current_task = state["current_task"]
        self.task_history = state["task_history"]

        # 手法固有の状態復元
        if self.ewc and "ewc_params" in state:
            self.ewc.optimal_params = state["ewc_params"]
            self.ewc.fisher_information = state["ewc_fisher"]
            self.ewc.task_count = len(self.ewc.optimal_params)

        logger.info(f"Continual learning state loaded from {path}")
