"""
Federated Learning for SAC v421
FedAvgアルゴリズムと差分プライバシーによる分散トレーニング
"""

import copy
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from torch.utils.data import DataLoader

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class FederatedConfig:
    """フェデレーテッドラーニング設定"""

    num_clients: int = 5
    num_rounds: int = 10
    client_fraction: float = 1.0  # 各ラウンドで参加するクライアントの割合
    local_epochs: int = 5
    local_batch_size: int = 32
    local_learning_rate: float = 0.01
    global_learning_rate: float = 1.0  # FedAvgでの重み付け

    # 差分プライバシー設定
    enable_privacy: bool = True
    privacy_budget: float = 1.0  # ε (epsilon)
    delta: float = 1e-5  # δ (delta)
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0

    # クライアント設定
    client_data_sizes: Optional[List[int]] = None  # 各クライアントのデータサイズ
    enable_client_weighting: bool = True  # データサイズによる重み付け


@dataclass
class ClientUpdate:
    """クライアント更新情報"""

    client_id: int
    model_state: Dict[str, torch.Tensor]
    num_samples: int
    loss_history: List[float]
    privacy_spent: Optional[float] = None


@dataclass
class FederatedRoundResult:
    """フェデレーテッドラウンド結果"""

    round_number: int
    participating_clients: List[int]
    global_loss: float
    client_updates: List[ClientUpdate]
    privacy_budget_spent: float
    convergence_metrics: Dict[str, float]


class FederatedClient:
    """フェデレーテッドクライアント"""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        config: FederatedConfig,
        local_data: Optional[DataLoader] = None,
    ):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.config = config
        self.local_data = local_data
        self.privacy_engine = None

        # プライバシー設定
        if config.enable_privacy:
            self._setup_privacy_engine()

    def _setup_privacy_engine(self):
        """プライバシーエンジン設定"""
        try:
            self.privacy_engine = PrivacyEngine()
            optimizer = optim.SGD(
                self.model.parameters(), lr=self.config.local_learning_rate
            )

            # プライバシー設定
            self.model, optimizer, self.local_data = self.privacy_engine.make_private(
                module=self.model,
                optimizer=optimizer,
                data_loader=self.local_data,
                noise_multiplier=self.config.noise_multiplier,
                max_grad_norm=self.config.max_grad_norm,
            )

            self.optimizer = optimizer

        except Exception as e:
            logger.warning(
                f"Privacy engine setup failed for client {self.client_id}: {e}"
            )
            # フォールバック: 通常のオプティマイザー
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.config.local_learning_rate
            )

    def load_global_model(self, global_state: Dict[str, torch.Tensor]):
        """グローバルモデル読み込み"""
        self.model.load_state_dict(global_state)

    def train_local_model(
        self, loss_fn: Callable, num_epochs: Optional[int] = None
    ) -> ClientUpdate:
        """ローカルモデル学習"""
        if num_epochs is None:
            num_epochs = self.config.local_epochs

        if self.local_data is None:
            logger.warning(f"Client {self.client_id}: No local data available")
            return ClientUpdate(
                client_id=self.client_id,
                model_state=self.model.state_dict(),
                num_samples=0,
                loss_history=[],
            )

        loss_history = []
        num_samples = 0

        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            batch_count = 0

            for batch_data in self.local_data:
                self.optimizer.zero_grad()

                # バッチ処理（データ形式に応じて調整）
                if isinstance(batch_data, (list, tuple)):
                    inputs, targets = batch_data
                else:
                    inputs, targets = batch_data, None

                outputs = self.model(inputs)
                if targets is not None:
                    loss = loss_fn(outputs, targets)
                else:
                    # 自己教師あり学習の場合
                    loss = loss_fn(outputs, inputs)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
                num_samples += len(inputs)

            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                loss_history.append(avg_loss)

                if (epoch + 1) % 2 == 0:
                    logger.debug(
                        f"Client {self.client_id}, epoch {epoch+1}/{num_epochs}, loss: {avg_loss:.6f}"
                    )

        # プライバシー消費量取得
        privacy_spent = None
        if self.privacy_engine is not None:
            privacy_spent = self.privacy_engine.accountant.get_privacy_spent(
                delta=self.config.delta
            )[0]  # ε値

        return ClientUpdate(
            client_id=self.client_id,
            model_state=self.model.state_dict(),
            num_samples=num_samples,
            loss_history=loss_history,
            privacy_spent=privacy_spent,
        )

    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """モデル状態取得"""
        return self.model.state_dict()


class FedAvgServer:
    """FedAvgサーバー"""

    def __init__(self, global_model: nn.Module, config: FederatedConfig):
        self.global_model = global_model
        self.config = config
        self.clients = []
        self.round_results = []
        self.global_optimizer = optim.SGD(
            global_model.parameters(), lr=config.global_learning_rate
        )

    def add_client(self, client: FederatedClient):
        """クライアント追加"""
        self.clients.append(client)

    def initialize_clients(self, client_data_loaders: List[DataLoader]):
        """クライアント初期化"""
        if len(client_data_loaders) != len(self.clients):
            raise ValueError("Number of data loaders must match number of clients")

        for client, data_loader in zip(self.clients, client_data_loaders):
            client.local_data = data_loader

    def select_clients(self) -> List[FederatedClient]:
        """ラウンド参加クライアント選択"""
        num_participants = max(1, int(len(self.clients) * self.config.client_fraction))
        selected_clients = random.sample(self.clients, num_participants)
        return selected_clients

    def aggregate_updates(
        self, client_updates: List[ClientUpdate]
    ) -> Dict[str, torch.Tensor]:
        """クライアント更新の集約（FedAvg）"""
        if not client_updates:
            return self.global_model.state_dict()

        # 重み計算（データサイズによる重み付け）
        total_samples = sum(update.num_samples for update in client_updates)
        if total_samples == 0:
            # データサイズが不明な場合は均等重み
            weights = [1.0 / len(client_updates)] * len(client_updates)
        else:
            weights = [update.num_samples / total_samples for update in client_updates]

        # パラメータ集約
        aggregated_state = {}
        param_names = client_updates[0].model_state.keys()

        for param_name in param_names:
            weighted_sum = None

            for update, weight in zip(client_updates, weights):
                param = update.model_state[param_name]

                if weighted_sum is None:
                    weighted_sum = param * weight
                else:
                    weighted_sum += param * weight

            aggregated_state[param_name] = weighted_sum

        return aggregated_state

    def run_federated_round(self, loss_fn: Callable) -> FederatedRoundResult:
        """フェデレーテッドラウンド実行"""
        round_num = len(self.round_results) + 1
        logger.info(f"Starting federated round {round_num}/{self.config.num_rounds}")

        # 参加クライアント選択
        participating_clients = self.select_clients()
        client_ids = [client.client_id for client in participating_clients]

        # グローバルモデルをクライアントに配布
        global_state = self.global_model.state_dict()
        for client in participating_clients:
            client.load_global_model(global_state)

        # クライアントローカル学習
        client_updates = []
        for client in participating_clients:
            logger.debug(f"Training client {client.client_id}")
            update = client.train_local_model(loss_fn)
            client_updates.append(update)

        # 更新集約
        aggregated_state = self.aggregate_updates(client_updates)
        self.global_model.load_state_dict(aggregated_state)

        # グローバル損失計算
        global_loss = self._compute_global_loss(loss_fn, participating_clients)

        # 収束メトリクス計算
        convergence_metrics = self._compute_convergence_metrics(client_updates)

        # プライバシー消費量計算
        privacy_budget_spent = sum(
            update.privacy_spent
            for update in client_updates
            if update.privacy_spent is not None
        )

        result = FederatedRoundResult(
            round_number=round_num,
            participating_clients=client_ids,
            global_loss=global_loss,
            client_updates=client_updates,
            privacy_budget_spent=privacy_budget_spent,
            convergence_metrics=convergence_metrics,
        )

        self.round_results.append(result)

        logger.info(
            f"Round {round_num} completed. Global loss: {global_loss:.6f}, "
            f"Privacy spent: {privacy_budget_spent:.6f}"
        )

        return result

    def _compute_global_loss(
        self, loss_fn: Callable, clients: List[FederatedClient]
    ) -> float:
        """グローバル損失計算"""
        self.global_model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for client in clients:
                if client.local_data is not None:
                    client_loss = 0
                    client_samples = 0

                    for batch_data in client.local_data:
                        if isinstance(batch_data, (list, tuple)):
                            inputs, targets = batch_data
                        else:
                            inputs, targets = batch_data, None

                        outputs = self.global_model(inputs)
                        if targets is not None:
                            loss = loss_fn(outputs, targets)
                        else:
                            loss = loss_fn(outputs, inputs)

                        client_loss += loss.item() * len(inputs)
                        client_samples += len(inputs)

                    if client_samples > 0:
                        total_loss += client_loss
                        total_samples += client_samples

        return total_loss / total_samples if total_samples > 0 else 0.0

    def _compute_convergence_metrics(
        self, client_updates: List[ClientUpdate]
    ) -> Dict[str, float]:
        """収束メトリクス計算"""
        if not client_updates:
            return {}

        # クライアント間パラメータ分散
        param_names = client_updates[0].model_state.keys()
        param_variances = {}

        for param_name in param_names:
            params = [update.model_state[param_name] for update in client_updates]
            param_tensor = torch.stack(params)

            # パラメータの分散を計算
            variance = torch.var(param_tensor, dim=0).mean().item()
            param_variances[param_name] = variance

        avg_param_variance = np.mean(list(param_variances.values()))

        # 損失の統計
        final_losses = [
            update.loss_history[-1] for update in client_updates if update.loss_history
        ]
        loss_std = np.std(final_losses) if final_losses else 0.0

        return {
            "avg_param_variance": avg_param_variance,
            "loss_std": loss_std,
            "num_participating_clients": len(client_updates),
        }

    def get_training_history(self) -> List[FederatedRoundResult]:
        """トレーニング履歴取得"""
        return self.round_results.copy()

    def get_global_model(self) -> nn.Module:
        """グローバルモデル取得"""
        return self.global_model


class MarketFederatedLearner:
    """市場特化フェデレーテッドラーニング"""

    def __init__(
        self, base_model: nn.Module, market_configs: Dict[str, FederatedConfig]
    ):
        self.base_model = base_model
        self.market_configs = market_configs
        self.market_servers = {}
        self.market_clients = defaultdict(list)

        # 各市場のサーバー初期化
        for market_name, config in market_configs.items():
            server = FedAvgServer(copy.deepcopy(base_model), config)
            self.market_servers[market_name] = server

    def add_market_client(
        self, market_name: str, client_data: DataLoader, client_id: Optional[int] = None
    ):
        """市場クライアント追加"""
        if market_name not in self.market_servers:
            raise ValueError(f"Market {market_name} not configured")

        if client_id is None:
            client_id = len(self.market_clients[market_name])

        config = self.market_configs[market_name]
        client = FederatedClient(
            client_id, copy.deepcopy(self.base_model), config, client_data
        )

        self.market_clients[market_name].append(client)
        self.market_servers[market_name].add_client(client)

        logger.info(f"Added client {client_id} to market {market_name}")

    def train_market_federated(
        self, market_name: str, loss_fn: Callable, num_rounds: Optional[int] = None
    ) -> List[FederatedRoundResult]:
        """市場別フェデレーテッド学習"""
        if market_name not in self.market_servers:
            raise ValueError(f"Market {market_name} not configured")

        server = self.market_servers[market_name]
        if num_rounds is None:
            num_rounds = server.config.num_rounds

        results = []
        for round_num in range(num_rounds):
            result = server.run_federated_round(loss_fn)
            results.append(result)

            # 収束チェック
            if self._check_convergence(result):
                logger.info(f"Market {market_name} converged at round {round_num + 1}")
                break

        return results

    def train_all_markets(
        self, loss_fn: Callable
    ) -> Dict[str, List[FederatedRoundResult]]:
        """全市場フェデレーテッド学習"""
        results = {}

        for market_name in self.market_servers.keys():
            logger.info(f"Training federated learning for market: {market_name}")
            market_results = self.train_market_federated(market_name, loss_fn)
            results[market_name] = market_results

        return results

    def _check_convergence(self, result: FederatedRoundResult) -> bool:
        """収束チェック"""
        # パラメータ分散が閾値以下で収束と判定
        param_variance_threshold = 1e-6
        loss_std_threshold = 0.01

        variance = result.convergence_metrics.get("avg_param_variance", float("inf"))
        loss_std = result.convergence_metrics.get("loss_std", float("inf"))

        return variance < param_variance_threshold and loss_std < loss_std_threshold

    def get_market_model(self, market_name: str) -> Optional[nn.Module]:
        """市場モデル取得"""
        server = self.market_servers.get(market_name)
        return server.get_global_model() if server else None

    def aggregate_cross_market_knowledge(self) -> nn.Module:
        """クロスマーケット知識集約"""
        if not self.market_servers:
            return self.base_model

        # 全市場モデルのアンサンブル平均
        aggregated_state = {}
        market_models = [
            server.get_global_model() for server in self.market_servers.values()
        ]

        for param_name in market_models[0].state_dict().keys():
            params = [model.state_dict()[param_name] for model in market_models]
            aggregated_state[param_name] = torch.stack(params).mean(dim=0)

        # 新しいモデルに適用
        aggregated_model = copy.deepcopy(self.base_model)
        aggregated_model.load_state_dict(aggregated_state)

        logger.info(f"Aggregated knowledge from {len(market_models)} markets")
        return aggregated_model

    def get_federated_stats(self) -> Dict[str, Any]:
        """フェデレーテッド統計取得"""
        stats = {}

        for market_name, server in self.market_servers.items():
            history = server.get_training_history()
            if history:
                latest_result = history[-1]
                stats[market_name] = {
                    "rounds_completed": len(history),
                    "final_global_loss": latest_result.global_loss,
                    "total_privacy_spent": sum(r.privacy_budget_spent for r in history),
                    "num_clients": len(server.clients),
                    "convergence_metrics": latest_result.convergence_metrics,
                }

        return stats
