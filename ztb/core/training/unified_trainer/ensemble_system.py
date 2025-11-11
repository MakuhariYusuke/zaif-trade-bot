#!/usr/bin/env python3
"""
Ensemble System for SAC v428 - Enhanced modularity and maintainability.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class EnsembleSpecialization(Enum):
    """アンサンブルメンバーの専門化タイプ"""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"


class VotingMechanism(Enum):
    """投票メカニズムの種類"""

    MAJORITY = "majority"
    WEIGHTED_CONFIDENCE = "weighted_confidence"
    CONSENSUS = "consensus"
    STABILITY_WEIGHTED = "stability_weighted"


@dataclass
class EnsembleConfig:
    """アンサンブル設定"""

    enabled: bool = True
    members: int = 5
    specializations: List[str] = field(
        default_factory=lambda: ["bull", "bear", "sideways", "high_vol", "low_vol"]
    )
    voting_mechanism: str = "weighted_confidence"
    diversity_weight: float = 0.3
    consensus_requirement: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "agreement_threshold": 0.6,
            "force_hold_on_disagreement": True,
        }
    )
    stability_voting: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "stability_weight": 0.4,
            "performance_weight": 0.6,
        }
    )
    adaptation: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "performance_threshold": 0.7,
            "rebalancing_interval": 1000,
        }
    )


@dataclass
class EnsembleMember:
    """アンサンブルメンバーの情報"""

    id: int
    specialization: EnsembleSpecialization
    model: Optional[Any] = None
    confidence: float = 0.5
    performance_score: float = 0.0
    stability_score: float = 0.0
    last_updated: float = field(default_factory=time.time)
    training_stats: Dict[str, Any] = field(default_factory=dict)


class EnsemblePredictor:
    """アンサンブル予測器 - 保守性と分析性を重視した設計"""

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.members: Dict[int, EnsembleMember] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.decision_log: List[Dict[str, Any]] = []
        self.logger = get_logger(f"{__name__}.EnsemblePredictor")

        self._initialize_members()

    def _initialize_members(self):
        """アンサンブルメンバーを初期化"""
        for i, spec in enumerate(self.config.specializations):
            member = EnsembleMember(
                id=i,
                specialization=EnsembleSpecialization(spec),
                confidence=0.5,
                performance_score=0.5,
                stability_score=0.5,
            )
            self.members[i] = member
            self.logger.info(f"Initialized ensemble member {i}: {spec}")

    def add_member(self, member: EnsembleMember):
        """新しいメンバーを追加"""
        self.members[member.id] = member
        self.logger.info(
            f"Added ensemble member {member.id}: {member.specialization.value}"
        )

    def remove_member(self, member_id: int):
        """メンバーを削除"""
        if member_id in self.members:
            del self.members[member_id]
            self.logger.info(f"Removed ensemble member {member_id}")

    def update_member_performance(
        self, member_id: int, performance: float, stability: float = None
    ):
        """メンバーのパフォーマンスを更新"""
        if member_id in self.members:
            member = self.members[member_id]
            member.performance_score = performance
            if stability is not None:
                member.stability_score = stability
            member.last_updated = time.time()

            # 信頼度を更新
            member.confidence = self._calculate_confidence(member)

    def _calculate_confidence(self, member: EnsembleMember) -> float:
        """メンバーの信頼度を計算"""
        # パフォーマンスと安定性の加重平均
        perf_weight = self.config.stability_voting.get("performance_weight", 0.6)
        stab_weight = self.config.stability_voting.get("stability_weight", 0.4)

        confidence = (
            member.performance_score * perf_weight
            + member.stability_score * stab_weight
        )

        return min(max(confidence, 0.0), 1.0)

    def predict(self, observation: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """
        アンサンブル予測を実行

        Args:
            observation: 観測データ

        Returns:
            Tuple[int, Dict[str, Any]]: (予測アクション, 分析情報)
        """
        if not self.members:
            self.logger.warning(
                "No ensemble members available, returning default action"
            )
            return 0, {"error": "no_members"}

        # 各メンバーの予測を取得
        predictions = {}
        member_info = {}

        for member_id, member in self.members.items():
            try:
                prediction = self._get_member_prediction(member, observation)
                confidence = member.confidence

                predictions[member_id] = {
                    "action": prediction,
                    "confidence": confidence,
                    "specialization": member.specialization.value,
                }

                member_info[member_id] = {
                    "prediction": prediction,
                    "confidence": confidence,
                    "performance": member.performance_score,
                    "stability": member.stability_score,
                }

            except Exception as e:
                self.logger.error(
                    f"Error getting prediction from member {member_id}: {e}"
                )
                continue

        if not predictions:
            return 0, {"error": "no_valid_predictions"}

        # 投票メカニズムに基づいて最終予測を決定
        final_action, analysis = self._aggregate_predictions(predictions)

        # 決定をログに記録
        decision_record = {
            "timestamp": time.time(),
            "observation_shape": observation.shape,
            "predictions": predictions,
            "final_action": final_action,
            "analysis": analysis,
        }
        self.decision_log.append(decision_record)

        # ログを制限（メモリ管理）
        if len(self.decision_log) > 10000:
            self.decision_log = self.decision_log[-5000:]

        return final_action, analysis

    def _get_member_prediction(
        self, member: EnsembleMember, observation: np.ndarray
    ) -> int:
        """個別メンバーの予測を取得"""
        # 実際の実装では、ここでメンバーのモデルを使って予測を行う
        # 現在はモック実装
        if member.model is None:
            # 専門化に基づくシンプルな予測ロジック
            if member.specialization == EnsembleSpecialization.BULL:
                return 1 if np.random.random() > 0.3 else 0
            elif member.specialization == EnsembleSpecialization.BEAR:
                return -1 if np.random.random() > 0.3 else 0
            else:
                return np.random.choice([-1, 0, 1])

        # 実際のモデル予測（未実装）
        return 0

    def _aggregate_predictions(
        self, predictions: Dict[int, Dict[str, Any]]
    ) -> Tuple[int, Dict[str, Any]]:
        """予測を集約して最終決定を下す"""
        mechanism = VotingMechanism(self.config.voting_mechanism)

        if mechanism == VotingMechanism.MAJORITY:
            return self._majority_vote(predictions)
        elif mechanism == VotingMechanism.WEIGHTED_CONFIDENCE:
            return self._weighted_confidence_vote(predictions)
        elif mechanism == VotingMechanism.CONSENSUS:
            return self._consensus_vote(predictions)
        elif mechanism == VotingMechanism.STABILITY_WEIGHTED:
            return self._stability_weighted_vote(predictions)
        else:
            self.logger.warning(
                f"Unknown voting mechanism: {mechanism}, using majority vote"
            )
            return self._majority_vote(predictions)

    def _majority_vote(
        self, predictions: Dict[int, Dict[str, Any]]
    ) -> Tuple[int, Dict[str, Any]]:
        """多数決"""
        action_counts = {}
        total_confidence = 0

        for pred_info in predictions.values():
            action = pred_info["action"]
            confidence = pred_info["confidence"]

            if action not in action_counts:
                action_counts[action] = 0
            action_counts[action] += 1
            total_confidence += confidence

        # 最多のアクションを選択
        final_action = max(action_counts, key=action_counts.get)

        analysis = {
            "method": "majority",
            "action_counts": action_counts,
            "total_members": len(predictions),
            "avg_confidence": total_confidence / len(predictions) if predictions else 0,
        }

        return final_action, analysis

    def _weighted_confidence_vote(
        self, predictions: Dict[int, Dict[str, Any]]
    ) -> Tuple[int, Dict[str, Any]]:
        """信頼度加重投票"""
        action_weights = {}
        total_weight = 0

        for pred_info in predictions.values():
            action = pred_info["action"]
            confidence = pred_info["confidence"]

            if action not in action_weights:
                action_weights[action] = 0
            action_weights[action] += confidence
            total_weight += confidence

        # 重みが最も高いアクションを選択
        final_action = max(action_weights, key=action_weights.get)

        analysis = {
            "method": "weighted_confidence",
            "action_weights": action_weights,
            "total_weight": total_weight,
            "normalized_weights": {
                k: v / total_weight for k, v in action_weights.items()
            },
        }

        return final_action, analysis

    def _consensus_vote(
        self, predictions: Dict[int, Dict[str, Any]]
    ) -> Tuple[int, Dict[str, Any]]:
        """合意ベース投票"""
        consensus_config = self.config.consensus_requirement

        if not consensus_config.get("enabled", False):
            return self._majority_vote(predictions)

        threshold = consensus_config.get("agreement_threshold", 0.6)
        total_members = len(predictions)

        action_counts = {}
        for pred_info in predictions.values():
            action = pred_info["action"]
            action_counts[action] = action_counts.get(action, 0) + 1

        # 合意アクションを探す
        for action, count in action_counts.items():
            agreement_ratio = count / total_members
            if agreement_ratio >= threshold:
                analysis = {
                    "method": "consensus",
                    "consensus_action": action,
                    "agreement_ratio": agreement_ratio,
                    "threshold": threshold,
                    "force_hold": False,
                }
                return action, analysis

        # 合意が得られなかった場合
        force_hold = consensus_config.get("force_hold_on_disagreement", True)
        final_action = 0 if force_hold else max(action_counts, key=action_counts.get)

        analysis = {
            "method": "consensus",
            "consensus_reached": False,
            "action_counts": action_counts,
            "threshold": threshold,
            "force_hold": force_hold,
        }

        return final_action, analysis

    def _stability_weighted_vote(
        self, predictions: Dict[int, Dict[str, Any]]
    ) -> Tuple[int, Dict[str, Any]]:
        """安定性加重投票"""
        stability_config = self.config.stability_voting

        action_scores = {}
        member_stability = {}

        # 各メンバーの安定性情報を取得
        for member_id, pred_info in predictions.items():
            if member_id in self.members:
                member = self.members[member_id]
                stability = member.stability_score
                confidence = pred_info["confidence"]
                action = pred_info["action"]

                # 安定性と信頼度の組み合わせスコア
                score = stability * stability_config.get(
                    "stability_weight", 0.4
                ) + confidence * stability_config.get("performance_weight", 0.6)

                if action not in action_scores:
                    action_scores[action] = 0
                action_scores[action] += score

                member_stability[member_id] = stability

        final_action = max(action_scores, key=action_scores.get)

        analysis = {
            "method": "stability_weighted",
            "action_scores": action_scores,
            "member_stability": member_stability,
            "stability_weights": stability_config,
        }

        return final_action, analysis

    def get_ensemble_stats(self) -> Dict[str, Any]:
        """アンサンブルの統計情報を取得（分析用）"""
        if not self.members:
            return {"error": "no_members"}

        member_stats = {}
        for member_id, member in self.members.items():
            member_stats[member_id] = {
                "specialization": member.specialization.value,
                "confidence": member.confidence,
                "performance_score": member.performance_score,
                "stability_score": member.stability_score,
                "last_updated": member.last_updated,
            }

        overall_stats = {
            "total_members": len(self.members),
            "avg_confidence": np.mean([m.confidence for m in self.members.values()]),
            "avg_performance": np.mean(
                [m.performance_score for m in self.members.values()]
            ),
            "avg_stability": np.mean(
                [m.stability_score for m in self.members.values()]
            ),
            "decision_log_size": len(self.decision_log),
            "performance_history_size": len(self.performance_history),
        }

        return {
            "member_stats": member_stats,
            "overall_stats": overall_stats,
            "config": {
                "voting_mechanism": self.config.voting_mechanism,
                "diversity_weight": self.config.diversity_weight,
                "consensus_enabled": self.config.consensus_requirement.get(
                    "enabled", False
                ),
            },
        }

    def adapt_ensemble(self, market_conditions: Dict[str, Any]):
        """市場条件に基づいてアンサンブルを適応"""
        if not self.config.adaptation.get("enabled", False):
            return

        # 市場条件に基づく適応ロジック
        # （実装は必要に応じて拡張）
        self.logger.info(f"Adapting ensemble to market conditions: {market_conditions}")

    def save_ensemble_state(self, path: str):
        """アンサンブルの状態を保存"""
        state = {
            "config": self.config.__dict__,
            "members": {
                k: {
                    "id": v.id,
                    "specialization": v.specialization.value,
                    "confidence": v.confidence,
                    "performance_score": v.performance_score,
                    "stability_score": v.stability_score,
                    "last_updated": v.last_updated,
                }
                for k, v in self.members.items()
            },
            "decision_log": self.decision_log[-1000:],  # 最新1000件のみ保存
            "performance_history": self.performance_history[-1000:],
        }

        import json

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        self.logger.info(f"Ensemble state saved to {path}")

    def load_ensemble_state(self, path: str):
        """アンサンブルの状態を読み込み"""
        import json

        with open(path, "r") as f:
            state = json.load(f)

        # 設定を復元
        config_dict = state.get("config", {})
        self.config = EnsembleConfig(**config_dict)

        # メンバーを復元
        self.members = {}
        for member_data in state.get("members", {}).values():
            member = EnsembleMember(
                id=member_data["id"],
                specialization=EnsembleSpecialization(member_data["specialization"]),
                confidence=member_data["confidence"],
                performance_score=member_data["performance_score"],
                stability_score=member_data["stability_score"],
                last_updated=member_data["last_updated"],
            )
            self.members[member.id] = member

        # ログを復元
        self.decision_log = state.get("decision_log", [])
        self.performance_history = state.get("performance_history", [])

        self.logger.info(f"Ensemble state loaded from {path}")
