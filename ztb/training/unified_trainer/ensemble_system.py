#!/usr/bin/env python3
"""
Ensemble System for SAC v428 - Enhanced modularity and maintainability.
"""

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TypedDict

import numpy as np

from ztb.io.json_io import read_json
from ztb.utils.file_utils import safe_json_dump
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

class ConsensusRequirementConfig(TypedDict, total=False):
    enabled: bool
    agreement_threshold: float
    force_hold_on_disagreement: bool

class StabilityVotingConfig(TypedDict, total=False):
    enabled: bool
    stability_weight: float
    performance_weight: float

class AdaptationConfig(TypedDict, total=False):
    enabled: bool
    performance_threshold: float
    rebalancing_interval: int

class PredictionInfo(TypedDict):
    action: int
    confidence: float
    specialization: str

@dataclass
class EnsembleConfig:
    """アンサンブル設定"""

    enabled: bool = True
    members: int = 5
    specializations: list[str] = field(
        default_factory=lambda: ["bull", "bear", "sideways", "high_vol", "low_vol"]
    )
    voting_mechanism: str = "weighted_confidence"
    diversity_weight: float = 0.3
    consensus_requirement: ConsensusRequirementConfig = field(
        default_factory=lambda: {
            "enabled": True,
            "agreement_threshold": 0.6,
            "force_hold_on_disagreement": True,
        }
    )
    stability_voting: StabilityVotingConfig = field(
        default_factory=lambda: {
            "enabled": True,
            "stability_weight": 0.4,
            "performance_weight": 0.6,
        }
    )
    adaptation: AdaptationConfig = field(
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
    model: object | None = None
    confidence: float = 0.5
    performance_score: float = 0.0
    stability_score: float = 0.0
    last_updated: float = field(default_factory=time.time)
    training_stats: dict[str, object] = field(default_factory=dict)

class EnsemblePredictor:
    """アンサンブル予測器 - 保守性と分析性を重視した設計"""

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.members: dict[int, EnsembleMember] = {}
        self.performance_history: list[dict[str, object]] = []
        self.decision_log: list[dict[str, object]] = []
        self.logger = get_logger(f"{__name__}.EnsemblePredictor")

        self._initialize_members()

    @staticmethod
    def _as_bool(value: object, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        return default

    @staticmethod
    def _as_float(value: object, default: float) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_int(value: object, default: int) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_object_map(value: object) -> dict[str, object]:
        if isinstance(value, dict):
            return {str(k): v for k, v in value.items()}
        return {}

    def _resolve_specialization(self, spec_name: str) -> EnsembleSpecialization:
        try:
            return EnsembleSpecialization(spec_name)
        except ValueError:
            self.logger.warning(
                "Unknown specialization '%s'; falling back to '%s'",
                spec_name,
                EnsembleSpecialization.SIDEWAYS.value,
            )
            return EnsembleSpecialization.SIDEWAYS

    def _append_bounded_record(
        self,
        records: list[dict[str, object]],
        record: dict[str, object],
        max_size: int = 10_000,
        retain_size: int = 5_000,
    ) -> None:
        records.append(record)
        if len(records) > max_size:
            del records[:-retain_size]

    def _initialize_members(self) -> None:
        """アンサンブルメンバーを初期化"""
        self.members.clear()
        requested_members = max(1, self._as_int(self.config.members, 5))
        configured_specs = (
            self.config.specializations
            if self.config.specializations
            else [spec.value for spec in EnsembleSpecialization]
        )

        for i in range(requested_members):
            spec_name = configured_specs[i % len(configured_specs)]
            member = EnsembleMember(
                id=i,
                specialization=self._resolve_specialization(spec_name),
                confidence=0.5,
                performance_score=0.5,
                stability_score=0.5,
            )
            self.members[i] = member
            self.logger.info(
                "Initialized ensemble member %s: %s", i, member.specialization.value
            )

    def add_member(self, member: EnsembleMember) -> None:
        """新しいメンバーを追加"""
        self.members[member.id] = member
        self.logger.info(
            f"Added ensemble member {member.id}: {member.specialization.value}"
        )

    def remove_member(self, member_id: int) -> None:
        """メンバーを削除"""
        if member_id in self.members:
            del self.members[member_id]
            self.logger.info(f"Removed ensemble member {member_id}")

    def update_member_performance(
        self, member_id: int, performance: float, stability: float | None = None
    ) -> None:
        """メンバーのパフォーマンスを更新"""
        if member_id in self.members:
            member = self.members[member_id]
            member.performance_score = performance
            if stability is not None:
                member.stability_score = stability
            member.last_updated = time.time()

            # 信頼度を更新
            member.confidence = self._calculate_confidence(member)
            self._append_bounded_record(
                self.performance_history,
                {
                    "timestamp": member.last_updated,
                    "member_id": member_id,
                    "performance_score": member.performance_score,
                    "stability_score": member.stability_score,
                    "confidence": member.confidence,
                },
            )

    def _calculate_confidence(self, member: EnsembleMember) -> float:
        """メンバーの信頼度を計算"""
        # パフォーマンスと安定性の加重平均
        perf_weight = self._as_float(
            self.config.stability_voting.get("performance_weight", 0.6), 0.6
        )
        stab_weight = self._as_float(
            self.config.stability_voting.get("stability_weight", 0.4), 0.4
        )

        confidence = (
            member.performance_score * perf_weight
            + member.stability_score * stab_weight
        )

        return min(max(confidence, 0.0), 1.0)

    def predict(self, observation: np.ndarray) -> tuple[int, dict[str, object]]:
        """
        アンサンブル予測を実行

        Args:
            observation: 観測データ

        Returns:
            tuple[int, dict[str, object]]: (予測アクション, 分析情報)
        """
        if not self.members:
            self.logger.warning(
                "No ensemble members available, returning default action"
            )
            return 0, {"error": "no_members"}

        # 各メンバーの予測を取得
        predictions: dict[int, PredictionInfo] = {}
        member_info: dict[int, dict[str, object]] = {}

        for member_id, member in self.members.items():
            try:
                prediction = int(self._get_member_prediction(member, observation))
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
                self.logger.error("Error getting prediction from member %s: %s", member_id, e)
                continue

        if not predictions:
            return 0, {"error": "no_valid_predictions"}

        # 投票メカニズムに基づいて最終予測を決定
        final_action, analysis = self._aggregate_predictions(predictions)

        # 決定をログに記録
        decision_record = {
            "timestamp": time.time(),
            "observation_shape": list(observation.shape),
            "predictions": predictions,
            "member_info": member_info,
            "final_action": final_action,
            "analysis": analysis,
        }
        self._append_bounded_record(self.decision_log, decision_record)

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
                return int(np.random.choice([-1, 0, 1]))

        # 実際のモデル予測（未実装）
        return 0

    def _aggregate_predictions(
        self, predictions: dict[int, PredictionInfo]
    ) -> tuple[int, dict[str, object]]:
        """予測を集約して最終決定を下す"""
        try:
            mechanism = VotingMechanism(self.config.voting_mechanism)
        except ValueError:
            self.logger.warning(
                "Unknown voting mechanism '%s'; using majority vote",
                self.config.voting_mechanism,
            )
            return self._majority_vote(predictions)

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
        self, predictions: dict[int, PredictionInfo]
    ) -> tuple[int, dict[str, object]]:
        """多数決"""
        action_counts: dict[int, int] = {}
        total_confidence = 0.0

        for pred_info in predictions.values():
            action = pred_info["action"]
            confidence = float(pred_info["confidence"])

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
        self, predictions: dict[int, PredictionInfo]
    ) -> tuple[int, dict[str, object]]:
        """信頼度加重投票"""
        action_weights: dict[int, float] = {}
        total_weight = 0.0

        for pred_info in predictions.values():
            action = pred_info["action"]
            confidence = float(pred_info["confidence"])

            if action not in action_weights:
                action_weights[action] = 0.0
            action_weights[action] += confidence
            total_weight += confidence

        if total_weight <= 0:
            fallback_action, fallback_analysis = self._majority_vote(predictions)
            fallback_analysis["method"] = "weighted_confidence_fallback_majority"
            fallback_analysis["reason"] = "non_positive_total_weight"
            return fallback_action, fallback_analysis

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
        self, predictions: dict[int, PredictionInfo]
    ) -> tuple[int, dict[str, object]]:
        """合意ベース投票"""
        consensus_config = self.config.consensus_requirement

        if not self._as_bool(consensus_config.get("enabled", False), False):
            return self._majority_vote(predictions)

        threshold = self._as_float(consensus_config.get("agreement_threshold", 0.6), 0.6)
        threshold = min(max(threshold, 0.0), 1.0)
        total_members = len(predictions)

        action_counts: dict[int, int] = {}
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
        force_hold = self._as_bool(
            consensus_config.get("force_hold_on_disagreement", True), True
        )
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
        self, predictions: dict[int, PredictionInfo]
    ) -> tuple[int, dict[str, object]]:
        """安定性加重投票"""
        stability_config = self.config.stability_voting
        stability_weight = self._as_float(
            stability_config.get("stability_weight", 0.4), 0.4
        )
        performance_weight = self._as_float(
            stability_config.get("performance_weight", 0.6), 0.6
        )

        action_scores: dict[int, float] = {}
        member_stability: dict[int, float] = {}

        # 各メンバーの安定性情報を取得
        for member_id, pred_info in predictions.items():
            if member_id in self.members:
                member = self.members[member_id]
                stability = member.stability_score
                confidence = float(pred_info["confidence"])
                action = pred_info["action"]

                # 安定性と信頼度の組み合わせスコア
                score = (
                    stability * stability_weight
                    + confidence * performance_weight
                )

                if action not in action_scores:
                    action_scores[action] = 0.0
                action_scores[action] += score

                member_stability[member_id] = stability

        if not action_scores:
            fallback_action, fallback_analysis = self._majority_vote(predictions)
            fallback_analysis["method"] = "stability_weighted_fallback_majority"
            fallback_analysis["reason"] = "empty_action_scores"
            return fallback_action, fallback_analysis

        final_action = max(action_scores, key=action_scores.get)

        analysis = {
            "method": "stability_weighted",
            "action_scores": action_scores,
            "member_stability": member_stability,
            "stability_weights": stability_config,
        }

        return final_action, analysis

    def get_ensemble_stats(self) -> dict[str, object]:
        """アンサンブルの統計情報を取得（分析用）"""
        if not self.members:
            return {"error": "no_members"}

        member_stats: dict[int, dict[str, object]] = {}
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

    def adapt_ensemble(self, market_conditions: dict[str, object]) -> None:
        """市場条件に基づいてアンサンブルを適応"""
        if not self._as_bool(self.config.adaptation.get("enabled", False), False):
            return

        # 市場条件に基づく適応ロジック
        # （実装は必要に応じて拡張）
        self.logger.info(f"Adapting ensemble to market conditions: {market_conditions}")

    def save_ensemble_state(self, path: str) -> None:
        """アンサンブルの状態を保存"""
        state = {
            "config": asdict(self.config),
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

        safe_json_dump(state, path, indent=2)

        self.logger.info(f"Ensemble state saved to {path}")

    def _coerce_consensus_requirement(
        self, raw: object, fallback: ConsensusRequirementConfig
    ) -> ConsensusRequirementConfig:
        source = self._as_object_map(raw)
        return {
            "enabled": self._as_bool(source.get("enabled"), fallback.get("enabled", True)),
            "agreement_threshold": min(
                max(
                    self._as_float(
                        source.get("agreement_threshold"),
                        fallback.get("agreement_threshold", 0.6),
                    ),
                    0.0,
                ),
                1.0,
            ),
            "force_hold_on_disagreement": self._as_bool(
                source.get("force_hold_on_disagreement"),
                fallback.get("force_hold_on_disagreement", True),
            ),
        }

    def _coerce_stability_voting(
        self, raw: object, fallback: StabilityVotingConfig
    ) -> StabilityVotingConfig:
        source = self._as_object_map(raw)
        return {
            "enabled": self._as_bool(source.get("enabled"), fallback.get("enabled", True)),
            "stability_weight": self._as_float(
                source.get("stability_weight"),
                fallback.get("stability_weight", 0.4),
            ),
            "performance_weight": self._as_float(
                source.get("performance_weight"),
                fallback.get("performance_weight", 0.6),
            ),
        }

    def _coerce_adaptation_config(
        self, raw: object, fallback: AdaptationConfig
    ) -> AdaptationConfig:
        source = self._as_object_map(raw)
        return {
            "enabled": self._as_bool(source.get("enabled"), fallback.get("enabled", True)),
            "performance_threshold": self._as_float(
                source.get("performance_threshold"),
                fallback.get("performance_threshold", 0.7),
            ),
            "rebalancing_interval": max(
                1,
                self._as_int(
                    source.get("rebalancing_interval"),
                    fallback.get("rebalancing_interval", 1000),
                ),
            ),
        }

    def _coerce_config(self, raw_config: object) -> EnsembleConfig:
        base = EnsembleConfig()
        config_map = self._as_object_map(raw_config)

        raw_specializations = config_map.get("specializations")
        if isinstance(raw_specializations, list):
            specializations = [str(spec) for spec in raw_specializations if str(spec)]
        else:
            specializations = list(base.specializations)
        if not specializations:
            specializations = [spec.value for spec in EnsembleSpecialization]

        return EnsembleConfig(
            enabled=self._as_bool(config_map.get("enabled"), base.enabled),
            members=max(1, self._as_int(config_map.get("members"), base.members)),
            specializations=specializations,
            voting_mechanism=str(
                config_map.get("voting_mechanism", base.voting_mechanism)
            ),
            diversity_weight=self._as_float(
                config_map.get("diversity_weight"), base.diversity_weight
            ),
            consensus_requirement=self._coerce_consensus_requirement(
                config_map.get("consensus_requirement"),
                base.consensus_requirement,
            ),
            stability_voting=self._coerce_stability_voting(
                config_map.get("stability_voting"),
                base.stability_voting,
            ),
            adaptation=self._coerce_adaptation_config(
                config_map.get("adaptation"),
                base.adaptation,
            ),
        )

    def _coerce_member_record(self, raw_member: object) -> EnsembleMember | None:
        member_map = self._as_object_map(raw_member)
        if not member_map:
            return None

        member_id = self._as_int(member_map.get("id"), -1)
        if member_id < 0:
            return None

        specialization = self._resolve_specialization(
            str(member_map.get("specialization", EnsembleSpecialization.SIDEWAYS.value))
        )
        return EnsembleMember(
            id=member_id,
            specialization=specialization,
            confidence=min(
                max(self._as_float(member_map.get("confidence"), 0.5), 0.0), 1.0
            ),
            performance_score=self._as_float(member_map.get("performance_score"), 0.5),
            stability_score=self._as_float(member_map.get("stability_score"), 0.5),
            last_updated=self._as_float(member_map.get("last_updated"), time.time()),
        )

    def _coerce_record_list(self, raw_records: object) -> list[dict[str, object]]:
        if not isinstance(raw_records, list):
            return []
        records: list[dict[str, object]] = []
        for item in raw_records:
            if isinstance(item, dict):
                records.append({str(k): v for k, v in item.items()})
        return records

    def load_ensemble_state(self, path: str) -> None:
        """アンサンブルの状態を読み込み"""
        try:
            state = read_json(path)
        except Exception as e:
            self.logger.error("Failed to load ensemble state from %s: %s", path, e)
            return
        if not isinstance(state, dict):
            self.logger.warning("Invalid ensemble state format: expected dict, got %s", type(state).__name__)
            return

        # 設定を復元
        self.config = self._coerce_config(state.get("config"))

        # メンバーを復元
        self.members = {}
        members_map = state.get("members", {})
        if isinstance(members_map, dict):
            for member_data in members_map.values():
                member = self._coerce_member_record(member_data)
                if member is not None:
                    self.members[member.id] = member

        if not self.members:
            self.logger.warning(
                "No valid members found in state file; reinitializing defaults"
            )
            self._initialize_members()

        # ログを復元
        self.decision_log = self._coerce_record_list(state.get("decision_log"))[-5000:]
        self.performance_history = self._coerce_record_list(
            state.get("performance_history")
        )[-5000:]

        self.logger.info(f"Ensemble state loaded from {path}")
