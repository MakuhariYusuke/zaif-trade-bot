"""
V433 Phase 5: Gradual Rollout Layer - Risk-based Allocator

リスクベースの取引量配分と段階的移行を行う。
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Awaitable, Callable

from ztb.trading.production.state_persistence import (
    read_state_payload,
    write_state_payload,
)

class AllocationStrategy(Enum):
    """配分戦略"""

    LINEAR = "linear"  # 線形増加
    EXPONENTIAL = "exponential"  # 指数増加
    STEPWISE = "stepwise"  # 段階的増加
    PERFORMANCE_BASED = "performance_based"  # パフォーマンスベース
    RISK_ADJUSTED = "risk_adjusted"  # リスク調整

class RiskThreshold(Enum):
    """リスク閾値"""

    CONSERVATIVE = "conservative"  # 保守的
    MODERATE = "moderate"  # 中間
    AGGRESSIVE = "aggressive"  # 積極的

@dataclass
class AllocationRule:
    """配分ルール"""

    rule_id: str
    system_id: str
    strategy: AllocationStrategy
    initial_percentage: float
    target_percentage: float
    increment_percentage: float
    time_window_hours: int
    risk_threshold: RiskThreshold
    conditions: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

@dataclass
class RiskMetrics:
    """リスク指標"""

    volatility: float
    max_drawdown: Decimal
    sharpe_ratio: float
    value_at_risk: Decimal
    correlation: float
    concentration_risk: float

@dataclass
class AllocationDecision:
    """配分決定"""

    decision_id: str
    timestamp: datetime
    system_id: str
    current_percentage: float
    proposed_percentage: float
    reason: str
    risk_assessment: dict[str, Any]
    approved: bool
    executed_at: datetime | None = None

@dataclass
class RolloutPhase:
    """移行フェーズ"""

    phase_id: str
    name: str
    description: str
    target_percentage: float
    duration_hours: int
    success_criteria: dict[str, Any]
    risk_limits: dict[str, Any]
    started_at: datetime | None = None
    completed_at: datetime | None = None
    status: str = "pending"

class RiskBasedAllocator:
    """
    リスクベースアロケーター

    リスク指標に基づいて取引量を段階的に配分し、
    安全な移行を実現する。
    """

    def __init__(
        self,
        risk_threshold: RiskThreshold = RiskThreshold.MODERATE,
        max_single_system_allocation: float = 0.8,
        reallocation_interval_minutes: int = 60,
    ):
        """
        初期化

        Args:
            risk_threshold: リスク閾値
            max_single_system_allocation: 単一システム最大配分率
            reallocation_interval_minutes: 再配分間隔（分）
        """
        self.risk_threshold = risk_threshold
        self.max_single_system_allocation = max_single_system_allocation
        self.reallocation_interval_minutes = reallocation_interval_minutes

        # 配分ルール
        self.allocation_rules: dict[str, AllocationRule] = {}

        # 現在の配分
        self.current_allocations: dict[str, float] = {}

        # 移行フェーズ
        self.rollout_phases: list[RolloutPhase] = []
        self.current_phase: RolloutPhase | None = None

        # 配分決定履歴
        self.allocation_decisions: list[AllocationDecision] = []

        # リスク閾値設定
        self.risk_limits = self._get_risk_limits(risk_threshold)

        # モニタリング
        self.monitoring_active = False
        self.monitoring_thread: threading.Thread | None = None
        self.last_reallocation = datetime.now()

        # コールバック
        self.allocation_callbacks: list[
            Callable[[AllocationDecision], Awaitable[None]]
        ] = []
        self.phase_callbacks: list[Callable[[RolloutPhase], Awaitable[None]]] = []

        # ロギング
        self.logger = logging.getLogger(__name__)

        self.logger.info("Risk-based Allocator initialized")

    def add_allocation_rule(self, rule: AllocationRule) -> None:
        """
        配分ルール追加

        Args:
            rule: 配分ルール
        """
        self.allocation_rules[rule.rule_id] = rule

        # 初期配分設定
        if rule.system_id not in self.current_allocations:
            self.current_allocations[rule.system_id] = rule.initial_percentage

        self.logger.info(
            f"Allocation rule added: {rule.system_id} ({rule.strategy.value})"
        )

    def define_rollout_phases(self, phases: list[RolloutPhase]) -> None:
        """
        移行フェーズ定義

        Args:
            phases: 移行フェーズリスト
        """
        self.rollout_phases = phases
        if phases and not self.current_phase:
            self.current_phase = phases[0]

        self.logger.info(f"Rollout phases defined: {len(phases)} phases")

    async def evaluate_allocation(
        self, system_id: str, risk_metrics: RiskMetrics
    ) -> AllocationDecision | None:
        """
        配分評価

        Args:
            system_id: システムID
            risk_metrics: リスク指標

        Returns:
            AllocationDecision | None: 配分決定
        """
        if system_id not in self.allocation_rules:
            return None

        rule = self.allocation_rules[system_id]
        current_allocation = self.current_allocations.get(system_id, 0.0)

        try:
            # リスク評価
            risk_assessment = self._assess_risk(risk_metrics)

            # 新配分率計算
            proposed_allocation = await self._calculate_proposed_allocation(
                system_id, rule, risk_assessment
            )

            # 制約チェック
            proposed_allocation = self._apply_allocation_constraints(
                proposed_allocation, system_id
            )

            # 決定作成
            decision = AllocationDecision(
                decision_id=f"ALLOC_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                system_id=system_id,
                current_percentage=current_allocation,
                proposed_percentage=proposed_allocation,
                reason=self._generate_allocation_reason(
                    rule, risk_assessment, proposed_allocation
                ),
                risk_assessment=risk_assessment,
                approved=self._should_approve_allocation(
                    proposed_allocation, risk_assessment
                ),
            )

            # 決定履歴保存
            self.allocation_decisions.append(decision)

            # 履歴サイズ制限
            if len(self.allocation_decisions) > 1000:
                self.allocation_decisions = self.allocation_decisions[-1000:]

            # コールバック実行
            for callback in self.allocation_callbacks:
                try:
                    await callback(decision)
                except Exception as e:
                    self.logger.error(f"Allocation callback error: {e}")

            return decision

        except Exception as e:
            self.logger.error(f"Allocation evaluation failed for {system_id}: {e}")
            return None

    async def execute_allocation_decision(self, decision: AllocationDecision) -> bool:
        """
        配分決定実行

        Args:
            decision: 配分決定

        Returns:
            bool: 実行成功フラグ
        """
        if not decision.approved:
            self.logger.warning(
                f"Cannot execute unapproved allocation decision: {decision.decision_id}"
            )
            return False

        try:
            # 配分更新
            self.current_allocations[decision.system_id] = decision.proposed_percentage
            decision.executed_at = datetime.now()

            # フェーズ進行チェック
            await self._check_phase_progression()

            self.logger.info(
                f"Allocation executed: {decision.system_id} = {decision.proposed_percentage:.1f}%"
            )

            return True

        except Exception as e:
            self.logger.error(f"Allocation execution failed: {e}")
            return False

    def _assess_risk(self, risk_metrics: RiskMetrics) -> dict[str, Any]:
        """
        リスク評価

        Args:
            risk_metrics: リスク指標

        Returns:
            dict[str, Any]: リスク評価
        """
        assessment = {
            "overall_risk_level": "low",
            "breaches": [],
            "warnings": [],
            "score": 0.0,
        }

        # リスクスコア計算
        score = 0.0

        # ボラティリティ評価
        if risk_metrics.volatility > self.risk_limits["max_volatility"]:
            assessment["breaches"].append("volatility")
            score += 0.3
        elif risk_metrics.volatility > self.risk_limits["max_volatility"] * 0.8:
            assessment["warnings"].append("volatility")
            score += 0.1

        # 最大ドローダウン評価
        if risk_metrics.max_drawdown > self.risk_limits["max_drawdown"]:
            assessment["breaches"].append("max_drawdown")
            score += 0.4
        elif risk_metrics.max_drawdown > self.risk_limits["max_drawdown"] * 0.8:
            assessment["warnings"].append("max_drawdown")
            score += 0.2

        # VaR評価
        if risk_metrics.value_at_risk < self.risk_limits["min_var"]:
            assessment["breaches"].append("value_at_risk")
            score += 0.3

        # シャープレシオ評価（低いほどリスク）
        if risk_metrics.sharpe_ratio < self.risk_limits["min_sharpe"]:
            assessment["warnings"].append("sharpe_ratio")
            score += 0.1

        # 相関係数評価（高い相関は集中リスク）
        if risk_metrics.correlation > self.risk_limits["max_correlation"]:
            assessment["warnings"].append("correlation")
            score += 0.1

        # 全体リスクレベル決定
        if score >= 0.5:
            assessment["overall_risk_level"] = "high"
        elif score >= 0.2:
            assessment["overall_risk_level"] = "medium"
        else:
            assessment["overall_risk_level"] = "low"

        assessment["score"] = score

        return assessment

    async def _calculate_proposed_allocation(
        self, system_id: str, rule: AllocationRule, risk_assessment: dict[str, Any]
    ) -> float:
        """
        新配分率計算

        Args:
            system_id: システムID
            rule: 配分ルール
            risk_assessment: リスク評価

        Returns:
            float: 提案配分率
        """
        current_allocation = self.current_allocations.get(system_id, 0.0)
        risk_level = risk_assessment["overall_risk_level"]

        # リスクレベルに基づく増分調整
        risk_multiplier = {"low": 1.0, "medium": 0.5, "high": 0.0}.get(
            risk_level, 0.0
        )  # 高リスク時は増分なし

        # 戦略別計算
        if rule.strategy == AllocationStrategy.LINEAR:
            increment = rule.increment_percentage * risk_multiplier
            proposed = min(current_allocation + increment, rule.target_percentage)

        elif rule.strategy == AllocationStrategy.EXPONENTIAL:
            # 指数関数的な増加（リスクが高いほど緩やか）
            growth_rate = rule.increment_percentage * risk_multiplier
            proposed = current_allocation * (1 + growth_rate / 100)
            proposed = min(proposed, rule.target_percentage)

        elif rule.strategy == AllocationStrategy.STEPWISE:
            # 段階的増加
            if risk_level == "low" and current_allocation < rule.target_percentage:
                proposed = min(
                    current_allocation + rule.increment_percentage,
                    rule.target_percentage,
                )
            else:
                proposed = current_allocation

        elif rule.strategy == AllocationStrategy.PERFORMANCE_BASED:
            # パフォーマンスベース（簡易実装）
            proposed = await self._performance_based_allocation(
                system_id, rule, risk_assessment
            )

        elif rule.strategy == AllocationStrategy.RISK_ADJUSTED:
            # リスク調整
            risk_adjustment = 1.0 - risk_assessment["score"]
            increment = rule.increment_percentage * risk_adjustment
            proposed = min(current_allocation + increment, rule.target_percentage)

        else:
            proposed = current_allocation

        return proposed

    async def _performance_based_allocation(
        self, system_id: str, rule: AllocationRule, risk_assessment: dict[str, Any]
    ) -> float:
        """
        パフォーマンスベース配分

        Args:
            system_id: システムID
            rule: 配分ルール
            risk_assessment: リスク評価

        Returns:
            float: 提案配分率
        """
        # 簡易実装：リスクが低い場合は積極的に増加
        current = self.current_allocations.get(system_id, 0.0)

        if risk_assessment["overall_risk_level"] == "low":
            return min(
                current + rule.increment_percentage * 1.5, rule.target_percentage
            )
        elif risk_assessment["overall_risk_level"] == "medium":
            return min(current + rule.increment_percentage, rule.target_percentage)
        else:
            return current  # 高リスク時は維持

    def _apply_allocation_constraints(self, proposed: float, system_id: str) -> float:
        """
        配分制約適用

        Args:
            proposed: 提案配分率
            system_id: システムID

        Returns:
            float: 制約適用後配分率
        """
        # 単一システム最大配分制約
        proposed = min(proposed, self.max_single_system_allocation)

        # 全体配分の制約（全システムの合計が100%を超えない）
        other_allocations = sum(
            alloc for sid, alloc in self.current_allocations.items() if sid != system_id
        )
        max_for_this_system = 1.0 - other_allocations
        proposed = min(proposed, max_for_this_system)

        # 最小配分制約
        rule = self.allocation_rules.get(system_id)
        if rule:
            proposed = max(proposed, rule.initial_percentage)

        return proposed

    def _should_approve_allocation(
        self, proposed: float, risk_assessment: dict[str, Any]
    ) -> bool:
        """
        配分承認判定

        Args:
            proposed: 提案配分率
            risk_assessment: リスク評価

        Returns:
            bool: 承認フラグ
        """
        # 高リスク時は配分増加を制限
        if risk_assessment["overall_risk_level"] == "high":
            return False

        # 重大なリスク違反時は拒否
        if risk_assessment["breaches"]:
            return False

        # 警告がある場合は慎重に
        if risk_assessment["warnings"] and risk_assessment["score"] > 0.3:
            return False

        return True

    def _generate_allocation_reason(
        self, rule: AllocationRule, risk_assessment: dict[str, Any], proposed: float
    ) -> str:
        """
        配分理由生成

        Args:
            rule: 配分ルール
            risk_assessment: リスク評価
            proposed: 提案配分率

        Returns:
            str: 配分理由
        """
        current = self.current_allocations.get(rule.system_id, 0.0)
        risk_level = risk_assessment["overall_risk_level"]

        if proposed > current:
            return f"Increasing allocation due to {risk_level} risk level (strategy: {rule.strategy.value})"
        elif proposed < current:
            return f"Decreasing allocation due to {risk_level} risk level"
        else:
            return f"Maintaining allocation (risk level: {risk_level})"

    async def _check_phase_progression(self) -> None:
        """フェーズ進行チェック"""
        if not self.current_phase:
            return

        # 現在のフェーズ目標達成チェック
        target_allocation = self.current_phase.target_percentage / 100.0
        current_total_new = sum(
            alloc for sid, alloc in self.current_allocations.items() if sid != "legacy"
        )  # legacy以外を新システムと仮定

        if current_total_new >= target_allocation:
            # フェーズ完了
            self.current_phase.completed_at = datetime.now()
            self.current_phase.status = "completed"

            # 次のフェーズへ
            current_index = self.rollout_phases.index(self.current_phase)
            if current_index + 1 < len(self.rollout_phases):
                self.current_phase = self.rollout_phases[current_index + 1]
                self.current_phase.started_at = datetime.now()
                self.current_phase.status = "active"

                # コールバック実行
                for callback in self.phase_callbacks:
                    try:
                        await callback(self.current_phase)
                    except Exception as e:
                        self.logger.error(f"Phase callback error: {e}")

                self.logger.info(
                    f"Rollout phase progressed to: {self.current_phase.name}"
                )

    def _get_risk_limits(self, threshold: RiskThreshold) -> dict[str, Any]:
        """
        リスク閾値取得

        Args:
            threshold: リスク閾値

        Returns:
            dict[str, Any]: リスク制限
        """
        limits = {
            RiskThreshold.CONSERVATIVE: {
                "max_volatility": 0.15,
                "max_drawdown": Decimal("0.05"),
                "min_var": Decimal("-0.03"),
                "min_sharpe": 0.5,
                "max_correlation": 0.7,
            },
            RiskThreshold.MODERATE: {
                "max_volatility": 0.25,
                "max_drawdown": Decimal("0.10"),
                "min_var": Decimal("-0.05"),
                "min_sharpe": 0.3,
                "max_correlation": 0.8,
            },
            RiskThreshold.AGGRESSIVE: {
                "max_volatility": 0.35,
                "max_drawdown": Decimal("0.15"),
                "min_var": Decimal("-0.08"),
                "min_sharpe": 0.1,
                "max_correlation": 0.9,
            },
        }

        return limits.get(threshold, limits[RiskThreshold.MODERATE])

    def get_current_allocations(self) -> dict[str, float]:
        """
        現在の配分取得

        Returns:
            dict[str, float]: 現在の配分
        """
        return self.current_allocations.copy()

    def get_allocation_history(
        self, system_id: str | None = None, limit: int | None = None
    ) -> list[AllocationDecision]:
        """
        配分履歴取得

        Args:
            system_id: システムID（指定なしの場合は全システム）
            limit: 取得件数制限

        Returns:
            list[AllocationDecision]: 配分決定履歴
        """
        history = self.allocation_decisions

        if system_id:
            history = [d for d in history if d.system_id == system_id]

        if limit:
            history = history[-limit:]

        return history

    def get_rollout_status(self) -> dict[str, Any]:
        """
        移行状況取得

        Returns:
            dict[str, Any]: 移行状況
        """
        return {
            "current_phase": {
                "phase_id": self.current_phase.phase_id if self.current_phase else None,
                "name": self.current_phase.name if self.current_phase else None,
                "target_percentage": self.current_phase.target_percentage
                if self.current_phase
                else None,
                "status": self.current_phase.status if self.current_phase else None,
                "started_at": self.current_phase.started_at.isoformat()
                if self.current_phase and self.current_phase.started_at
                else None,
            }
            if self.current_phase
            else None,
            "all_phases": [
                {
                    "phase_id": p.phase_id,
                    "name": p.name,
                    "target_percentage": p.target_percentage,
                    "status": p.status,
                    "started_at": p.started_at.isoformat() if p.started_at else None,
                    "completed_at": p.completed_at.isoformat()
                    if p.completed_at
                    else None,
                }
                for p in self.rollout_phases
            ],
            "current_allocations": self.current_allocations,
        }

    def start_monitoring(self) -> None:
        """モニタリング開始"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        self.logger.info("Allocation monitoring started")

    def stop_monitoring(self) -> None:
        """モニタリング停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        self.logger.info("Allocation monitoring stopped")

    def _monitoring_loop(self) -> None:
        """モニタリングループ"""
        while self.monitoring_active:
            try:
                now = datetime.now()

                # 定期的な再配分チェック
                if (
                    now - self.last_reallocation
                ).total_seconds() >= self.reallocation_interval_minutes * 60:
                    # 実際の実装ではここでリスク指標を取得して評価
                    # asyncio.run(self._perform_reallocation())
                    self.last_reallocation = now

                time.sleep(60)  # 1分間隔

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)

    def add_allocation_callback(
        self, callback: Callable[[AllocationDecision], Awaitable[None]]
    ) -> None:
        """
        配分コールバック追加

        Args:
            callback: コールバック関数
        """
        self.allocation_callbacks.append(callback)

    def add_phase_callback(
        self, callback: Callable[[RolloutPhase], Awaitable[None]]
    ) -> None:
        """
        フェーズコールバック追加

        Args:
            callback: コールバック関数
        """
        self.phase_callbacks.append(callback)

    def save_state(self, filepath: str) -> None:
        """
        状態保存

        Args:
            filepath: 保存ファイルパス
        """
        state = {
            "risk_threshold": self.risk_threshold.value,
            "max_single_system_allocation": self.max_single_system_allocation,
            "reallocation_interval_minutes": self.reallocation_interval_minutes,
            "current_allocations": self.current_allocations,
            "allocation_decisions": [
                {
                    "decision_id": d.decision_id,
                    "timestamp": d.timestamp.isoformat(),
                    "system_id": d.system_id,
                    "current_percentage": d.current_percentage,
                    "proposed_percentage": d.proposed_percentage,
                    "reason": d.reason,
                    "approved": d.approved,
                    "executed_at": d.executed_at.isoformat() if d.executed_at else None,
                }
                for d in self.allocation_decisions[-200:]  # 最新200件
            ],
            "rollout_phases": [
                {
                    "phase_id": p.phase_id,
                    "name": p.name,
                    "description": p.description,
                    "target_percentage": p.target_percentage,
                    "duration_hours": p.duration_hours,
                    "success_criteria": p.success_criteria,
                    "risk_limits": p.risk_limits,
                    "started_at": p.started_at.isoformat() if p.started_at else None,
                    "completed_at": p.completed_at.isoformat()
                    if p.completed_at
                    else None,
                    "status": p.status,
                }
                for p in self.rollout_phases
            ],
            "current_phase_id": self.current_phase.phase_id
            if self.current_phase
            else None,
            "last_reallocation": self.last_reallocation.isoformat(),
        }

        write_state_payload(filepath, state)

        self.logger.info(f"Allocator state saved to {filepath}")

    def load_state(self, filepath: str) -> bool:
        """
        状態読み込み

        Args:
            filepath: 読み込みファイルパス

        Returns:
            bool: 読み込み成功フラグ
        """
        try:
            state = read_state_payload(filepath)

            self.risk_threshold = RiskThreshold(state["risk_threshold"])
            self.max_single_system_allocation = state["max_single_system_allocation"]
            self.reallocation_interval_minutes = state["reallocation_interval_minutes"]
            self.current_allocations = state["current_allocations"]
            self.last_reallocation = datetime.fromisoformat(state["last_reallocation"])

            # リスク制限再設定
            self.risk_limits = self._get_risk_limits(self.risk_threshold)

            # 配分決定履歴復元
            self.allocation_decisions = []
            for d_data in state.get("allocation_decisions", []):
                decision = AllocationDecision(
                    decision_id=d_data["decision_id"],
                    timestamp=datetime.fromisoformat(d_data["timestamp"]),
                    system_id=d_data["system_id"],
                    current_percentage=d_data["current_percentage"],
                    proposed_percentage=d_data["proposed_percentage"],
                    reason=d_data["reason"],
                    risk_assessment={},  # 簡易復元
                    approved=d_data["approved"],
                    executed_at=datetime.fromisoformat(d_data["executed_at"])
                    if d_data["executed_at"]
                    else None,
                )
                self.allocation_decisions.append(decision)

            # 移行フェーズ復元
            self.rollout_phases = []
            for p_data in state.get("rollout_phases", []):
                phase = RolloutPhase(
                    phase_id=p_data["phase_id"],
                    name=p_data["name"],
                    description=p_data["description"],
                    target_percentage=p_data["target_percentage"],
                    duration_hours=p_data["duration_hours"],
                    success_criteria=p_data["success_criteria"],
                    risk_limits=p_data["risk_limits"],
                    started_at=datetime.fromisoformat(p_data["started_at"])
                    if p_data["started_at"]
                    else None,
                    completed_at=datetime.fromisoformat(p_data["completed_at"])
                    if p_data["completed_at"]
                    else None,
                    status=p_data["status"],
                )
                self.rollout_phases.append(phase)

            # 現在のフェーズ設定
            current_phase_id = state.get("current_phase_id")
            if current_phase_id:
                self.current_phase = next(
                    (p for p in self.rollout_phases if p.phase_id == current_phase_id),
                    None,
                )

            self.logger.info(f"Allocator state loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load allocator state: {e}")
            return False
