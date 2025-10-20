"""
Type definitions for SAC training reports and analysis.

This module provides comprehensive type definitions for SAC (Soft Actor-Critic)
training reports and analysis results to improve type safety across the codebase.
"""

from typing import Any, Dict, List, TypedDict


class Metadata(TypedDict):
    """Metadata section of training report."""

    timestamp: str
    algorithm: str
    model_name: str
    success: bool


class SACHyperparameters(TypedDict):
    """SAC hyperparameters configuration."""

    learning_rate: float
    buffer_size: int
    learning_starts: int
    batch_size: int
    tau: float
    gamma: float
    train_freq: int
    gradient_steps: int
    ent_coef: float
    target_update_interval: int
    target_entropy: float


class EnvironmentConfig(TypedDict):
    """Environment configuration."""

    initial_balance: float
    transaction_cost: float
    max_position_size: float
    enable_action_masking: bool
    use_continuous_actions: bool
    use_standardized_observations: bool
    random_start: bool
    curriculum_stage: str
    continuous_to_discrete_threshold: float


class RewardSettings(TypedDict):
    """Reward function settings."""

    reward_scale: float
    reward_clip_min: float
    reward_clip_max: float
    profit_bonuses: Dict[str, Any]  # Complex nested structure
    action_bonuses: Dict[str, Any]  # Complex nested structure
    behavior_penalties: Dict[str, Any]  # Complex nested structure
    risk_penalties: Dict[str, Any]  # Complex nested structure
    flags: Dict[str, bool]


class Configuration(TypedDict):
    """Configuration section of training report."""

    model_name: str
    algorithm: str
    total_timesteps: int
    data_source: str
    data_path: str
    data_config: Dict[str, Any]
    sac_hyperparameters: SACHyperparameters
    environment: EnvironmentConfig
    reward_settings: RewardSettings
    checkpoint_interval: int
    notes: str
    fixes_implemented: Dict[str, str]


class ActionDistribution(TypedDict):
    """Action distribution statistics."""

    HOLD: float
    BUY: float
    SELL: float


class OptimizationStats(TypedDict):
    """Optimization and performance statistics."""

    memory_stats: str
    performance_profile: str
    parallel_processing_enabled: bool
    cache_size: int
    data_optimization_applied: bool


class TrainingStats(TypedDict):
    """Training statistics section."""

    total_timesteps: int
    training_time: float
    steps_per_second: float
    model_path: str
    final_reward: float
    action_distribution: ActionDistribution
    optimization: OptimizationStats


class PerformanceMetrics(TypedDict):
    """Performance metrics section."""

    steps_per_second: float
    training_efficiency: float
    action_diversity: float
    dominant_action: str
    dominant_action_ratio: float


class SystemInfo(TypedDict):
    """System information section."""

    platform: str
    python_version: str
    cpu_count: int
    memory_total: int
    memory_available: int


class TrainingReport(TypedDict):
    """Complete training report structure."""

    metadata: Metadata
    configuration: Configuration
    training_stats: TrainingStats
    performance_metrics: PerformanceMetrics
    system_info: SystemInfo


# Analysis result types
class AnalysisSummary(TypedDict):
    """Summary of analysis results."""

    total_reports: int
    successful_runs: int
    failed_runs: int
    average_training_time: float
    average_steps_per_second: float
    dominant_actions: Dict[str, int]


class ActionPattern(TypedDict):
    """Action pattern analysis result."""

    action: str
    frequency: float
    percentage: float
    trend: str


class PerformanceComparison(TypedDict):
    """Performance comparison between runs."""

    model_name: str
    total_timesteps: int
    training_time: float
    steps_per_second: float
    action_diversity: float
    dominant_action: str


class SACAnalysisResult(TypedDict):
    """Complete SAC analysis result."""

    summary: AnalysisSummary
    action_patterns: List[ActionPattern]
    performance_comparisons: List[PerformanceComparison]
    recommendations: List[str]


class TrainingMetrics(TypedDict):
    """Training metrics extracted from reports."""

    final_episode_reward: float
    best_episode_reward: float
    training_time_seconds: float
    action_distribution: ActionDistribution
    total_timesteps: int


class ModelConfig(TypedDict):
    """Model configuration extracted from reports."""

    model_name: str
    algorithm: str
    total_timesteps: int
    data_source: str
    data_path: str
    sac_hyperparameters: SACHyperparameters
    environment: EnvironmentConfig
    reward_settings: RewardSettings


class ActionAnalysisResult(TypedDict):
    """Result of action distribution analysis."""

    total_actions: int
    action_counts: ActionDistribution
    action_percentages: Dict[str, float]
    action_ratios: Dict[str, float]
    dominant_action: str
    dominant_ratio: float
    action_diversity: float
    trading_intensity: float


class PositionHoldingAnalysis(TypedDict):
    """Result of position holding interval analysis."""

    avg_hold_interval: float
    trades_per_timestep: float
    position_turnover_rate: float
    estimated_avg_position_duration: float
    trading_frequency_category: str
