"""
Signal Reward Integrator - Unified Signal Integration for Reward Functions

This module provides unified integration between technical signals and reward functions,
combining ActionSignalGuide functionality with reward calculation logic.
Enhanced to support new pattern recognition systems (Granville's Law, Dow Theory, Heikin-Ashi).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuide,
)
from ztb.utils.logging_utils import get_logger


class SignalRewardIntegrator:
    """
    Unified integrator for technical signals and reward functions.

    This class combines signal evaluation and reward modification logic,
    providing a clean interface for signal-guided reinforcement learning.
    Enhanced to support advanced pattern recognition systems.
    """

    def __init__(
        self,
        signal_guide: ActionSignalGuide,
        signal_bonus_weight: float = 0.1,
        signal_penalty_weight: float = 0.05,
        granville_weight: float = 1.2,  # Higher weight for volume-price analysis
        dow_theory_weight: float = 1.5,  # Higher weight for multi-timeframe confirmation
        heikin_ashi_weight: float = 1.0,  # Standard weight for smoothed trend analysis
        # New technical indicator weights
        rsi_weight: float = 1.0,
        macd_weight: float = 1.2,
        atr_weight: float = 0.8,
        ichimoku_weight: float = 1.1,
        cci_weight: float = 1.0,
        stochastic_weight: float = 1.0,
        williams_r_weight: float = 1.0,
        mfi_weight: float = 1.0,
        chaikin_ad_weight: float = 1.0,
        # Bollinger Bands and ADX weights
        bollinger_weight: float = 1.3,  # Higher weight for volatility-based signals
        adx_weight: float = 1.4,  # Higher weight for trend strength signals
        enable_advanced_integration: bool = True,
    ):
        """
        Initialize signal reward integrator.

        Args:
            signal_guide: ActionSignalGuide instance for signal evaluation
            signal_bonus_weight: Base weight for signal alignment bonuses
            signal_penalty_weight: Base weight for signal contradiction penalties
            granville_weight: Multiplier for Granville's Law signals
            dow_theory_weight: Multiplier for Dow Theory signals
            heikin_ashi_weight: Multiplier for Heikin-Ashi signals
            rsi_weight: Multiplier for RSI signals
            macd_weight: Multiplier for MACD signals
            atr_weight: Multiplier for ATR signals
            ichimoku_weight: Multiplier for Ichimoku Cloud signals
            cci_weight: Multiplier for CCI signals
            stochastic_weight: Multiplier for Stochastic signals
            williams_r_weight: Multiplier for Williams %R signals
            mfi_weight: Multiplier for MFI signals
            chaikin_ad_weight: Multiplier for Chaikin AD signals
            bollinger_weight: Multiplier for Bollinger Bands signals (default: 1.3)
            adx_weight: Multiplier for ADX signals (default: 1.4)
            enable_advanced_integration: Enable advanced integration features
        """
        self.logger = get_logger("SignalRewardIntegrator")
        self.signal_guide = signal_guide
        self.signal_bonus_weight = signal_bonus_weight
        self.signal_penalty_weight = signal_penalty_weight

        # Pattern-specific weights
        self.granville_weight = granville_weight
        self.dow_theory_weight = dow_theory_weight
        self.heikin_ashi_weight = heikin_ashi_weight
        # New technical indicator weights
        self.rsi_weight = rsi_weight
        self.macd_weight = macd_weight
        self.atr_weight = atr_weight
        self.ichimoku_weight = ichimoku_weight
        self.cci_weight = cci_weight
        self.stochastic_weight = stochastic_weight
        self.williams_r_weight = williams_r_weight
        self.mfi_weight = mfi_weight
        self.chaikin_ad_weight = chaikin_ad_weight
        # Bollinger Bands and ADX weights
        self.bollinger_weight = bollinger_weight
        self.adx_weight = adx_weight
        self.enable_advanced_integration = enable_advanced_integration

        # Tracking for analysis
        self.signal_bonuses_applied = 0
        self.signal_penalties_applied = 0
        self.total_steps = 0

        # Advanced integration tracking
        self.granville_signals_used = 0
        self.dow_theory_signals_used = 0
        self.heikin_ashi_signals_used = 0
        self.multi_timeframe_confirmations = 0
        # New indicator tracking
        self.rsi_signals_used = 0
        self.macd_signals_used = 0
        self.atr_signals_used = 0
        self.ichimoku_signals_used = 0
        self.cci_signals_used = 0
        self.stochastic_signals_used = 0
        self.williams_r_signals_used = 0
        self.mfi_signals_used = 0
        self.chaikin_ad_signals_used = 0
        # Bollinger Bands and ADX tracking
        self.bollinger_signals_used = 0
        self.adx_signals_used = 0

        self.logger.info("Initialized SignalRewardIntegrator with advanced integration")

    def integrate_signal_reward(
        self, reward: float, observation: Optional[np.ndarray], action: int, step: int
    ) -> float:
        """
        Apply advanced signal integration to the reward.

        Enhanced integration that considers pattern-specific characteristics:
        - Granville's Law: Volume-price relationship analysis
        - Dow Theory: Multi-timeframe trend confirmation
        - Heikin-Ashi: Smoothed trend analysis
        - RSI: Momentum oscillator signals
        - MACD: Trend-following momentum indicator
        - ATR: Volatility-based signals
        - Ichimoku Cloud: Comprehensive trend analysis
        - CCI: Commodity Channel Index signals
        - Stochastic: Momentum oscillator signals
        - Williams %R: Momentum oscillator signals
        - MFI: Money Flow Index signals
        - Chaikin AD: Accumulation/Distribution signals
        - Bollinger Bands: Volatility-based support/resistance signals
        - ADX: Trend strength and direction signals

        Args:
            reward: Base reward before signal integration
            observation: Current market observation
            action: Action taken (0=HOLD, 1=BUY, 2=SELL)
            step: Current training step

        Returns:
            Modified reward with advanced signal integration
        """
        self.logger.debug(
            f"integrate_signal_reward called: action={action}, step={step}"
        )

        if observation is None or not self.enable_advanced_integration:
            return reward

        try:
            if self.enable_advanced_integration:
                return self._advanced_signal_integration(reward, observation, action, step)
            else:
                return self._basic_signal_integration(reward, observation, action, step)

        except Exception as e:
            self.logger.warning(f"Signal reward integration failed: {e}")
            return reward

    def _basic_signal_integration(
        self, reward: float, observation: np.ndarray, action: int, step: int
    ) -> float:
        """
        Basic signal integration using consolidated signal strength.

        Args:
            reward: Base reward
            observation: Current observation
            action: Action taken
            step: Current step

        Returns:
            Modified reward
        """
        # Get signal strength for this action
        signal_strength = self.signal_guide.get_signal_strength(
            observation, action, step
        )

        if signal_strength > 0.0:
            # Signal supports this action - apply bonus
            signal_bonus = self.signal_bonus_weight * signal_strength
            modified_reward = reward + signal_bonus
            self.signal_bonuses_applied += 1
            self.logger.debug(
                f"Applied signal bonus: {signal_bonus:.4f} (strength: {signal_strength:.4f})"
            )
        elif signal_strength < 0.0:
            # Signal contradicts this action - apply penalty
            signal_penalty = self.signal_penalty_weight * abs(signal_strength)
            modified_reward = reward - signal_penalty
            self.signal_penalties_applied += 1
            self.logger.debug(
                f"Applied signal penalty: {signal_penalty:.4f} (strength: {abs(signal_strength):.4f})"
            )
        else:
            # No signal influence
            modified_reward = reward

        self.total_steps += 1
        return modified_reward

    def _advanced_signal_integration(
        self, reward: float, observation: np.ndarray, action: int, step: int
    ) -> float:
        """
        Advanced signal integration considering pattern-specific characteristics.

        This method analyzes individual signals from different pattern recognition systems
        and applies specialized weighting and logic for each pattern type.

        Args:
            reward: Base reward
            observation: Current observation
            action: Action taken
            step: Current step

        Returns:
            Modified reward with advanced integration
        """
        try:
            # Get individual signals from the signal guide
            individual_signals = self._get_individual_signals(observation, action, step)

            if not individual_signals:
                self.total_steps += 1
                return reward

            # Analyze signals by pattern type
            pattern_analysis = self._analyze_pattern_signals(individual_signals, action)

            # Calculate advanced reward modification
            reward_modifier = self._calculate_advanced_reward_modifier(
                pattern_analysis, action
            )

            modified_reward = reward + reward_modifier

            # Update tracking
            self._update_advanced_tracking(pattern_analysis)

            self.logger.debug(
                f"Advanced integration: modifier={reward_modifier:.4f}, "
                f"patterns={len(individual_signals)}"
            )

            self.total_steps += 1
            return modified_reward

        except Exception as e:
            self.logger.warning(f"Advanced signal integration failed: {e}")
            # Fall back to basic integration
            return self._basic_signal_integration(reward, observation, action, step)

    def _get_individual_signals(
        self, observation: np.ndarray, action: int, step: int
    ) -> List[Dict[str, Any]]:
        """
        Get individual signals from different pattern recognition systems.

        Args:
            observation: Current observation
            action: Action taken
            step: Current step

        Returns:
            List of individual signal dictionaries
        """
        signals = []

        # Get signals from each pattern group
        if hasattr(self.signal_guide, 'granville_recognizers'):
            for recognizer in self.signal_guide.granville_recognizers:
                signal = self._get_recognizer_signal(recognizer, observation, action, step, "granville")
                if signal:
                    signals.append(signal)

        if hasattr(self.signal_guide, 'dow_theory_recognizers'):
            for recognizer in self.signal_guide.dow_theory_recognizers:
                signal = self._get_recognizer_signal(recognizer, observation, action, step, "dow_theory")
                if signal:
                    signals.append(signal)

        if hasattr(self.signal_guide, 'heikin_ashi_recognizers'):
            for recognizer in self.signal_guide.heikin_ashi_recognizers:
                signal = self._get_recognizer_signal(recognizer, observation, action, step, "heikin_ashi")
                if signal:
                    signals.append(signal)

        # New oscillator patterns
        if hasattr(self.signal_guide, 'oscillator_recognizers'):
            for recognizer in self.signal_guide.oscillator_recognizers:
                pattern_type = self._get_oscillator_pattern_type(recognizer)
                signal = self._get_recognizer_signal(recognizer, observation, action, step, pattern_type)
                if signal:
                    signals.append(signal)

        # New volume patterns
        if hasattr(self.signal_guide, 'volume_recognizers'):
            for recognizer in self.signal_guide.volume_recognizers:
                pattern_type = self._get_volume_pattern_type(recognizer)
                signal = self._get_recognizer_signal(recognizer, observation, action, step, pattern_type)
                if signal:
                    signals.append(signal)

        # Bollinger Bands patterns
        if hasattr(self.signal_guide, 'bollinger_recognizers'):
            for recognizer in self.signal_guide.bollinger_recognizers:
                signal = self._get_recognizer_signal(recognizer, observation, action, step, "bollinger")
                if signal:
                    signals.append(signal)

        # ADX patterns
        if hasattr(self.signal_guide, 'adx_recognizers'):
            for recognizer in self.signal_guide.adx_recognizers:
                signal = self._get_recognizer_signal(recognizer, observation, action, step, "adx")
                if signal:
                    signals.append(signal)

        return signals

    def _get_recognizer_signal(
        self, recognizer: Any, observation: np.ndarray, action: int, step: int, pattern_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get signal from a specific recognizer.

        Args:
            recognizer: Pattern recognizer instance
            observation: Current observation
            action: Action taken
            step: Current step
            pattern_type: Type of pattern ("granville", "dow_theory", "heikin_ashi")

        Returns:
            Signal dictionary or None
        """
        try:
            # Convert observation to DataFrame for recognizer
            df = self.signal_guide._observation_to_dataframe(observation)
            if df is None or df.empty:
                return None

            # Get signal from recognizer
            signal_result = recognizer.recognize(df, current_index=-1)
            if not signal_result or signal_result.strength < 0.1:
                return None

            # Convert action to direction
            action_direction = {0: 0, 1: 1, 2: -1}.get(action, 0)

            # Determine alignment
            alignment = 0
            if action_direction == signal_result.direction:
                alignment = 1  # Supportive
            elif action_direction == -signal_result.direction:
                alignment = -1  # Contradictory

            return {
                'pattern_type': pattern_type,
                'signal_type': signal_result.signal_type,
                'direction': signal_result.direction,
                'strength': signal_result.strength,
                'alignment': alignment,
                'description': signal_result.description,
                'metadata': signal_result.metadata,
            }

        except Exception as e:
            self.logger.debug(f"Failed to get signal from {recognizer.__class__.__name__}: {e}")
            return None

    def _analyze_pattern_signals(
        self, signals: List[Dict[str, Any]], action: int
    ) -> Dict[str, Any]:
        """
        Analyze signals by pattern type and calculate pattern-specific metrics.

        Args:
            signals: List of individual signals
            action: Action taken

        Returns:
            Analysis dictionary with pattern metrics
        """
        analysis = {
            'granville': {'signals': [], 'weighted_strength': 0.0, 'alignment_score': 0.0},
            'dow_theory': {'signals': [], 'weighted_strength': 0.0, 'alignment_score': 0.0},
            'heikin_ashi': {'signals': [], 'weighted_strength': 0.0, 'alignment_score': 0.0},
            'rsi': {'signals': [], 'weighted_strength': 0.0, 'alignment_score': 0.0},
            'macd': {'signals': [], 'weighted_strength': 0.0, 'alignment_score': 0.0},
            'atr': {'signals': [], 'weighted_strength': 0.0, 'alignment_score': 0.0},
            'ichimoku': {'signals': [], 'weighted_strength': 0.0, 'alignment_score': 0.0},
            'cci': {'signals': [], 'weighted_strength': 0.0, 'alignment_score': 0.0},
            'stochastic': {'signals': [], 'weighted_strength': 0.0, 'alignment_score': 0.0},
            'williams_r': {'signals': [], 'weighted_strength': 0.0, 'alignment_score': 0.0},
            'mfi': {'signals': [], 'weighted_strength': 0.0, 'alignment_score': 0.0},
            'chaikin_ad': {'signals': [], 'weighted_strength': 0.0, 'alignment_score': 0.0},
            # Bollinger Bands and ADX patterns
            'bollinger': {'signals': [], 'weighted_strength': 0.0, 'alignment_score': 0.0},
            'adx': {'signals': [], 'weighted_strength': 0.0, 'alignment_score': 0.0},
            'total_signals': len(signals),
            'supporting_signals': 0,
            'contradicting_signals': 0,
        }

        for signal in signals:
            pattern_type = signal['pattern_type']
            if pattern_type in analysis:
                analysis[pattern_type]['signals'].append(signal)

                # Apply pattern-specific weighting
                weight = self._get_pattern_weight(pattern_type)
                weighted_strength = signal['strength'] * weight

                analysis[pattern_type]['weighted_strength'] += weighted_strength
                analysis[pattern_type]['alignment_score'] += signal['alignment'] * weighted_strength

                if signal['alignment'] == 1:
                    analysis['supporting_signals'] += 1
                elif signal['alignment'] == -1:
                    analysis['contradicting_signals'] += 1

        # Calculate averages
        for pattern_type in ['granville', 'dow_theory', 'heikin_ashi', 'rsi', 'macd', 'atr', 'ichimoku', 'cci', 'stochastic', 'williams_r', 'mfi', 'chaikin_ad']:
            signal_count = len(analysis[pattern_type]['signals'])
            if signal_count > 0:
                analysis[pattern_type]['avg_strength'] = (
                    analysis[pattern_type]['weighted_strength'] / signal_count
                )
                analysis[pattern_type]['avg_alignment'] = (
                    analysis[pattern_type]['alignment_score'] / analysis[pattern_type]['weighted_strength']
                    if analysis[pattern_type]['weighted_strength'] > 0 else 0
                )

        return analysis

    def _get_pattern_weight(self, pattern_type: str) -> float:
        """
        Get the weight multiplier for a pattern type.

        Args:
            pattern_type: Type of pattern

        Returns:
            Weight multiplier
        """
        weights = {
            'granville': self.granville_weight,
            'dow_theory': self.dow_theory_weight,
            'heikin_ashi': self.heikin_ashi_weight,
            'rsi': self.rsi_weight,
            'macd': self.macd_weight,
            'atr': self.atr_weight,
            'ichimoku': self.ichimoku_weight,
            'cci': self.cci_weight,
            'stochastic': self.stochastic_weight,
            'williams_r': self.williams_r_weight,
            'mfi': self.mfi_weight,
            'chaikin_ad': self.chaikin_ad_weight,
            # Bollinger Bands and ADX weights
            'bollinger': self.bollinger_weight,
            'adx': self.adx_weight,
        }
        return weights.get(pattern_type, 1.0)

    def _calculate_advanced_reward_modifier(
        self, pattern_analysis: Dict[str, Any], action: int
    ) -> float:
        """
        Calculate advanced reward modifier based on pattern analysis.

        This method implements sophisticated logic considering:
        - Pattern-specific characteristics
        - Multi-timeframe confirmation (Dow Theory)
        - Volume-price relationships (Granville's Law)
        - Trend smoothing (Heikin-Ashi)

        Args:
            pattern_analysis: Pattern analysis results
            action: Action taken

        Returns:
            Reward modifier value
        """
        modifier = 0.0

        # Dow Theory multi-timeframe confirmation bonus
        dow_modifier = self._calculate_dow_theory_modifier(pattern_analysis)
        modifier += dow_modifier

        # Granville's Law volume-price analysis
        granville_modifier = self._calculate_granville_modifier(pattern_analysis)
        modifier += granville_modifier

        # Heikin-Ashi trend smoothing
        heikin_modifier = self._calculate_heikin_ashi_modifier(pattern_analysis)
        modifier += heikin_modifier

        # Cross-pattern synergy bonus
        synergy_modifier = self._calculate_synergy_modifier(pattern_analysis)
        modifier += synergy_modifier

        # Apply base weights
        modifier *= self.signal_bonus_weight

        # Ensure reasonable bounds
        modifier = max(-0.5, min(0.5, modifier))

        return modifier

    def _calculate_dow_theory_modifier(self, pattern_analysis: Dict[str, Any]) -> float:
        """Calculate Dow Theory specific modifier."""
        dow_data = pattern_analysis['dow_theory']
        if not dow_data['signals']:
            return 0.0

        # Dow Theory gets bonus for strong alignment (multi-timeframe confirmation)
        alignment = dow_data.get('avg_alignment', 0.0)
        strength = dow_data.get('avg_strength', 0.0)

        # Multi-timeframe confirmation is valuable
        if alignment > 0.7 and strength > 0.6:
            return 0.3  # Strong confirmation bonus
        elif alignment > 0.5 and strength > 0.4:
            return 0.15  # Moderate confirmation bonus
        elif alignment < -0.5:
            return -0.2  # Strong contradiction penalty

        return 0.0

    def _calculate_granville_modifier(self, pattern_analysis: Dict[str, Any]) -> float:
        """Calculate Granville's Law specific modifier."""
        granville_data = pattern_analysis['granville']
        if not granville_data['signals']:
            return 0.0

        # Granville's Law is good for volume-price analysis
        alignment = granville_data.get('avg_alignment', 0.0)
        strength = granville_data.get('avg_strength', 0.0)

        # Volume confirmation is valuable but can be noisy
        if alignment > 0.6 and strength > 0.5:
            return 0.2  # Good volume-price confirmation
        elif alignment < -0.6:
            return -0.15  # Volume contradiction penalty

        return 0.0

    def _calculate_heikin_ashi_modifier(self, pattern_analysis: Dict[str, Any]) -> float:
        """Calculate Heikin-Ashi specific modifier."""
        heikin_data = pattern_analysis['heikin_ashi']
        if not heikin_data['signals']:
            return 0.0

        # Heikin-Ashi provides smoothed trend signals
        alignment = heikin_data.get('avg_alignment', 0.0)
        strength = heikin_data.get('avg_strength', 0.0)

        # Trend continuation signals are moderately valuable
        if alignment > 0.5 and strength > 0.4:
            return 0.1  # Trend continuation support
        elif alignment < -0.5:
            return -0.1  # Trend contradiction penalty

        return 0.0

    def _calculate_synergy_modifier(self, pattern_analysis: Dict[str, Any]) -> float:
        """Calculate cross-pattern synergy modifier."""
        # Check if multiple pattern types agree
        supporting_patterns = 0
        contradicting_patterns = 0

        for pattern_type in ['granville', 'dow_theory', 'heikin_ashi']:
            avg_alignment = pattern_analysis[pattern_type].get('avg_alignment', 0.0)
            if avg_alignment > 0.4:
                supporting_patterns += 1
            elif avg_alignment < -0.4:
                contradicting_patterns += 1

        # Synergy bonus for multiple patterns agreeing
        if supporting_patterns >= 2:
            self.multi_timeframe_confirmations += 1
            return 0.1  # Multi-pattern confirmation bonus
        elif contradicting_patterns >= 2:
            return -0.1  # Multi-pattern contradiction penalty

        return 0.0

    def _update_advanced_tracking(self, pattern_analysis: Dict[str, Any]) -> None:
        """Update advanced integration tracking statistics."""
        for pattern_type, data in pattern_analysis.items():
            if pattern_type in [
                'granville', 
                'dow_theory', 
                'heikin_ashi', 
                'rsi', 
                'macd', 
                'atr', 
                'ichimoku', 
                'cci', 
                'stochastic', 
                'williams_r', 
                'mfi', 
                'chaikin_ad',
                # Bollinger Bands and ADX patterns
                'bollinger',
                'adx'
                ]:
                signal_count = len(data['signals'])
                if signal_count > 0:
                    if pattern_type == 'granville':
                        self.granville_signals_used += signal_count
                    elif pattern_type == 'dow_theory':
                        self.dow_theory_signals_used += signal_count
                    elif pattern_type == 'heikin_ashi':
                        self.heikin_ashi_signals_used += signal_count
                    elif pattern_type == 'rsi':
                        self.rsi_signals_used += signal_count
                    elif pattern_type == 'macd':
                        self.macd_signals_used += signal_count
                    elif pattern_type == 'atr':
                        self.atr_signals_used += signal_count
                    elif pattern_type == 'ichimoku':
                        self.ichimoku_signals_used += signal_count
                    elif pattern_type == 'cci':
                        self.cci_signals_used += signal_count
                    elif pattern_type == 'stochastic':
                        self.stochastic_signals_used += signal_count
                    elif pattern_type == 'williams_r':
                        self.williams_r_signals_used += signal_count
                    elif pattern_type == 'mfi':
                        self.mfi_signals_used += signal_count
                    elif pattern_type == 'chaikin_ad':
                        self.chaikin_ad_signals_used += signal_count
                    elif pattern_type == 'bollinger':
                        self.bollinger_signals_used += signal_count
                    elif pattern_type == 'adx':
                        self.adx_signals_used += signal_count

    def _get_oscillator_pattern_type(self, recognizer: Any) -> str:
        """Get pattern type for oscillator recognizers."""
        from .action_signal_guide.pattern_recognition.oscillator_patterns import (
            CCIRecognizer, StochasticRecognizer, WilliamsRRecognizer, MFIRecognizer
        )
        if isinstance(recognizer, CCIRecognizer):
            return "cci"
        elif isinstance(recognizer, StochasticRecognizer):
            return "stochastic"
        elif isinstance(recognizer, WilliamsRRecognizer):
            return "williams_r"
        elif isinstance(recognizer, MFIRecognizer):
            return "mfi"
        else:
            return "oscillator"

    def _get_volume_pattern_type(self, recognizer: Any) -> str:
        """Get pattern type for volume recognizers."""
        from .action_signal_guide.pattern_recognition.volume_patterns import ChaikinADRecognizer
        if isinstance(recognizer, ChaikinADRecognizer):
            return "chaikin_ad"
        else:
            return "volume"

    def get_signal_strength(
        self, observation: np.ndarray, action: int, step: int = 0
    ) -> float:
        """
        Get signal strength for a given observation and action.

        This is a convenience method that delegates to the signal guide.

        Args:
            observation: Current market observation
            action: Action taken
            step: Current training step

        Returns:
            Signal strength (-1.0 to 1.0)
        """
        return self.signal_guide.get_signal_strength(observation, action, step)

    def get_integration_stats(self) -> Dict[str, Any]:
        """
        Get integration statistics including advanced pattern tracking.

        Returns:
            Dictionary with integration statistics
        """
        total_actions = self.signal_bonuses_applied + self.signal_penalties_applied
        bonus_rate = self.signal_bonuses_applied / max(total_actions, 1)
        penalty_rate = self.signal_penalties_applied / max(total_actions, 1)

        return {
            "total_steps": self.total_steps,
            "signal_bonuses_applied": self.signal_bonuses_applied,
            "signal_penalties_applied": self.signal_penalties_applied,
            "bonus_rate": bonus_rate,
            "penalty_rate": penalty_rate,
            "signal_guide_mode": self.signal_guide.guidance_level.value
            if self.signal_guide
            else None,
            "signal_bonus_weight": self.signal_bonus_weight,
            "signal_penalty_weight": self.signal_penalty_weight,
            # Advanced integration stats
            "advanced_integration_enabled": self.enable_advanced_integration,
            "granville_signals_used": self.granville_signals_used,
            "dow_theory_signals_used": self.dow_theory_signals_used,
            "heikin_ashi_signals_used": self.heikin_ashi_signals_used,
            "multi_timeframe_confirmations": self.multi_timeframe_confirmations,
            # New indicator stats
            "rsi_signals_used": self.rsi_signals_used,
            "macd_signals_used": self.macd_signals_used,
            "atr_signals_used": self.atr_signals_used,
            "ichimoku_signals_used": self.ichimoku_signals_used,
            "cci_signals_used": self.cci_signals_used,
            "stochastic_signals_used": self.stochastic_signals_used,
            "williams_r_signals_used": self.williams_r_signals_used,
            "mfi_signals_used": self.mfi_signals_used,
            "chaikin_ad_signals_used": self.chaikin_ad_signals_used,
            # Bollinger Bands and ADX stats
            "bollinger_signals_used": self.bollinger_signals_used,
            "adx_signals_used": self.adx_signals_used,
            "pattern_weights": {
                "granville": self.granville_weight,
                "dow_theory": self.dow_theory_weight,
                "heikin_ashi": self.heikin_ashi_weight,
                "rsi": self.rsi_weight,
                "macd": self.macd_weight,
                "atr": self.atr_weight,
                "ichimoku": self.ichimoku_weight,
                "cci": self.cci_weight,
                "stochastic": self.stochastic_weight,
                "williams_r": self.williams_r_weight,
                "mfi": self.mfi_weight,
                "chaikin_ad": self.chaikin_ad_weight,
                # Bollinger Bands and ADX weights
                "bollinger": self.bollinger_weight,
                "adx": self.adx_weight,
            },
        }

    def reset_stats(self) -> None:
        """Reset integration statistics including advanced tracking."""
        self.signal_bonuses_applied = 0
        self.signal_penalties_applied = 0
        self.total_steps = 0

        # Reset advanced integration tracking
        self.granville_signals_used = 0
        self.dow_theory_signals_used = 0
        self.heikin_ashi_signals_used = 0
        self.multi_timeframe_confirmations = 0
        # Reset new indicator tracking
        self.rsi_signals_used = 0
        self.macd_signals_used = 0
        self.atr_signals_used = 0
        self.ichimoku_signals_used = 0
        self.cci_signals_used = 0
        self.stochastic_signals_used = 0
        self.williams_r_signals_used = 0
        self.mfi_signals_used = 0
        self.chaikin_ad_signals_used = 0
        # Reset Bollinger Bands and ADX tracking
        self.bollinger_signals_used = 0
        self.adx_signals_used = 0

        self.logger.debug("Reset SignalRewardIntegrator stats")


# Legacy compatibility - SignalIntegration class for backward compatibility
class SignalIntegration(SignalRewardIntegrator):
    """
    Legacy SignalIntegration class for backward compatibility.

    This class maintains the old interface while delegating to SignalRewardIntegrator.
    """

    def __init__(
        self,
        signal_guide: ActionSignalGuide,
        base_reward_function: Callable,
        signal_bonus_weight: float = 0.1,
        signal_penalty_weight: float = 0.05,
    ):
        """
        Initialize legacy signal integration.

        Args:
            signal_guide: ActionSignalGuide instance
            base_reward_function: Original reward function (ignored for compatibility)
            signal_bonus_weight: Weight for signal alignment bonuses
            signal_penalty_weight: Weight for signal contradiction penalties
        """
        super().__init__(signal_guide, signal_bonus_weight, signal_penalty_weight)
        self.base_reward_function = base_reward_function  # Kept for compatibility

    def integrated_reward_function(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        info: Dict[str, Any],
        step: int,
    ) -> float:
        """
        Legacy integrated reward function interface.

        Args:
            observation: Current observation
            action: Action taken
            reward: Current reward
            next_observation: Next observation
            done: Episode done flag
            info: Additional info
            step: Current step

        Returns:
            Modified reward
        """
        return self.integrate_signal_reward(reward, observation, action, step)
