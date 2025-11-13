#!/usr/bin/env python3
"""
SIGNAL_GUIDANCE Phase 1-4 Enhanced Backtest
Phase 1-4の改善を統合したバックテスト検証
"""

import json
import logging
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import asyncio

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import SAC

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.utils.logging_utils import setup_logging, get_logger, setup_logging_from_config
from ztb.utils.config_loader import safe_json_load

# ロギング設定 - デフォルトはINFO、configで上書き可能
logger = get_logger(__name__)
setup_logging(level=logging.INFO)

# SIGNAL_GUIDANCE統合
from ztb.trading.signal.quality_scorer import SignalQualityScorer
from ztb.trading.signal.timeframe.phase4_manager import Phase4MinuteTradingManager

# 環境をインポート
sys.path.append(str(Path(__file__).parent))

from ztb.features.unified_feature import UnifiedFeatureEngineer as V4FeatureExtractor
from ztb.config.unified_config import UnifiedConfig
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.utils.analysis_formatters import print_formatted_metrics
from backtest.data_generator import generate_synthetic_data


class SignalGuidanceBacktestEnv(HeavyTradingEnv):
    """
    SIGNAL_GUIDANCE Phase 1-4統合バックテスト環境
    """

    def __init__(self, df: pd.DataFrame, config: Dict[str, Any], initial_balance: int = 1000000):
        super().__init__(df, config, initial_balance)

        # SIGNAL_GUIDANCE統合
        self.signal_scorer = SignalQualityScorer()
        self.phase4_manager = Phase4MinuteTradingManager({
            'enable_phase4_integration': True,
            'primary_timeframe': '5m'
        })

        # 価格履歴をリアルタイムで維持（移動平均計算用）
        self.price_history: List[float] = []
        self.max_history_length = 100  # 十分な履歴を保持

        # バックテスト統計
        self.guidance_signals: List[Dict[str, Any]] = []
        self.phase4_signals: List[Dict[str, Any]] = []

        # initial_balanceを保存
        self.initial_balance: int = initial_balance

        # ログレベル設定
        self.debug_enabled = config.get('debug_logging', False)
        self.logger = get_logger(f"{__name__}.SignalGuidanceBacktestEnv")
        
        self.logger.info("🎯 SIGNAL_GUIDANCE Phase 1-4 Backtest Environment initialized")
        if self.debug_enabled:
            with open('debug_init.log', 'a') as f:
                f.write("SignalGuidanceBacktestEnv instance created\n")

    def _get_signal_guidance_score(self, observation: np.ndarray, continuous_action: float) -> Tuple[int, float]:
        """
        SIGNAL_GUIDANCEスコアを取得
        """
        self.logger.debug(f"🎯 SIGNAL_GUIDANCE: Starting score calculation for observation shape: {observation.shape}")

        try:
            # デフォルトの離散アクションを初期化
            discrete_action = continuous_to_discrete_action(continuous_action)

            # 観測データを技術指標に変換
            tech_signals = self._extract_technical_signals(observation)

            self.logger.debug(f"📊 Extracted technical signals: {tech_signals}")

            # 技術指標が十分かチェック
            if not tech_signals or len(tech_signals) < 3:
                self.logger.warning(f"Insufficient technical signals: {len(tech_signals) if tech_signals else 0}")
                return continuous_to_discrete_action(continuous_action), 50.0

            # SignalQualityScorerの内部メソッドを使用して直接スコア計算
            bollinger_score = self._calculate_bollinger_score_simple(tech_signals)
            supertrend_score = self._calculate_supertrend_score_simple(tech_signals)
            obv_score = self._calculate_obv_score_simple(tech_signals)

            # SIGNAL_GUIDANCEの原点: ゴールデンクロス・デッドクロスを追加
            ma_cross_score = self._calculate_ma_cross_score_simple(tech_signals)

            self.logger.debug(f"🔢 Individual scores - BB: {bollinger_score:.2f}, ST: {supertrend_score:.2f}, OBV: {obv_score:.2f}, MA_Cross: {ma_cross_score:.2f}")

            # 重み付き平均で最終スコア計算 (V4特徴量に合わせた重み)
            # SIGNAL_GUIDANCEの原点: 基本シグナルを重視した重み調整
            weights = {
                'bollinger': 0.2,   # BB_Position (トレンド強度)
                'supertrend': 0.3,  # Supertrend (トレンドフォロー)
                'obv': 0.2,         # OBV (出来高分析)
                'ma_cross': 0.3,    # 移動平均クロス (ゴールデンクロス・デッドクロス)
            }

            total_score = (
                bollinger_score * weights['bollinger'] +
                supertrend_score * weights['supertrend'] +
                obv_score * weights['obv'] +
                ma_cross_score * weights['ma_cross']
            )

            self.logger.debug(f"⚖️ Weighted calculation: BB({bollinger_score:.2f}*{weights['bollinger']:.1f}) + ST({supertrend_score:.2f}*{weights['supertrend']:.1f}) + OBV({obv_score:.2f}*{weights['obv']:.1f}) + MA({ma_cross_score:.2f}*{weights['ma_cross']:.1f}) = {total_score:.2f}")

            # スコアに基づいて離散アクション決定（逆転ロジック）
            # SIGNAL_GUIDANCEの設計思想に基づき、高いスコア = 強い売りシグナル
            # より保守的な取引のため、閾値をさらに調整: 70以上をSELL、30以下をBUY、30-70をHOLDとする
            if total_score >= 60:
                discrete_action = -1  # SELL (高いスコアは売りシグナル)
                self.logger.debug(f"🛒 SELL signal triggered: score {total_score:.2f} >= 60")
            elif total_score <= 40:
                discrete_action = 1   # BUY (低いスコアは買いシグナル)
                self.logger.debug(f"🛍️ BUY signal triggered: score {total_score:.2f} <= 40")
            else:
                discrete_action = 0   # HOLD (中間スコアはホールド)
                self.logger.debug(f"⏸️ HOLD signal: score {total_score:.2f} in (40, 60)")

            self.logger.info(f"🎯 SIGNAL_GUIDANCE Total Score: {total_score:.1f} (Action: {discrete_action})")

            return discrete_action, total_score

        except Exception as e:
            self.logger.error(f"❌ Error in SIGNAL_GUIDANCE scoring: {e}", exc_info=self.debug_enabled)
            return continuous_to_discrete_action(continuous_action), 50.0

    def _calculate_rsi_score_simple(self, signals):
        """簡易RSIスコア計算"""
        rsi = signals.get('rsi', 50.0)
        if rsi <= 10:  # 極端なオーバーソールド
            return 90
        elif rsi <= 30:  # 通常のオーバーソールド
            return 70
        elif rsi <= 40:  # ややオーバーソールド
            return 55
        elif rsi <= 60:  # 中立
            return 50
        elif rsi <= 70:  # ややオーバーバイト
            return 45
        elif rsi <= 90:  # 通常のオーバーバイト
            return 30
        else:  # 極端なオーバーバイト
            return 10

    def _calculate_macd_score_simple(self, signals):
        """簡易MACDスコア計算"""
        macd_line = signals.get('macd_line', 0.0)
        signal_line = signals.get('signal_line', 0.0)
        histogram = signals.get('histogram', 0.0)

        if macd_line > signal_line and histogram > 0:
            return 80  # 強い買いシグナル
        elif macd_line > signal_line:
            return 65  # 買いシグナル
        elif macd_line < signal_line and histogram < 0:
            return 20  # 強い売りシグナル
        elif macd_line < signal_line:
            return 35  # 売りシグナル
        else:
            return 50  # 中立

    def _calculate_bollinger_score_simple(self, signals: Dict[str, float]) -> float:
        """簡易ボリンジャーバンドスコア計算"""
        bb_position = signals.get('bb_position', 0.5)
        if bb_position <= 0.1:  # 下限近く
            return 80
        elif bb_position <= 0.3:  # 下限寄り
            return 65
        elif bb_position <= 0.7:  # 中間
            return 50
        elif bb_position <= 0.9:  # 上限寄り
            return 35
        else:  # 上限近く
            return 20

    def _calculate_supertrend_score_simple(self, signals: Dict[str, float]) -> float:
        """簡易スーパートレンドスコア計算"""
        supertrend_direction = signals.get('supertrend_direction', 0.0)

        # Supertrend directionは連続値なので、適切なスコアリングを行う
        # 値域分析: -1.67 ～ 0.87 の範囲を取る
        # 正の値: 上昇トレンド（高スコア = 売りシグナル）
        # 負の値: 下降トレンド（低スコア = 買いシグナル）
        # 0付近: 中立

        if supertrend_direction >= 0.5:  # 強い上昇トレンド
            return 80
        elif supertrend_direction >= 0.2:  # 上昇トレンド
            return 65
        elif supertrend_direction >= -0.2:  # 中立
            return 50
        elif supertrend_direction >= -0.5:  # 下降トレンド
            return 35
        else:  # 強い下降トレンド
            return 20

    def _calculate_obv_score_simple(self, signals: Dict[str, float]) -> float:
        """簡易OBVスコア計算"""
        obv = signals.get('obv', 0.0)
        # OBVの変化に基づいて判断（正規化されていると仮定）
        # SIGNAL_GUIDANCE設計思想: 高いスコア = 売りシグナル
        # 買い圧力が高い場合 → 低スコア（買いシグナル抑制）
        # 売り圧力が高い場合 → 高スコア（売りシグナル強化）
        if obv > 0.6:
            return 25  # 強い買い圧力 → 低スコア（買い抑制）
        elif obv > 0.4:
            return 40  # 買い圧力 → 低スコア（買い抑制）
        elif obv < 0.4:
            return 60  # 売り圧力 → 高スコア（売り強化）
        elif obv < 0.2:
            return 75  # 強い売り圧力 → 高スコア（売り強化）
        else:
            return 50  # 中立

    def _calculate_atr_score_simple(self, signals):
        """簡易ATRスコア計算"""
        atr = signals.get('atr', 0.02)
        # ATRが大きいほどボラティリティが高い
        if atr > 0.05:  # 高ボラティリティ
            return 70
        elif atr > 0.03:  # 中ボラティリティ
            return 55
        elif atr > 0.01:  # 低ボラティリティ
            return 45
        else:  # 非常に低いボラティリティ
            return 30

    def _calculate_trend_score_simple(self, signals):
        """簡易トレンドスコア計算"""
        trend_strength = signals.get('trend_strength', 0.0)
        if trend_strength > 0.7:  # 強い上昇トレンド
            return 80
        elif trend_strength > 0.3:  # 上昇トレンド
            return 65
        elif trend_strength > -0.3:  # 横ばい
            return 50
        elif trend_strength > -0.7:  # 下落トレンド
            return 35
        else:  # 強い下落トレンド
            return 20

    def _calculate_momentum_score_simple(self, signals):
        """簡易モメンタムスコア計算"""
        momentum = signals.get('momentum', 0.0)
        if momentum > 0.02:  # 強い上昇モメンタム
            return 75
        elif momentum > 0.01:  # 上昇モメンタム
            return 60
        elif momentum > -0.01:  # 中立
            return 50
        elif momentum > -0.02:  # 下落モメンタム
            return 40
        else:  # 強い下落モメンタム
            return 25

    def _calculate_ma_cross_score_simple(self, signals: Dict[str, float]) -> float:
        """
        簡易移動平均クロススコア計算（SIGNAL_GUIDANCEの原点: ゴールデンクロス・デッドクロス）
        """
        try:
            golden_cross = signals.get('golden_cross', 0.0)
            dead_cross = signals.get('dead_cross', 0.0)
            sma_trend = signals.get('sma_trend', 0.0)

            # SIGNAL_GUIDANCEの設計思想に基づき、高いスコア = 強い売りシグナル
            # ゴールデンクロス（短期SMAが長期SMAを上抜く）= 買いシグナル → 低スコア
            # デッドクロス（短期SMAが長期SMAを下抜く）= 売りシグナル → 高スコア

            if dead_cross > 0.5:  # デッドクロス発生
                return 80  # 強い売りシグナル
            elif golden_cross > 0.5:  # ゴールデンクロス発生
                return 20  # 買いシグナル（売りシグナルを抑制）
            elif sma_trend > 0.5:  # 上昇トレンド
                return 45  # 弱い売りシグナル（上昇トレンド中は売りを控えめ）
            elif sma_trend < -0.5:  # 下降トレンド
                return 65  # 売りシグナル強化
            else:  # 中立
                return 50

        except Exception as e:
            self.logger.warning(f"Error calculating MA cross score: {e}")
            return 50

    def _extract_technical_signals(self, observation: np.ndarray) -> Dict[str, float]:
        """
        観測データから技術指標を抽出
        SIGNAL_GUIDANCEの原点: ゴールデンクロス・デッドクロスなどの基本シグナル
        """
        self.logger.debug(f"🔍 Extracting technical signals from observation shape: {observation.shape}")

        try:
            tech_signals = {}

            # observationがnumpy配列の場合
            if isinstance(observation, np.ndarray):
                # V4FeatureExtractorの実際の特徴量名を使用
                # V4FeatureExtractorは以下の特徴量を生成: ['Supertrend', 'Supertrend_Direction', 'OBV']
                v4_feature_names = ['Supertrend', 'Supertrend_Direction', 'OBV']

                self.logger.debug(f"📋 V4 features expected: {v4_feature_names}")

                # 各特徴量を直接マッピング
                if len(observation) >= 3:
                    tech_signals['supertrend'] = float(observation[0])  # Supertrend
                    tech_signals['supertrend_direction'] = float(observation[1])  # Supertrend_Direction
                    tech_signals['obv'] = float(observation[2])  # OBV

                    # BB_Positionは利用できないので、Supertrendに基づいて代替
                    # Supertrend_DirectionをBB_Positionとして使用（トレンドの強さを示す）
                    tech_signals['bb_position'] = (tech_signals['supertrend_direction'] + 1.0) / 2.0  # -1~1 を 0~1 に正規化

                    self.logger.debug(f"✅ Extracted signals: supertrend={tech_signals['supertrend']:.4f}, direction={tech_signals['supertrend_direction']:.4f}, obv={tech_signals['obv']:.4f}, bb_position={tech_signals['bb_position']:.4f}")
                else:
                    self.logger.warning(f"Observation too short: {len(observation)}, expected at least 3")

                # SIGNAL_GUIDANCEの原点: ゴールデンクロス・デッドクロスを実装
                # 観測データから価格情報を取得（observationの構造による）
                if len(observation) >= 4:
                    # observation[3] を現在の価格として仮定
                    current_price = float(observation[3]) if len(observation) > 3 else 100.0

                    # 移動平均線を計算するための価格履歴が必要
                    # 簡易実装: 直近の価格変化からSMAを推定
                    tech_signals.update(self._calculate_moving_averages(current_price))

            else:
                self.logger.warning(f"Unsupported observation type: {type(observation)}")
                return tech_signals

            self.logger.debug(f"📊 Final extracted technical signals: {list(tech_signals.keys())}")
            return tech_signals

        except Exception as e:
            self.logger.error(f"❌ Error extracting technical signals: {e}", exc_info=self.debug_enabled)
            return {}

    def _calculate_moving_averages(self, current_price: float) -> Dict[str, float]:
        """
        移動平均線を計算（SIGNAL_GUIDANCEの原点: ゴールデンクロス・デッドクロス用）
        """
        try:
            # リアルタイム価格履歴を使用
            if len(self.price_history) >= 25:  # 最低25期間の履歴が必要
                recent_prices = np.array(self.price_history[-25:])  # 直近25期間

                # SMA計算
                sma_short = np.mean(recent_prices[-5:])   # 5期間SMA
                sma_medium = np.mean(recent_prices[-20:]) # 20期間SMA

                # ゴールデンクロス・デッドクロス検出
                prev_sma_short = np.mean(recent_prices[-6:-1])   # 1期間前の5SMA
                prev_sma_medium = np.mean(recent_prices[-21:-1]) # 1期間前の20SMA

                # クロスオーバー検出
                short_cross_up = prev_sma_short <= prev_sma_medium and sma_short > sma_medium  # ゴールデンクロス
                short_cross_down = prev_sma_short >= prev_sma_medium and sma_short < sma_medium  # デッドクロス

                return {
                    'sma_short': sma_short,
                    'sma_medium': sma_medium,
                    'golden_cross': 1.0 if short_cross_up else 0.0,
                    'dead_cross': 1.0 if short_cross_down else 0.0,
                    'sma_trend': 1.0 if sma_short > sma_medium else (-1.0 if sma_short < sma_medium else 0.0)
                }

            # 履歴が不十分な場合はフォールバック
            return {
                'sma_short': current_price,
                'sma_medium': current_price,
                'golden_cross': 0.0,
                'dead_cross': 0.0,
                'sma_trend': 0.0
            }

            return {
                'sma_short': sma_short,
                'sma_medium': sma_medium,
                'golden_cross': 0.0,  # 簡易実装ではクロス検出なし
                'dead_cross': 0.0,
                'sma_trend': base_trend
            }

        except Exception as e:
            self.logger.warning(f"Error calculating moving averages: {e}")
            return {
                'sma_short': current_price,
                'sma_medium': current_price,
                'golden_cross': 0.0,
                'dead_cross': 0.0,
                'sma_trend': 0.0
            }

    async def _get_phase4_signal(self, symbol='BTC/JPY'):
        """
        Phase 4分足シグナルを取得
        """
        logger = logging.getLogger(__name__)

        try:
            # 現在のポートフォリオ状態
            portfolio = {
                'position': self.position,
                'cash': self.cash,
                'value': self.portfolio_value
            }

            # Phase 4マネージャーでシグナル処理
            action, score, details = await self.phase4_manager.process_minute_signal(
                symbol=symbol,
                continuous_action=0.0,  # 中立アクション
                portfolio=portfolio
            )

            return action, score, details

        except Exception as e:
            logger.warning(f"Error in Phase 4 signal processing: {e}")
            return 0, 50.0, {'status': 'error'}

    def step(self, action):
        """
        SIGNAL_GUIDANCE統合ステップ
        """
        self.logger.info("=== SIGNAL_GUIDANCE STEP METHOD CALLED ===")
        if self.debug_enabled:
            with open('debug_step.log', 'a') as f:
                f.write(f"=== SIGNAL_GUIDANCE STEP METHOD CALLED at step {self.current_step} ===\n")

        # 型安全チェック
        if not isinstance(action, (int, np.ndarray)):
            self.logger.error(f"Invalid action type: {type(action)}, expected int or np.ndarray")
            raise TypeError(f"Action must be int or np.ndarray, got {type(action)}")

        # Get current observation for SIGNAL_GUIDANCE
        observation = self._get_observation()

        # 現在の価格を価格履歴に追加（移動平均計算用）
        if hasattr(observation, '__len__') and len(observation) > 3:
            current_price = float(observation[3])
            self.price_history.append(current_price)
            # 履歴長を制限
            if len(self.price_history) > self.max_history_length:
                self.price_history.pop(0)

        self.logger.debug(f"📊 Step called with action: {action}, observation shape: {observation.shape if hasattr(observation, 'shape') else 'no shape'}")

        try:
            # SIGNAL_GUIDANCE統合（修正版）
            try:
                if hasattr(observation, '__len__') and len(observation) > 0:
                    guidance_action, guidance_score = self._get_signal_guidance_score(
                        observation, action
                    )
                else:
                    guidance_action, guidance_score = continuous_to_discrete_action(action), 50.0
                    self.logger.warning("Empty observation, using fallback action")

                # デバッグ: SIGNAL_GUIDANCEスコアとアクションを確認
                self.logger.info(f"🎯 SIGNAL_GUIDANCE: action={action} -> guidance_action={guidance_action}, guidance_score={guidance_score:.2f}")
                if self.debug_enabled:
                    with open('debug_score.log', 'a') as f:
                        f.write(f"Step {self.current_step}: action={action}, guidance_action={guidance_action}, guidance_score={guidance_score:.2f}\n")

                # 型安全チェック
                if not isinstance(guidance_action, int):
                    self.logger.error(f"guidance_action must be int, got {type(guidance_action)}")
                    raise TypeError(f"guidance_action must be int, got {type(guidance_action)}")
                if not isinstance(guidance_score, (int, float)):
                    self.logger.error(f"guidance_score must be numeric, got {type(guidance_score)}")
                    raise TypeError(f"guidance_score must be numeric, got {type(guidance_score)}")

                # Phase 4シグナル取得（非同期）
                # Note: 同期コンテキストなので簡易的に処理
                phase4_action, phase4_score, phase4_details = 0, 50.0, {'status': 'sync_context'}

                # SIGNAL_GUIDANCEアクションを使用（強制適用）
                final_action = guidance_action  # SIGNAL_GUIDANCE有効化

                # デバッグ: 最終アクションを確認
                self.logger.info(f"✅ FINAL ACTION: final_action={final_action} (from guidance_action={guidance_action})")
                if self.debug_enabled:
                    with open('debug_final.log', 'a') as f:
                        f.write(f"Step {self.current_step}: final_action={final_action}\n")

                self.logger.debug(f"🎯 SIGNAL_GUIDANCE: Original action {action:.3f} -> Guidance action {guidance_action} (score: {guidance_score:.1f})")

                # SIGNAL_GUIDANCEスコアに基づくポジションサイズ調整
                # より保守的なアプローチ: スコアが高いほど小さなポジションサイズ
                base_max_position = getattr(self.config, 'max_position_size', 0.05)
                if guidance_score >= 70:
                    adjusted_max_position = base_max_position * 0.075  # 0.375% (高スコアでも超保守的)
                elif guidance_score >= 50:
                    adjusted_max_position = base_max_position * 0.125  # 0.625% (中程度)
                else:
                    adjusted_max_position = base_max_position * 0.0625  # 0.3125% (低スコアは最小)

                # 一時的にmax_position_sizeを調整
                original_max_position = getattr(self.config, 'max_position_size', 0.05)
                self.config.max_position_size = adjusted_max_position

                self.logger.info(f"🎯 POSITION SIZE: guidance_score={guidance_score:.2f}, adjusted_max_position={adjusted_max_position:.3f}")

                # SIGNAL_GUIDANCEアクションでステップ実行
                result = super().step(final_action)

                # max_position_sizeを元に戻す
                self.config.max_position_size = original_max_position

            except Exception as e:
                self.logger.error(f"❌ Error in SIGNAL_GUIDANCE step: {e}", exc_info=self.debug_enabled)
                final_action = action
                guidance_action = continuous_to_discrete_action(action)
                guidance_score = 50.0
                phase4_score = 50.0
                phase4_details = {'status': 'error', 'error': str(e)}

                # エラーの場合もステップ実行
                result = super().step(final_action)

            # HeavyTradingEnvは5つの値を返す (observation, reward, done, truncated, info)
            if len(result) == 5:
                observation, reward, done, truncated, info = result
            elif len(result) == 4:
                observation, reward, done, info = result
                truncated = False
            else:
                # 予期しない返り値の場合
                self.logger.warning(f"Unexpected step result length: {len(result)}")
                return result[0] if len(result) > 0 else None, 0.0, True, False, {}

            # 統計記録
            self.guidance_signals.append({
                'step': self.current_step,
                'original_action': action,
                'guidance_action': guidance_action,
                'guidance_score': guidance_score,
                'phase4_score': phase4_score,
                'portfolio_value': self.portfolio_value
            })

            # infoにSIGNAL_GUIDANCE情報を追加
            info['signal_guidance'] = {
                'action': guidance_action,
                'score': guidance_score,
                'phase4_score': phase4_score,
                'phase4_status': phase4_details.get('status', 'unknown')
            }

            self.logger.debug(f"📈 Step completed: reward={reward:.4f}, portfolio={self.portfolio_value:.2f}, done={done}")

            return observation, reward, done, truncated, info

        except Exception as e:
            self.logger.error(f"💥 Critical error in step: {e}", exc_info=self.debug_enabled)
            # フォールバック
            return None, 0.0, True, False, {'error': str(e)}


def create_signal_guidance_backtest_env(data_df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[SignalGuidanceBacktestEnv, pd.DataFrame]:
    """
    SIGNAL_GUIDANCE Phase 1-4統合バックテスト環境を作成
    """
    logger = logging.getLogger(__name__)

    # V4FeatureExtractorで特徴量を拡張
    logger.info("🔧 Applying V4FeatureExtractor with SIGNAL_GUIDANCE integration...")
    feature_extractor = V4FeatureExtractor(config=config)

    # 特徴量抽出
    enhanced_df = feature_extractor.generate_features(data_df, feature_set="full", model_type="sac")

    logger.info(f"✅ Enhanced features: {len(enhanced_df.columns)} columns")
    logger.info(f"📊 New features added: {len(enhanced_df.columns) - len(data_df.columns)}")

    # 特徴量名を取得
    feature_names = feature_extractor.get_available_features(model_type="sac")
    logger.info(f"🎯 Total features: {len(feature_names)}")

    # 環境設定
    env_config = EnvironmentConfig(
        transaction_cost=0.001,    # 0.1% 手数料
        max_position_size=0.05,    # 最大ポジションサイズ 5% (リスク低減)
        feature_names=list(enhanced_df.columns),  # データフレームの実際の特徴量を使用
        reward_scaling=1.0,
        max_steps=len(enhanced_df),
    )

    # SIGNAL_GUIDANCE統合環境の作成
    env = SignalGuidanceBacktestEnv(
        df=enhanced_df,
        config=env_config.__dict__ if hasattr(env_config, '__dict__') else env_config,
        initial_balance=1000000,  # 100万円スタート
    )

    return env, enhanced_df


def run_signal_guidance_backtest(model_path: Optional[str] = None, config_path: Optional[Path] = None, n_episodes: int = 1) -> Dict[str, Any]:
    """
    SIGNAL_GUIDANCE Phase 1-4統合バックテストを実行
    """
    logger = get_logger(__name__)
    logger.info("🚀 SIGNAL_GUIDANCE Phase 1-4 Enhanced Backtest")
    logger.info("=" * 80)

    # 設定ファイルの読み込み
    if config_path is None:
        config_path = Path(__file__).parent / "config.py"

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using default config")
        config = {}
    else:
        config = safe_json_load(config_path)
        logger.info(f"✅ Loaded config from {config_path}")

    # ログレベル設定
    if 'logging' in config:
        setup_logging_from_config(config)
        logger.info("📝 Logging configured from config")
    else:
        # デフォルト設定
        setup_logging(level=logging.INFO)
        logger.info("📝 Using default logging configuration")

    # デバッグモード設定
    debug_logging = config.get('debug_logging', False)
    if debug_logging:
        logger.info("🐛 Debug logging enabled")

    # データ生成
    logger.info("📊 Generating synthetic market data...")
    data_df = generate_synthetic_data(
        n_periods=5000,  # 5000期間分のデータ
        start_price=50000.0,
        volatility=500
    )

    logger.info(f"✅ Generated {len(data_df)} data points")

    # SIGNAL_GUIDANCE統合環境作成
    env, enhanced_df = create_signal_guidance_backtest_env(data_df, config)

    # デバッグ設定を環境に反映
    if hasattr(env, 'debug_enabled'):
        env.debug_enabled = debug_logging

    # 環境インスタンス確認
    logger.info(f"Environment type: {type(env)}")
    logger.info(f"Is SignalGuidanceBacktestEnv: {isinstance(env, SignalGuidanceBacktestEnv)}")

    # モデル読み込みまたはランダムエージェント使用
    if model_path and Path(model_path).exists():
        logger.info(f"🤖 Loading model from {model_path}")
        model = SAC.load(model_path)
    else:
        logger.info("🎲 Using random actions (no model loaded)")
        model = None

    # バックテスト実行
    results = []
    total_rewards = []
    total_returns = []

    for episode in range(n_episodes):
        logger.info(f"🏃 Episode {episode + 1}/{n_episodes}")

        obs = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        portfolio_history = [env.portfolio_value]

        while not done:
            # モデルがある場合は予測、ない場合はランダム
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # ランダムアクション（離散アクションに変換）
                continuous_action = np.random.uniform(-1, 1)
                action = continuous_to_discrete_action(continuous_action)

            # 環境ステップ (HeavyTradingEnvは5つの値を返す)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
            elif len(step_result) == 4:
                obs, reward, done, info = step_result
                truncated = False
            else:
                logger.warning(f"Unexpected step result length: {len(step_result)}")
                continue

            episode_reward += reward
            episode_steps += 1

            # ポートフォリオ履歴記録
            portfolio_history.append(env.portfolio_value)

            # 進捗表示
            if episode_steps % 100 == 0:
                logger.info(f"Step {episode_steps}: Portfolio = ¥{env.portfolio_value:,.0f}, "
                          f"Reward = {episode_reward:.2f}")

        # エピソード結果
        initial_balance = env.initial_balance
        final_balance = env.portfolio_value
        total_return_pct = ((final_balance - initial_balance) / initial_balance) * 100

        episode_result = {
            'episode': episode + 1,
            'total_reward': episode_reward,
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'total_return_pct': total_return_pct,
            'steps': episode_steps,
            'portfolio_values': portfolio_history,
            'guidance_signals': env.guidance_signals[-100:] if env.guidance_signals else []  # 最新100件
        }

        results.append(episode_result)
        total_rewards.append(episode_reward)
        total_returns.append(total_return_pct)

        logger.info(f"✅ Episode {episode + 1} completed: "
                  f"Return = {total_return_pct:.2f}%, "
                  f"Final Balance = ¥{final_balance:,.0f}")

    # 総合結果
    avg_reward = np.mean(total_rewards)
    avg_return = np.mean(total_returns)
    std_return = np.std(total_returns)

    summary = {
        'test_type': 'SIGNAL_GUIDANCE_Phase1-4_Backtest',
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'n_episodes': n_episodes,
        'avg_total_reward': avg_reward,
        'avg_total_return_pct': avg_return,
        'std_total_return_pct': std_return,
        'config': config,
        'results': results
    }

    # SIGNAL_GUIDANCE統計
    if results and results[0].get('guidance_signals'):
        guidance_scores = []
        for result in results:
            for signal in result.get('guidance_signals', []):
                guidance_scores.append(signal.get('guidance_score', 50.0))

        if guidance_scores:
            summary['signal_guidance_stats'] = {
                'avg_guidance_score': np.mean(guidance_scores),
                'std_guidance_score': np.std(guidance_scores),
                'min_guidance_score': np.min(guidance_scores),
                'max_guidance_score': np.max(guidance_scores)
            }

    # 結果保存
    output_file = f"signal_guidance_backtest_results_{summary['timestamp']}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Results saved to {output_file}")

    # 結果表示
    print_formatted_metrics(summary)

    logger.info("🎉 SIGNAL_GUIDANCE Phase 1-4 Backtest completed!")
    logger.info(f"📊 Average Return: {avg_return:.2f}% ± {std_return:.2f}%")
    logger.info(f"🎯 Average Reward: {avg_reward:.2f}")

    return summary


if __name__ == "__main__":
    # バックテスト実行
    results = run_signal_guidance_backtest(
        model_path=None,  # モデルなしでランダムアクション
        config_path=None,
        n_episodes=3
    )