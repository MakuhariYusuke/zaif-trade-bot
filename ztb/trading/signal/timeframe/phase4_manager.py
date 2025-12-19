"""
Phase 4: Minute-level Trading Integration Manager

分足対応の統合マネージャー
"""

from typing import Dict, Optional, Tuple, Any
import asyncio

from ztb.utils.logging_utils import get_logger
from ztb.trading.signal.timeframe.adaptive_timeframe_manager import AdaptiveTimeframeManager
from ztb.trading.signal.timeframe.multi_timeframe_validator import MultiTimeframeSignalValidator
from ztb.trading.signal.timeframe.minute_data_pipeline import MinuteDataPipeline
from ztb.trading.signal.quality_scorer import SignalQualityScorer

logger = get_logger(__name__)


class Phase4MinuteTradingManager:
    """
    Phase 4 分足取引統合マネージャー

    分足対応の完全な取引システムを統合管理
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Phase 4 minute trading manager

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()

        # Phase 4コンポーネント初期化
        self.timeframe_manager = AdaptiveTimeframeManager(
            self.config.get('timeframe_config', {})
        )
        self.signal_validator = MultiTimeframeSignalValidator(
            self.config.get('validator_config', {})
        )
        self.data_pipeline = MinuteDataPipeline(
            self.config.get('pipeline_config', {})
        )

        # SignalQualityScorerとの統合設定
        self.enable_phase4_integration = self.config.get('enable_phase4_integration', True)
        self.primary_timeframe = self.config.get('primary_timeframe', '5m')

        logger.info("Phase4MinuteTradingManager initialized")

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'enable_phase4_integration': True,
            'primary_timeframe': '5m',
            'timeframe_config': {},
            'validator_config': {},
            'pipeline_config': {},
            'min_data_points': {
                '1m': 50,
                '5m': 20,
                '15m': 10,
                '1h': 5
            }
        }

    async def process_minute_signal(self, symbol: str,
                                  continuous_action: float = 0.0,
                                  portfolio: Optional[Dict] = None) -> Tuple[int, float, Dict[str, Any]]:
        """
        分足シグナルを処理（Phase 4完全統合）

        Args:
            symbol: 取引シンボル
            continuous_action: 連続アクション値
            portfolio: ポートフォリオ状態

        Returns:
            Tuple[int, float, Dict]: (アクション, スコア, 処理詳細)
        """
        portfolio = portfolio or {'position': 0, 'cash': 10000, 'value': 10000}

        try:
            # ステップ1: マルチタイムフレームデータ取得
            multi_tf_data = await self.data_pipeline.get_multi_timeframe_data(
                symbol, timeframes=['1m', '5m', '15m']
            )

            if not multi_tf_data:
                logger.warning("Failed to retrieve multi-timeframe data")
                return 0, 50.0, {'status': 'data_unavailable'}

            # ステップ2: 適応型タイムフレーム選択
            primary_data = multi_tf_data.get(self.primary_timeframe)
            if primary_data is None:
                logger.warning(f"Primary timeframe {self.primary_timeframe} data not available")
                return 0, 50.0, {'status': 'primary_data_unavailable'}

            optimal_timeframe, market_condition = self.timeframe_manager.select_optimal_timeframe(
                primary_data, self.primary_timeframe
            )

            # ステップ3: 最適タイムフレームのデータを使用したシグナル生成
            optimal_data = multi_tf_data.get(optimal_timeframe, primary_data)

            # SignalQualityScorerでシグナル生成（Phase 1-3統合）
            scorer_config = self.config.get('signal_scorer_config', {})
            scorer = SignalQualityScorer(scorer_config)

            action, score = scorer.calculate_signal_quality(optimal_data, continuous_action, portfolio)

            # ステップ4: マルチタイムフレーム検証（Phase 4）
            if self.enable_phase4_integration and len(multi_tf_data) > 1:
                validated_score, validation_result = self.signal_validator.validate_signal_consistency(
                    score, multi_tf_data, optimal_timeframe
                )
                score = validated_score
            else:
                validation_result = {'status': 'validation_disabled'}

            # ステップ5: 適応パラメータ取得
            adaptive_params = self.timeframe_manager.get_adaptive_parameters(market_condition)

            # 結果統合
            processing_details = {
                'status': 'success',
                'phase4_enabled': self.enable_phase4_integration,
                'optimal_timeframe': optimal_timeframe,
                'market_condition': market_condition.value,
                'multi_tf_data_points': {tf: len(data) for tf, data in multi_tf_data.items()},
                'validation_result': validation_result,
                'adaptive_params': adaptive_params,
                'data_quality': self.data_pipeline.get_data_quality_metrics(optimal_data)
            }

            logger.info(f"Phase 4 signal processed: {symbol} | Action: {action} | Score: {score:.2f} | "
                       f"Timeframe: {optimal_timeframe} | Condition: {market_condition.value}")

            return action, score, processing_details

        except Exception as e:
            logger.error(f"Error in Phase 4 signal processing: {e}")
            return 0, 50.0, {'status': 'error', 'error': str(e)}

    async def get_minute_trading_context(self, symbol: str) -> Dict[str, Any]:
        """
        分足取引コンテキストを取得

        Args:
            symbol: 取引シンボル

        Returns:
            Dict: 取引コンテキスト情報
        """
        try:
            # マルチタイムフレームデータ取得
            multi_tf_data = await self.data_pipeline.get_multi_timeframe_data(symbol)

            if not multi_tf_data:
                return {'status': 'data_unavailable'}

            # 各タイムフレームの分析
            context = {
                'status': 'success',
                'symbol': symbol,
                'timeframes': {},
                'market_analysis': {},
                'recommendations': {}
            }

            for tf, data in multi_tf_data.items():
                if len(data) > 0:
                    # 市場条件分析
                    market_condition = self.timeframe_manager.analyze_market_condition(data)

                    # データ品質評価
                    quality_metrics = self.data_pipeline.get_data_quality_metrics(data)

                    # 適応パラメータ
                    adaptive_params = self.timeframe_manager.get_adaptive_parameters(market_condition)

                    context['timeframes'][tf] = {
                        'data_points': len(data),
                        'latest_price': data['close'].iloc[-1],
                        'market_condition': market_condition.value,
                        'data_quality': quality_metrics,
                        'adaptive_params': adaptive_params
                    }

            # 全体的な市場分析
            primary_tf = self.primary_timeframe
            if primary_tf in multi_tf_data:
                primary_data = multi_tf_data[primary_tf]
                market_condition = self.timeframe_manager.analyze_market_condition(primary_data)

                context['market_analysis'] = {
                    'primary_timeframe': primary_tf,
                    'overall_condition': market_condition.value,
                    'recommended_timeframe': self.timeframe_manager.select_optimal_timeframe(
                        primary_data, primary_tf
                    )[0]
                }

            # 取引推奨
            context['recommendations'] = self._generate_trading_recommendations(context)

            return context

        except Exception as e:
            logger.error(f"Error getting trading context: {e}")
            return {'status': 'error', 'error': str(e)}

    def _generate_trading_recommendations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        取引推奨を生成

        Args:
            context: 取引コンテキスト

        Returns:
            Dict: 推奨事項
        """
        try:
            recommendations = {
                'timeframe_priority': [],
                'risk_level': 'medium',
                'confidence_level': 'medium'
            }

            timeframes = context.get('timeframes', {})
            market_analysis = context.get('market_analysis', {})

            # タイムフレーム優先順位付け
            tf_scores = {}
            for tf, info in timeframes.items():
                score = info.get('data_quality', {}).get('quality_score', 0)
                condition = info.get('market_condition', '')

                # 高ボラティリティ時は短いタイムフレームを優先
                if condition in ['high_volatility', 'trending']:
                    score += 20
                elif condition == 'low_volatility':
                    score += 10

                tf_scores[tf] = score

            # スコア順にソート
            sorted_tfs = sorted(tf_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations['timeframe_priority'] = [tf for tf, _ in sorted_tfs]

            # リスクレベル判定
            overall_condition = market_analysis.get('overall_condition', '')
            if overall_condition == 'high_volatility':
                recommendations['risk_level'] = 'high'
            elif overall_condition == 'low_volatility':
                recommendations['risk_level'] = 'low'

            # 信頼性レベル判定
            avg_quality = np.mean([info.get('data_quality', {}).get('quality_score', 0)
                                 for info in timeframes.values()])
            if avg_quality > 80:
                recommendations['confidence_level'] = 'high'
            elif avg_quality < 60:
                recommendations['confidence_level'] = 'low'

            return recommendations

        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
            return {'error': str(e)}

    async def validate_phase4_system(self) -> Dict[str, Any]:
        """
        Phase 4システム全体の検証

        Returns:
            Dict: 検証結果
        """
        validation_results = {
            'components': {},
            'integration': {},
            'performance': {},
            'overall_status': 'unknown'
        }

        try:
            # コンポーネント検証
            validation_results['components'] = {
                'timeframe_manager': 'initialized' if self.timeframe_manager else 'failed',
                'signal_validator': 'initialized' if self.signal_validator else 'failed',
                'data_pipeline': 'initialized' if self.data_pipeline else 'failed'
            }

            # 統合テスト
            test_symbol = 'btc_jpy'
            test_context = await self.get_minute_trading_context(test_symbol)

            validation_results['integration'] = {
                'context_retrieval': 'success' if test_context.get('status') == 'success' else 'failed',
                'multi_tf_data': len(test_context.get('timeframes', {})) > 0,
                'market_analysis': bool(test_context.get('market_analysis')),
                'recommendations': bool(test_context.get('recommendations'))
            }

            # パフォーマンステスト
            start_time = asyncio.get_event_loop().time()
            for _ in range(5):  # 5回テスト
                await self.process_minute_signal(test_symbol)
            end_time = asyncio.get_event_loop().time()

            avg_processing_time = (end_time - start_time) / 5
            validation_results['performance'] = {
                'avg_processing_time': avg_processing_time,
                'performance_rating': 'good' if avg_processing_time < 2.0 else 'needs_improvement'
            }

            # 全体ステータス判定
            all_components_ok = all(status == 'initialized'
                                  for status in validation_results['components'].values())
            integration_ok = all(validation_results['integration'].values())
            performance_ok = validation_results['performance']['performance_rating'] == 'good'

            if all_components_ok and integration_ok and performance_ok:
                validation_results['overall_status'] = 'healthy'
            elif all_components_ok and integration_ok:
                validation_results['overall_status'] = 'degraded_performance'
            else:
                validation_results['overall_status'] = 'needs_attention'

            logger.info(f"Phase 4 system validation completed: {validation_results['overall_status']}")

            return validation_results

        except Exception as e:
            logger.error(f"Error in Phase 4 system validation: {e}")
            validation_results['overall_status'] = 'error'
            validation_results['error'] = str(e)
            return validation_results
