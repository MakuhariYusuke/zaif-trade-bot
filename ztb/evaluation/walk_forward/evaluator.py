"""
Walk-Forward モデル評価モジュール

各ウィンドウでのSAC訓練・評価を実施。

時系列ウィンドウごとに強化学習（SAC）モデルを訓練し、
検証セット（in-sample）とテストセット（out-of-sample）で性能を評価します。

## 既存実装との統合

- メトリクス計算: ztb.metrics.metrics を活用
- チェックポイント: ztb.utils.checkpoint のパターンに統合した
  ztb.evaluation.walk_forward.checkpoint を活用
- エラーハンドリング: ztb.utils.error_utils.safe_operation による統一的な例外隔離

## 依存注入パターン（旧API）

```python
from ztb.evaluation.walk_forward import WalkForwardModelEvaluator

# 1. デフォルト（自動初期化）
evaluator = WalkForwardModelEvaluator()

# 2. カスタム環境工場を注入
evaluator = WalkForwardModelEvaluator(env_factory=my_custom_env_factory)

# 3. カスタムアルゴリズムファクトリを注入
evaluator = WalkForwardModelEvaluator(algorithm_factory=my_sac_factory)

# 4. 両方カスタム + チェックポイント
evaluator = WalkForwardModelEvaluator(
    env_factory=my_env_factory,
    algorithm_factory=my_algo_factory,
    checkpoint_dir="./checkpoints"  # ztb.utils.checkpoint パターンに統合
)
```

UnifiedEvaluator 経由の実行を推奨します。
"""

import warnings
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from stable_baselines3 import SAC  # type: ignore
except Exception:  # pragma: no cover - defensive fallback for test environments
    SAC = None  # type: ignore
from ztb.metrics.metrics import max_drawdown as calculate_max_drawdown
from ztb.metrics.metrics import sharpe_ratio as calculate_sharpe_ratio
from ztb.metrics.metrics import win_rate as calculate_win_rate
from ztb.utils.error_utils import safe_operation


from .reporter import BacktestReporter

from .checkpoint import CheckpointManager
from .types import TimeSeriesWindow, WindowPerformance

logger = logging.getLogger(__name__)


class WindowEvaluationError(Exception):
    """ウィンドウ評価エラー"""

    pass


class WalkForwardModelEvaluator:
    """各ウィンドウでのSAC訓練・評価

    Walk-Forward 分析用にウィンドウごとにモデルを訓練・評価し、
    過学習の有無を検出します。

    依存注入パターンで環境とアルゴリズムを外部から提供可能。
    ウィンドウごとのエラーを隔離し、他のウィンドウの評価を継続。
    """

    def __init__(
        self,
        env_factory: Optional[Callable[[pd.DataFrame], Any]] = None,
        algorithm_factory: Optional[Callable[[Any], SAC]] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        """初期化

        Args:
            env_factory: 環境作成関数（df -> env）
                デフォルト: scripts/v456 の create_environment_wrapper
            algorithm_factory: アルゴリズム作成関数（env -> model）
                デフォルト: SAC デフォルトパラメータ
            checkpoint_dir: チェックポイント保存ディレクトリ。
                指定時、自動でチェックポイント管理が有効化される。
                デフォルト: None（チェックポイント機能無効）
        """
        warnings.warn(
            "WalkForwardModelEvaluator is deprecated. "
            "Use ztb.evaluation.unified_evaluation.UnifiedEvaluator with "
            "EvaluationType.WALK_FORWARD instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.models: Dict[int, SAC] = {}
        self.results: Dict[int, WindowPerformance] = {}
        self.errors: Dict[int, Exception] = {}  # エラー追跡
        self.env_factory = env_factory or self._default_env_factory
        self.algorithm_factory = algorithm_factory or self._default_algorithm_factory
        
        # チェックポイント管理
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_manager = (
            CheckpointManager(checkpoint_dir) if checkpoint_dir else None
        )

    @staticmethod
    def _default_env_factory(df: pd.DataFrame) -> Any:
        """デフォルト環境工場

        Args:
            df: 時系列データ

        Returns:
            Any: 環境オブジェクト
        """
        import sys

        project_root: Path = Path(__file__).resolve().parent.parent.parent.parent
        scripts_path: Path = project_root / "scripts" / "v456"
        if str(scripts_path) not in sys.path:
            sys.path.insert(0, str(scripts_path))

        from train_and_evaluate_v456_phase3 import create_environment_wrapper

        return create_environment_wrapper(df, None)

    @staticmethod
    def _default_algorithm_factory(env: Any) -> SAC:
        """デフォルトアルゴリズム工場

        Args:
            env: 環境オブジェクト

        Returns:
            SAC: 訓練済みSACモデル
        """
        from ztb.config.environment_config import TrainingConfig

        model: SAC = SAC(
            "MlpPolicy",
            env,
            learning_rate=TrainingConfig.LEARNING_RATE,
            batch_size=256,
            buffer_size=TrainingConfig.BUFFER_SIZE,
            tau=0.005,
            gamma=0.99,
            verbose=0,
        )
        return model

    def train_and_evaluate_window(
        self,
        df: pd.DataFrame,
        window: TimeSeriesWindow,
        timesteps: int = 10000,
        continue_on_error: bool = True,
    ) -> Tuple[Optional[WindowPerformance], Optional[BacktestReporter]]:
        """ウィンドウ内で訓練・評価（例外処理強化版）

        Args:
            df: 時系列データフレーム
            window: 訓練対象のウィンドウ
            timesteps: SAC訓練のタイムステップ数
            continue_on_error: Trueの場合、エラーが発生してもNoneを返すのみ
                              Falseの場合、例外を発生させる

        Returns:
            Optional[WindowPerformance]: ウィンドウの性能結果、エラーの場合はNone
        """
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"Window {window.window_id}: Training & Evaluation")
            logger.info(f"{'='*70}")

            # データ分割
            train_df: pd.DataFrame = df.iloc[window.train_start : window.train_end]
            val_df: pd.DataFrame = df.iloc[window.val_start : window.val_end]
            test_df: pd.DataFrame = df.iloc[window.test_start : window.test_end]

            # ★ Z1 P0-3: Train/Val/Test 汚染チェック
            if window.train_end > window.val_start:
                logger.warning(
                    f"P0-3: Train/Val overlap ({window.train_end - window.val_start} bars)"
                )
                train_df = df.iloc[window.train_start : window.val_start]
            if window.val_end > window.test_start:
                logger.warning(
                    f"P0-3: Val/Test overlap ({window.val_end - window.test_start} bars)"
                )
                val_df = df.iloc[window.val_start : window.test_start]

            # データ検証
            if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
                raise ValueError(
                    f"Empty data detected: train={len(train_df)}, "
                    f"val={len(val_df)}, test={len(test_df)}"
                )

            logger.info(f"Train: {len(train_df)} bars")
            logger.info(f"Val:   {len(val_df)} bars")
            logger.info(f"Test:  {len(test_df)} bars")

            # 訓練環境作成
            try:
                train_env = self.env_factory(train_df)
                logger.info("✓ Training environment created")
            except Exception as e:
                raise WindowEvaluationError(
                    f"Failed to create training environment: {str(e)}"
                ) from e

            # SAC訓練
            logger.info("\n[Training]")
            try:
                model: SAC = self.algorithm_factory(train_env)
                model.learn(total_timesteps=timesteps)
                logger.info(f"✓ Training completed ({timesteps} timesteps)")
            except Exception as e:
                raise WindowEvaluationError(f"Training failed: {str(e)}") from e

            # 評価実行
            perf_tuple = self.evaluate_window_with_model(df, window, model, continue_on_error)

            # evaluate_window_with_model returns either (performance, reporter) on
            # success or a tuple of (None, None) on failure. For compatibility with
            # older callers/tests that expect None on failure, normalize the return
            # value here so that None is returned when both elements are None.
            if isinstance(perf_tuple, tuple) and perf_tuple == (None, None):
                return None

            return perf_tuple

        except WindowEvaluationError as e:
            # ウィンドウ固有のエラー
            logger.error(f"❌ Window {window.window_id} evaluation error: {str(e)}")
            self.errors[window.window_id] = e

            if continue_on_error:
                logger.info(f"⚠️  Continuing with other windows...")
                return None
            else:
                raise

        except Exception as e:
            # 予期しないエラー
            logger.error(
                f"❌ Window {window.window_id} unexpected error: {str(e)}",
                exc_info=True,
            )
            wrapped_error = WindowEvaluationError(f"Unexpected error: {str(e)}")
            self.errors[window.window_id] = wrapped_error

            if continue_on_error:
                logger.info(f"⚠️  Continuing with other windows...")
                return None
            else:
                raise wrapped_error from e

    def evaluate_window_with_model(
        self,
        df: pd.DataFrame,
        window: TimeSeriesWindow,
        model: SAC,
        continue_on_error: bool = True,
    ) -> Tuple[Optional[WindowPerformance], Optional[BacktestReporter]]:
        """事前トレーニング済みモデルでウィンドウを評価

        Args:
            df: 時系列データフレーム
            window: 評価対象のウィンドウ
            model: 事前トレーニング済み SAC モデル
            continue_on_error: Trueの場合、エラーが発生してもNoneを返すのみ
                              Falseの場合、例外を発生させる

        Returns:
            Optional[WindowPerformance]: ウィンドウの性能結果、エラーの場合はNone
        """
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"Window {window.window_id}: Evaluation with Pre-trained Model")
            logger.info(f"{'='*70}")

            # データ分割
            val_df: pd.DataFrame = df.iloc[window.val_start : window.val_end]
            test_df: pd.DataFrame = df.iloc[window.test_start : window.test_end]

            # ★ Z1 P0-3: Val/Test 汚染チェック — 重複を検出
            if window.val_end > window.test_start:
                overlap = window.val_end - window.test_start
                logger.warning(
                    f"P0-3: Val/Test overlap detected ({overlap} bars). "
                    f"val_end={window.val_end}, test_start={window.test_start}. "
                    f"Truncating val to prevent contamination."
                )
                val_df = df.iloc[window.val_start : window.test_start]

            # データ検証
            if len(val_df) == 0 or len(test_df) == 0:
                raise ValueError(
                    f"Empty data detected: val={len(val_df)}, test={len(test_df)}"
                )

            logger.info(f"Val:   {len(val_df)} bars")
            logger.info(f"Test:  {len(test_df)} bars")

            # ★ Z1 P1-2: 正規 BacktestReporter を使用 (v457 ローカル import 廃止)
            # モジュールレベルの from .reporter import BacktestReporter を再利用
            val_reporter = BacktestReporter()
            test_reporter = BacktestReporter()

            # 検証評価
            logger.info("\n[Validation Evaluation]")
            try:
                val_result: Dict[str, Any] = self._evaluate_on_df(model, val_df, val_reporter, max_steps=len(val_df))
                logger.info(f"  Val ROI: {val_result['roi']:.4f}")
                logger.info(f"  Val Balance: {val_result['final_balance']:,.0f} JPY")
            except Exception as e:
                raise WindowEvaluationError(f"Validation evaluation failed: {str(e)}") from e

            # テスト評価
            logger.info("\n[Test Evaluation (Out-of-Sample)]")
            try:
                test_result: Dict[str, Any] = self._evaluate_on_df(model, test_df, test_reporter, max_steps=len(test_df))
                logger.info(f"  Test ROI: {test_result['roi']:.4f}")
                logger.info(f"  Test Balance: {test_result['final_balance']:,.0f} JPY")
            except Exception as e:
                raise WindowEvaluationError(f"Test evaluation failed: {str(e)}") from e

            # 性能オブジェクト作成
            try:
                performance: WindowPerformance = WindowPerformance(
                    window_id=window.window_id,
                    val_roi=val_result["roi"],
                    test_roi=test_result["roi"],
                    val_final_balance=val_result["final_balance"],
                    test_final_balance=test_result["final_balance"],
                    sharpe_ratio=test_result.get("sharpe", 0.0),
                    max_drawdown=test_result.get("max_drawdown", 0.0),
                    win_rate=test_result.get("win_rate", 0.0),
                    trades=test_result.get("trades", 0),
                    profit_factor=test_result.get("profit_factor", 0.0),
                    expectancy=test_result.get("expectancy", 0.0),
                    avg_win=test_result.get("avg_win", 0.0),
                    avg_loss=test_result.get("avg_loss", 0.0),
                    val_reporter=val_reporter,
                    test_reporter=test_reporter,
                )
                # 検証実行
                performance.validate()
                logger.info("✓ Window evaluation successful")
            except Exception as e:
                raise WindowEvaluationError(f"Performance object creation failed: {str(e)}") from e

            # Reporter統計最終化
            val_reporter.finalize_stats()
            test_reporter.finalize_stats()

            # モデル保存
            self.models[window.window_id] = model
            self.results[window.window_id] = performance

            return performance, test_reporter

        except WindowEvaluationError as e:
            # ウィンドウ固有のエラー
            logger.error(f"❌ Window {window.window_id} evaluation error: {str(e)}")
            self.errors[window.window_id] = e

            if continue_on_error:
                logger.info(f"⚠️  Continuing with other windows...")
                return None, None
            else:
                raise

        except Exception as e:
            # 予期しないエラー
            logger.error(
                f"❌ Window {window.window_id} unexpected error: {str(e)}",
                exc_info=True,
            )
            wrapped_error = WindowEvaluationError(f"Unexpected error: {str(e)}")
            self.errors[window.window_id] = wrapped_error

            if continue_on_error:
                logger.info(f"⚠️  Continuing with other windows...")
                return None, None
            else:
                raise wrapped_error from e

    def _evaluate_on_df(
        self,
        model: SAC,
        df: pd.DataFrame,
        reporter: BacktestReporter,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """データフレーム上で評価（BacktestReporter統合版）

        v457 BacktestReporterを使用して正確な取引統計を計算します。
        
        ★ P0-4対応: Val/Test Leakage Prevention
        - 各評価ごとに新しい環境インスタンスを生成（env_factory呼び出し）
        - 環境内のscalerは独立したstateを持つ（prewarmで再構築）
        - Val/Test評価間でscaler統計の汚染なし

        Args:
            model: 訓練済み SAC モデル
            df: 評価用データフレーム
            max_steps: 最大ステップ数（指定時は全期間評価）

        Returns:
            Dict[str, Any]: ROI, balance, metrics など

        Raises:
            ValueError: 評価に失敗した場合
        """
        if len(df) == 0:
            raise ValueError("DataFrame is empty")

        # ★ P0-4: 環境を毎回生成（Val/Test分離）
        # 新しいインスタンス → 独立したscaler state → リーク防止
        eval_env = self.env_factory(df)

        # 環境にreporterアタッチ
        eval_env.recorder = reporter

        obs, _ = eval_env.reset(seed=42, options={"start_step": 0, "max_steps": max_steps})  # 固定seed、全期間評価
        done: bool = False
        episode_reward: float = 0.0
        balances: List[float] = [eval_env.initial_balance]
        step_count: int = 0
        prev_position: float = 0.0  # 初期position
        prev_balance: float = eval_env.initial_balance  # 初期balance

        # エピソード実行
        try:
            while not done and (max_steps is None or step_count < max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated

                episode_reward += float(reward)
                balances.append(eval_env.balance)
                step_count += 1

                # Reporter更新
                reporter.update_step(
                    step=step_count,
                    portfolio_value=eval_env.balance,
                    action=action,
                    env_info=info
                )

                # Position変化でtrade検出（reporter.record_trade用）
                current_position = info.get("position", eval_env.position)
                if abs(current_position - prev_position) > 1e-6:  # position変化
                    # PnL計算: infoから取得、なければbalance変化から推定
                    pnl = info.get("trade_pnl", eval_env.balance - prev_balance)
                    fee = info.get("fee_paid", 0.0)
                    slippage = info.get("slippage_paid", 0.0)
                    
                    # ★ Doc21指摘[Major]: 反転時はクローズ側にprev_entry_priceを使用
                    # 反転判定: 符号が逆転
                    is_reversal = (abs(prev_position) > 1e-6 and 
                                  abs(current_position) > 1e-6 and 
                                  prev_position * current_position < 0)
                    
                    if is_reversal:
                        # 反転時: クローズ側はprev_entry_price、新規側は現在のentry_price
                        entry_price = info.get("prev_entry_price", eval_env.entry_price)
                    else:
                        # 通常取引: 現在のentry_price
                        entry_price = info.get("entry_price", eval_env.close_prices[eval_env.current_step])
                    
                    exit_price = info.get("exit_price", eval_env.close_prices[eval_env.current_step])
                    size = abs(current_position - prev_position)
                    
                    # ★ P1-1: close_reasonをinfoから取得
                    close_reason = info.get("close_reason", None)
                    
                    # ★ 修正: 新しいrecord_tradeシグネチャに合わせる
                    reporter.record_trade(
                        position_before=prev_position,
                        position_after=current_position,
                        pnl=pnl,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        size=size,
                        fee=fee,
                        slippage=slippage,
                        timestamp=None,
                        close_reason=close_reason,
                    )
                
                prev_position = current_position
                prev_balance = eval_env.balance  # 更新
        except Exception as e:
            raise ValueError(f"Episode execution failed: {str(e)}") from e

        # Reporter統計最終化
        reporter.finalize_stats()

        # パフォーマンス計算
        try:
            final_balance: float = eval_env.balance
            total_pnl: float = final_balance - eval_env.initial_balance
            roi: float = (
                total_pnl / eval_env.initial_balance
                if eval_env.initial_balance != 0
                else 0.0
            )

            # 既存の ztb.metrics.metrics を使用（balancesベース）
            balances_array = np.array(balances)
            returns = np.diff(balances_array) / np.maximum(balances_array[:-1], 1e-12)

            sharpe = calculate_sharpe_ratio(returns)
            max_dd = calculate_max_drawdown(balances_array)

            # Reporterから取引統計取得
            win_rate = (reporter.stats.get("winning_trades", 0) / max(1, reporter.stats.get("total_trades", 1)))
            profit_factor = reporter.stats.get("profit_factor", 0.0)
            expectancy = reporter.stats.get("expectancy", 0.0)
            avg_win = reporter.stats.get("avg_win", 0.0)
            avg_loss = reporter.stats.get("avg_loss", 0.0)
            trades = reporter.stats.get("total_trades", 0)

            # Reporter統計最終化
            reporter.finalize_stats()

            return {
                "episode_reward": episode_reward,
                "final_balance": final_balance,
                "total_pnl": total_pnl,
                "roi": roi,
                "trades": trades,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "expectancy": expectancy,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
            }
        except Exception as e:
            raise ValueError(f"Metrics calculation failed: {str(e)}") from e

    def evaluate_multiple_windows(
        self,
        df: pd.DataFrame,
        windows: List[TimeSeriesWindow],
        timesteps: int = 10000,
        continue_on_error: bool = True,
        run_id: Optional[str] = None,
        resume_from_checkpoint: bool = False,
    ) -> Tuple[List[Tuple[Optional[WindowPerformance], Optional[BacktestReporter]]], Dict[int, Exception]]:
        """複数ウィンドウを連続評価（例外分離・チェックポイント対応）

        Args:
            df: 時系列データフレーム
            windows: ウィンドウリスト
            timesteps: 各ウィンドウのタイムステップ数
            continue_on_error: Trueの場合、エラーが発生してもスキップして続行
            run_id: 実行ID（チェックポイント用）。指定時、自動で保存される。
                   デフォルト: None（チェックポイント機能使用しない）
            resume_from_checkpoint: Trueの場合、既存チェックポイントから復元して続行

        Returns:
            Tuple[List[WindowPerformance], Dict[int, Exception]]:
                - 成功したウィンドウの結果リスト
                - エラーが発生したウィンドウの辞書（ウィンドウID -> 例外）
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Evaluating {len(windows)} windows...")
        logger.info(f"{'='*70}")

        # チェックポイント復元（オプション）
        if resume_from_checkpoint and run_id and self.checkpoint_manager:
            try:
                self.checkpoint_manager.restore(self, run_id, restore_models=False)
                logger.info(f"✓ Restored checkpoint: run_id={run_id}")
            except Exception as e:
                logger.warning(f"Failed to restore checkpoint: {e}")

        # 評価対象ウィンドウを決定（既に完了したものはスキップ）
        completed_window_ids = set(self.results.keys())
        target_windows = [
            w for w in windows if w.window_id not in completed_window_ids
        ]

        if len(target_windows) < len(windows):
            logger.info(
                f"Skipping {len(windows) - len(target_windows)} already completed windows"
            )

        successful_results: List[Tuple[Optional[WindowPerformance], Optional[BacktestReporter]]] = []
        errors: Dict[int, Exception] = {}

        for i, window in enumerate(target_windows):
            logger.info(
                f"\n[{i+1}/{len(target_windows)}] Processing window {window.window_id}"
            )

            result = self.train_and_evaluate_window(
                df=df,
                window=window,
                timesteps=timesteps,
                continue_on_error=continue_on_error,
            )

            if result is not None:
                # Append to successful results (support both tuple and raw WindowPerformance)
                successful_results.append(result)

                # If result contains a performance object, ensure self.results is populated
                perf = None
                if isinstance(result, tuple):
                    perf = result[0]
                elif isinstance(result, WindowPerformance):
                    perf = result

                if isinstance(perf, WindowPerformance):
                    self.results[perf.window_id] = perf
            else:
                if window.window_id in self.errors:
                    errors[window.window_id] = self.errors[window.window_id]

            # 定期的にチェックポイント保存
            if run_id and self.checkpoint_manager and (i + 1) % 5 == 0:
                try:
                    self.checkpoint_manager.save(self, run_id)
                    logger.debug(f"Saved checkpoint after window {i+1}")
                except Exception as e:
                    logger.warning(f"Failed to save checkpoint: {e}")

        # 最終チェックポイント保存
        if run_id and self.checkpoint_manager:
            try:
                self.checkpoint_manager.save(self, run_id)
                logger.info(f"✓ Final checkpoint saved: run_id={run_id}")
            except Exception as e:
                logger.warning(f"Failed to save final checkpoint: {e}")

        logger.info(f"\n{'='*70}")
        logger.info(f"Evaluation Summary:")
        logger.info(f"  Successful: {len(successful_results)}/{len(windows)}")
        logger.info(f"  Failed: {len(self.errors)}/{len(windows)}")
        if self.errors:
            for window_id, error in self.errors.items():
                logger.warning(f"    Window {window_id}: {str(error)}")
        logger.info(f"{'='*70}\n")

        return successful_results, self.errors

    def get_results_summary(self) -> Dict[str, Any]:
        """評価結果のサマリーを取得

        Returns:
            Dict[str, Any]: 結果統計
        """
        if not self.results:
            return {
                "total_windows": 0,
                "successful_windows": 0,
                "failed_windows": len(self.errors),
                "avg_val_roi": 0.0,
                "avg_test_roi": 0.0,
                "avg_sharpe": 0.0,
            }

        window_ids = list(self.results.keys())
        val_rois = [self.results[wid].val_roi for wid in window_ids]
        test_rois = [self.results[wid].test_roi for wid in window_ids]
        sharpes = [self.results[wid].sharpe_ratio for wid in window_ids]

        return {
            "total_windows": len(window_ids) + len(self.errors),
            "successful_windows": len(window_ids),
            "failed_windows": len(self.errors),
            "avg_val_roi": float(np.mean(val_rois)) if val_rois else 0.0,
            "std_val_roi": float(np.std(val_rois)) if val_rois else 0.0,
            "avg_test_roi": float(np.mean(test_rois)) if test_rois else 0.0,
            "std_test_roi": float(np.std(test_rois)) if test_rois else 0.0,
            "avg_sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
            "std_sharpe": float(np.std(sharpes)) if sharpes else 0.0,
        }

    @staticmethod
    def _calculate_sharpe(
        returns: List[float],
        rf_rate: float = 0.0,
    ) -> float:
        """廃止予定: ztb.metrics.metrics.sharpe_ratio() を使用してください

        このメソッドは後方互換性のためのみに存在します。
        新規コードは ztb.metrics.metrics.sharpe_ratio() を直接使用してください。
        """
        return calculate_sharpe_ratio(np.array(returns), rf=rf_rate)

    @staticmethod
    def _calculate_max_drawdown(balances: List[float]) -> float:
        """廃止予定: ztb.metrics.metrics.max_drawdown() を使用してください

        このメソッドは後方互換性のためのみに存在します。
        新規コードは ztb.metrics.metrics.max_drawdown() を直接使用してください。
        """
        return calculate_max_drawdown(np.array(balances))
