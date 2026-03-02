#!/usr/bin/env python3
"""
市場レジーム評価モジュール

市場をトレンド/レンジ/高ボラ/低ボラに分類し、
各レジームにおけるモデルの性能を比較します。
"""

import warnings
import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

warnings.filterwarnings("ignore")
warnings.warn(
    "ztb.analysis.regime.regime_evaluation is deprecated; "
    "use ztb.analysis.regime.regime_eval or UnifiedEvaluator with EvaluationType.REGIME.",
    DeprecationWarning,
    stacklevel=2,
)

# 年間取引日数
from ztb.trading.constants import TRADING_DAYS_PER_YEAR  # = 252
from ztb.training.policy_utils import predict_with_masks
from ztb.io.data_loader import DataLoader
from ztb.utils.file_utils import safe_json_dump
from ztb.utils.path_utils import ensure_dir

@dataclass

@dataclass
class RegimeAnalysisResult:
    """レジーム分析結果"""

    regime_labels: NDArray[np.str_]
    regime_counts: dict[str, int]
    regime_metrics: dict[str, dict[str, RegimeMetrics]]
    regime_transitions: pd.DataFrame

class RegimeEvaluator:
    """市場レジーム評価クラス"""
        self.volatility_window = volatility_window
        self.trend_window = trend_window

    def classify_market_regime(
        self, price_data: pd.DataFrame
    ) -> tuple[NDArray[np.str_], dict[str, int]]:
        """
        市場レジームを分類

        Args:
            price_data: OHLCVデータ

        # リターンの計算
        returns = price_data["close"].pct_change().fillna(0)

        # ボラティリティの計算（標準偏差）
        from ztb.metrics.technical import calculate_rolling_volatility

        volatility = calculate_rolling_volatility(
            returns, window=self.volatility_window, annualize=True
        )

        # トレンド方向の計算（移動平均の傾き）
        ma_short = price_data["close"].rolling(window=self.trend_window // 2).mean()
        ma_long = price_data["close"].rolling(window=self.trend_window).mean()
        trend_strength = (ma_short - ma_long) / ma_long

        # ボラティリティの閾値
        vol_median = volatility.median()
        vol_high_threshold = vol_median * 1.5
        vol_low_threshold = vol_median * 0.5

        # トレンド強度の閾値
        trend_threshold = 0.02  # 2%のトレンド

        # レジーム分類
        regime_labels = []

        for i in range(len(price_data)):
            vol = volatility.iloc[i] if i >= self.volatility_window else vol_median
            trend = abs(trend_strength.iloc[i]) if i >= self.trend_window else 0

            if trend > trend_threshold:
                regime = "trend"
            elif vol > vol_high_threshold:
                regime = "high_vol"
            elif vol < vol_low_threshold:
                regime = "low_vol"
            else:
                regime = "range"

            regime_labels.append(regime)

        regime_array = np.array(regime_labels)

        # レジームカウント
        unique, counts = np.unique(regime_array, return_counts=True)
        regime_counts = dict(zip(unique, counts))

        return regime_array, regime_counts

    def get_model_actions(
        self, model_path: str, price_data: pd.DataFrame
    ) -> tuple[NDArray[np.int_], int]:
        """
        モデルからアクションを生成

        Args:
            model_path: モデルファイルのパス
            price_data: 価格データ

        Returns:
            アクション配列と非法アクション数
        """
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker

        from ztb.trading import HeavyTradingEnv

        # モデルをロード
        model = MaskablePPO.load(model_path)

        # 環境を作成
        env = HeavyTradingEnv(df=price_data, config={})

        # Wrap with ActionMasker
        def mask_fn(env: Any) -> Any:
            return env.get_legal_actions().astype(bool)

        env = ActionMasker(env, mask_fn)

        # アクションを生成
        actions = []
        illegal_count = 0
        obs, _ = env.reset()

        for _ in range(len(price_data)):
            # Use predict_with_masks for proper action mask handling
            action, _ = predict_with_masks(model, obs, env.env, deterministic=False)
            actions.append(int(action))

            # ActionMasker適用後は非法アクションが発生しないはず
            # legal_actions = env.env.get_legal_actions()
            # if not legal_actions[action]:
            #     illegal_count += 1

            # 次の観測を取得
            obs, _, done, truncated, _ = env.step(action.item())
            if done or truncated:
                obs, _ = env.reset()

        return np.array(actions), illegal_count

    def calculate_regime_metrics(
        self,
        returns: pd.Series,
        regime_labels: NDArray[np.str_],
        regime: str,
        actions: NDArray[np.int_] | None = None,
        illegal_actions: int = 0,
    ) -> RegimeMetrics:
        """
        指定レジームのメトリクスを計算

        Args:
            returns: リターン系列
            regime_labels: レジームラベル配列
            regime: 対象レジーム

        Returns:
            レジームメトリクス
        """
        # 指定レジームのデータを抽出
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]

        if len(regime_returns) == 0:
            return RegimeMetrics(0, 0, 0, 0, 0, 0, {"BUY": 0, "SELL": 0, "HOLD": 0}, 0)

        # 各モデルのレジーム別メトリクス
        regime_metrics = {}

        for model_name in models.keys():
            model_regime_metrics = {}

            for regime in ["trend", "range", "high_vol", "low_vol"]:
                actions = None
                illegal_actions = 0
                if actions_dict and model_name in actions_dict:
                    actions, illegal_actions = actions_dict[model_name]
                metrics = self.calculate_regime_metrics(
                    returns, regime_labels, regime, actions, illegal_actions
                )
                model_regime_metrics[regime] = metrics

            regime_metrics[model_name] = model_regime_metrics

        # レジーム遷移マトリックス
        regime_transitions = self._calculate_regime_transitions(regime_labels)

        return RegimeAnalysisResult(
            regime_labels=regime_labels,
            regime_counts=regime_counts,
            regime_metrics=regime_metrics,
            regime_transitions=regime_transitions,
        )

    def _calculate_regime_transitions(
        self, regime_labels: NDArray[np.str_]
    ) -> pd.DataFrame:
        """レジーム遷移マトリックスを計算"""
        regimes = ["trend", "range", "high_vol", "low_vol"]
        transition_matrix = pd.DataFrame(0, index=regimes, columns=regimes)

        for i in range(1, len(regime_labels)):
            from_regime = regime_labels[i - 1]
            to_regime = regime_labels[i]
            if from_regime in regimes and to_regime in regimes:
                transition_matrix.loc[from_regime, to_regime] += 1

        # 確率に変換
        transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

        return transition_matrix

    def plot_regime_analysis(
        self, result: RegimeAnalysisResult, save_path: str | None = None
    ) -> None:
        """レジーム分析結果を可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Market Regime Analysis", fontsize=16)

        # レジーム分布
        regimes = list(result.regime_counts.keys())
        counts = list(result.regime_counts.values())

        axes[0, 0].bar(regimes, counts, color="skyblue")
        axes[0, 0].set_title("Regime Distribution")
        axes[0, 0].set_ylabel("Count")

        # Sharpe比率比較
        model_names = list(result.regime_metrics.keys())
        regimes = ["trend", "range", "high_vol", "low_vol"]

        sharpe_data = []
        for model in model_names:
            sharpe_row = [
                result.regime_metrics[model][regime].sharpe_ratio for regime in regimes
            ]
            sharpe_data.append(sharpe_row)

        sharpe_df = pd.DataFrame(sharpe_data, columns=regimes, index=model_names)
        sharpe_df.plot(kind="bar", ax=axes[0, 1])
        axes[0, 1].set_title("Sharpe Ratio by Regime")
        axes[0, 1].set_ylabel("Sharpe Ratio")
        "--volatility-window",
        type=int,
        default=20,
        help="Window size for volatility calculation",
    )
    parser.add_argument(
        "--trend-window", type=int, default=50, help="Window size for trend detection"
    )

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # モデル辞書の作成
    models = {}
    for model_spec in args.models:
        if ":" in model_spec:
            name, path = model_spec.split(":", 1)
            models[name] = path
        else:
            name = Path(model_spec).stem
            models[name] = model_spec

    # 価格データの読み込み
    try:
        price_data = DataLoader.load_csv_strict(args.price_data, index_col=0, parse_dates=True)
        print(f"Loaded price data with {len(price_data)} rows")
    except Exception as e:
        print(f"Error loading price data: {e}")
        return

    # レジーム評価器の初期化
    evaluator = RegimeEvaluator(
        volatility_window=args.volatility_window, trend_window=args.trend_window
    )

    # レジーム分類
    print("Classifying market regimes...")
    regime_labels, regime_counts = evaluator.classify_market_regime(price_data)
    print(f"Regime distribution: {regime_counts}")

    # モデルからアクションを取得
    print("Loading models and generating actions...")
    actions_dict = {}
    for model_name, model_path in models.items():
        try:
            actions, illegal_count = evaluator.get_model_actions(model_path, price_data)
            actions_dict[model_name] = (actions, illegal_count)
            print(
                f"Generated actions for {model_name}: {len(actions)} steps, illegal actions: {illegal_count}"
            )
        except Exception as e:
        for regime, metric in metrics.items():
            result_dict["regime_metrics"][model_name][regime] = {  # type: ignore
                "sharpe_ratio": metric.sharpe_ratio,
                "win_rate": metric.win_rate,
                "total_return": metric.total_return,
                "max_drawdown": metric.max_drawdown,
                "volatility": metric.volatility,
                "trade_count": metric.trade_count,
                "action_distribution": metric.action_distribution,
                "illegal_actions": metric.illegal_actions,
            }

    # JSONとして保存
    safe_json_dump(result_dict, Path(result_file), indent=2, default=str)

    print(f"Results saved to {result_file}")

    # 可視化
    plot_file = output_dir / "regime_analysis.png"
    evaluator.plot_regime_analysis(result, str(plot_file))

if __name__ == "__main__":
    main()
