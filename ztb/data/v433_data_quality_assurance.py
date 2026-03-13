#!/usr/bin/env python3
"""
V433 Data Quality Assurance System
現実データ中心主義に基づく包括的なデータ品質管理
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ztb.metrics.metrics import kurtosis, skewness, test_normality
from ztb.utils.error_utils import safe_execute
from ztb.utils.file_utils import safe_json_dump

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityAssurance:
    """
    包括的なデータ品質保証システム
    v433設計原則: 現実データ中心主義
    """

    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: データ保存ディレクトリ
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def comprehensive_quality_check(
        self, df: pd.DataFrame, market: str = "BTC-JPY"
    ) -> dict[str, Any]:
        """
        包括的なデータ品質チェックを実行

        Args:
            df: チェック対象のデータ
            market: 市場識別子

        Returns:
            品質チェック結果
        """
        logger.info(f"Starting comprehensive quality check for {market}")

        results = {
            "basic_integrity": self._check_basic_integrity(df),
            "statistical_properties": self._check_statistical_properties(df),
            "market_realism": self._check_market_realism(df, market),
            "temporal_consistency": self._check_temporal_consistency(df),
            "anomaly_detection": self._detect_anomalies(df),
            "data_completeness": self._assess_completeness(df),
            "overall_score": 0.0,
            "recommendations": [],
        }

        # 全体スコアの計算
        results["overall_score"] = self._calculate_overall_score(results)

        # 改善推奨事項の生成
        results["recommendations"] = self._generate_recommendations(results)

        logger.info(
            f"Quality check completed. Overall score: {results['overall_score']:.2f}"
        )
        return results

    def _check_basic_integrity(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        基本的なデータ整合性チェック
        """
        integrity = {
            "total_records": len(df),
            "missing_values": {},
            "duplicate_timestamps": 0,
            "invalid_prices": {},
            "data_types": {},
            "score": 0.0,
        }

        # 欠損値チェック
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            integrity["missing_values"][col] = missing_count

        # 重複タイムスタンプ
        if df.index.name == "Date" or isinstance(df.index, pd.DatetimeIndex):
            integrity["duplicate_timestamps"] = df.index.duplicated().sum()

        # 価格の妥当性チェック
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                invalid_count = (df[col] <= 0).sum()
                integrity["invalid_prices"][col] = invalid_count

        # データ型チェック
        for col in df.columns:
            integrity["data_types"][col] = str(df[col].dtype)

        # スコア計算
        total_missing = sum(integrity["missing_values"].values())
        total_invalid = sum(integrity["invalid_prices"].values())

        missing_penalty = min(total_missing / len(df), 1.0) if len(df) > 0 else 1.0
        invalid_penalty = min(total_invalid / len(df), 1.0) if len(df) > 0 else 1.0
        duplicate_penalty = (
            min(integrity["duplicate_timestamps"] / len(df), 1.0)
            if len(df) > 0
            else 1.0
        )

        integrity["score"] = 1.0 - (
            missing_penalty * 0.4 + invalid_penalty * 0.4 + duplicate_penalty * 0.2
        )

        return integrity

    def _check_statistical_properties(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        統計的特性のチェック
        """
        stats_check = {
            "price_distribution": {},
            "volatility_analysis": {},
            "autocorrelation": {},
            "score": 0.0,
        }

        if "close" not in df.columns or len(df) < 30:
            stats_check["score"] = 0.0
            return stats_check

        close_prices = df["close"].dropna()

        # 価格分布のチェック
        def check_price_distribution():
            # 正規性の検定 (Shapiro-Wilk)
            if len(close_prices) <= 5000:  # Shapiro-Wilkの制限
                normality_results = test_normality(close_prices.values)
                shapiro_result = normality_results.get("shapiro_wilk", {})
                stat = shapiro_result.get("statistic")
                p_value = shapiro_result.get("p_value")
                stats_check["price_distribution"]["normality_test"] = {
                    "statistic": stat,
                    "p_value": p_value,
                    "is_normal": p_value is not None and p_value > 0.05,
                }

            # 歪度と尖度
            skewness_val = skewness(close_prices)
            kurtosis_val = kurtosis(close_prices)
            stats_check["price_distribution"]["skewness"] = skewness_val
            stats_check["price_distribution"]["kurtosis"] = kurtosis_val

            # 現実的な範囲チェック
            reasonable_skew = abs(skewness_val) < 2.0  # 過度な歪みなし
            reasonable_kurt = kurtosis_val < 10.0  # 過度な尖りなし
            stats_check["price_distribution"]["reasonable_distribution"] = (
                reasonable_skew and reasonable_kurt
            )

        safe_execute(check_price_distribution, error_msg="Error in distribution analysis")

        # ボラティリティ分析
        def check_volatility():
            returns = close_prices.pct_change().dropna()
            from ztb.metrics.technical import calculate_volatility_from_returns

            volatility = calculate_volatility_from_returns(
                returns, window=len(returns), annualize=True
            )

            stats_check["volatility_analysis"]["annualized_volatility"] = volatility
            stats_check["volatility_analysis"]["realistic_volatility"] = (
                0.1 <= volatility <= 2.0
            )  # 10%-200%の範囲

        safe_execute(check_volatility, error_msg="Error in volatility analysis")

        # 自己相関
        def check_autocorrelation():
            if len(returns) > 10:
                autocorr_1 = returns.autocorr(lag=1)
                autocorr_5 = returns.autocorr(lag=5)
                stats_check["autocorrelation"]["lag_1"] = autocorr_1
                stats_check["autocorrelation"]["lag_5"] = autocorr_5
                stats_check["autocorrelation"]["weak_autocorr"] = (
                    abs(autocorr_1) < 0.3
                )  # 弱い自己相関

        safe_execute(check_autocorrelation, error_msg="Error in autocorrelation analysis")

        # スコア計算
        score_components = []

        if "reasonable_distribution" in stats_check["price_distribution"]:
            score_components.append(
                stats_check["price_distribution"]["reasonable_distribution"]
            )

        if "realistic_volatility" in stats_check["volatility_analysis"]:
            score_components.append(
                stats_check["volatility_analysis"]["realistic_volatility"]
            )

        if "weak_autocorr" in stats_check["autocorrelation"]:
            score_components.append(stats_check["autocorrelation"]["weak_autocorr"])

        stats_check["score"] = np.mean(score_components) if score_components else 0.0

        return stats_check

    def _check_market_realism(self, df: pd.DataFrame, market: str) -> dict[str, Any]:
        """
        市場現実性のチェック
        """
        realism = {
            "price_range": {},
            "volume_analysis": {},
            "market_behavior": {},
            "score": 0.0,
        }

        if market == "BTC-JPY":
            # BTC/JPYの現実的な価格範囲 (2024年時点)
            expected_min = 5000000  # 500万円
            expected_max = 20000000  # 2000万円

            if "close" in df.columns:
                actual_min = df["close"].min()
                actual_max = df["close"].max()

                realism["price_range"] = {
                    "expected_min": expected_min,
                    "expected_max": expected_max,
                    "actual_min": actual_min,
                    "actual_max": actual_max,
                    "in_expected_range": expected_min <= actual_min
                    and actual_max <= expected_max,
                }

            # 出来高分析
            if "volume" in df.columns:
                avg_volume = df["volume"].mean()
                volume_variability = (
                    df["volume"].std() / df["volume"].mean()
                    if df["volume"].mean() > 0
                    else 0
                )

                realism["volume_analysis"] = {
                    "average_volume": avg_volume,
                    "volume_variability": volume_variability,
                    "realistic_volume": avg_volume > 1000000,  # 一定の出来高
                }

            # 市場行動のチェック
            if len(df) > 5:
                # 価格変動の連続性
                price_changes = df["close"].pct_change().dropna()
                extreme_changes = (abs(price_changes) > 0.5).sum()  # 50%以上の変動
                realism["market_behavior"]["extreme_changes"] = extreme_changes
                realism["market_behavior"]["reasonable_volatility"] = (
                    extreme_changes == 0
                )

        # スコア計算
        score_components = []

        if "in_expected_range" in realism["price_range"]:
            score_components.append(realism["price_range"]["in_expected_range"])

        if "realistic_volume" in realism["volume_analysis"]:
            score_components.append(realism["volume_analysis"]["realistic_volume"])

        if "reasonable_volatility" in realism["market_behavior"]:
            score_components.append(realism["market_behavior"]["reasonable_volatility"])

        realism["score"] = np.mean(score_components) if score_components else 0.0

        return realism

    def _check_temporal_consistency(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        時系列の一貫性チェック
        """
        temporal = {
            "gaps_analysis": {},
            "frequency_consistency": {},
            "business_days": {},
            "score": 0.0,
        }

        if not isinstance(df.index, pd.DatetimeIndex):
            temporal["score"] = 0.0
            return temporal

        # データのギャップ分析
        if len(df) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            expected_diff = (
                time_diffs.mode().iloc[0]
                if not time_diffs.mode().empty
                else pd.Timedelta(days=1)
            )

            gaps = time_diffs[time_diffs > expected_diff * 2]
            temporal["gaps_analysis"] = {
                "total_gaps": len(gaps),
                "max_gap_days": gaps.max().days if len(gaps) > 0 else 0,
                "expected_frequency": str(expected_diff),
                "significant_gaps": len(gaps) < len(df) * 0.1,  # 10%以内のギャップ
            }

        # 頻度の一貫性
        if len(df) > 10:
            # 週次パターン分析
            df["weekday"] = df.index.weekday
            weekday_counts = df["weekday"].value_counts()
            temporal["frequency_consistency"] = {
                "weekday_distribution": weekday_counts.to_dict(),
                "consistent_weekly_pattern": weekday_counts.std()
                / weekday_counts.mean()
                < 0.5,
            }

        # 営業日チェック (市場休場日の考慮)
        if len(df) > 30:
            # 土日データのチェック
            weekend_data = df[df.index.weekday >= 5]
            temporal["business_days"] = {
                "weekend_records": len(weekend_data),
                "mostly_business_days": len(weekend_data)
                < len(df) * 0.3,  # 30%以内の週末データ
            }

        # スコア計算
        score_components = []

        if "significant_gaps" in temporal["gaps_analysis"]:
            score_components.append(temporal["gaps_analysis"]["significant_gaps"])

        if "consistent_weekly_pattern" in temporal["frequency_consistency"]:
            score_components.append(
                temporal["frequency_consistency"]["consistent_weekly_pattern"]
            )

        if "mostly_business_days" in temporal["business_days"]:
            score_components.append(temporal["business_days"]["mostly_business_days"])

        temporal["score"] = np.mean(score_components) if score_components else 0.0

        return temporal

    def _detect_anomalies(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        異常値検知
        """
        anomalies = {
            "outlier_detection": {},
            "sudden_changes": {},
            "stale_prices": {},
            "score": 0.0,
        }

        if len(df) < 10:
            anomalies["score"] = 1.0  # 小さなデータセットは異常なしと仮定
            return anomalies

        # 外れ値検知 (Z-scoreベース)
        if "close" in df.columns:
            prices = df["close"].dropna()
            if len(prices) > 0:
                z_scores = np.abs(stats.zscore(prices))
                outliers = (z_scores > 3).sum()
                anomalies["outlier_detection"] = {
                    "outlier_count": outliers,
                    "outlier_percentage": outliers / len(prices),
                    "acceptable_outliers": outliers / len(prices)
                    < 0.05,  # 5%以内の外れ値
                }

        # 突然の変化検知
        if "close" in df.columns and len(df) > 5:
            returns = df["close"].pct_change().dropna()
            sudden_changes = (abs(returns) > 0.2).sum()  # 20%以上の変化
            anomalies["sudden_changes"] = {
                "sudden_change_count": sudden_changes,
                "acceptable_changes": sudden_changes
                < len(returns) * 0.1,  # 10%以内の突然変化
            }

        # 停滞価格検知 (同じ価格が続く)
        if "close" in df.columns:
            price_stability = df["close"].rolling(window=5).std()
            stale_periods = (price_stability == 0).sum()
            anomalies["stale_prices"] = {
                "stale_periods": stale_periods,
                "acceptable_staleness": stale_periods < len(df) * 0.1,  # 10%以内の停滞
            }

        # スコア計算
        score_components = []

        if "acceptable_outliers" in anomalies["outlier_detection"]:
            score_components.append(
                anomalies["outlier_detection"]["acceptable_outliers"]
            )

        if "acceptable_changes" in anomalies["sudden_changes"]:
            score_components.append(anomalies["sudden_changes"]["acceptable_changes"])

        if "acceptable_staleness" in anomalies["stale_prices"]:
            score_components.append(anomalies["stale_prices"]["acceptable_staleness"])

        anomalies["score"] = np.mean(score_components) if score_components else 0.0

        return anomalies

    def _assess_completeness(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        データ完全性の評価
        """
        completeness = {
            "expected_records": 0,
            "actual_records": len(df),
            "completeness_ratio": 0.0,
            "score": 0.0,
        }

        if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
            completeness["score"] = 0.0
            return completeness

        # 期待されるレコード数の計算
        start_date = df.index.min()
        end_date = df.index.max()
        time_diff = end_date - start_date

        # 頻度の推定
        if len(df) > 1:
            avg_interval = (df.index[1:] - df.index[:-1]).mean()
            if avg_interval < pd.Timedelta(days=1):
                # 日次データと仮定
                completeness["expected_records"] = time_diff.days + 1
            else:
                # それ以外の頻度
                completeness["expected_records"] = int(time_diff / avg_interval) + 1

        if completeness["expected_records"] > 0:
            completeness["completeness_ratio"] = (
                len(df) / completeness["expected_records"]
            )
            completeness["score"] = min(completeness["completeness_ratio"], 1.0)

        return completeness

    def _calculate_overall_score(self, results: dict[str, Any]) -> float:
        """
        全体品質スコアの計算
        """
        weights = {
            "basic_integrity": 0.3,
            "statistical_properties": 0.2,
            "market_realism": 0.25,
            "temporal_consistency": 0.15,
            "anomaly_detection": 0.1,
        }

        overall_score = 0.0
        total_weight = 0.0

        for component, weight in weights.items():
            if component in results and "score" in results[component]:
                score = results[component]["score"]
                if not np.isnan(score):
                    overall_score += score * weight
                    total_weight += weight

        return overall_score / total_weight if total_weight > 0 else 0.0

    def _generate_recommendations(self, results: dict[str, Any]) -> list[str]:
        """
        改善推奨事項の生成
        """
        recommendations = []

        # 基本整合性に関する推奨
        integrity = results.get("basic_integrity", {})
        if integrity.get("score", 1.0) < 0.9:
            if sum(integrity.get("missing_values", {}).values()) > 0:
                recommendations.append("欠損値の補完または除去を検討してください")
            if sum(integrity.get("invalid_prices", {}).values()) > 0:
                recommendations.append("無効な価格データをクリーニングしてください")
            if integrity.get("duplicate_timestamps", 0) > 0:
                recommendations.append("重複タイムスタンプを除去してください")

        # 統計的特性に関する推奨
        stats = results.get("statistical_properties", {})
        if stats.get("score", 1.0) < 0.8:
            if not stats.get("price_distribution", {}).get(
                "reasonable_distribution", True
            ):
                recommendations.append("価格分布の異常性を調査してください")
            if not stats.get("volatility_analysis", {}).get(
                "realistic_volatility", True
            ):
                recommendations.append("ボラティリティの妥当性を確認してください")

        # 市場現実性に関する推奨
        realism = results.get("market_realism", {})
        if realism.get("score", 1.0) < 0.8:
            if not realism.get("price_range", {}).get("in_expected_range", True):
                recommendations.append("価格範囲が市場の実勢と乖離しています")
            if not realism.get("volume_analysis", {}).get("realistic_volume", True):
                recommendations.append("出来高データの妥当性を確認してください")

        # 時系列一貫性に関する推奨
        temporal = results.get("temporal_consistency", {})
        if temporal.get("score", 1.0) < 0.8:
            if not temporal.get("gaps_analysis", {}).get("significant_gaps", True):
                recommendations.append("データギャップを補完してください")
            if not temporal.get("business_days", {}).get("mostly_business_days", True):
                recommendations.append("週末データの妥当性を確認してください")

        # 異常値に関する推奨
        anomalies = results.get("anomaly_detection", {})
        if anomalies.get("score", 1.0) < 0.8:
            recommendations.append("異常値の除去または調査を検討してください")

        # 全体スコアに基づく推奨
        overall_score = results.get("overall_score", 0.0)
        if overall_score < 0.7:
            recommendations.append(
                "データ品質が低いため、データソースの見直しを推奨します"
            )
        elif overall_score < 0.9:
            recommendations.append("軽微な品質問題がありますが、使用可能です")

        return recommendations

    def save_quality_report(self, results: dict[str, Any], filename: str) -> str:
        """
        品質レポートを保存

        Args:
            results: 品質チェック結果
            filename: ファイル名

        Returns:
            保存されたファイルのパス
        """

        def make_serializable(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            elif isinstance(obj, pd.Timedelta):
                return str(obj)
            else:
                return obj

        serializable_results = make_serializable(results)

        report_path = self.data_dir / f"{filename}_quality_report.json"
        safe_json_dump(serializable_results, str(report_path), indent=2, ensure_ascii=False)

        logger.info(f"Quality report saved to {report_path}")
        return str(report_path)

def main():
    """
    メイン実行関数
    """
    dqa = DataQualityAssurance()

    # 最新のYahoo Financeデータを読み込み
    data_files = list(Path("data").glob("btc_jpy_yahoo_real_*.csv"))
    if not data_files:
        logger.error("No BTC/JPY data files found")
        return

    # 最新のファイルを使用
    latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Analyzing {latest_file}")

    # データ読み込み
    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)

    # 包括的な品質チェック
    quality_results = dqa.comprehensive_quality_check(df, market="BTC-JPY")

    # 結果表示
    print("\n=== Data Quality Assessment Results ===")
    print(f"Overall Quality Score: {quality_results['overall_score']:.3f}")
    print(f"Total Records: {quality_results['basic_integrity']['total_records']}")

    print("\nRecommendations:")
    for rec in quality_results["recommendations"]:
        print(f"- {rec}")

    # レポート保存
    filename = latest_file.stem
    dqa.save_quality_report(quality_results, filename)

if __name__ == "__main__":
    main()
