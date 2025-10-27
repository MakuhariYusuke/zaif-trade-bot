"""
Feature Importance Analysis for SAC v427

Analyzes current 156D feature set to identify:
- High-importance features that should be preserved
- Potentially redundant features for removal consideration
- Feature correlation analysis
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class FeatureImportanceAnalyzer:
    """Analyzes feature importance and redundancy in SAC v427 feature set."""

    def __init__(self, feature_engineer):
        self.feature_engineer = feature_engineer
        self.importance_scores = {}
        self.correlation_matrix = None

    def analyze_feature_importance(
        self, df: pd.DataFrame, target_col: str = "future_return"
    ) -> Dict[str, float]:
        """
        Analyze feature importance using correlation with target and feature interactions.

        Args:
            df: DataFrame with features and target
            target_col: Target column for importance analysis

        Returns:
            Dictionary of feature importance scores
        """
        if target_col not in df.columns:
            # Create synthetic target for analysis
            df = df.copy()
            df[target_col] = df["close"].shift(-1) / df["close"] - 1

        # Calculate correlation with target
        feature_cols = [
            col
            for col in df.columns
            if col not in ["close", "high", "low", "open", "volume", target_col]
        ]
        correlations = {}

        for col in feature_cols:
            corr = abs(df[col].corr(df[target_col]))
            if not np.isnan(corr):
                correlations[col] = corr

        # Sort by importance
        self.importance_scores = dict(
            sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        )

        return self.importance_scores

    def analyze_feature_correlations(
        self, df: pd.DataFrame, threshold: float = 0.95
    ) -> Dict[str, List[str]]:
        """
        Identify highly correlated feature groups for redundancy analysis.

        Args:
            df: DataFrame with features
            threshold: Correlation threshold for redundancy

        Returns:
            Dictionary mapping features to their highly correlated counterparts
        """
        feature_cols = [
            col
            for col in df.columns
            if col not in ["close", "high", "low", "open", "volume"]
        ]
        feature_df = df[feature_cols].fillna(0)

        # Calculate correlation matrix
        self.correlation_matrix = feature_df.corr()

        # Find highly correlated pairs
        correlated_pairs = {}
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i + 1, len(self.correlation_matrix.columns)):
                corr_val = abs(self.correlation_matrix.iloc[i, j])
                if corr_val > threshold:
                    col_i = self.correlation_matrix.columns[i]
                    col_j = self.correlation_matrix.columns[j]

                    if col_i not in correlated_pairs:
                        correlated_pairs[col_i] = []
                    if col_j not in correlated_pairs:
                        correlated_pairs[col_j] = []

                    correlated_pairs[col_i].append((col_j, corr_val))
                    correlated_pairs[col_j].append((col_i, corr_val))

        return correlated_pairs

    def categorize_features_by_importance(
        self, importance_threshold: float = 0.1
    ) -> Dict[str, List[str]]:
        """
        Categorize features by importance levels.

        Args:
            importance_threshold: Threshold for high importance

        Returns:
            Dictionary with high, medium, low importance feature lists
        """
        high_importance = []
        medium_importance = []
        low_importance = []

        for feature, score in self.importance_scores.items():
            if score >= importance_threshold:
                high_importance.append(feature)
            elif score >= importance_threshold * 0.5:
                medium_importance.append(feature)
            else:
                low_importance.append(feature)

        return {
            "high_importance": high_importance,
            "medium_importance": medium_importance,
            "low_importance": low_importance,
        }

    def generate_removal_recommendations(
        self,
        correlated_pairs: Dict[str, List[str]],
        importance_scores: Dict[str, float],
    ) -> List[Tuple[str, str, float]]:
        """
        Generate recommendations for feature removal based on redundancy and importance.

        Returns:
            List of (feature_to_remove, reason, correlation_score) tuples
        """
        recommendations = []

        for feature, correlated_list in correlated_pairs.items():
            if not correlated_list:
                continue

            # Find the most correlated feature
            most_correlated, corr_score = max(correlated_list, key=lambda x: x[1])

            # Only recommend removal if both features exist in importance scores
            if feature in importance_scores and most_correlated in importance_scores:
                feature_imp = importance_scores[feature]
                correlated_imp = importance_scores[most_correlated]

                # Recommend removing the less important one
                if feature_imp < correlated_imp:
                    recommendations.append(
                        (
                            feature,
                            f"Highly correlated with {most_correlated} (r={corr_score:.3f}), lower importance",
                            corr_score,
                        )
                    )
                elif correlated_imp < feature_imp:
                    recommendations.append(
                        (
                            most_correlated,
                            f"Highly correlated with {feature} (r={corr_score:.3f}), lower importance",
                            corr_score,
                        )
                    )

        # Remove duplicates and sort by correlation strength
        unique_recommendations = []
        seen = set()
        for rec in recommendations:
            if rec[0] not in seen:
                unique_recommendations.append(rec)
                seen.add(rec[0])

        return sorted(unique_recommendations, key=lambda x: x[2], reverse=True)


def analyze_current_features(feature_engineer, sample_data_path: str = None):
    """
    Main analysis function for current SAC v427 features.

    Args:
        feature_engineer: SACv427FeatureEngineer instance
        sample_data_path: Path to sample market data for analysis
    """
    print("=== SAC v427 Feature Importance Analysis ===\n")

    # Create sample data if not provided
    if sample_data_path is None:
        dates = pd.date_range("2023-01-01", periods=1000, freq="1H")
        np.random.seed(42)

        # Generate synthetic OHLCV data
        close = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.02, 1000)))
        high = close * (1 + np.random.uniform(0, 0.01, 1000))
        low = close * (1 - np.random.uniform(0, 0.01, 1000))
        open_price = close + np.random.normal(0, close * 0.005, 1000)
        volume = np.random.uniform(1000, 10000, 1000)

        sample_df = pd.DataFrame(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )
    else:
        sample_df = pd.read_csv(sample_data_path)

    print(f"Analyzing features with {len(sample_df)} data points...")

    # Generate features
    feature_df = feature_engineer.generate_v427_features(sample_df.copy())
    total_features = len(feature_df.columns) - len(sample_df.columns)

    print(f"Generated {total_features} features")

    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer(feature_engineer)

    # Analyze importance
    importance_scores = analyzer.analyze_feature_importance(feature_df)

    # Categorize features
    categories = analyzer.categorize_features_by_importance()

    print("\n=== Feature Importance Summary ===")
    print(
        f"High importance features (>0.1 correlation): {len(categories['high_importance'])}"
    )
    print(
        f"Medium importance features (0.05-0.1): {len(categories['medium_importance'])}"
    )
    print(f"Low importance features (<0.05): {len(categories['low_importance'])}")

    # Analyze correlations
    correlated_pairs = analyzer.analyze_feature_correlations(feature_df, threshold=0.95)

    print(f"\nHighly correlated feature pairs (r>0.95): {len(correlated_pairs)}")

    # Generate removal recommendations
    recommendations = analyzer.generate_removal_recommendations(
        correlated_pairs, importance_scores
    )

    print("\n=== Feature Removal Recommendations ===")
    print(f"Potential redundant features to consider: {len(recommendations)}")

    if recommendations:
        print("\nTop 10 recommendations:")
        for i, (feature, reason, corr) in enumerate(recommendations[:10]):
            print(f"{i+1}. Remove '{feature}' - {reason}")

    return {
        "importance_scores": importance_scores,
        "categories": categories,
        "correlated_pairs": correlated_pairs,
        "recommendations": recommendations,
        "total_features": total_features,
    }


if __name__ == "__main__":
    # This would be run with actual feature engineer instance
    print("Feature importance analysis script created.")
    print(
        'Run with: python -c "from feature_importance_analysis import analyze_current_features; analyze_current_features(feature_engineer_instance)"'
    )
