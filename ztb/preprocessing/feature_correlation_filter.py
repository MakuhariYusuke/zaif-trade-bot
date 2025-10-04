#!/usr/bin/env python3
"""Feature correlation analysis and preprocessing pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from ztb.utils.data_utils import load_csv_data
from ztb.utils.errors import safe_operation
from ztb.utils.logging_utils import get_logger

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = get_logger(__name__)


class FeatureCorrelationProcessor:
    """Process and filter features based on correlation analysis."""

    def __init__(self, correlation_threshold: float = 0.95):
        """
        Initialize feature correlation processor.

        Args:
            correlation_threshold: Threshold for removing highly correlated features
        """
        self.correlation_threshold = correlation_threshold
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.features_to_remove: set[str] = set()
        self.scaler = StandardScaler()

    def analyze_correlations(
        self, df: pd.DataFrame, feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        Analyze correlations between features.

        Args:
            df: Input dataframe
            feature_columns: List of feature column names

        Returns:
            Correlation matrix
        """
        return safe_operation(
            logger=LOGGER,
            operation=lambda: self._analyze_correlations_impl(df, feature_columns),
            context="correlation_analysis",
            default_result=pd.DataFrame(),
        )

    def _analyze_correlations_impl(
        self, df: pd.DataFrame, feature_columns: List[str]
    ) -> pd.DataFrame:
        """Implementation of correlation analysis."""
        # Select only feature columns
        feature_df = df[feature_columns].copy()

        # Handle missing values
        feature_df = feature_df.dropna()

        # Scale features for better correlation analysis
        scaled_features = self.scaler.fit_transform(feature_df)
        scaled_df = pd.DataFrame(scaled_features, columns=feature_columns)

        # Calculate correlation matrix
        self.correlation_matrix = scaled_df.corr()

        assert self.correlation_matrix is not None
        return self.correlation_matrix

    def identify_highly_correlated_features(self) -> Set[str]:
        """
        Identify features that are highly correlated and should be removed.

        Returns:
            Set of feature names to remove
        """
        if self.correlation_matrix is None:
            raise ValueError(
                "Correlation matrix not computed. Run analyze_correlations first."
            )

        features_to_remove = set()

        # Get upper triangle of correlation matrix
        upper = self.correlation_matrix.where(
            np.triu(np.ones_like(self.correlation_matrix), k=1).astype(bool)
        )

        # Find features with correlation above threshold
        high_corr_pairs = []
        for col in upper.columns:
            for idx in upper.index:
                if abs(upper.loc[idx, col]) > self.correlation_threshold:
                    high_corr_pairs.append((idx, col, upper.loc[idx, col]))

        # Sort by absolute correlation (highest first)
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        # Remove features (keep first occurrence, remove subsequent highly correlated ones)
        for feature1, feature2, corr in high_corr_pairs:
            if (
                feature1 not in features_to_remove
                and feature2 not in features_to_remove
            ):
                # Remove the feature with higher mean absolute correlation
                corr_with_others_1 = self.correlation_matrix[feature1].abs().sum()
                corr_with_others_2 = self.correlation_matrix[feature2].abs().sum()

                if corr_with_others_1 > corr_with_others_2:
                    features_to_remove.add(feature1)
                else:
                    features_to_remove.add(feature2)

        self.features_to_remove = features_to_remove
        return features_to_remove

    def filter_features(
        self, df: pd.DataFrame, feature_columns: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Filter out highly correlated features from dataframe.

        Args:
            df: Input dataframe
            feature_columns: Original feature column names

        Returns:
            Tuple of (filtered_dataframe, remaining_feature_columns)
        """
        if not self.features_to_remove:
            self.identify_highly_correlated_features()

        remaining_features = [
            f for f in feature_columns if f not in self.features_to_remove
        ]

        LOGGER.info(
            f"Removed {len(self.features_to_remove)} highly correlated features: {sorted(self.features_to_remove)}"
        )
        LOGGER.info(f"Remaining features: {len(remaining_features)}")

        return df[remaining_features], remaining_features

    def create_correlation_heatmap(self, output_path: Path) -> None:
        """
        Create and save correlation heatmap.

        Args:
            output_path: Path to save the heatmap
        """
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix not computed.")

        plt.figure(figsize=(20, 16))  # type: ignore[unreachable]

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))

        # Create heatmap
        sns.heatmap(
            self.correlation_matrix,
            mask=mask,
            annot=False,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )

        plt.title(
            f"Feature Correlation Matrix (Threshold: {self.correlation_threshold})",
            fontsize=16,
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        # Highlight features to remove
        if self.features_to_remove:
            for feature in self.features_to_remove:
                if feature in self.correlation_matrix.columns:
                    idx = self.correlation_matrix.columns.get_loc(feature)
                    plt.axhline(y=idx, color="red", alpha=0.3, linewidth=2)
                    plt.axvline(x=idx, color="red", alpha=0.3, linewidth=2)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def get_correlation_report(self) -> Dict[str, Any]:
        """
        Generate correlation analysis report.

        Returns:
            Dictionary containing analysis results
        """
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix not computed.")

        # Find most correlated feature pairs
        upper = self.correlation_matrix.where(  # type: ignore[unreachable]
            np.triu(np.ones_like(self.correlation_matrix), k=1).astype(bool)
        )

        # Get top correlations
        correlations = []
        for col in upper.columns:
            for idx in upper.index:
                corr_val = upper.loc[idx, col]
                if not np.isnan(corr_val):
                    correlations.append(
                        {"feature1": idx, "feature2": col, "correlation": corr_val}
                    )

        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return {
            "total_features": len(self.correlation_matrix.columns),
            "features_removed": len(self.features_to_remove),
            "features_remaining": len(self.correlation_matrix.columns)
            - len(self.features_to_remove),
            "correlation_threshold": self.correlation_threshold,
            "top_correlations": correlations[:20],  # Top 20 correlations
            "removed_features": sorted(list(self.features_to_remove)),
        }


def process_dataset(
    input_path: Path,
    output_path: Path,
    correlation_threshold: float = 0.95,
    feature_prefixes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Process dataset by removing highly correlated features.

    Args:
        input_path: Path to input dataset
        output_path: Path to save processed dataset
        correlation_threshold: Correlation threshold for feature removal
        feature_prefixes: List of feature prefixes to consider (e.g., ['rsi', 'macd'])

    Returns:
        Processing report
    """
    LOGGER.info(f"Loading dataset from {input_path}")
    df = load_csv_data(input_path)

    # Identify feature columns
    if feature_prefixes:
        feature_columns = []
        for prefix in feature_prefixes:
            feature_columns.extend(
                [col for col in df.columns if col.startswith(prefix)]
            )
    else:
        # Assume all numeric columns except timestamp/target are features
        feature_columns = [
            col
            for col in df.columns
            if col
            not in ["timestamp", "target", "close", "high", "low", "open", "volume"]
            and df[col].dtype in ["float64", "int64"]
        ]

    LOGGER.info(f"Identified {len(feature_columns)} feature columns")

    # Initialize processor
    processor = FeatureCorrelationProcessor(correlation_threshold)

    # Analyze correlations
    LOGGER.info("Analyzing feature correlations...")
    corr_matrix = processor.analyze_correlations(df, feature_columns)

    # Identify features to remove
    features_to_remove = processor.identify_highly_correlated_features()

    # Filter features
    filtered_df, remaining_features = processor.filter_features(df, feature_columns)

    # Save processed dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, index=False)

    # Create visualizations
    viz_dir = output_path.parent / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    processor.create_correlation_heatmap(viz_dir / "correlation_heatmap.png")

    # Generate report
    report = processor.get_correlation_report()
    report["input_file"] = str(input_path)
    report["output_file"] = str(output_path)
    report["original_features"] = feature_columns

    # Save report
    report_path = output_path.parent / "correlation_analysis_report.json"
    import json

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    LOGGER.info(f"Processing completed. Results saved to {output_path.parent}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze and filter highly correlated features"
    )
    parser.add_argument(
        "--input-path", type=Path, required=True, help="Path to input dataset"
    )
    parser.add_argument(
        "--output-path", type=Path, required=True, help="Path to save processed dataset"
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Correlation threshold for feature removal",
    )
    parser.add_argument(
        "--feature-prefixes",
        nargs="+",
        help="Feature prefixes to analyze (e.g., rsi macd)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    report = process_dataset(
        args.input_path,
        args.output_path,
        args.correlation_threshold,
        args.feature_prefixes,
    )

    LOGGER.info("Feature correlation analysis completed!")
    LOGGER.info(f"Original features: {report['total_features']}")
    LOGGER.info(f"Features removed: {report['features_removed']}")
    LOGGER.info(f"Features remaining: {report['features_remaining']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
