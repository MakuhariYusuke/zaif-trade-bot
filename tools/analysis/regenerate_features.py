#!/usr/bin/env python3
"""
Feature Re-generation Script

This script re-generates features from corrected volume data.
Uses the SAC v427 feature engineering system to create quality-filtered features.

Usage:
    python tools/analysis/regenerate_features.py --input <corrected_csv> --output <featured_csv>
"""

import argparse
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureRegenerator:
    """Re-generates features from corrected data"""

    def __init__(self):
        self.stats = {
            "original_features": 0,
            "generated_features": 0,
            "total_features": 0,
            "rows_processed": 0,
        }

    def regenerate_features(
        self, df: pd.DataFrame, feature_set: str = "full"
    ) -> pd.DataFrame:
        """
        Re-generate features from corrected data

        Args:
            df: DataFrame with corrected OHLCV data
            feature_set: Feature set to generate ("full", "high_quality", etc.)

        Returns:
            DataFrame with generated features
        """
        logger.info(f"Starting feature re-generation with {feature_set} feature set...")

        # Validate required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.stats["rows_processed"] = len(df)
        self.stats["original_features"] = len(df.columns)

        try:
            # Import the feature engineering system
            from ztb.features.sac_v427_feature_engineering import (
                generate_v427_quality_filtered_features,
            )

            # Generate features
            logger.info("Generating v427 quality-filtered features...")
            featured_df = generate_v427_quality_filtered_features(
                df.copy(), feature_set=feature_set
            )

            self.stats["total_features"] = len(featured_df.columns)
            self.stats["generated_features"] = (
                self.stats["total_features"] - self.stats["original_features"]
            )

            logger.info("Feature re-generation completed.")
            logger.info(f"Statistics: {self.stats}")

            return featured_df

        except ImportError as e:
            logger.error(f"Failed to import feature engineering module: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during feature generation: {e}")
            raise

    def validate_features(
        self, featured_df: pd.DataFrame, original_df: pd.DataFrame
    ) -> dict:
        """
        Validate generated features

        Args:
            featured_df: DataFrame with generated features
            original_df: Original DataFrame

        Returns:
            Validation results dictionary
        """
        validation_results = {
            "feature_count": {
                "original": len(original_df.columns),
                "generated": len(featured_df.columns) - len(original_df.columns),
                "total": len(featured_df.columns),
            },
            "data_integrity": {},
            "feature_quality": {},
        }

        # Check data integrity
        validation_results["data_integrity"] = {
            "rows_match": len(featured_df) == len(original_df),
            "index_preserved": featured_df.index.equals(original_df.index),
            "price_columns_preserved": all(
                col in featured_df.columns for col in ["open", "high", "low", "close"]
            ),
        }

        # Basic feature quality checks
        numeric_columns = featured_df.select_dtypes(include=[np.number]).columns
        validation_results["feature_quality"] = {
            "numeric_features": len(numeric_columns),
            "features_with_nan": featured_df.isna().any().sum(),
            "features_with_inf": np.isinf(
                featured_df.select_dtypes(include=[np.number])
            )
            .any()
            .any(),
            "constant_features": (featured_df.std() == 0).sum(),
        }

        return validation_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Re-generate features from corrected volume data"
    )
    # add common args (log-level etc)
    from ztb.utils.cli import add_common_cli_args

    add_common_cli_args(parser)
    parser.add_argument(
        "--input", "-i", required=True, help="Input CSV file path (corrected data)"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output CSV file path (featured data)"
    )
    parser.add_argument(
        "--feature-set",
        default="full",
        choices=["minimal", "high_quality", "no_harmful", "full"],
        help="Feature set to generate",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Perform validation after feature generation",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level",
    )

    args = parser.parse_args()

    # Set log level using centralized CLI helper
    from ztb.utils.cli import configure_logging_from_args

    configure_logging_from_args(args)

    try:
        # Read corrected data
        logger.info(f"Reading corrected data from {args.input}")
        df = pd.read_csv(args.input, index_col=0, parse_dates=True)

        logger.info(f"Loaded {len(df)} rows of corrected data")
        logger.info(f"Columns: {list(df.columns)}")

        # Initialize feature regenerator
        regenerator = FeatureRegenerator()

        # Re-generate features
        featured_df = regenerator.regenerate_features(df, feature_set=args.feature_set)

        logger.info(f"Generated {regenerator.stats['generated_features']} features")
        logger.info(f"Total features: {regenerator.stats['total_features']}")

        # Save featured data
        logger.info(f"Saving featured data to {args.output}")
        featured_df.to_csv(args.output)
        logger.info("Featured data saved successfully")

        # Validation if requested
        if args.validate:
            logger.info("Performing validation...")
            validation_results = regenerator.validate_features(featured_df, df)

            logger.info("Validation Results:")
            logger.info(f"Feature counts: {validation_results['feature_count']}")
            logger.info(f"Data integrity: {validation_results['data_integrity']}")
            logger.info(f"Feature quality: {validation_results['feature_quality']}")

            # Check for issues
            if validation_results["feature_quality"]["features_with_nan"] > 0:
                logger.warning(
                    f"Found {validation_results['feature_quality']['features_with_nan']} features with NaN values"
                )
            if validation_results["feature_quality"]["features_with_inf"] > 0:
                logger.warning("Found features with infinite values")
            if validation_results["feature_quality"]["constant_features"] > 0:
                logger.warning(
                    f"Found {validation_results['feature_quality']['constant_features']} constant features"
                )

        logger.info("✅ Feature re-generation completed successfully!")

    except Exception as e:
        logger.error(f"❌ Error during feature re-generation: {e}")
        return 1


if __name__ == "__main__":
    from ztb.utils.cli import run_main

    run_main(main)
