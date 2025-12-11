#!/usr/bin/env python3
"""
Volume Data Correction Script

This script corrects Yahoo Finance BTC-JPY volume data by converting
BTC quantities to JPY amounts using the corresponding close prices.

Usage:
    python tools/analysis/correct_volume_data.py --input <input_csv> --output <output_csv>
"""

import argparse
import logging

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VolumeDataCorrector:
    """Corrects Yahoo Finance volume data from BTC quantities to JPY amounts"""

    def __init__(self):
        self.correction_stats = {
            "total_rows": 0,
            "corrected_rows": 0,
            "invalid_prices": 0,
            "invalid_volumes": 0,
        }

    def correct_volume_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Correct volume data by converting BTC quantities to JPY amounts

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with corrected volume data
        """
        logger.info("Starting volume data correction...")

        # Make a copy to avoid modifying original
        corrected_df = df.copy()
        self.correction_stats["total_rows"] = len(corrected_df)

        # Validate required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [
            col for col in required_columns if col not in corrected_df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert volume from BTC quantity to JPY amount
        corrected_volumes = []
        for idx, row in corrected_df.iterrows():
            try:
                close_price = float(row["close"])
                volume_btc = float(row["volume"])

                # Skip invalid prices
                if close_price <= 0:
                    logger.warning(f"Invalid close price at index {idx}: {close_price}")
                    self.correction_stats["invalid_prices"] += 1
                    corrected_volumes.append(volume_btc)  # Keep original
                    continue

                # Skip invalid volumes
                if volume_btc <= 0:
                    logger.warning(f"Invalid volume at index {idx}: {volume_btc}")
                    self.correction_stats["invalid_volumes"] += 1
                    corrected_volumes.append(volume_btc)  # Keep original
                    continue

                # Check if volume is likely a JPY amount (very large numbers)
                # Yahoo Finance BTC-JPY volumes are typically in the range of 1e12 to 1e15
                # If volume is in this range, it's likely JPY amount, not BTC quantity
                if 1e10 <= volume_btc <= 1e16:
                    # Convert JPY amount to BTC quantity
                    volume_btc_corrected = volume_btc / close_price
                    corrected_volumes.append(volume_btc_corrected)
                    self.correction_stats["corrected_rows"] += 1
                    logger.debug(
                        f"Corrected volume at {idx}: {volume_btc:.2f} JPY -> {volume_btc_corrected:.6f} BTC"
                    )
                else:
                    # Volume is already in reasonable BTC range, keep as is
                    corrected_volumes.append(volume_btc)

            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing row {idx}: {e}")
                corrected_volumes.append(row["volume"])  # Keep original

        corrected_df["volume"] = corrected_volumes

        logger.info("Volume correction completed.")
        logger.info(f"Statistics: {self.correction_stats}")

        return corrected_df

    def validate_correction(
        self, original_df: pd.DataFrame, corrected_df: pd.DataFrame
    ) -> dict:
        """
        Validate the volume correction results

        Args:
            original_df: Original DataFrame
            corrected_df: Corrected DataFrame

        Returns:
            Validation results dictionary
        """
        validation_results = {
            "original_volume_range": {
                "min": float(original_df["volume"].min()),
                "max": float(original_df["volume"].max()),
                "mean": float(original_df["volume"].mean()),
            },
            "corrected_volume_range": {
                "min": float(corrected_df["volume"].min()),
                "max": float(corrected_df["volume"].max()),
                "mean": float(corrected_df["volume"].mean()),
            },
            "volume_ratio_check": [],
            "correction_stats": self.correction_stats,
        }

        # Check volume ratios for reasonableness
        for idx, (orig_vol, corr_vol, close_price) in enumerate(
            zip(original_df["volume"], corrected_df["volume"], corrected_df["close"])
        ):
            if orig_vol > 0 and close_price > 0:
                ratio = corr_vol / orig_vol
                expected_ratio = close_price

                # If original volume was JPY amount, ratio should be close to 1/close_price
                if 1e10 <= orig_vol <= 1e16:
                    expected_ratio = 1.0 / close_price
                    ratio_diff = (
                        abs(ratio - expected_ratio) / expected_ratio
                        if expected_ratio > 0
                        else 0
                    )
                    if ratio_diff > 0.1:  # More than 10% difference
                        validation_results["volume_ratio_check"].append(
                            {
                                "index": idx,
                                "original_volume": float(orig_vol),
                                "corrected_volume": float(corr_vol),
                                "close_price": float(close_price),
                                "ratio": float(ratio),
                                "expected_ratio": float(expected_ratio),
                                "ratio_difference_pct": float(ratio_diff * 100),
                            }
                        )

        return validation_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Correct Yahoo Finance BTC-JPY volume data"
    )
    from ztb.utils.cli import add_common_cli_args

    add_common_cli_args(parser)
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file path")
    parser.add_argument(
        "--validate", action="store_true", help="Perform validation after correction"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level",
    )

    args = parser.parse_args()

    from ztb.utils.cli import configure_logging_from_args

    configure_logging_from_args(args)

    try:
        # Read input data
        logger.info(f"Reading data from {args.input}")
        df = pd.read_csv(args.input, index_col=0, parse_dates=True)

        logger.info(f"Loaded {len(df)} rows of data")
        logger.info(
            f"Original volume range: {df['volume'].min():.2e} - {df['volume'].max():.2e}"
        )

        # Initialize corrector
        corrector = VolumeDataCorrector()

        # Correct volume data
        corrected_df = corrector.correct_volume_data(df)

        logger.info(
            f"Corrected volume range: {corrected_df['volume'].min():.2e} - {corrected_df['volume'].max():.2e}"
        )

        # Save corrected data
        logger.info(f"Saving corrected data to {args.output}")
        corrected_df.to_csv(args.output)
        logger.info("Data saved successfully")

        # Validation if requested
        if args.validate:
            logger.info("Performing validation...")
            validation_results = corrector.validate_correction(df, corrected_df)

            logger.info("Validation Results:")
            logger.info(
                f"Original volume range: {validation_results['original_volume_range']}"
            )
            logger.info(
                f"Corrected volume range: {validation_results['corrected_volume_range']}"
            )
            logger.info(
                f"Correction statistics: {validation_results['correction_stats']}"
            )

            if validation_results["volume_ratio_check"]:
                logger.warning(
                    f"Found {len(validation_results['volume_ratio_check'])} suspicious volume ratios"
                )
                for check in validation_results["volume_ratio_check"][
                    :5
                ]:  # Show first 5
                    logger.warning(
                        f"Index {check['index']}: ratio diff {check['ratio_difference_pct']:.1f}%"
                    )
            else:
                logger.info("All volume ratios appear reasonable")

        logger.info("✅ Volume data correction completed successfully!")

    except Exception as e:
        logger.error(f"❌ Error during volume correction: {e}")
        return 1


if __name__ == "__main__":
    from ztb.utils.cli import run_main

    run_main(main)
