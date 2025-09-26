#!/usr/bin/env python3
"""
Test script for Step 6: Data pipeline enhancement
"""

import tempfile
from pathlib import Path
from ztb.data.integrity_checker import ParquetIntegrityChecker
from ztb.data.binance_data import save_parquet_chunked, load_parquet_pattern
import pandas as pd
import numpy as np

def test_parquet_integrity_checker():
    print('=== Step 6 Dry-run: Data Pipeline Enhancement ===')

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "test_data"
        data_dir.mkdir()

        # Create test data with some issues
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)

        # Introduce some issues
        # Add duplicate
        df = pd.concat([df, df.iloc[:5]])  # Duplicate first 5 rows

        # Create gap (remove some rows)
        df = df.drop(df.index[10:15])

        # Save test data
        save_parquet_chunked(df, str(data_dir))

        print(f'Created test data with {len(df)} records (including duplicates and gaps)')

        # Test integrity checker
        checker = ParquetIntegrityChecker(data_dir=str(data_dir))
        # Use mock notifier
        class MockNotifier:
            def notify(self, message, level="info", data=None):
                print(f"[{level.upper()}] {message}")
        checker.notifier = MockNotifier()

        # Check integrity
        report = checker.check_integrity()
        print(f'Integrity check results:')
        print(f'  Total records: {report["total_records"]}')
        print(f'  Duplicate records: {report["duplicate_records"]}')
        print(f'  Gaps found: {len(report["gaps"])}')
        print(f'  Integrity OK: {report["is_integrity_ok"]}')

        # Test repair
        print('\nTesting auto-repair...')
        repair_success = checker.repair_integrity(report, auto_repair=True)
        print(f'Repair successful: {repair_success}')

        # Re-check after repair
        if repair_success:
            final_report = checker.check_integrity()
            print(f'After repair:')
            print(f'  Total records: {final_report["total_records"]}')
            print(f'  Duplicate records: {final_report["duplicate_records"]}')
            print(f'  Gaps found: {len(final_report["gaps"])}')
            print(f'  Integrity OK: {final_report["is_integrity_ok"]}')

    print('=== Step 6 Test Completed ===')

if __name__ == "__main__":
    test_parquet_integrity_checker()