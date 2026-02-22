#!/usr/bin/env python3
"""Quick script to check schema status"""
from ztb.training.core.feature_schema_manager import FeatureSchemaManager

if __name__ == "__main__":
    FeatureSchemaManager.print_schema_summary()
