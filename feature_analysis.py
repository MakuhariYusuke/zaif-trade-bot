import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split

# Import existing utilities
from ztb.preprocessing.feature_correlation_filter import FeatureCorrelationProcessor
from ztb.features.registry import FeatureRegistry
from ztb.utils.data_utils import load_csv_data


def load_data(file_path: str) -> pd.DataFrame:
    """Load the dataset using common utility"""
    return load_csv_data(file_path)


def identify_features(df: pd.DataFrame) -> list[str]:
    """Identify feature columns using FeatureRegistry"""
    # Use existing feature registry to get available features
    try:
        available_features = FeatureRegistry.list()
        # Filter to features present in dataframe
        features = [f for f in available_features if f in df.columns and df[f].dtype in ["float64", "int64"]]
        return features
    except:
        # Fallback to original logic
        exclude_cols = ["ts", "pair", "side", "pnl", "win", "source", "timestamp"]
        features = [
            col
            for col in df.columns
            if col not in exclude_cols and df[col].dtype in ["float64", "int64"]
        ]
        return features


def analyze_correlation(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Create correlation matrix and identify multicollinearity using existing processor"""
    processor = FeatureCorrelationProcessor()
    corr_matrix = processor.analyze_correlations(df, features)
    
    # Get high correlation pairs
    high_corr_pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            corr_value = corr_matrix.loc[features[i], features[j]]
            if isinstance(corr_value, (int, float)) and abs(corr_value) > 0.8:
                high_corr_pairs.append(
                    {
                        "feature1": features[i],
                        "feature2": features[j],
                        "correlation": float(corr_value),
                    }
                )
    return corr_matrix, high_corr_pairs


def analyze_distributions(df: pd.DataFrame, features: list[str]) -> dict[str, dict[str, object]]:
    """Analyze distributions: normality, missing values, outliers"""
    distribution_analysis = {}
    for feature in features:
        data = df[feature].dropna()
        missing_count = df[feature].isnull().sum()
        missing_percent = missing_count / len(df) * 100

        # Normality test (Shapiro-Wilk)
        if len(data) > 5000:
            # For large samples, use subset
            data_sample = data.sample(5000, random_state=42)
        else:
            data_sample = data
        try:
            stat, p_value = stats.shapiro(data_sample)
            normality = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "is_normal": bool(p_value > 0.05),
            }
        except:
            normality = {"statistic": 0.0, "p_value": 0.0, "is_normal": False}

        # Outliers (IQR method)
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_count = ((data < lower_bound) | (data > upper_bound)).sum()
        outliers_percent = outliers_count / len(data) * 100

        distribution_analysis[feature] = {
            "missing_count": int(missing_count),
            "missing_percent": float(missing_percent),
            "normality": normality,
            "outliers_count": int(outliers_count),
            "outliers_percent": float(outliers_percent),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
        }
    return distribution_analysis


def calculate_permutation_importance(
        df: pd.DataFrame, 
        features: list[str], 
        target: str = "win"
    ) -> tuple[dict[str, dict[str, float]], list[tuple[str, dict[str, float]]]]:
    """Calculate permutation importance"""
    # Prepare data
    X = df[features].fillna(0)  # Simple imputation
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a simple model (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42
    )

    # Create importance dict
    importance_dict = {}
    for i, feature in enumerate(features):
        importance_dict[feature] = {
            "importance_mean": float(perm_importance.importances_mean[i]),  # type: ignore
            "importance_std": float(perm_importance.importances_std[i]),  # type: ignore
        }

    # Sort by importance
    sorted_importance = sorted(
        importance_dict.items(), key=lambda x: x[1]["importance_mean"], reverse=True
    )

    return importance_dict, sorted_importance


def suggest_feature_reduction(importance_dict: dict[str, dict[str, float]], threshold_percentile: float = 25) -> tuple[list[str], float]:
    """Suggest features for reduction based on low importance"""
    importances = [v["importance_mean"] for v in importance_dict.values()]
    threshold = np.percentile(importances, threshold_percentile)
    low_importance_features = [
        k for k, v in importance_dict.items() if v["importance_mean"] < threshold
    ]
    return low_importance_features, float(threshold)


def main() -> None:
    # Load data
    data_path = "ml-dataset-enhanced.csv"
    df = load_data(data_path)

    # Identify features
    features = identify_features(df)
    print(f"Identified {len(features)} features")

    # Correlation analysis
    corr_matrix, high_corr_pairs = analyze_correlation(df, features)
    print(f"Found {len(high_corr_pairs)} highly correlated pairs (>0.8)")

    # Distribution analysis
    distribution_analysis = analyze_distributions(df, features)
    print("Distribution analysis completed")

    # Permutation importance
    importance_dict, sorted_importance = calculate_permutation_importance(df, features)
    print("Permutation importance calculated")

    # Feature reduction suggestions
    low_importance_features, threshold = suggest_feature_reduction(importance_dict)
    print(
        f"Suggested {len(low_importance_features)} features for reduction (below {float(threshold):.6f})"
    )

    # Prepare report
    report = {
        "analysis_date": datetime.now().isoformat(),
        "dataset_info": {
            "file": data_path,
            "total_rows": len(df),
            "features_count": len(features),
        },
        "correlation_analysis": {
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlation_pairs": high_corr_pairs
        },
        "distribution_analysis": distribution_analysis,
        "permutation_importance": {
            "importance_scores": importance_dict,
            "sorted_importance": sorted_importance,
            "low_importance_features": low_importance_features,
            "reduction_threshold": float(threshold),
        },
    }

    # Save report
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/feature_analysis_20251003.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
