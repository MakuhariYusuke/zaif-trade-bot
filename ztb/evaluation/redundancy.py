"""
Feature redundancy reduction utilities.

This module provides functionality for detecting and removing redundant features
using correlation clustering and other redundancy metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.cluster import AgglomerativeClustering


def calculate_feature_correlations(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for features

    Args:
        feature_df: DataFrame of features

    Returns:
        Correlation matrix
    """
    # Remove constant columns
    feature_df = feature_df.loc[:, feature_df.nunique() > 1]

    if feature_df.empty:
        return pd.DataFrame()

    # Calculate correlation matrix
    corr_matrix = feature_df.corr()

    # Fill NaN values (from constant columns) with 0
    corr_matrix = corr_matrix.fillna(0)

    return corr_matrix


def find_highly_correlated_features(corr_matrix: pd.DataFrame,
                                   threshold: float = 0.8) -> List[Tuple[str, str, float]]:
    """
    Find pairs of highly correlated features

    Args:
        corr_matrix: Feature correlation matrix
        threshold: Correlation threshold

    Returns:
        List of (feature1, feature2, correlation) tuples
    """
    if corr_matrix.empty:
        return []

    correlated_pairs = []

    # Get upper triangle of correlation matrix
    corr_upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))

    # Find correlations above threshold
    high_corr = corr_upper.abs() > threshold
    high_corr_pairs = high_corr.stack()

    for idx, flag in high_corr_pairs.items():
        # 値は通常 bool（相関判定）か NaN。型安全にフィルタ。
        if not isinstance(flag, (bool, np.bool_)):
            continue  # NaN や想定外型は無視
        if not flag:
            continue
        if isinstance(idx, tuple) and len(idx) == 2:
            feature1, feature2 = idx
            correlation = corr_matrix.loc[feature1, feature2]
            correlated_pairs.append((feature1, feature2, correlation))
        # それ以外は無視（想定外フォーマット対策）

    return correlated_pairs


def cluster_features_by_correlation(feature_df: pd.DataFrame,
                                   distance_threshold: float = 0.3) -> Dict[int, List[str]]:
    """
    Cluster features based on correlation distance

    Args:
        feature_df: DataFrame of features
        distance_threshold: Distance threshold for clustering (lower = more clusters)

    Returns:
        Dictionary mapping cluster IDs to feature names
    """
    if feature_df.empty or len(feature_df.columns) < 2:
        return {0: list(feature_df.columns)}

    # Calculate correlation matrix
    corr_matrix = calculate_feature_correlations(feature_df)

    if corr_matrix.empty:
        return {0: list(feature_df.columns)}

    # Convert correlation to distance (1 - |correlation|)
    distance_matrix = 1 - corr_matrix.abs()

    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage='average'
    )

    try:
        cluster_labels = clustering.fit_predict(distance_matrix)

        # Group features by cluster
        clusters = {}
        for feature, cluster_id in zip(corr_matrix.columns, cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(feature)

        return clusters

    except Exception as e:
        print(f"Error in correlation clustering: {e}")
        return {0: list(feature_df.columns)}


def select_representative_features_from_clusters(clusters: Dict[int, List[str]],
                                               feature_df: pd.DataFrame,
                                               target_returns: pd.Series,
                                               method: str = 'correlation') -> List[str]:
    """
    Select representative features from each cluster

    Args:
        clusters: Feature clusters
        feature_df: DataFrame of features
        target_returns: Target returns for evaluation
        method: Selection method ('correlation', 'variance', 'random')

    Returns:
        List of selected feature names
    """
    selected_features = []

    for cluster_id, features in clusters.items():
        if len(features) == 1:
            # Single feature in cluster - keep it
            selected_features.extend(features)
            continue

        if method == 'correlation':
            # Select feature with highest absolute correlation to target
            best_feature = None
            best_corr = 0

            for feature in features:
                if feature in feature_df.columns:
                    aligned_data = pd.concat([feature_df[feature], target_returns], axis=1).dropna()
                    if len(aligned_data) > 10:
                        corr = aligned_data[feature].corr(aligned_data[target_returns.name])
                        if abs(corr) > abs(best_corr):
                            best_corr = corr
                            best_feature = feature

            if best_feature:
                selected_features.append(best_feature)

        elif method == 'variance':
            # Select feature with highest variance
            best_feature = None
            best_variance = 0

            for feature in features:
                if feature in feature_df.columns:
                    variance = feature_df[feature].var()
                    if pd.notna(variance) and isinstance(variance, (int, float)) and variance > best_variance:
                        best_variance = variance
                        best_feature = feature

            if best_feature:
                selected_features.append(best_feature)

        elif method == 'random':
            # Random selection
            selected_features.append(np.random.choice(features))

        else:
            # Default: select first feature
            selected_features.append(features[0])

    return selected_features


def remove_redundant_features(feature_df: pd.DataFrame,
                            target_returns: pd.Series,
                            corr_threshold: float = 0.8,
                            method: str = 'correlation') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Remove redundant features using correlation-based clustering

    Args:
        feature_df: DataFrame of features
        target_returns: Target returns for evaluation
        corr_threshold: Correlation threshold for clustering
        method: Feature selection method within clusters

    Returns:
        Tuple of (reduced_feature_df, reduction_info)
    """
    if feature_df.empty:
        return feature_df, {'status': 'empty_input'}

    # Calculate correlation matrix
    corr_matrix = calculate_feature_correlations(feature_df)

    if corr_matrix.empty:
        return feature_df, {'status': 'no_valid_features'}

    # Find highly correlated pairs
    correlated_pairs = find_highly_correlated_features(corr_matrix, corr_threshold)

    # Convert correlation threshold to distance threshold for clustering
    # Higher correlation = lower distance
    distance_threshold = 1 - corr_threshold

    # Cluster features
    clusters = cluster_features_by_correlation(feature_df, distance_threshold)

    # Select representative features
    selected_features = select_representative_features_from_clusters(
        clusters, feature_df, target_returns, method
    )

    # Create reduced feature DataFrame
    reduced_df = feature_df[selected_features].copy()

    # Create reduction info
    reduction_info = {
        'status': 'success',
        'original_features': len(feature_df.columns),
        'reduced_features': len(selected_features),
        'reduction_ratio': len(selected_features) / len(feature_df.columns) if feature_df.columns.size > 0 else 0,
        'correlated_pairs': correlated_pairs,
        'clusters': clusters,
        'selected_features': selected_features,
        'removed_features': [f for f in feature_df.columns if f not in selected_features]
    }

    return reduced_df, reduction_info


def analyze_feature_redundancy(feature_df: pd.DataFrame,
                              target_returns: pd.Series) -> Dict[str, Any]:
    """
    Comprehensive analysis of feature redundancy

    Args:
        feature_df: DataFrame of features
        target_returns: Target returns for evaluation

    Returns:
        Redundancy analysis results
    """
    if feature_df.empty:
        return {'status': 'empty_input'}

    # Calculate correlation matrix
    corr_matrix = calculate_feature_correlations(feature_df)

    if corr_matrix.empty:
        return {'status': 'no_valid_features'}

    # Basic statistics
    n_features = len(corr_matrix.columns)
    mean_abs_corr = corr_matrix.abs().mean().mean()
    max_corr = corr_matrix.abs().max().max()
    min_corr = corr_matrix.abs().min().min()

    # Find highly correlated pairs at different thresholds
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    correlation_counts = {}

    for threshold in thresholds:
        pairs = find_highly_correlated_features(corr_matrix, threshold)
        correlation_counts[f'corr_gt_{threshold}'] = len(pairs)

    # Cluster analysis
    clusters_03 = cluster_features_by_correlation(feature_df, 0.3)
    clusters_05 = cluster_features_by_correlation(feature_df, 0.5)
    clusters_07 = cluster_features_by_correlation(feature_df, 0.7)

    # Feature importance based on correlation to target
    target_correlations = {}
    for col in feature_df.columns:
        if col in feature_df.columns:
            aligned_data = pd.concat([feature_df[col], target_returns], axis=1).dropna()
            if len(aligned_data) > 10:
                corr = aligned_data[col].corr(aligned_data[target_returns.name])
                target_correlations[col] = abs(corr) if not np.isnan(corr) else 0

    # Sort features by importance
    sorted_features = sorted(target_correlations.items(), key=lambda x: x[1], reverse=True)

    return {
        'status': 'success',
        'n_features': n_features,
        'correlation_stats': {
            'mean_abs_correlation': mean_abs_corr,
            'max_correlation': max_corr,
            'min_correlation': min_corr
        },
        'correlation_counts': correlation_counts,
        'clusters': {
            'distance_0.3': clusters_03,
            'distance_0.5': clusters_05,
            'distance_0.7': clusters_07
        },
        'target_correlations': target_correlations,
        'sorted_by_importance': sorted_features,
        'cluster_sizes': {
            'distance_0.3': [len(cluster) for cluster in clusters_03.values()],
            'distance_0.5': [len(cluster) for cluster in clusters_05.values()],
            'distance_0.7': [len(cluster) for cluster in clusters_07.values()]
        }
    }