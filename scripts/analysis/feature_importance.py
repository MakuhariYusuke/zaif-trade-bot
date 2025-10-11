#!/usr/bin/env python3
"""
特徴量重要度分析モジュール

SHAP (SHapley Additive exPlanations) と Permutation Importance を使用して、
機械学習モデルの特徴量重要度を分析します。
"""

import argparse
import json
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
from numpy.typing import NDArray

# SHAP のインポート（利用可能な場合）
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP is not installed. Install with: pip install shap")

@dataclass
class FeatureImportanceResult:
    """特徴量重要度分析結果"""
    feature_names: List[str]
    shap_values: Optional[NDArray[np.floating]] = None
    shap_base_value: Optional[float] = None
    permutation_importance: Optional[Dict[str, NDArray[np.floating]]] = None
    mean_shap_importance: Optional[NDArray[np.floating]] = None
    std_shap_importance: Optional[NDArray[np.floating]] = None
    mean_permutation_importance: Optional[NDArray[np.floating]] = None
    std_permutation_importance: Optional[NDArray[np.floating]] = None

class FeatureImportanceAnalyzer:
    """特徴量重要度分析クラス"""

    def __init__(self, random_state: int = 42):  # type: ignore
        """
        Args:
            random_state: 乱数シード
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def calculate_shap_importance(self, model: BaseEstimator,
                                 X: Union[pd.DataFrame, NDArray[np.floating]],
                                 max_evals: int = 1000) -> Tuple[NDArray[np.floating], float]:
        """
        SHAP 値を使用して特徴量重要度を計算

        Args:
            model: 学習済みモデル
            X: 特徴量データ
            max_evals: SHAP 計算の最大評価回数

        Returns:
            SHAP 値とベース値のタプル
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for SHAP importance calculation")

        # データが DataFrame の場合、numpy 配列に変換
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # SHAP Explainer の作成
        try:
            explainer = shap.Explainer(model, X_array[:100])  # type: ignore
        except:
            # Tree モデル以外の場合
            explainer = shap.KernelExplainer(model.predict, X_array[:50])  # type: ignore

        # SHAP 値の計算
        shap_values = explainer(X_array, max_evals=max_evals)

        if hasattr(shap_values, 'base_values'):
            base_value = float(np.mean(shap_values.base_values))
        else:
            base_value = 0.0

        if hasattr(shap_values, 'values'):
            values = shap_values.values
        else:
            values = shap_values

        return values, base_value

    def calculate_permutation_importance(self, model: BaseEstimator,
                                        X: Union[pd.DataFrame, NDArray[np.floating]],
                                        y: Union[pd.Series, NDArray[np.floating]],
                                        n_repeats: int = 10,
                                        scoring: str = 'neg_mean_squared_error') -> Dict[str, NDArray[np.floating]]:
        """
        Permutation Importance を計算

        Args:
            model: 学習済みモデル
            X: 特徴量データ
            y: ターゲットデータ
            n_repeats: 繰り返し回数
            scoring: スコアリング関数

        Returns:
            Permutation Importance の結果
        """
        # データの準備
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        # Permutation Importance の計算
        perm_importance = permutation_importance(
            model, X_array, y_array,  # type: ignore
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring=scoring
        )

        return {
            'importances': perm_importance.importances,
            'importances_mean': perm_importance.importances_mean,
            'importances_std': perm_importance.importances_std
        }

    def aggregate_shap_importance(self, shap_values: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        SHAP 値を集約して特徴量重要度を計算

        Args:
            shap_values: SHAP 値

        Returns:
            平均重要度と標準偏差のタプル
        """
        # 各特徴量の SHAP 値の絶対値の平均
        mean_importance = np.mean(np.abs(shap_values), axis=0)
        std_importance = np.std(np.abs(shap_values), axis=0)

        return mean_importance, std_importance

    def run_feature_importance_analysis(self, model: BaseEstimator,
                                       X: Union[pd.DataFrame, NDArray[np.floating]],
                                       y: Optional[Union[pd.Series, NDArray[np.floating]]] = None,
                                       calculate_shap: bool = True,
                                       calculate_permutation: bool = True,
                                       n_repeats: int = 10) -> FeatureImportanceResult:
        """
        特徴量重要度分析を実行

        Args:
            model: 学習済みモデル
            X: 特徴量データ
            y: ターゲットデータ（Permutation Importance 用）
            calculate_shap: SHAP 重要度を計算するか
            calculate_permutation: Permutation Importance を計算するか
            n_repeats: Permutation Importance の繰り返し回数

        Returns:
            分析結果
        """
        # 特徴量名の取得
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        result = FeatureImportanceResult(feature_names=feature_names)

        # SHAP 重要度の計算
        if calculate_shap and SHAP_AVAILABLE:
            try:
                shap_values, base_value = self.calculate_shap_importance(model, X)
                result.shap_values = shap_values
                result.shap_base_value = base_value

                # 集約重要度の計算
                mean_shap, std_shap = self.aggregate_shap_importance(shap_values)
                result.mean_shap_importance = mean_shap
                result.std_shap_importance = std_shap

            except Exception as e:
                warnings.warn(f"SHAP calculation failed: {e}")

        # Permutation Importance の計算
        if calculate_permutation and y is not None:
            try:
                perm_result = self.calculate_permutation_importance(model, X, y, n_repeats)
                result.permutation_importance = perm_result
                result.mean_permutation_importance = perm_result['importances_mean']
                result.std_permutation_importance = perm_result['importances_std']

            except Exception as e:
                warnings.warn(f"Permutation importance calculation failed: {e}")

        return result

    def plot_feature_importance(self, result: FeatureImportanceResult,
                               save_path: Optional[str] = None):
        """特徴量重要度を可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Feature Importance Analysis', fontsize=16)

        # SHAP 重要度のプロット
        if result.mean_shap_importance is not None:
            sorted_idx = np.argsort(result.mean_shap_importance)[::-1]
            feature_names_sorted = [result.feature_names[i] for i in sorted_idx]
            importance_sorted = result.mean_shap_importance[sorted_idx]
            std_sorted = result.std_shap_importance[sorted_idx] if result.std_shap_importance is not None else None

            axes[0, 0].barh(range(len(feature_names_sorted)), importance_sorted,
                           xerr=std_sorted, color='skyblue', alpha=0.7)
            axes[0, 0].set_yticks(range(len(feature_names_sorted)))
            axes[0, 0].set_yticklabels(feature_names_sorted)
            axes[0, 0].set_xlabel('Mean |SHAP Value|')
            axes[0, 0].set_title('SHAP Feature Importance')
            axes[0, 0].grid(True, alpha=0.3)

            # SHAP サマリープロット
            if result.shap_values is not None and SHAP_AVAILABLE:
                axes[0, 1].set_title('SHAP Summary Plot')
                # SHAP summary plot の簡易版
                shap_importance = np.abs(result.shap_values).mean(axis=0)
                axes[0, 1].barh(range(len(result.feature_names)), shap_importance,
                               color='lightgreen', alpha=0.7)
                axes[0, 1].set_yticks(range(len(result.feature_names)))
                axes[0, 1].set_yticklabels(result.feature_names)
                axes[0, 1].set_xlabel('Mean |SHAP Value|')
        else:
            axes[0, 0].text(0.5, 0.5, 'SHAP not available', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 1].text(0.5, 0.5, 'SHAP not available', ha='center', va='center', transform=axes[0, 1].transAxes)

        # Permutation Importance のプロット
        if result.mean_permutation_importance is not None:
            sorted_idx = np.argsort(result.mean_permutation_importance)[::-1]
            feature_names_sorted = [result.feature_names[i] for i in sorted_idx]
            importance_sorted = result.mean_permutation_importance[sorted_idx]
            std_sorted = result.std_permutation_importance[sorted_idx] if result.std_permutation_importance is not None else None

            axes[1, 0].barh(range(len(feature_names_sorted)), importance_sorted,
                           xerr=std_sorted, color='salmon', alpha=0.7)
            axes[1, 0].set_yticks(range(len(feature_names_sorted)))
            axes[1, 0].set_yticklabels(feature_names_sorted)
            axes[1, 0].set_xlabel('Mean Permutation Importance')
            axes[1, 0].set_title('Permutation Feature Importance')
            axes[1, 0].grid(True, alpha=0.3)

            # 重要度の比較
            if result.mean_shap_importance is not None:
                # 正規化して比較
                shap_norm = (result.mean_shap_importance - np.min(result.mean_shap_importance)) / \
                           (np.max(result.mean_shap_importance) - np.min(result.mean_shap_importance))
                perm_norm = (result.mean_permutation_importance - np.min(result.mean_permutation_importance)) / \
                           (np.max(result.mean_permutation_importance) - np.min(result.mean_permutation_importance))

                axes[1, 1].scatter(shap_norm, perm_norm, alpha=0.6)
                axes[1, 1].set_xlabel('Normalized SHAP Importance')
                axes[1, 1].set_ylabel('Normalized Permutation Importance')
                axes[1, 1].set_title('SHAP vs Permutation Importance')
                axes[1, 1].grid(True, alpha=0.3)

                # 対角線
                min_val = min(np.min(shap_norm), np.min(perm_norm))
                max_val = max(np.max(shap_norm), np.max(perm_norm))
                axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        else:
            axes[1, 0].text(0.5, 0.5, 'Permutation Importance not calculated', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 1].text(0.5, 0.5, 'Comparison not available', ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def export_results(self, result: FeatureImportanceResult,
                      output_path: str):
        """分析結果を JSON ファイルにエクスポート"""
        export_data: Dict[str, Any] = {
            'feature_names': result.feature_names,
            'shap_available': result.shap_values is not None,
            'permutation_available': result.permutation_importance is not None
        }

        if result.shap_values is not None:
            export_data['shap_base_value'] = result.shap_base_value
            export_data['mean_shap_importance'] = result.mean_shap_importance.tolist() if result.mean_shap_importance is not None else None
            export_data['std_shap_importance'] = result.std_shap_importance.tolist() if result.std_shap_importance is not None else None

        if result.permutation_importance is not None:
            export_data['permutation_importance'] = {
                'importances_mean': result.mean_permutation_importance.tolist() if result.mean_permutation_importance is not None else None,
                'importances_std': result.std_permutation_importance.tolist() if result.std_permutation_importance is not None else None
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"Results exported to {output_path}")

def main():
    """コマンドラインインターフェース"""
    parser = argparse.ArgumentParser(description='Feature Importance Analysis')
    parser.add_argument('--model-path', required=True, help='Path to trained model (pickle file)')
    parser.add_argument('--data-path', required=True, help='Path to feature data (CSV or pickle)')
    parser.add_argument('--target-column', help='Target column name (for permutation importance)')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--no-shap', action='store_true', help='Skip SHAP calculation')
    parser.add_argument('--no-permutation', action='store_true', help='Skip permutation importance')
    parser.add_argument('--n-repeats', type=int, default=10, help='Number of permutation repeats')

    args = parser.parse_args()

    # モデルの読み込み
    import pickle
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)

    # データの読み込み
    if args.data_path.endswith('.csv'):
        data = pd.read_csv(args.data_path)
    elif args.data_path.endswith('.pkl'):
        with open(args.data_path, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError("Unsupported data format")

    # 特徴量とターゲットの分離
    if args.target_column and args.target_column in data.columns:
        X = data.drop(columns=[args.target_column])
        y = data[args.target_column]
    else:
        X = data
        y = None

    # 分析の実行
    analyzer = FeatureImportanceAnalyzer()
    result = analyzer.run_feature_importance_analysis(
        model=model,
        X=X,
        y=y,
        calculate_shap=not args.no_shap,
        calculate_permutation=not args.no_permutation,
        n_repeats=args.n_repeats
    )

    # 結果の保存
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # プロットの保存
    plot_path = os.path.join(args.output_dir, 'feature_importance_analysis.png')
    analyzer.plot_feature_importance(result, save_path=plot_path)

    # JSON エクスポート
    json_path = os.path.join(args.output_dir, 'feature_importance_results.json')
    analyzer.export_results(result, json_path)

    print("Feature importance analysis completed!")

if __name__ == '__main__':
    main()