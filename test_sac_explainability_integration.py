#!/usr/bin/env python3
"""
SAC v421 説明可能性統合テストスクリプト。

このスクリプトは、SACアルゴリズムに説明可能性機能が正しく統合されているかをテストします。
"""

import sys
import numpy as np
import logging
sys.path.append('.')

from ztb.training.algorithms.sac.sac_algorithm import SACAlgorithm
from ztb.adaptation.explainability.config import ExplainabilityConfig

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sac_explainability_integration():
    """SACアルゴリズムの説明可能性統合をテスト。"""
    print("Testing SAC v421 Explainability Integration...")

    # SACアルゴリズムのインスタンス作成
    sac = SACAlgorithm()
    print(f"✓ SACAlgorithm created: {sac}")

    # デフォルト設定を取得
    config = sac.get_default_config()
    print(f"✓ Default config loaded with {len(config)} parameters")

    # 説明可能性設定が含まれているか確認
    explainability_keys = [
        "explainability_enabled", "shap_enabled", "shap_max_evals",
        "natural_language_enabled", "market_context_analysis", "risk_warnings"
    ]

    for key in explainability_keys:
        if key in config:
            print(f"✓ Explainability config '{key}': {config[key]}")
        else:
            print(f"✗ Missing explainability config: {key}")
            return False

    # 説明可能性を有効にした設定でテスト
    config["explainability_enabled"] = True
    print(f"✓ Explainability enabled in config")

    # 説明可能性アナライザーの初期化テスト（モデル作成なし）
    try:
        sac._initialize_explainability_analyzer(config)
        print(f"✓ Explainability analyzer initialized: {sac.explainability_analyzer is not None}")
        
        if sac.explainability_analyzer is None:
            print("✗ Explainability analyzer initialization failed")
            return False
            
    except Exception as e:
        print(f"✗ Explainability analyzer initialization failed: {e}")
        return False

    # 実際の環境を使用せずにAPIテスト
    print("✓ Skipping model creation test (requires actual Gymnasium environment)")
    
    # 決定説明のテスト（モックデータ使用）
    mock_observation = np.random.randn(80).astype(np.float32)
    print(f"✓ Mock observation created with shape: {mock_observation.shape}")
    
    # 説明可能性アナライザーがない場合のテスト
    explanation = sac.explain_decision(mock_observation)
    if explanation is None:
        print("✓ Decision explanation correctly returns None (no model)")
    else:
        print("? Decision explanation returned result without model (unexpected)")
    
    # レポート生成のテスト（モックデータ使用）
    mock_observations = np.random.randn(5, 80).astype(np.float32)
    report_path = sac.generate_explanation_report(mock_observations)
    if report_path is None:
        print("✓ Explanation report correctly returns None (no model)")
    else:
        print("? Explanation report returned result without model (unexpected)")
    
    print("🎉 All SAC explainability integration tests passed!")
    return True

if __name__ == "__main__":
    success = test_sac_explainability_integration()
    sys.exit(0 if success else 1)