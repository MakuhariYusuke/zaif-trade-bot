#!/usr/bin/env python3
"""
Test adaptation system integration
"""

try:
    from ztb.adaptation.hyperparameter_adaptation_system import (
        HyperparameterAdaptationSystem,
    )
    from ztb.trading.backtest.runner import (
        MockEvaluationManager,
        MockOnlineLearningPipeline,
    )

    # モックコンポーネント作成
    mock_online = MockOnlineLearningPipeline()
    mock_eval = MockEvaluationManager()

    # 適応システム初期化
    adaptation_system = HyperparameterAdaptationSystem(mock_online, mock_eval)
    print("適応システム初期化成功")

    # システム開始
    if adaptation_system.start():
        print("適応システム開始成功")

        # テストデータ作成
        import numpy as np
        import pandas as pd

        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        prices = np.random.normal(30000, 1000, 100)
        market_data = pd.DataFrame(
            {
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "close": prices,
                "volume": np.random.uniform(1000, 10000, 100),
            },
            index=dates,
        )

        # 適応実行
        result = adaptation_system.adapt_hyperparameters(market_data)
        print(f"適応実行結果: {result}")

        # システム停止
        adaptation_system.stop()
        print("適応システム停止成功")
    else:
        print("適応システム開始失敗")

except Exception as e:
    print(f"エラー: {e}")
