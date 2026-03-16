import pandas as pd

from ztb.data.v433_feature_engineering import MarketRegimeDetector


def test_classify_regime_vectorized_paths() -> None:
    detector = MarketRegimeDetector()
    df = pd.DataFrame(
        {
            "trend_strength_short": [0.10, -0.10, 0.10, 0.01, 0.0],
            "trend_strength_medium": [0.04, -0.04, -0.04, 0.0, 0.0],
            "volatility_short": [0.01, 0.01, 0.01, 0.30, 0.02],
            "volatility_medium": [0.01, 0.01, 0.01, 0.10, 0.02],
        }
    )

    result = detector._classify_regime(df)

    assert result.tolist() == ["bull", "bear", "mixed", "volatile", "sideways"]


def test_regime_confidence_is_bounded_and_nan_safe() -> None:
    detector = MarketRegimeDetector()
    df = pd.DataFrame(
        {
            "trend_strength_short": [None, 0.03, 0.20],
            "trend_strength_medium": [None, 0.02, 0.20],
            "volatility_short": [None, 0.10, 0.40],
            "volatility_medium": [None, 0.05, 0.10],
        }
    )

    result = detector._calculate_regime_confidence(df)

    assert result.iloc[0] == 0.0
    assert ((result >= 0.0) & (result <= 1.0)).all()
