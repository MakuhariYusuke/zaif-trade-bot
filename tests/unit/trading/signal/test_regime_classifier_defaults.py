from ztb.trading.signal.regime.classifier import MarketRegimeClassifier


def test_regime_classifier_default_config_contains_confidence_threshold():
    c = MarketRegimeClassifier()
    assert "confidence_threshold" in c.config
    assert isinstance(c.config["confidence_threshold"], float)
    assert c.config["confidence_threshold"] == 0.6
