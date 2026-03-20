from __future__ import annotations

import scripts.v460.lib.regime_detector as legacy_regime
import ztb.trading.signal.regime.regime_detector as canonical_regime


class TestRegimeDetectorMigration:
    def test_legacy_and_canonical_share_detector_class(self) -> None:
        assert legacy_regime.FillTestRegimeDetector is canonical_regime.FillTestRegimeDetector

    def test_legacy_and_canonical_share_protocol_and_enums(self) -> None:
        assert legacy_regime.RegimeDetectorLike is canonical_regime.RegimeDetectorLike
        assert legacy_regime.FillTestRegime is canonical_regime.FillTestRegime
        assert legacy_regime.RegimeConfig is canonical_regime.RegimeConfig

    def test_legacy_docstring_keeps_theory_references(self) -> None:
        doc = legacy_regime.__doc__ or ''
        assert 'Hamilton' in doc
        assert ('Adaptive Market' in doc) or ('Lo (2004)' in doc) or ('AMH' in doc)
