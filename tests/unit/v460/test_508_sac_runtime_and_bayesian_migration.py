from __future__ import annotations

from scripts.v460.lib import bayesian_regime_filter as legacy_brf
from scripts.v460.lib import sac_common as legacy_sac
from ztb.training.sac import runtime as canonical_sac
from ztb.trading.signal.regime import bayesian_regime_filter as canonical_brf


class TestSACRuntimeMigration:
    def test_legacy_and_canonical_share_evaluate_model_oos(self) -> None:
        assert legacy_sac.evaluate_model_oos is canonical_sac.evaluate_model_oos

    def test_legacy_and_canonical_share_train_val_split(self) -> None:
        assert legacy_sac.train_val_split is canonical_sac.train_val_split

    def test_legacy_and_canonical_share_cleanup_training_resources(self) -> None:
        assert (
            legacy_sac.cleanup_training_resources
            is canonical_sac.cleanup_training_resources
        )


class TestBayesianRegimeMigration:
    def test_legacy_and_canonical_share_filter_class(self) -> None:
        assert legacy_brf.BayesianRegimeFilter is canonical_brf.BayesianRegimeFilter

    def test_legacy_and_canonical_share_config_and_state(self) -> None:
        assert legacy_brf.BayesianRegimeConfig is canonical_brf.BayesianRegimeConfig
        assert legacy_brf.RegimeState is canonical_brf.RegimeState

    def test_legacy_shim_reexports_internal_constants_for_old_tests(self) -> None:
        assert legacy_brf._N_STATES == canonical_brf._N_STATES
        assert legacy_brf._STATE_TO_REGIME_STR == canonical_brf._STATE_TO_REGIME_STR
        assert legacy_brf._REGIME_STR_TO_STATE == canonical_brf._REGIME_STR_TO_STATE
