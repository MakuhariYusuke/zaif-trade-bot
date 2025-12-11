def test_v450_scripts_importable():
    # Ensure experiment scripts can import without executing heavy work
    from experiments.v450 import generate_range_data_v450 as gen
    from experiments.v450 import run_ab_test_threshold_v450 as ab
    from experiments.v450 import run_short_training_v450 as st

    assert st is not None
    assert ab is not None
    assert gen is not None
