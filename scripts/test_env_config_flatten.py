import os
import sys

# Ensure repo root is on sys.path so 'ztb' package can be imported when running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ztb.trading.environment.utils.config import EnvironmentConfig

cases = [
    ({"use_continuous_actions": True}, "top-level direct"),
    (
        {"training": {"environment": {"use_continuous_actions": True}}},
        "training.environment",
    ),
    (
        {"training": {"environment": {"config": {"use_continuous_actions": True}}}},
        "training.environment.config",
    ),
    ({}, "empty"),
]

for c, name in cases:
    obj = EnvironmentConfig.from_dict(c)
    print(f"Case {name}: use_continuous_actions={obj.use_continuous_actions}")
