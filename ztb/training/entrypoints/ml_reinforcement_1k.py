#!/usr/bin/env python3
"""
1k step reinforcement learning experiment execution script
ExperimentBase class-based feature evaluation experiment with strategy support
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Local module imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from .base_ml_reinforcement import MLReinforcementExperiment
from ztb.utils.experiment_cli import create_experiment_main


class MLReinforcement1KExperiment(MLReinforcementExperiment):
    """1k step reinforcement learning experiment class with strategy support"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, total_steps=1000)


# Create main function using the utility
main = create_experiment_main(
    MLReinforcement1KExperiment,
    "1k step reinforcement learning experiment",
    1000,
    "1k"
)


if __name__ == "__main__":
    main()
