#!/usr/bin/env python3
"""
100k step reinforcement learning experiment execution script
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Local module imports
current_dir = Path(__file__).parent.parent
project_root = current_dir.parent  # Go up one more level to project root
sys.path.insert(0, str(project_root))

from .base_ml_reinforcement import MLReinforcementExperiment
from ztb.utils.experiment_cli import ExperimentCLI
from typing import Any, Dict

# Local module imports
current_dir = Path(__file__).parent.parent
project_root = current_dir.parent  # Go up one more level to project root
sys.path.insert(0, str(project_root))

from .base_ml_reinforcement import MLReinforcementExperiment
from ztb.utils.experiment_cli import ExperimentCLI


class MLReinforcement100KExperiment(MLReinforcementExperiment):
    """100k step reinforcement learning experiment"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, total_steps=100000)


def main() -> None:
    """Main function"""
    parser = ExperimentCLI.create_parser(
        "100k step reinforcement learning experiment",
        100000,
        "100k"
    )
    args = parser.parse_args()
    config = ExperimentCLI.create_config(args, "100k")
    ExperimentCLI.run_experiment(MLReinforcement100KExperiment, config, "100k")


if __name__ == "__main__":
    main()
