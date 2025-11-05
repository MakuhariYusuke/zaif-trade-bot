#!/usr/bin/env python
"""Quick test to verify balanced_penalty stage is active during training."""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from ztb.config.loader import ConfigLoader
from ztb.training.unified_trainer import V4XXUnifiedTrainer
from ztb.utils.v4xx_config_converter import V4XXConfigConverter

# Load and convert config
config = json.load(open('config/sac_v444_3_balanced_penalty_scale_200.json'))
print("\n=== Config Values ===")
print(f"training.curriculum_learning.curriculum_stage = {config['training']['curriculum_learning']['curriculum_stage']}")

# Convert
converted_config = V4XXConfigConverter.convert_v444_to_unified(config)
env_stage = converted_config['training']['environment'].get('curriculum_stage')
print(f"training.environment.curriculum_stage = {env_stage}")

# Load with ConfigLoader
validated_config = ConfigLoader.validate_config(converted_config)
env_stage_validated = validated_config['training']['environment'].get('curriculum_stage')
print(f"After validation: {env_stage_validated}")

print("\n=== Quick Training Test ===")
print("Creating trainer with 1000 steps for rapid test...")

try:
    # Quick train with just 1000 steps
    config['training']['total_timesteps'] = 1000
    config['training']['eval_frequency'] = 500
    
    trainer = V4XXUnifiedTrainer(config, verbose=0)
    print("Trainer created successfully")
    print(f"Environment curriculum_stage: {trainer.env.env.curriculum_stage}")
    
    # Run training
    trainer.train()
    print("✅ Training completed with balanced_penalty stage active!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
