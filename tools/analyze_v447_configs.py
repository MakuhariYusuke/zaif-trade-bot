#!/usr/bin/env python3
"""Validate and analyze v447 config files."""
import json
from pathlib import Path
from typing import Dict, Any


def validate_json(file_path: Path) -> tuple[bool, str]:
    """Validate JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True, "OK"
    except json.JSONDecodeError as e:
        return False, f"JSON Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def extract_config_summary(file_path: Path) -> Dict[str, Any]:
    """Extract key configuration parameters."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        training = config.get('training', {})
        env = training.get('environment', {})
        behavior = env.get('behavior_optimization', {})
        sac = training.get('sac_hyperparameters', {})
        
        return {
            'model_name': training.get('model_name', 'N/A'),
            'total_timesteps': training.get('total_timesteps', 'N/A'),
            'learning_rate': sac.get('learning_rate', 'N/A'),
            'ent_coef': sac.get('ent_coef', 'N/A'),
            'balance_penalty': behavior.get('balance_penalty', 'N/A'),
            'entropy_regularization': behavior.get('entropy_regularization', 'N/A'),
            'action_balance_target': behavior.get('action_balance_target', 'N/A'),
        }
    except Exception as e:
        return {'error': str(e)}


def main():
    v447_dir = Path('config/v447')
    
    if not v447_dir.exists():
        print(f"Directory not found: {v447_dir}")
        return
    
    json_files = sorted(v447_dir.glob('*.json'))
    
    print("="*80)
    print("V447 Configuration Files Validation & Analysis")
    print("="*80)
    
    print("\n1. JSON Validity Check:")
    print("-"*80)
    
    valid_files = []
    invalid_files = []
    
    for json_file in json_files:
        is_valid, message = validate_json(json_file)
        status = "✓" if is_valid else "✗"
        print(f"{status} {json_file.name:50s} {message}")
        
        if is_valid:
            valid_files.append(json_file)
        else:
            invalid_files.append((json_file, message))
    
    if not valid_files:
        print("\n❌ No valid JSON files found!")
        return
    
    print(f"\n2. Configuration Summary (Valid files: {len(valid_files)}):")
    print("-"*80)
    print(f"{'File Name':<50} {'LR':<8} {'EntCoef':<8} {'BalPen':<8} {'EntReg':<8}")
    print("-"*80)
    
    summaries = []
    for json_file in valid_files:
        summary = extract_config_summary(json_file)
        summaries.append((json_file.name, summary))
        
        lr = summary.get('learning_rate', 'N/A')
        ent = summary.get('ent_coef', 'N/A')
        bal = summary.get('balance_penalty', 'N/A')
        ent_reg = summary.get('entropy_regularization', 'N/A')
        
        print(f"{json_file.name:<50} {str(lr):<8} {str(ent):<8} {str(bal):<8} {str(ent_reg):<8}")
    
    print("\n3. Recommendation:")
    print("-"*80)
    
    # Analyze which config to use
    base_config = None
    for name, summary in summaries:
        if 'config.json' in name:
            base_config = (name, summary)
            break
    
    if base_config:
        print(f"✓ Base config found: {base_config[0]}")
        print("  - This is the baseline configuration for v447")
        print(f"  - Model: {base_config[1].get('model_name', 'N/A')}")
        print("  - Use this for standard AB testing with CLI overrides")
    
    print("\n4. Usage Recommendations:")
    print("-"*80)
    print("For AB testing with reward_components persistence:")
    print("")
    print("  Option A: Use base config + CLI overrides")
    print("  python tools/ab_param_search.py \\")
    print("    --template config/v447/sac_v447_1m_multiframe_config.json \\")
    print("    --grid config/ab/ab_grid_fine_tuning.json \\")
    print("    --timesteps 3000 --seeds 1 --fast-mode")
    print("")
    print("  Option B: Use specific variant configs directly")
    print("  python tools/ab_test_runner.py \\")
    print("    --configs config/v447/sac_v447_1m_multiframe_entropy_lr_lower.json \\")
    print("              config/v447/sac_v447_1m_multiframe_balance_shaping.json \\")
    print("    --seeds 3 --jobs 1")
    
    if invalid_files:
        print("\n5. Files Requiring Attention:")
        print("-"*80)
        for file, error in invalid_files:
            print(f"✗ {file.name}")
            print(f"  Error: {error}")


if __name__ == "__main__":
    main()
