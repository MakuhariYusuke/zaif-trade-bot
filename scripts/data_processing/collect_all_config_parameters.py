#!/usr/bin/env python3
"""
Collect all parameters from SAC/PPO config JSON files
全ての設定JSONファイルからパラメータを収集
"""

import glob
import json
import os
from collections import defaultdict


def collect_all_parameters():
    """Collect all parameters from all config JSON files"""
    config_dirs = ["config", "configs", "ztb/config", "ztb/config/versions"]

    all_params = defaultdict(set)
    file_count = 0

    for config_dir in config_dirs:
        if not os.path.exists(config_dir):
            continue

        # Find all JSON files
        json_files = []
        json_files.extend(glob.glob(f"{config_dir}/**/*.json", recursive=True))
        json_files.extend(glob.glob(f"{config_dir}/*.json"))

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Recursively collect all parameter paths
                def collect_params(obj, path=""):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            current_path = f"{path}.{key}" if path else key
                            all_params[current_path].add(json_file)
                            collect_params(value, current_path)
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            current_path = f"{path}[{i}]"
                            collect_params(item, current_path)

                collect_params(data)
                file_count += 1
                print(f"Processed: {json_file}")

            except Exception as e:
                print(f"Error processing {json_file}: {e}")

    print(f"\nTotal files processed: {file_count}")
    return all_params


def categorize_parameters(all_params):
    """Categorize parameters by their structure"""
    categories = {
        "basic": [],
        "sac_hyperparams": [],
        "ppo_hyperparams": [],
        "environment": [],
        "reward": [],
        "data": [],
        "training": [],
        "regime_adaptation": [],
        "checkpoint": [],
        "analysis": [],
        "other": [],
    }

    for param in sorted(all_params.keys()):
        if any(
            keyword in param.lower()
            for keyword in ["model_name", "algorithm", "version", "description"]
        ):
            categories["basic"].append(param)
        elif param.startswith("sac_hyperparameters") or "sac." in param.lower():
            categories["sac_hyperparams"].append(param)
        elif param.startswith("ppo_hyperparameters") or "ppo." in param.lower():
            categories["ppo_hyperparams"].append(param)
        elif param.startswith("environment") or "env" in param.lower():
            categories["environment"].append(param)
        elif "reward" in param.lower():
            categories["reward"].append(param)
        elif "data" in param.lower() or "csv" in param.lower():
            categories["data"].append(param)
        elif "training" in param.lower() or "timesteps" in param.lower():
            categories["training"].append(param)
        elif "regime" in param.lower() or "adaptation" in param.lower():
            categories["regime_adaptation"].append(param)
        elif "checkpoint" in param.lower():
            categories["checkpoint"].append(param)
        elif "analysis" in param.lower() or "notes" in param.lower():
            categories["analysis"].append(param)
        else:
            categories["other"].append(param)

    return categories


def generate_documentation(all_params, categories):
    """Generate comprehensive documentation"""
    doc = "# SAC/PPO設定ファイル完全パラメータ解説ドキュメント\n\n"
    doc += "## 概要\n\n"
    doc += f"このドキュメントは、{len(all_params)}個の一意なパラメータを調査し、"
    doc += "全ての設定JSONファイルから収集した完全なパラメータリファレンスです。\n\n"

    # Summary statistics
    total_files = set()
    for files in all_params.values():
        total_files.update(files)
    doc += f"- 調査した設定ファイル数: {len(total_files)}\n"
    doc += f"- 検出された一意パラメータ数: {len(all_params)}\n\n"

    # Categories overview
    doc += "## パラメータカテゴリ概要\n\n"
    for category, params in categories.items():
        doc += f"- **{category}**: {len(params)} パラメータ\n"
    doc += "\n"

    # Detailed parameter documentation
    category_names = {
        "basic": "基本パラメータ",
        "sac_hyperparams": "SACハイパーパラメータ",
        "ppo_hyperparams": "PPOハイパーパラメータ",
        "environment": "環境パラメータ",
        "reward": "報酬パラメータ",
        "data": "データパラメータ",
        "training": "学習パラメータ",
        "regime_adaptation": "市場レジーム適応パラメータ",
        "checkpoint": "チェックポイントパラメータ",
        "analysis": "分析パラメータ",
        "other": "その他パラメータ",
    }

    for category, params in categories.items():
        if not params:
            continue

        doc += f"## {category_names[category]}\n\n"
        doc += "| パラメータ | 説明 | 使用ファイル数 | 例 |\n"
        doc += "|-----------|------|---------------|------|\n"

        for param in sorted(params):
            file_count = len(all_params[param])
            # Get example value from one of the files
            example = "N/A"
            for json_file in list(all_params[param])[:1]:  # Just use first file
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    value = data
                    for key in param.split("."):
                        if "[" in key and "]" in key:
                            # Handle array indices
                            base_key = key.split("[")[0]
                            index = int(key.split("[")[1].split("]")[0])
                            value = value[base_key][index]
                        else:
                            value = value[key]
                    example = (
                        str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    )
                except Exception:
                    example = "N/A"

            doc += f"| `{param}` | {get_parameter_description(param)} | {file_count} | `{example}` |\n"

        doc += "\n"

    return doc


def get_parameter_description(param):
    """Get description for a parameter"""
    descriptions = {
        "model_name": "モデル識別子",
        "algorithm": "使用アルゴリズム",
        "version": "設定バージョン",
        "description": "設定の説明",
        "total_timesteps": "総学習ステップ数",
        "data_source": "データソース種別",
        "data_path": "データファイルパス",
        "learning_rate": "学習率",
        "buffer_size": "リプレイバッファサイズ",
        "learning_starts": "学習開始ステップ",
        "batch_size": "バッチサイズ",
        "tau": "ターゲット更新率",
        "gamma": "割引率",
        "ent_coef": "エントロピー係数",
        "target_entropy": "目標エントロピー",
        "initial_balance": "初期残高",
        "transaction_cost": "取引コスト",
        "max_position_size": "最大ポジションサイズ",
        "enable_action_masking": "アクションmasking有効化",
        "use_continuous_actions": "連続アクション使用",
        "random_start": "ランダム開始",
        "hold_penalty_weight": "HOLD罰則重み",
        "profit_reward_multiplier": "利益報酬倍率",
        "checkpoint_interval": "チェックポイント間隔",
        "regime_scheme": "レジーム分類方式",
        "confidence_threshold": "信頼性閾値",
        "adaptation_frequency": "適応頻度",
        "n_steps": "ステップ数 per 更新",
        "n_epochs": "エポック数",
        "gae_lambda": "GAEラムダ",
        "clip_range": "クリップ範囲",
        "vf_coef": "価値関数係数",
        "max_grad_norm": "最大勾配ノルム",
        "lagrange_constraint.enabled": "Lagrange制約有効化",
        "r_target": "目標リターン",
        "tolerance": "許容誤差",
        "eta": "イータ値",
        "eta_min": "イータ最小値",
        "eta_max": "イータ最大値",
        "eta_lr": "イータ学習率",
    }

    # Try exact match first
    if param in descriptions:
        return descriptions[param]

    # Try partial match
    for key, desc in descriptions.items():
        if key in param:
            return desc

    return "詳細不明"


if __name__ == "__main__":
    print("Collecting all parameters from config files...")
    all_params = collect_all_parameters()

    print("Categorizing parameters...")
    categories = categorize_parameters(all_params)

    print("Generating documentation...")
    doc = generate_documentation(all_params, categories)

    output_file = "docs/sac_ppo_complete_parameter_reference.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(doc)

    print(f"Documentation generated: {output_file}")
    print(f"Total unique parameters: {len(all_params)}")
