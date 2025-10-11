"""
ディレクトリ構造移行スクリプト

段階的にファイルを新しいディレクトリ構造に移行します。
"""

from pathlib import Path
import shutil
import json
from typing import List, Dict


class StructureMigrator:
    """構造移行を管理するクラス"""
    
    def __init__(self, root_path: Path, dry_run: bool = True):
        self.root = root_path
        self.dry_run = dry_run
        self.migration_log = []
    
    def migrate_phase_1_configs(self) -> Dict:
        """Phase 1: configs/ディレクトリへの移行"""
        print("\n" + "="*80)
        print("Phase 1: configs/ディレクトリへの移行")
        print("="*80)
        
        # 移行するファイル
        config_files = [
            '.eslintrc.json',
            '.markdownlint.json',
            'action_distribution_comparison.json',
            'backtest_balanced_test.json',
            'backtest_debug.json',
            'backtest_fixed_v2.json',
            'backtest_fixed.json',
            'backtest_results_ppo_100k.json',
            'config_consistency_report.json',
            'curriculum_test_config.json',
            'custom_ppo_rollout_10k_test.json',
        ]
        
        # サブディレクトリ構成
        target_structure = {
            'configs/algorithms/': [
                'bitflyer-config.json',
            ],
            'configs/environments/': [],
            'configs/optimization/': [],
            'configs/training/': [
                'curriculum_test_config.json',
            ],
            'configs/backtest/': [
                'backtest_balanced_test.json',
                'backtest_debug.json',
                'backtest_fixed_v2.json',
                'backtest_fixed.json',
            ],
            'configs/results/': [
                'backtest_results_ppo_100k.json',
                'custom_ppo_rollout_10k_test.json',
                'action_distribution_comparison.json',
                'config_consistency_report.json',
            ],
            'configs/linting/': [
                '.eslintrc.json',
                '.markdownlint.json',
            ],
        }
        
        return self._execute_migration(target_structure, "configs")
    
    def migrate_phase_2_docs(self) -> Dict:
        """Phase 2: docs/ディレクトリへの移行"""
        print("\n" + "="*80)
        print("Phase 2: docs/ディレクトリへの移行")
        print("="*80)
        
        target_structure = {
            'docs/development/': [
                'CHANGELOG.md',
                'CODE_QUALITY_IMPROVEMENT_PLAN.md',
                'COMPREHENSIVE_TRAINING_GUIDE.md',
                'INTEGRATION_GUIDE_100K_AND_QUALITY.md',
            ],
            'docs/operations/': [
                '1M_ENSEMBLE_OPERATIONS_MANUAL.md',
                'LIVE_TRADE_OPTIMIZATION_SUMMARY.md',
                'LOG_LEVEL_CONTROL.md',
            ],
            'docs/proposals/': [
                'ADDITIONAL_FEATURES_PROPOSAL.md',
                'ADVANCED_IMPROVEMENTS_PROPOSAL.md',
                'ACTION_BIAS_MITIGATION_CONSULTATION.md',
            ],
            'docs/reports/': [
                'BINARY_SEARCH_OPTIMIZER_UPDATE.md',
                'BUG_47_48_49_COMPREHENSIVE_FIX.md',
                'BUG_48_HORIZONTAL_DEPLOYMENT.md',
                'BUG_48_REWARD_SETTINGS_NOT_PASSED.md',
                'BUG_49_PROFIT_BONUS_ORDER_ERROR.md',
                'BUG_50_BUY_SCARCITY.md',
                'CHECKPOINT_INTERVAL_EXTENSION.md',
                'CHECKPOINT_INTERVAL_INTEGRATION_SUMMARY.md',
                'CUSTOM_PPO_ROLLOUT_REPORT.md',
                'CUSTOM_PPO_SUCCESS_REPORT.md',
                'EMERGENCY_FIX_SUMMARY.md',
                'EMERGENCY_FIX_v3.6.3.md',
                'FEATURE_IMPLEMENTATION_SUMMARY.md',
                'FINAL_ROLLOUT_AND_ROADMAP.md',
                'FINAL_SUMMARY.md',
                'executive_summary.md',
                'integration_failure_diagnosis.md',
            ],
            'docs/': [
                'DISCLAIMER.md',
                'LICENSE',
                'BACKLOG.md',
            ],
        }
        
        return self._execute_migration(target_structure, "docs")
    
    def migrate_phase_3_scripts(self) -> Dict:
        """Phase 3: scripts/ディレクトリへの移行"""
        print("\n" + "="*80)
        print("Phase 3: scripts/ディレクトリへの移行")
        print("="*80)
        
        target_structure = {
            'scripts/analysis/': [
                'analyze_action_distribution.py',
                'analyze_feature_selection.py',
                'analyze_features_quality.py',
                'analyze_file_sizes.py',
                'analyze_reward_function.py',
                'analyze_reward_improvements.py',
                'analyze_risk_metrics.py',
                'analyze_v364_console.py',
                'analyze_v365_result.py',
                'analyze_v366_bugs.py',
                'analyze_v394_training.py',
            ],
            'scripts/training/': [
                'backtest_model.py',
                'cloud_training.sh',
                'linux_canary.sh',
                'live_trade.py',
            ],
            'scripts/debugging/': [
                'debug_action_masking.py',
                'debug_model_predictions.py',
                'debug_sell_rate_detail.py',
                'debug_tiebreaker.py',
                'debug_timesteps_bug.py',
            ],
            'scripts/data/': [
                'create_test_dataset.py',
                'curated_features.py',
            ],
            'scripts/comparison/': [
                'compare_training_metrics.py',
            ],
            'scripts/investigation/': [
                'inspect_tensorboard_tags.py',
                'investigate_sb3.py',
                'list_tensorboard_scalars.py',
            ],
            'scripts/utilities/': [
                'fix_run_1m.py',
            ],
        }
        
        return self._execute_migration(target_structure, "scripts")
    
    def _execute_migration(self, target_structure: Dict[str, List[str]], category: str) -> Dict:
        """移行を実行"""
        results = {
            'category': category,
            'total_files': 0,
            'success': 0,
            'skipped': 0,
            'errors': 0,
            'operations': []
        }
        
        for target_dir, files in target_structure.items():
            target_path = self.root / target_dir
            
            # ディレクトリ作成
            if not self.dry_run:
                target_path.mkdir(parents=True, exist_ok=True)
            print(f"\n📁 {target_dir}")
            
            for filename in files:
                source = self.root / filename
                dest = target_path / filename
                
                results['total_files'] += 1
                
                if not source.exists():
                    print(f"  ⚠️  スキップ (存在しない): {filename}")
                    results['skipped'] += 1
                    continue
                
                if dest.exists():
                    print(f"  ⚠️  スキップ (既存): {filename}")
                    results['skipped'] += 1
                    continue
                
                try:
                    if self.dry_run:
                        print(f"  [DRY-RUN] {filename} → {target_dir}")
                    else:
                        shutil.move(str(source), str(dest))
                        print(f"  ✅ {filename} → {target_dir}")
                    
                    results['success'] += 1
                    results['operations'].append({
                        'action': 'move',
                        'source': str(source.relative_to(self.root)),
                        'destination': str(dest.relative_to(self.root))
                    })
                except Exception as e:
                    print(f"  ❌ エラー: {filename} - {e}")
                    results['errors'] += 1
        
        # サマリー
        print(f"\n{'[DRY-RUN] ' if self.dry_run else ''}Phase完了:")
        print(f"  総ファイル数: {results['total_files']}")
        print(f"  成功: {results['success']}")
        print(f"  スキップ: {results['skipped']}")
        print(f"  エラー: {results['errors']}")
        
        self.migration_log.append(results)
        return results
    
    def save_log(self, output_path: Path):
        """移行ログを保存"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.migration_log, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 移行ログを保存: {output_path}")


def main():
    """メイン実行"""
    root_path = Path(__file__).parent.parent.parent
    
    print("="*80)
    print("  ディレクトリ構造移行スクリプト")
    print("="*80)
    print(f"\nプロジェクトルート: {root_path}")
    
    # DRY-RUNモード確認
    import sys
    dry_run = '--execute' not in sys.argv
    
    if dry_run:
        print("\n⚠️  DRY-RUNモード（実際の移行は行いません）")
        print("実際に移行するには: python migrate_structure.py --execute")
    else:
        print("\n🚀 実行モード（実際にファイルを移動します）")
        confirm = input("続行しますか？ (yes/no): ")
        if confirm.lower() != 'yes':
            print("キャンセルしました。")
            return
    
    # 移行実行
    migrator = StructureMigrator(root_path, dry_run=dry_run)
    
    migrator.migrate_phase_1_configs()
    migrator.migrate_phase_2_docs()
    migrator.migrate_phase_3_scripts()
    
    # ログ保存
    log_path = root_path / 'docs' / 'architecture' / 'migration_log.json'
    migrator.save_log(log_path)
    
    print("\n" + "="*80)
    print("移行スクリプト完了")
    print("="*80)


if __name__ == '__main__':
    main()
