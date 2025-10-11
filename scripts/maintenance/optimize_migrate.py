"""
最適化された構造化実行スクリプト

既存のUtilを活用し、高速・メモリ効率的に構造化を実行します。
- 並列処理による高速化
- バッチ操作
- プログレス表示
- ロールバック機能
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

# プロジェクトルートをPATHに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.file_operations import BatchFileOperator, scan_files_fast, categorize_files
from ztb.utils.path_utils import get_project_root, ensure_dir
from ztb.utils.performance_utils import timed_with_memory

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class OptimizedMigrator:
    """最適化された構造移行クラス"""
    
    def __init__(self, dry_run: bool = True, max_workers: int = 4):
        """
        初期化
        
        Args:
            dry_run: True の場合は実行せずログ出力のみ
            max_workers: 並列処理のワーカー数
        """
        self.root = get_project_root()
        self.dry_run = dry_run
        self.operator = BatchFileOperator(
            root_path=self.root,
            max_workers=max_workers,
            dry_run=dry_run,
            verbose=True
        )
        self.migration_plan = {}
    
    @timed_with_memory
    def analyze_and_plan(self) -> Dict[str, Any]:
        """
        高速分析と移行計画立案
        
        Returns:
            移行計画の詳細
        """
        logger.info("=" * 80)
        logger.info("Phase 0: 高速分析と移行計画立案")
        logger.info("=" * 80)
        
        # ルート直下のファイルのみをスキャン（高速）
        logger.info("ルート直下のファイルをスキャン中...")
        root_files = scan_files_fast(
            self.root,
            pattern='*',
            max_depth=0  # ルート直下のみ
        )
        
        logger.info(f"発見: {len(root_files)}ファイル")
        
        # カテゴリ分類
        logger.info("ファイルをカテゴリ分類中...")
        categories = categorize_files(root_files)
        
        # 移行計画を生成
        self.migration_plan = {
            'configs': self._plan_configs(categories['configs']),
            'docs': self._plan_docs(categories['docs']),
            'scripts': self._plan_scripts(categories['scripts']),
        }
        
        # サマリー表示
        total_files = sum(len(plan['files']) for plan in self.migration_plan.values())
        logger.info(f"\n移行計画: {total_files}ファイル")
        for category, plan in self.migration_plan.items():
            logger.info(f"  {category}: {len(plan['files'])}ファイル")
        
        return self.migration_plan
    
    def _plan_configs(self, config_files: List[Path]) -> Dict[str, Any]:
        """configs/への移行計画"""
        plan: Dict[str, Any] = {'files': [], 'destinations': {}}
        
        for file_path in config_files:
            name = file_path.name
            
            # サブディレクトリ判定
            if name.startswith('.'):
                dest = self.root / 'configs' / 'linting' / name
            elif 'backtest' in name:
                dest = self.root / 'configs' / 'backtest' / name
            elif 'curriculum' in name or 'training' in name:
                dest = self.root / 'configs' / 'training' / name
            elif 'bitflyer' in name or 'exchange' in name:
                dest = self.root / 'configs' / 'algorithms' / name
            elif 'result' in name or 'comparison' in name or 'ppo' in name:
                dest = self.root / 'configs' / 'results' / name
            else:
                dest = self.root / 'configs' / name
            
            plan['files'].append(file_path)
            plan['destinations'][file_path] = dest
        
        return plan
    
    def _plan_docs(self, doc_files: List[Path]) -> Dict[str, Any]:
        """docs/への移行計画"""
        plan: Dict[str, Any] = {'files': [], 'destinations': {}}
        
        for file_path in doc_files:
            name = file_path.name.upper()
            
            # サブディレクトリ判定
            if any(kw in name for kw in ['CHANGELOG', 'CODE_QUALITY', 'COMPREHENSIVE', 'INTEGRATION_GUIDE']):
                dest = self.root / 'docs' / 'development' / file_path.name
            elif any(kw in name for kw in ['OPERATIONS', 'LIVE_TRADE', 'LOG_LEVEL']):
                dest = self.root / 'docs' / 'operations' / file_path.name
            elif any(kw in name for kw in ['PROPOSAL', 'CONSULTATION', 'ADDITIONAL', 'ADVANCED']):
                dest = self.root / 'docs' / 'proposals' / file_path.name
            elif any(kw in name for kw in ['BUG', 'FIX', 'REPORT', 'SUMMARY', 'CHECKPOINT', 'EMERGENCY', 'FEATURE', 'ROLLOUT', 'DIAGNOSIS']):
                dest = self.root / 'docs' / 'reports' / file_path.name
            elif name in ['LICENSE', 'DISCLAIMER.MD', 'BACKLOG.MD']:
                dest = self.root / 'docs' / file_path.name
            else:
                dest = self.root / 'docs' / file_path.name
            
            plan['files'].append(file_path)
            plan['destinations'][file_path] = dest
        
        return plan
    
    def _plan_scripts(self, script_files: List[Path]) -> Dict[str, Any]:
        """scripts/への移行計画"""
        plan: Dict[str, Any] = {'files': [], 'destinations': {}}
        
        for file_path in script_files:
            name = file_path.name.lower()
            
            # サブディレクトリ判定
            if name.startswith('analyze_'):
                dest = self.root / 'scripts' / 'analysis' / file_path.name
            elif name.startswith('debug_'):
                dest = self.root / 'scripts' / 'debugging' / file_path.name
            elif any(kw in name for kw in ['train', 'backtest', 'live_trade']):
                dest = self.root / 'scripts' / 'training' / file_path.name
            elif any(kw in name for kw in ['compare', 'comparison']):
                dest = self.root / 'scripts' / 'comparison' / file_path.name
            elif any(kw in name for kw in ['inspect', 'investigate', 'list_']):
                dest = self.root / 'scripts' / 'investigation' / file_path.name
            elif any(kw in name for kw in ['create_', 'curated']):
                dest = self.root / 'scripts' / 'data' / file_path.name
            else:
                dest = self.root / 'scripts' / 'utilities' / file_path.name
            
            plan['files'].append(file_path)
            plan['destinations'][file_path] = dest
        
        return plan
    
    @timed_with_memory
    def execute_migration(self) -> Dict[str, int]:
        """
        移行を並列実行
        
        Returns:
            実行結果の統計情報
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"{'[DRY-RUN] ' if self.dry_run else ''}移行実行")
        logger.info("=" * 80)
        
        # 操作をバッチに追加
        total_ops = 0
        for category, plan in self.migration_plan.items():
            logger.info(f"\n📦 {category.upper()}")
            for source_path in plan['files']:
                dest_path = plan['destinations'][source_path]
                self.operator.add_move(source_path, dest_path)
                total_ops += 1
        
        logger.info(f"\n総操作数: {total_ops}")
        
        # 並列実行
        stats = self.operator.execute()
        
        # サマリー
        logger.info("\n" + "=" * 80)
        logger.info("実行完了")
        logger.info("=" * 80)
        logger.info(self.operator.get_summary())
        
        return stats
    
    def save_log(self, output_path: Path):
        """実行ログを保存"""
        ensure_dir(output_path.parent)
        
        log_data = {
            'dry_run': self.dry_run,
            'migration_plan': {
                category: {
                    'file_count': len(plan['files']),
                    'files': [str(f) for f in plan['files']]
                }
                for category, plan in self.migration_plan.items()
            },
            'results': self.operator.results,
            'summary': self.operator.get_summary()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ ログ保存: {output_path}")


def main():
    """メイン実行"""
    print("=" * 80)
    print("  高速構造化実行スクリプト")
    print("  - 並列処理による高速化")
    print("  - メモリ最適化")
    print("  - 既存Util活用")
    print("=" * 80)
    
    # 引数確認
    dry_run = '--execute' not in sys.argv
    max_workers = 8  # 並列度を上げて高速化
    
    if dry_run:
        print("\n⚠️  DRY-RUNモード（実際の移行は行いません）")
        print("実際に移行するには: python optimize_migrate.py --execute")
    else:
        print("\n🚀 実行モード（実際にファイルを移動します）")
        confirm = input("続行しますか？ (yes/no): ")
        if confirm.lower() != 'yes':
            print("キャンセルしました。")
            return
    
    # 移行実行
    migrator = OptimizedMigrator(dry_run=dry_run, max_workers=max_workers)
    
    # 分析・計画
    migrator.analyze_and_plan()
    
    # 実行
    migrator.execute_migration()
    
    # ログ保存
    log_path = get_project_root() / 'docs' / 'architecture' / 'optimized_migration_log.json'
    migrator.save_log(log_path)
    
    print("\n" + "=" * 80)
    print("完了")
    print("=" * 80)


if __name__ == '__main__':
    main()
