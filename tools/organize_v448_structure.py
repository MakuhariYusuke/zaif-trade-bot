"""
v448ディレクトリ構造整理スクリプト

使用方法:
    python tools/utilities/organize_v448_structure.py --create      # 新ディレクトリ作成
    python tools/utilities/organize_v448_structure.py --archive-old # 古いバージョン整理
    python tools/utilities/organize_v448_structure.py --all         # 全実行
"""
import argparse
import shutil
from pathlib import Path
from typing import List
import json


class V448StructureOrganizer:
    """v448実装のためのディレクトリ構造整理"""
    
    def __init__(self, root_path: Path):
        self.root = root_path
        self.config_dir = root_path / "config"
        self.docs_dir = root_path / "docs"
        self.tools_dir = root_path / "tools"
        
    def create_new_structure(self):
        """新しいディレクトリ構造を作成"""
        print("📁 新しいディレクトリ構造を作成中...")
        
        # config/の新構造
        new_dirs = [
            self.config_dir / "active" / "v447",
            self.config_dir / "active" / "v448" / "emergency",
            self.config_dir / "active" / "v448" / "balanced",
            self.config_dir / "active" / "v448" / "experimental",
            self.config_dir / "active" / "v448" / "templates",
            self.config_dir / "archived",
            
            # docs/の新構造
            self.docs_dir / "current",
            self.docs_dir / "versions" / "v447",
            self.docs_dir / "versions" / "v448",
            self.docs_dir / "guides",
            self.docs_dir / "api",
            self.docs_dir / "archived",
            
            # tools/の新構造
            self.tools_dir / "analysis",
            self.tools_dir / "training",
            self.tools_dir / "utilities",
            
            # 新ディレクトリ
            self.root / "experiments" / "v448_phase0_emergency",
            self.root / "experiments" / "v448_phase1_config",
        ]
        
        for dir_path in new_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {dir_path.relative_to(self.root)}")
        
        # README作成
        self._create_readmes()
        
        print("✅ 新しい構造の作成完了\n")
    
    def _create_readmes(self):
        """各ディレクトリにREADME作成"""
        readmes = {
            self.config_dir / "active" / "README.md": 
                "# Active Configurations\n\n現在使用中のv447, v448設定ファイル",
            
            self.config_dir / "archived" / "README.md":
                "# Archived Configurations\n\nv367-v446の古いバージョン設定",
            
            self.docs_dir / "current" / "README.md":
                "# Current Documentation\n\n最新版（v448）のドキュメント",
            
            self.tools_dir / "analysis" / "README.md":
                "# Analysis Tools\n\nレポート分析、統計分析ツール",
            
            self.tools_dir / "training" / "README.md":
                "# Training Tools\n\nABテスト、パラメータ探索ツール",
            
            self.tools_dir / "utilities" / "README.md":
                "# Utilities\n\n汎用ユーティリティスクリプト",
        }
        
        for path, content in readmes.items():
            path.write_text(content, encoding='utf-8')
    
    def archive_old_versions(self, dry_run: bool = False):
        """古いバージョン設定を整理"""
        print("📦 古いバージョンをアーカイブ中...")
        
        # v367-v446をアーカイブ
        archived_dir = self.config_dir / "archived"
        
        old_versions = []
        for v in range(367, 447):
            version_dir = self.config_dir / f"v{v}"
            if version_dir.exists():
                old_versions.append(version_dir)
        
        print(f"  対象: {len(old_versions)}個のバージョン")
        
        if not old_versions:
            print("  ℹ️  アーカイブ対象なし")
            return
        
        for version_dir in old_versions:
            target = archived_dir / version_dir.name
            
            if dry_run:
                print(f"  [DRY-RUN] {version_dir.name} → archived/")
            else:
                if target.exists():
                    print(f"  ⚠️  {version_dir.name} 既に存在、スキップ")
                else:
                    shutil.move(str(version_dir), str(target))
                    print(f"  ✅ {version_dir.name} → archived/")
        
        if not dry_run:
            print(f"✅ {len(old_versions)}個のバージョンをアーカイブ完了\n")
        else:
            print(f"ℹ️  [DRY-RUN] 実際には移動していません\n")
    
    def move_active_configs(self, dry_run: bool = False):
        """v447をactive/に移動"""
        print("📂 アクティブ設定を移動中...")
        
        v447_dir = self.config_dir / "v447"
        active_v447 = self.config_dir / "active" / "v447"
        
        if not v447_dir.exists():
            print("  ⚠️  config/v447が存在しません")
            return
        
        if active_v447.exists():
            print("  ℹ️  active/v447は既に存在します")
            return
        
        if dry_run:
            print(f"  [DRY-RUN] v447/ → active/v447/")
        else:
            shutil.copytree(str(v447_dir), str(active_v447))
            print(f"  ✅ v447/ → active/v447/")
        
        print("✅ アクティブ設定の移動完了\n")
    
    def organize_docs(self, dry_run: bool = False):
        """ドキュメント整理"""
        print("📚 ドキュメントを整理中...")
        
        # 最新版ドキュメントをcurrent/に移動
        current_docs = [
            "SAC_v448_DEVELOPMENT_PLAN.md",
            "SAC_v448_IMPLEMENTATION_ROADMAP.md",
            "BALANCE_EXPLORATION_AND_MEMORY_OPTIMIZATION.md",
        ]
        
        for doc in current_docs:
            src = self.docs_dir / doc
            dst = self.docs_dir / "current" / doc
            
            if not src.exists():
                print(f"  ℹ️  {doc} が見つかりません（後で作成予定）")
                continue
            
            if dst.exists():
                print(f"  ℹ️  {doc} は既に存在します")
                continue
            
            if dry_run:
                print(f"  [DRY-RUN] {doc} → current/")
            else:
                shutil.copy(str(src), str(dst))
                print(f"  ✅ {doc} → current/")
        
        # v447ドキュメントをversions/v447/に
        v447_docs = [
            "SAC_v447_DEVELOPMENT_PLAN.md",
        ]
        
        for doc in v447_docs:
            src = self.root / doc
            dst = self.docs_dir / "versions" / "v447" / doc
            
            if src.exists() and not dst.exists():
                if dry_run:
                    print(f"  [DRY-RUN] {doc} → versions/v447/")
                else:
                    shutil.copy(str(src), str(dst))
                    print(f"  ✅ {doc} → versions/v447/")
        
        print("✅ ドキュメント整理完了\n")
    
    def organize_tools(self, dry_run: bool = False):
        """ツール整理"""
        print("🔧 ツールを整理中...")
        
        # 分析ツール
        analysis_tools = [
            "analyze_recent_reports.py",
            "analyze_profitability_vs_balance.py",
        ]
        
        for tool in analysis_tools:
            src = self.tools_dir / tool
            dst = self.tools_dir / "analysis" / tool
            
            if src.exists() and not dst.exists():
                if dry_run:
                    print(f"  [DRY-RUN] {tool} → analysis/")
                else:
                    shutil.move(str(src), str(dst))
                    print(f"  ✅ {tool} → analysis/")
        
        # トレーニングツール
        training_tools = [
            "ab_test_runner.py",
            "ab_param_search.py",
        ]
        
        for tool in training_tools:
            src = self.tools_dir / tool
            dst = self.tools_dir / "training" / tool
            
            if src.exists() and not dst.exists():
                if dry_run:
                    print(f"  [DRY-RUN] {tool} → training/")
                else:
                    shutil.move(str(src), str(dst))
                    print(f"  ✅ {tool} → training/")
        
        print("✅ ツール整理完了\n")
    
    def update_gitignore(self):
        """gitignore更新"""
        print("📝 .gitignore を更新中...")
        
        gitignore_path = self.root / ".gitignore"
        
        additions = [
            "\n# v448 structure",
            "config/archived/",
            "docs/archived/",
            "experiments/",
            "reports/training/",
            "reports/analysis/",
            "reports/experiments/",
        ]
        
        current_content = gitignore_path.read_text(encoding='utf-8')
        
        new_entries = []
        for entry in additions:
            if entry not in current_content:
                new_entries.append(entry)
        
        if new_entries:
            with gitignore_path.open('a', encoding='utf-8') as f:
                f.write('\n'.join(new_entries))
            print(f"  ✅ {len(new_entries)}個のエントリを追加")
        else:
            print("  ℹ️  更新不要")
        
        print("✅ .gitignore更新完了\n")
    
    def generate_summary(self):
        """整理サマリーを生成"""
        print("\n" + "="*60)
        print("📊 整理サマリー")
        print("="*60)
        
        summary = {
            "config/active": len(list((self.config_dir / "active").rglob("*.json"))) if (self.config_dir / "active").exists() else 0,
            "config/archived": len(list((self.config_dir / "archived").iterdir())) if (self.config_dir / "archived").exists() else 0,
            "docs/current": len(list((self.docs_dir / "current").glob("*.md"))) if (self.docs_dir / "current").exists() else 0,
            "tools/analysis": len(list((self.tools_dir / "analysis").glob("*.py"))) if (self.tools_dir / "analysis").exists() else 0,
            "tools/training": len(list((self.tools_dir / "training").glob("*.py"))) if (self.tools_dir / "training").exists() else 0,
        }
        
        for path, count in summary.items():
            print(f"  {path}: {count}個")
        
        print("\n✅ 整理完了！")


def main():
    parser = argparse.ArgumentParser(description="v448ディレクトリ構造整理")
    parser.add_argument("--create", action="store_true", help="新ディレクトリ作成")
    parser.add_argument("--archive-old", action="store_true", help="古いバージョン整理")
    parser.add_argument("--move-active", action="store_true", help="v447をactiveに移動")
    parser.add_argument("--organize-docs", action="store_true", help="ドキュメント整理")
    parser.add_argument("--organize-tools", action="store_true", help="ツール整理")
    parser.add_argument("--all", action="store_true", help="全実行")
    parser.add_argument("--dry-run", action="store_true", help="実際には変更しない")
    
    args = parser.parse_args()
    
    # ルートディレクトリ検出
    root = Path(__file__).resolve().parent.parent.parent
    
    organizer = V448StructureOrganizer(root)
    
    if args.all or args.create:
        organizer.create_new_structure()
    
    if args.all or args.archive_old:
        organizer.archive_old_versions(dry_run=args.dry_run)
    
    if args.all or args.move_active:
        organizer.move_active_configs(dry_run=args.dry_run)
    
    if args.all or args.organize_docs:
        organizer.organize_docs(dry_run=args.dry_run)
    
    if args.all or args.organize_tools:
        organizer.organize_tools(dry_run=args.dry_run)
    
    if args.all and not args.dry_run:
        organizer.update_gitignore()
    
    organizer.generate_summary()


if __name__ == "__main__":
    main()
