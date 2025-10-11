"""
プロジェクト構造分析スクリプト

現在のディレクトリ構造を分析し、問題点と改善点を特定します。
"""

from pathlib import Path
from collections import defaultdict
import json


def analyze_directory_structure(root_path: Path) -> dict:
    """ディレクトリ構造を分析"""
    
    stats = {
        'total_files': 0,
        'total_dirs': 0,
        'by_extension': defaultdict(int),
        'by_directory': defaultdict(int),
        'depth_distribution': defaultdict(int),
        'large_directories': [],  # 10ファイル以上
        'scattered_files': [],     # ルート直下のファイル
        'orphan_files': [],        # 適切な場所がないファイル
    }
    
    # ルート直下のファイルをチェック
    root_files = [f for f in root_path.iterdir() if f.is_file()]
    for f in root_files:
        if f.suffix in ['.py', '.json', '.md', '.txt']:
            stats['scattered_files'].append(str(f.name))
    
    # ディレクトリを走査
    for path in root_path.rglob('*'):
        # 除外ディレクトリ
        if any(exclude in str(path) for exclude in ['.git', '.venv', 'node_modules', '__pycache__']):
            continue
        
        if path.is_file():
            stats['total_files'] += 1
            stats['by_extension'][path.suffix] += 1
            
            # 親ディレクトリをカウント
            rel_parent = path.parent.relative_to(root_path)
            stats['by_directory'][str(rel_parent)] += 1
            
            # 深さを計算
            depth = len(path.relative_to(root_path).parts) - 1
            stats['depth_distribution'][depth] += 1
            
        elif path.is_dir():
            stats['total_dirs'] += 1
    
    # 大きなディレクトリを特定（10ファイル以上）
    for dir_path, count in stats['by_directory'].items():
        if count >= 10:
            stats['large_directories'].append({
                'path': dir_path,
                'file_count': count
            })
    
    # リストをソート
    stats['large_directories'].sort(key=lambda x: x['file_count'], reverse=True)
    
    return stats


def identify_issues(stats: dict) -> list:
    """問題点を特定"""
    issues = []
    
    # Issue 1: ルート直下のファイルが多すぎる
    if len(stats['scattered_files']) > 20:
        issues.append({
            'severity': 'high',
            'category': '構造の混乱',
            'issue': f"ルート直下に{len(stats['scattered_files'])}個のファイルが散在",
            'impact': '新規開発者の混乱、ファイル検索の困難',
            'recommendation': 'configs/, docs/, scripts/等に分類'
        })
    
    # Issue 2: 大きすぎるディレクトリ
    if stats['large_directories']:
        top_dir = stats['large_directories'][0]
        if top_dir['file_count'] > 15:
            issues.append({
                'severity': 'medium',
                'category': 'モジュールサイズ',
                'issue': f"{top_dir['path']}に{top_dir['file_count']}個のファイル",
                'impact': 'モジュールの責任が不明確、保守性低下',
                'recommendation': 'サブディレクトリに分割'
            })
    
    # Issue 3: 深さの不均一性
    max_depth = max(stats['depth_distribution'].keys()) if stats['depth_distribution'] else 0
    min_depth = min(stats['depth_distribution'].keys()) if stats['depth_distribution'] else 0
    if max_depth - min_depth > 3:
        issues.append({
            'severity': 'low',
            'category': '階層の不均一性',
            'issue': f"階層の深さが{min_depth}から{max_depth}まで変動",
            'impact': '予測可能性の低下',
            'recommendation': '階層の深さを3-5レベルに統一'
        })
    
    return issues


def generate_recommendations(stats: dict, issues: list) -> list:
    """改善提案を生成"""
    recommendations = []
    
    # ルート直下のファイル分類
    if stats['scattered_files']:
        file_categories = {
            'configs': [f for f in stats['scattered_files'] if f.endswith('.json')],
            'docs': [f for f in stats['scattered_files'] if f.endswith('.md')],
            'scripts': [f for f in stats['scattered_files'] if f.endswith('.py') and 'train' in f.lower()],
            'analysis': [f for f in stats['scattered_files'] if f.endswith('.py') and 'analyze' in f.lower()],
        }
        
        for category, files in file_categories.items():
            if files:
                recommendations.append({
                    'action': 'move',
                    'category': category,
                    'files': files,
                    'destination': f'{category}/',
                    'priority': 'high' if len(files) > 5 else 'medium'
                })
    
    # 大きなディレクトリの分割提案
    for large_dir in stats['large_directories'][:3]:  # トップ3のみ
        recommendations.append({
            'action': 'split',
            'path': large_dir['path'],
            'file_count': large_dir['file_count'],
            'suggestion': f"機能ごとにサブディレクトリを作成（例: base/, utils/, models/）",
            'priority': 'medium'
        })
    
    return recommendations


def print_analysis_report(stats: dict, issues: list, recommendations: list):
    """分析レポートを出力"""
    print("=" * 80)
    print("  プロジェクト構造分析レポート")
    print("=" * 80)
    print()
    
    # 基本統計
    print("📊 基本統計:")
    print(f"  総ファイル数: {stats['total_files']}")
    print(f"  総ディレクトリ数: {stats['total_dirs']}")
    print(f"  ルート直下のファイル: {len(stats['scattered_files'])}")
    print()
    
    # ファイルタイプ分布
    print("📁 ファイルタイプ分布:")
    for ext, count in sorted(stats['by_extension'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {ext or '(拡張子なし)'}: {count}ファイル")
    print()
    
    # 深さの分布
    print("📏 階層の深さ分布:")
    for depth, count in sorted(stats['depth_distribution'].items()):
        print(f"  レベル{depth}: {count}ファイル")
    print()
    
    # 大きなディレクトリ
    print("📦 ファイル数が多いディレクトリ (トップ10):")
    for dir_info in stats['large_directories'][:10]:
        print(f"  {dir_info['path']}: {dir_info['file_count']}ファイル")
    print()
    
    # 問題点
    print("⚠️  特定された問題点:")
    if not issues:
        print("  問題なし ✅")
    else:
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. [{issue['severity'].upper()}] {issue['category']}")
            print(f"     問題: {issue['issue']}")
            print(f"     影響: {issue['impact']}")
            print(f"     推奨: {issue['recommendation']}")
            print()
    
    # 改善提案
    print("💡 改善提案:")
    if not recommendations:
        print("  提案なし")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. [{rec.get('priority', 'medium').upper()}] {rec['action'].upper()}")
            if rec['action'] == 'move':
                print(f"     カテゴリ: {rec['category']}")
                print(f"     移動先: {rec['destination']}")
                print(f"     ファイル数: {len(rec['files'])}")
                if len(rec['files']) <= 5:
                    for f in rec['files']:
                        print(f"       - {f}")
                else:
                    for f in rec['files'][:3]:
                        print(f"       - {f}")
                    print(f"       ... 他{len(rec['files']) - 3}ファイル")
            elif rec['action'] == 'split':
                print(f"     パス: {rec['path']}")
                print(f"     ファイル数: {rec['file_count']}")
                print(f"     提案: {rec['suggestion']}")
            print()
    
    print("=" * 80)


def save_to_json(stats: dict, issues: list, recommendations: list, output_path: Path):
    """結果をJSONに保存"""
    result = {
        'statistics': {
            'total_files': stats['total_files'],
            'total_dirs': stats['total_dirs'],
            'by_extension': dict(stats['by_extension']),
            'depth_distribution': {str(k): v for k, v in stats['depth_distribution'].items()},
            'scattered_files_count': len(stats['scattered_files']),
            'large_directories_count': len(stats['large_directories'])
        },
        'issues': issues,
        'recommendations': recommendations
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 分析結果を保存: {output_path}")


def main():
    """メイン実行"""
    # プロジェクトルート
    root_path = Path(__file__).parent.parent.parent
    
    print(f"プロジェクトルート: {root_path}")
    print("分析中...\n")
    
    # 分析実行
    stats = analyze_directory_structure(root_path)
    issues = identify_issues(stats)
    recommendations = generate_recommendations(stats, issues)
    
    # レポート出力
    print_analysis_report(stats, issues, recommendations)
    
    # JSON保存
    output_path = root_path / 'docs' / 'architecture' / 'structure_analysis.json'
    save_to_json(stats, issues, recommendations, output_path)


if __name__ == '__main__':
    main()
