# リポジトリ整理タスク — zaif-trade-bot

## 背景

BTC/JPY 自動売買ボット (SAC強化学習ベース) のリポジトリ。開発が長期化し、6,208 追跡ファイル・80+ ルートディレクトリに肥大化。IDE・Git操作が著しく遅延しており、整理が急務。

## 現在のルート構造（問題のあるもの）

### 即座に削除可能（一時ファイル・ログ・旧成果物）

ルート直下の一時ファイル群（全てGitから除外してよい）:
```
*.txt (action_analysis_*, all_objects*, big_objects*, blob_sizes_*, stash_diff*, syntax_*, temp_*, test_d0_*, training_*_log*, training_*txt)
*.log (scan_*, test_*, training_*, temp_*)
*.csv (backtest_gate_log*, backtest_trades_*, test_synthetic_dataset*)
*.png (test_plot*)
*.json (test_results*)
tmp_*.py, temp_*.py
```

ルート直下のスクリプト（archived/ へ移動すべき）:
```
alert_system.py, analyze_*.py, backtest_v45*.py, circuit_breaker.py,
debug_*.py, diagnose_*.py, emergency_stop.py, health_checker.py,
inspect_model.py, market_data_simulator.py, paper_trading_manager.py,
performance_monitor.py, performance_validator.py, real_time_metrics.py,
recovery_system.py, result_comparator.py, risk_based_allocator.py,
rollback_manager.py, sac.py, virtual_portfolio_manager.py,
test_reward_simplified.py, test_scale_verification.py,
test_short_step_training.py
```

### 削除可能ディレクトリ（重複・旧バージョン・一時）

```
.tmp/                    # 一時ファイル
.tmp-strategies/         # 一時ファイル
.tmp-utils-stats/        # 一時ファイル
.hypothesis/             # pytest-hypothesis キャッシュ
.mypy_cache/             # mypy キャッシュ
.ruff_cache/             # ruff キャッシュ
.pytest_cache/           # pytest キャッシュ
.benchmarks/             # ベンチマーク結果
htmlcov/                 # カバレッジHTML
build/                   # ビルド成果物
zaif_trade_bot.egg-info/ # egg-info
__pycache__/             # Python キャッシュ（ルート）
node_modules/            # Node.js パッケージ
venv/                    # 古い仮想環境（.venvが正）
venv311/                 # 古い仮想環境
venv311_new/             # 古い仮想環境
.venv311/                # 古い仮想環境
zaif-trade-bot-mirror/   # ミラーリポジトリ
git-filter-repo/         # filter-repo 一時ファイル
v435/                    # 旧バージョン
```

### 統合・整理すべきディレクトリ

```
# 結果系（results/ に統合）
analysis_results/ → results/analysis/
backtest_results/ → results/backtest/
backtest_analysis_plots/ → results/backtest/plots/
experiment_plots/ → results/experiments/plots/
optimization_results/ → results/optimization/
phase3_comparison_results/ → results/phase3/
statistical_sampling_results/ → results/statistical/
test_backtest_results/ → results/test_backtest/
test_results/ → results/test/
training_results/ → results/training/
coverage/ → results/coverage/

# チェックポイント系（checkpoints/ に統合）
test_checkpoints/ → checkpoints/test/
test_checkpoints_phase2/ → checkpoints/test_phase2/
best_model/ → checkpoints/best/
models/ → checkpoints/models/
models_test/ → checkpoints/models_test/
temp_model/ → 削除

# ログ系（logs/ に統合）
eval_logs/ → logs/eval/
sac_action_test_logs/ → logs/sac_action_test/
tensorboard/ → logs/tensorboard/

# スクリプト系
temp_scripts/ → 削除
backtest_experiments/ → archived/backtest_experiments/

# 設定系（configs/ に統合）
config/ → configs/  （内容をマージ）
schema/ → configs/schema/
jsonschema/ → configs/jsonschema/

# Python パッケージ系
stable_baselines3/ → 削除 (.venv 内のパッケージを使用)
sb3_contrib/ → 削除 (.venv 内のパッケージを使用)
_stable_baselines3_shim/ → ztb/compat/sb3_shim/ へ移動
python/ → 内容確認の上削除 or ztb/ へ統合
src/ → 内容確認の上削除 or ztb/ へ統合
utils/ → ztb/utils/ へ統合
bundles/ → 内容確認の上 archived/ or 削除
websockets/ → ztb/api/websockets/ or 削除
venues/ → ztb/api/venues/ or 削除
```

### 保持すべきコアディレクトリ

```
.devcontainer/     # Docker開発環境
.github/           # CI/CD
.vscode/           # エディタ設定
archived/          # アーカイブ済みコード
assets/            # 静的アセット
configs/           # 設定ファイル
data/              # データ（.gitignore推奨）
docker/            # Dockerファイル
docs/              # ドキュメント
notebooks/         # Jupyter
ops/               # 運用スクリプト
plots/             # 図表
prompts/           # AIプロンプト
reports/           # レポート
scripts/           # 実行スクリプト
tests/             # テスト
tools/             # ツール
ztb/               # メインパッケージ
```

## .gitignore に追加すべきパターン

```gitignore
# 一時ファイル（ルート）
/*.txt
/*.log
/*.csv
/*.png
/tmp_*.py
/temp_*.py

# キャッシュ・ビルド
.mypy_cache/
.ruff_cache/
.hypothesis/
.benchmarks/
htmlcov/
build/
*.egg-info/

# 大容量データ（Git LFS or 除外）
data/*.parquet
checkpoints/**/*.zip
models/**/*.zip

# 旧仮想環境
venv/
venv311*/
.venv311/
```

## 実行手順

1. **バックアップ**: `git stash` または別ブランチ作成
2. **削除**: 上記「即座に削除可能」のファイル・ディレクトリを削除
3. **移動**: 「統合・整理すべきディレクトリ」を段階的に実行
4. **import パス更新**: 移動したモジュールの import を grep で検索・更新
5. **.gitignore 更新**: 上記パターンを追加
6. **テスト実行**: `pytest tests/ -x --timeout=60` で回帰確認
7. **コミット**: `--no-verify` で段階的にコミット

## 制約

- `ztb/` パッケージの内部構造は変更しない（活発に開発中）
- `scripts/v459/` は変更しない（現在のPhase D実験に使用中）
- `docs/v459/` は変更しない（ドキュメント履歴）
- `tests/` の構造は変更しない
- `configs/` は内容の確認のみ（削除しない）
- `.positions/` は本番ポジショントラッキングに使用中の可能性 → 確認してから判断

## 期待される結果

- ルート直下のファイル: 80+ → 10 以下 (pyproject.toml, README.md, CHANGELOG.md, conftest.py, pytest.ini, mypy.ini, .gitignore, .env)
- ルートディレクトリ: 80+ → 20 以下
- Git追跡ファイル: 6,208 → 3,000 以下
- `git status` 実行時間: 大幅改善
