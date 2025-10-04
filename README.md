# Zaif Trade Bot (exchange-agnostic)

長時間学習・実運用を見据えた、**強化学習 + バックテスト + ペーパートレード**一体型の取引基盤。
リポジトリ名に"Zaif"を含みますが、**取引所非依存（Zaif/Coincheck 等を切替可能）**な

## 現在のトレーニング設定

- **トレーニングステップ**: 1,000,000ステップ（1Mステップ）
- **特徴量次元**: 66次元（テクニカル指標ベースの特徴量セット）
- **アルゴリズム**: PPO (Proximal Policy Optimization)
- **環境**: HeavyTradingEnv (カスタム強化学習環境)
- **データ**: ml-dataset-enhanced.csv (強化された学習データセット)

## 評価フレームワーク (v3.4.0)

包括的なモデル評価システムで、伝統的な取引指標と高度なリスク・パフォーマンス分析を統合：

### 評価モジュール

- **Performance Attribution**: 収益源泉の詳細分析
- **Monte Carlo Simulation**: 確率的シナリオ分析とVaR計算
- **Strategy Robustness**: ストレステストと安定性評価
- **Benchmark Comparison**: ベンチマークとの相対比較
- **Risk Parity Analysis**: リスク分散最適化分析
- **Cost Sensitivity**: 取引コスト影響評価

### 包括的スコアリング

- **総合スコア**: 伝統指標と拡張評価の統合
- **リスク調整スコア**: リスク管理重視の評価
- **堅牢性スコア**: 安定性と一貫性の評価
- **進捗バー**: リアルタイム評価進捗表示

```bash
# 包括的ベンチマーク実行例
python comprehensive_benchmark.py --data ml-dataset-enhanced.csv --single-model models/final_model.zip --episodes 5 --output-dir benchmark_results
```

詳細: [comprehensive_benchmark.py](./comprehensive_benchmark.py), [CHANGELOG.md](./CHANGELOG.md) 2025-10-04エントリ

```bash
python -m ztb.training.rollup_artifacts --correlation-id  --interval-minutes 5
```

### 再開 & 停止

- 自動再開: supervisor が最新チェックポイントから再開
- 手動停止: プロジェクトルートに ztb.stop を作成 → 速やかに安全停止

---

## Run_1M Training Performance Analysis & Optimization

### 実行パス分析（2025-10-01実施）

`run_1m.py`スクリプトの実行時における12のクリティカルパスを分析し、重要度・ネック度に基づく対応優先順位を付与しました。

#### 優先順位1: クリティカル (毎回通る・失敗で停止)

1. **データファイル読み込みと検証** ⭐⭐⭐⭐⭐ - ファイル不在で即死
2. **特徴量計算 (全データ処理)** ⭐⭐⭐⭐⭐ - 数万行×100特徴量の計算、50-80%短縮可能
3. **環境初期化 (HeavyTradingEnv)** ⭐⭐⭐⭐⭐ - RL環境の基盤、メモリ使用量最適化が必要
4. **PPOモデル初期化と学習ループ** ⭐⭐⭐⭐⭐ - メイン学習処理、GPU/CPU負荷

#### 優先順位2: 高頻度/高コスト (最適化必須)

1. **チェックポイント保存** ⭐⭐⭐⭐ - I/O負荷、非同期化で90%短縮可能
2. **特徴量マネージャー初期化** ⭐⭐⭐⭐ - インポート/関数登録
3. **引数解析と検証** ⭐⭐⭐⭐ - 設定の基盤

#### 優先順位3: 条件付き/周辺処理

1. **ストリーミングパイプライン初期化** ⭐⭐⭐ - 接続安定性
2. **Discord通知** ⭐⭐⭐ - 監視/通知
3. **長時間実行確認** ⭐⭐ - ユーザー確認
4. **ログ設定と監視** ⭐⭐ - デバッグ/監視
5. **エラーハンドリングとクリーンアップ** ⭐⭐ - 異常時対応

#### 最適化計画

- **並列特徴量計算**: ThreadPoolExecutor使用でCPU使用率向上
- **メモリ最適化**: DataFrame dtype最適化、不要データ破棄（30-50%削減）
- **チェックポイントI/O最適化**: 非同期保存、増分保存、圧縮アルゴリズム動的選択
- **リソース監視**: CPU/メモリ/GPU使用率の定期監視
- **データ事前検証**: 特徴量計算前のデータ品質チェック

詳細: [CHANGELOG.md](./CHANGELOG.md) 2025-10-01エントリ

### Quick Start: live:minimal / DRY_RUN

実取引前に最小検証を行う手順:

1. **DRY_RUN=1 (ドライラン)**: モック取引でロジック検証

   ```bash
   export DRY_RUN=1
   python -m ztb.live.service_runner --config scalping-config.json
   ```

2. **live:minimal**: 最小発注→即キャンセルでAPI接続検証

   ```bash
   export LIVE_MINIMAL=1
   export TEST_FLOW_QTY=0.001  # 小量
   python -m ztb.live.service_runner --config scalping-config.json
   ```

3. **実取引移行**: DRY_RUN=0で本番開始

   ```bash
   unset DRY_RUN
   python -m ztb.live.service_runner --config scalping-config.json
   ```

---

## Evaluation & Validation (DSR/Bootstrap/Benchmarks)
> See also: [DISCLAIMER.md](./DISCLAIMER.md)

---

## Project Overview

- 目的：**実取引可能な Bot** を作ること。1M ステップ学習はその達成手段の一つ。
- 特徴：長時間学習を前提に、**再開性・安全性・可観測性**を重視。
- 交換可能な実装：ブローカー層はアダプタ方式（Zaif/Coincheck 等）。
  参考: [docs/runbook.md](./docs/runbook.md), [ztb/trading/README.md](./ztb/trading/README.md)

## Key Features

- **学習基盤**: PPO トレーナ、1M ステップ前提のチェックポイント運用（async + zstd 圧縮）
- **評価**: DSR（Deflated Sharpe Ratio）、Bootstrap による信頼区間、基準戦略比較
- **データ**: ストリーミング + 循環バッファ、オフライン/オンライン両対応
- **運用**: 監視（watcher）、自動再開（supervisor）、定期要約（rollup）
- **安全**: Circuit Breakers、Kill-file、冪等な注文状態機械、ドライラン
- **検証**: Canary（Linux/PS 同等動作）と障害注入（テーブル駆動）
- **インフラ**: 取引所ヘルスチェック、統一データソース管理、一貫したCLI、回帰スモークテスト

---

## File Organization Migration

As part of the refactoring to organize Python files into proper packages, the following files have been moved from `ztb/scripts/` to organized packages:

| Original Path | New Path | Category |
|---------------|----------|----------|
| alert_notifier.py | ztb/ops/monitoring/alert_notifier.py | monitoring/notification |
| health_monitor.py | ztb/ops/monitoring/health_monitor.py | monitoring/notification |
| disk_health.py | ztb/ops/monitoring/disk_health.py | monitoring/notification |
| ops_doctor.py | ztb/ops/monitoring/ops_doctor.py | monitoring/notification |
| collect_last_errors.py | ztb/ops/monitoring/collect_last_errors.py | monitoring/notification |
| artifacts_janitor.py | ztb/ops/artifacts/artifacts_janitor.py | artifacts/reports |
| compact_jsonl.py | ztb/ops/artifacts/compact_jsonl.py | artifacts/reports |
| cost_estimator.py | ztb/ops/artifacts/cost_estimator.py | artifacts/reports |
| dump_contract_schemas.py | ztb/ops/config/dump_contract_schemas.py | config/schema |
| benchmark_checkpoint.py | ztb/ops/benchmark/benchmark_checkpoint.py | benchmark/observability |
| benchmark_streaming.py | ztb/ops/benchmark/benchmark_streaming.py | benchmark/observability |
| compat_wrapper.py | ztb/ops/benchmark/compat_wrapper.py | benchmark/observability |
| smoke_tests.py | ztb/ops/benchmark/smoke_tests.py | benchmark/observability |
| fix_train.py | ztb/training/entrypoints/fix_train.py | training entrypoints |
| generate_sample_data.py | ztb/training/entrypoints/generate_sample_data.py | training entrypoints |
| validate_data_loading.py | ztb/training/entrypoints/validate_data_loading.py | training entrypoints |
| profile_features.py | ztb/training/evaluation/profile_features.py | evaluation/regime |
| re_evaluate_list.yaml | ztb/training/evaluation/re_evaluate_list.yaml | evaluation/regime |

All imports have been updated to use absolute paths. The `ztb/scripts/` directory now contains only shell scripts and the `trading_service.py` shim for backward compatibility.

---

## Architecture Overview

- **学習/環境**: ztb/trading/（PPO、環境、チェックポイント、評価フック）
  → 詳細: [ztb/trading/README.md](./ztb/trading/README.md)
- **データ**: ztb/data/（ストリーミング、バッファ、バリデーション）
  → 詳細: [ztb/data/README.md](./ztb/data/README.md)
- **ユーティリティ**: ztb/util/（テスト支援、設定、観測、スキーマ等）
  → 詳細: [ztb/util/README.md](./ztb/util/README.md)
- **開発者向け**: アーキテクチャ/セットアップ/テスト
  → 詳細: [docs/contributing/architecture.md](./docs/contributing/architecture.md),
           [docs/contributing/setup.md](./docs/contributing/setup.md),
           [docs/contributing/testing.md](./docs/contributing/testing.md)

---

## Quick Start

> Python 3.11 で動作確認済。3.13 対応の検証は CI マトリクスで進行中。

### 環境セットアップ

```bash
# リポジトリをクローン
git clone <repository-url>
cd zaif-trade-bot

# 開発環境のセットアップ（Makefile使用推奨）
make setup

# または手動で
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install types-requests types-psutil
pre-commit install
npm install
```

### 基本的なテスト実行

```bash
# 全テスト実行（ユニット + 統合テスト）
make test

# ユニットテストのみ（高速）
npm run test:unit

# 統合テストのみ（低速だが包括的）
npm run test:int-fast

# コード品質チェック（mypy, black, isort）
make check

# セキュリティ監査
make audit
```

### トレーニング実行の詳細手順

#### 1. 環境準備
```bash
# Python仮想環境のアクティベート
source .venv/Scripts/activate  # Windows
# または
.venv/bin/activate  # Linux/Mac

# 依存関係の確認
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

#### 2. データ準備
```bash
# データファイルの存在確認
ls -la ml-dataset-enhanced.csv

# データ品質チェック（オプション）
python -m ztb.data.validate_data ml-dataset-enhanced.csv
```

#### 3. 設定ファイルの確認
```bash
# 設定ファイルの検証
python -c "import json; print(json.load(open('unified_training_config.json')))"

# 特徴量数の確認
python -c "
import pandas as pd
df = pd.read_csv('ml-dataset-enhanced.csv')
exclude_cols = ['ts', 'pair', 'side', 'pnl', 'win', 'source', 'timestamp']
features = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
print(f'特徴量数: {len(features)}')
"
```

#### 4. トレーニング実行
```bash
# 基本実行（1Mステップ）
python -m ztb.training.unified_trainer --config unified_training_config.json

# ドライラン（設定検証のみ）
python -m ztb.training.unified_trainer --config unified_training_config.json --dry-run

# 短時間テスト実行（10kステップ）
python -m ztb.training.unified_trainer --config unified_training_config.json --total-timesteps 10000
```

#### 5. トレーニング監視
```bash
# リアルタイム監視（別ターミナルで）
python -m ztb.training.watch_1m --correlation-id scalping_15s_ultra_aggressive_1M

# 定期要約（5分ごと）
python -m ztb.training.rollup_artifacts --correlation-id scalping_15s_ultra_aggressive_1M --interval-minutes 5
```

#### 6. 結果確認
```bash
# チェックポイント確認
ls -la checkpoints/scalping_15s_ultra_aggressive_1M/

# ログ確認
tail -f logs/scalping_15s_ultra_aggressive_1M.log

# TensorBoard（オプション）
tensorboard --logdir logs/
```

### よく使うユーティリティ

- 監視: python -m ztb.training.watch_1m --correlation-id `<ID>` --run-once
- 要約: python -m ztb.training.rollup_artifacts --correlation-id `<ID>`
- Canary: linux_canary.sh / ps_canary.ps1（同等のフロー・出力）

---

## Live Trading

### 準備

1. **APIキー設定** (オプション - 設定しない場合はデモモード)
```bash
export COINCHECK_API_KEY="your_api_key"
export COINCHECK_API_SECRET="your_api_secret"
export DISCORD_WEBHOOK="your_webhook_url"  # 通知用（オプション）
```

2. **実行**
```bash
# デモモード（APIキーなし）
python live_trade.py --model-path models/scalping_iterative_v1_final.zip --duration-hours 1

# 本番モード（APIキー設定済み）
python live_trade.py --model-path models/scalping_iterative_v1_final.zip --duration-hours 24

# リスク制限を無効化（テスト/上級者向け）
python live_trade.py --model-path models/scalping_iterative_v1_final.zip --duration-hours 1 --disable-risk-limits
```

### リスク管理機能

- **日次損失制限**: 10,000円（デフォルト）
- **日次トレード数制限**: 50回（デフォルト）
- **緊急ストップロス**: 5%（デフォルト）
- **自動停止システム**: 高度なリスク管理

### クロスプラットフォーム対応

WindowsおよびRaspberry Piで動作可能です。自動的に環境を検知して最適化されます。

### ログ確認

```bash
# ライブトレーディングログ
tail -f logs/live_trading_*.log
```

---

## 1M Step Training Execution

### 推奨の起動方法（セッション ID = correlation-id）

```bash
## 例: UTC タイムスタンプを ID に
CORR=070929T160549Z

## 自動再開つき起動（存在すれば run_1m.py、なければ PPO 直呼び）
python -m ztb.training.supervise_1m --correlation-id  \
  --ppo-cli-args "--resume-from latest --total-timesteps 1000000 \
  --n-envs 4 --seed 42 --eval-interval 10000 --log-interval 1000 \
  --ckpt-async --ckpt-compress zstd --ckpt-max-pending 1"
```

### 進行中の監視

```bash
## 一度だけチェック
python -m ztb.training.watch_1m --correlation-id  --run-once

## 連続監視（閾値は環境変数で調整）
ZTB_WATCH_STALL_MIN=10 ZTB_WATCH_RSS_MB=2048 ZTB_WATCH_VRAM_MB=4096 \
python -m ztb.training.watch_1m --correlation-id
```

### 定期要約（5 分ごと）

```bash
python -m ztb.training.rollup_artifacts --correlation-id  --interval-minutes 5
`

### 再開 & 停止

- 自動再開: supervisor が最新チェックポイントから再開
- 手動停止: プロジェクトルートに ztb.stop を作成 → 速やかに安全停止

---

## Unified Training Runner

複数のトレーニングアプローチを統合した統一インターフェース。詳細: [UNIFIED_TRAINING_README.md](./UNIFIED_TRAINING_README.md)

### サポートされるアルゴリズム

- **PPO Training**: 標準的な PPO 強化学習トレーニング
- **Base ML Reinforcement**: カスタム実験フレームワーク
- **Iterative Training**: 反復トレーニングセッション

### 使用例

```bash
# PPO トレーニング
python -m ztb.training.unified_trainer --config unified_training_config.json --algorithm ppo

# 反復トレーニング
python -m ztb.training.unified_trainer --config unified_training_config.json --algorithm iterative

# 設定上書き
python -m ztb.training.unified_trainer --config unified_training_config.json --total-timesteps 500000
```

---

## Evaluation & Validation (DSR/Bootstrap/Benchmarks)

- **DSR**: 多重検定を考慮した Sharpe の有意性指標。--dsr-trials（既定 cap=1000）
- **Bootstrap**: --bootstrap-resamples（既定 1000 / CI は 200）、--bootstrap-block、--bootstrap-overlap
- **定期評価**: 既定 50k ステップごとに Sharpe/DSR/p 値を算出し、基準戦略（SMA/Buy&Hold）と比較
- **ベンチ**

  - ストリーミング: python ztb/benchmarks/streaming_benchmark.py
  - チェックポイント I/O: python ztb/benchmarks/checkpoint_benchmark.py

結果は artifacts/`<ID>`/reports/eval_*.json として保存、summary.* に集約されます。

---

## Streaming & Checkpoints

- **ストリーミング**: 既定 OFF。有効化時は --enable-streaming
  --stream-batch-size 64 --stream-buffer-policy drop_oldest
- **チェックポイント**: 10k ステップ間隔、保持 5、非同期保存、zstd 圧縮、max-pending=1
- **重複防止**: global_step を用いた **duplication guard** で再開時の二重学習を防止

---

## Production Safety (Risk Management & Shutdown)

- **Circuit Breakers**: 異常時は新規建てを禁止（既存ポジの縮小は許可）
- **Kill-file**: ztb.stop により全コンポーネントが安全停止
- **サイジング**: 年率 10% ターゲットボラ + Kelly 0.5、Decimal 丸め、最小数量/Notional 準拠

詳細: [docs/runbook.md](./docs/runbook.md)

---

## Canary & Fault Injection

- **Canary**: Linux/PowerShell で同等のフェーズ（replay → live-lite → kill/resume）と同一アーティファクト
- **障害注入**: テーブル駆動（切断/タイムアウト/メモリ圧 等）で回帰を検出
  実行例と使い方: [docs/deployment/canary.md](./docs/deployment/canary.md)

---

## Troubleshooting

### トレーニング実行時の一般的な問題

#### メモリ不足エラー
```
RuntimeError: CUDA out of memory
```
**解決策:**
- バッチサイズを小さくする: `--batch-size 32` または `--batch-size 16`
- ストリーミングを有効化: `--enable-streaming --stream-batch-size 64`
- CPU専用モード: `export CUDA_VISIBLE_DEVICES=""`

#### データ読み込みエラー
```
FileNotFoundError: ml-dataset-enhanced.csv
```
**解決策:**
- データファイルの存在を確認: `ls -la ml-dataset-enhanced.csv`
- パスが正しいか確認: プロジェクトルートから実行するか確認
- データ生成: `python generate_enhanced_training_data.py`

#### 特徴量計算エラー
```
ValueError: Input contains NaN
```
**解決策:**
- データの欠損値チェック: `python -c "import pandas as pd; df = pd.read_csv('ml-dataset-enhanced.csv'); print(df.isnull().sum())"`
- 欠損値補完: スクリプト内でfillnaを使用

#### GPU関連エラー
```
CUDA error: no kernel image is available
```
**解決策:**
- GPU互換性確認: `nvidia-smi`
- PyTorchバージョン確認: `python -c "import torch; print(torch.version.cuda)"`
- CPUフォールバック: `export CUDA_VISIBLE_DEVICES=""`

### ライブトレーディングの問題

#### API接続エラー
```
ConnectionError: HTTPSConnectionPool
```
**解決策:**
- APIキーの確認: `echo $COINCHECK_API_KEY`
- ネットワーク接続確認: `curl -I https://coincheck.com`
- レート制限チェック: 短時間に複数リクエストしていないか

#### 注文エラー
```
InvalidOrder: Minimum quantity not met
```
**解決策:**
- 最小注文数量確認: 取引所仕様を確認
- 数量計算の確認: `TEST_FLOW_QTY` でテスト

### パフォーマンス問題

#### トレーニングが遅い
**診断:**
```bash
# CPU使用率確認
top -p $(pgrep -f unified_trainer)

# GPU使用率確認
nvidia-smi

# メモリ使用量確認
python -c "import psutil; print(psutil.virtual_memory())"
```

**最適化:**
- 並列環境数調整: `--n-envs 2` から `--n-envs 1`
- チェックポイント間隔延長: `--ckpt-interval 20000`
- 非同期チェックポイント有効化: `--ckpt-async`

#### ログが大きくなる
**解決策:**
- ログローテーション設定
- 定期的なログ整理: `find logs/ -name "*.log" -mtime +7 -delete`

### モニタリングとデバッグ

#### トレーニング状態確認
```bash
# 現在のステップ数確認
python -c "
import glob
checkpoints = glob.glob('checkpoints/**/*.zip')
if checkpoints:
    latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f'Latest checkpoint: {latest}')
"
```

#### エラーログ解析
```bash
# エラーパターン検索
grep -r "ERROR" logs/ | tail -10

# メモリ使用量トレンド
grep "Memory usage" logs/*.log | tail -20
```

#### 設定検証
```bash
# JSON構文チェック
python -c "import json; json.load(open('unified_training_config.json')); print('Config OK')"

# 必須フィールド確認
python -c "
import json
config = json.load(open('unified_training_config.json'))
required = ['algorithm', 'total_timesteps', 'data_path']
missing = [k for k in required if k not in config]
if missing:
    print(f'Missing fields: {missing}')
else:
    print('All required fields present')
"
```

## Artifacts & Schema

- ルート: rtifacts/<correlation_id>/logs|metrics|reports|config|meta/
- **run_metadata.json**: git SHA / Python/OS/CPU / seeds /
  package & config ハッシュ / correlation_id
- **results_schema.json** に準拠（schema/ 配下）。
esults_validator.py で検証可能。

---

## Configuration, CLI & Environment Variables

- 代表的な CLI:

  - --total-timesteps, --n-envs, --seed, --eval-interval, --log-interval
  - --ckpt-async, --ckpt-compress zstd, --ckpt-max-pending 1
  - --enable-streaming, --stream-batch-size, --stream-buffer-policy
  - --dsr-trials, --bootstrap-resamples, --bootstrap-block, --bootstrap-overlap
- 代表的な環境変数:

  - ZTB_WATCH_*（監視閾値）, ZTB_KILL（kill-file 即時反映）, 他
    詳細: [docs/configuration.md](./docs/configuration.md)

| 環境変数 | 説明 | デフォルト値 | 例 |
|----------|------|--------------|----|
| ZTB_WATCH_CPU_PCT | CPU使用率監視閾値 | 80 | 90 |
| ZTB_WATCH_MEM_PCT | メモリ使用率監視閾値 | 85 | 90 |
| ZTB_KILL | kill-fileパス | ztb.stop | /tmp/ztb.stop |
| DRY_RUN | ドライラン有効化 | 0 | 1 |
| LIVE_MINIMAL | 最小ライブモード | 0 | 1 |
| ZTB_MAX_MEMORY_GB | 最大メモリ使用量 | 8 | 16 |

### Infrastructure Scripts

- **Venue Health Check**: `python -m ztb.ops.check_venue_health --venue coincheck --symbol BTC_JPY`
  - 取引所APIの接続性、レイテンシ、レート制限をチェック
- **Regression Smoke Tests**: `python -m ztb.ops.smoke_tests`
  - 合成データを使った基本機能の回帰テスト
- **CLI Consistency**: 全スクリプトで統一されたヘルプテキストとバリデーション
  - 共通の引数定義（--artifacts-dir, --correlation-id, --timeout等）
  - 標準化されたエラーメッセージとデフォルト値

---

## 移行ガイド: 旧パス→新パス

v2.5.1 より、スクリプトが適切なパッケージに再編成されました。`ztb/scripts/` はシェルスクリプト専用となり、Pythonファイルは削除されています。

| 旧パス | 新パス | 備考 |
|--------|--------|------|
| `python scripts/supervise_1m.py` | `python -m ztb.training.supervise_1m` | 1M学習監督 |
| `python scripts/watch_1m.py` | `python -m ztb.training.watch_1m` | 1M学習監視 |
| `python scripts/rollup_artifacts.py` | `python -m ztb.training.rollup_artifacts` | アーティファクト集約 |
| `python scripts/ops.py` | `python -m ztb.ops.cli` | 運用CLI |
| `python scripts/check_schema_version.py` | `python -m ztb.ops.check_schema_version` | スキーマチェック |
| `python scripts/check_links.py` | `python -m ztb.ops.check_links` | リンクチェック |
| `python scripts/generate_weekly_report.py` | `python -m ztb.ops.generate_weekly_report` | 週次レポート |

**注意**: `ztb/scripts/` には `.sh` および `.ps1` ファイルのみ配置可能です。PythonファイルはCIで拒否されます。

**将来のエントリーポイント** (packaging時):
- `ztb-ops` → `ztb.ops.cli:main`
- `ztb-run1m` → `ztb.training.run_1m:main`
- `ztb-supervise` → `ztb.training.supervise_1m:main`

---

---

## License

本リポジトリは MIT ライセンスです。詳細は [LICENSE](./LICENSE) を参照してください。
