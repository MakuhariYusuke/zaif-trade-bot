# Zaif Trade Bot (exchange-agnostic)

長時間学習・実運用を見据えた、**強化学習 + バックテスト + ペーパートレード**一体型の取引基盤。
リポジトリ名に"Zaif"を含みますが、**取引所非依存（Zaif/Coincheck 等を切替可能）**な

## プロジェクト進捗状況 (2025-10-04 更新)

### ✅ 解決した課題

- **報酬関数の基本設計**: ステップ単位PnL + ATR正規化 + アクション考慮の報酬関数を実装
- **reward_scaling最適化**: 二分探索で6.0を最適値として特定（71.10%リターン達成）
- **PPOハイパーパラメータ最適化**: learning_rate=5e-4, gamma=0.95, gae_lambda=0.8などの最適値を特定
- **評価フレームワーク構築**: comprehensive_benchmark.py, ablation_study.py, trade_analysis.pyなどのモジュールを実装
- **特徴量計算最適化**: FeatureRegistryの構築とパフォーマンスプロファイリング
- **トレーニングインフラ整備**: unified_trainer.pyと設定ファイルの統合
- **型安全性向上**: mypyエラーを237個から154個に削減（35%改善）、cast()使用による安全な型変換、設定キャッシュ機能の実装

#### 型安全性向上の詳細


実施内容:

- mypyの段階的実行で問題パターンを分類し、不要な ``# type: ignore`` を削除
- 必要箇所で `typing.cast()` を使用し、安全に型を明示
- 設定ローダー (`ztb/config/loader.py`) にファイルキャッシュと環境変数ベースのフォールバックを実装
- 共通ユーティリティ (`ztb/utils/errors.py`, `ztb/utils/data_utils.py`) に型注釈と安全なラッパーを追加

効果:

- mypyエラーを237→154に削減し、型の信頼性が向上
- テストの安定化（特に設定/ロード周り）とリファクタリングの容易化

```bash
# 全テスト実行（ユニット + 統合テスト）
make test

# ```bash

#### 1. 環境準備 (続)
-
```bash
# Python仮想環境のアクティベート
source .venv/Scripts/activate  # Windows
# または
.venv/bin/activate  # Linux/Mac

# 依存関係の確認
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

- **トレーニングステップ**: 1,000,000ステップ（1Mステップ）
#### 2. データ準備 (続)


```bash
# データファイルの存在確認
ls -la ml-dataset-enhanced.csv

# データ品質チェック（オプション）
python -m ztb.data.validate_data ml-dataset-enhanced.csv
```
包括的なモデル評価システムで、伝統的な取引指標と高度なリスク・パフォーマンス分析を統合：
#### 3. 設定ファイルの確認 (続)


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

- **リスク調整スコア**: リスク管理重視の評価
#### 4. トレーニング実行 (続)


```bash
# 基本実行（1Mステップ）
python -m ztb.training.unified_trainer --config unified_training_config.json

# ドライラン（設定検証のみ）
python -m ztb.training.unified_trainer --config unified_training_config.json --dry-run

# 短時間テスト実行（10kステップ）
python -m ztb.training.unified_trainer --config unified_training_config.json --total-timesteps 10000
```

#### 5. トレーニング監視 (続)


```bash
# リアルタイム監視（別ターミナルで）
python -m ztb.training.watch_1m --correlation-id scalping_15s_ultra_aggressive_1M

# 定期要約（5分ごと）
python -m ztb.training.rollup_artifacts --correlation-id scalping_15s_ultra_aggressive_1M --interval-minutes 5
```

#### 6. 結果確認 (続)


```bash
# チェックポイント確認
ls -la checkpoints/scalping_15s_ultra_aggressive_1M/

# ログ確認
tail -f logs/scalping_15s_ultra_aggressive_1M.log

# TensorBoard（オプション）
tensorboard --logdir logs/
```

1. **データファイル読み込みと検証** ⭐⭐⭐⭐⭐ - ファイル不在で即死
#### メモリ不足エラー (続)


```text
RuntimeError: CUDA out of memory
```

**解決策:**

- バッチサイズを小さくする: `--batch-size 32` または `--batch-size 16`
- ストリーミングを有効化: `--enable-streaming --stream-batch-size 64`
- CPU専用モード: `export CUDA_VISIBLE_DEVICES=""`
3. **引数解析と検証** ⭐⭐⭐⭐ - 設定の基盤
#### データ読み込みエラー (続)


```text
FileNotFoundError: ml-dataset-enhanced.csv
```

**解決策:**

- データファイルの存在を確認: `ls -la ml-dataset-enhanced.csv`
- パスが正しいか確認: プロジェクトルートから実行するか確認
- データ生成: `python generate_enhanced_training_data.py`

#### 特徴量計算エラー (続)


```text
ValueError: Input contains NaN
```

**解決策:**

- データの欠損値チェック: `python -c "import pandas as pd; df = pd.read_csv('ml-dataset-enhanced.csv'); print(df.isnull().sum())"`
- 欠損値補完: スクリプト内でfillnaを使用

#### GPU関連エラー (続)


```text
CUDA error: no kernel image is available
```

**解決策:**

- GPU互換性確認: `nvidia-smi`
- PyTorchバージョン確認: `python -c "import torch; print(torch.version.cuda)"`
- CPUフォールバック: `export CUDA_VISIBLE_DEVICES=""`

#### API接続エラー (続)

```text
ConnectionError: HTTPSConnectionPool
```

**解決策:**

- APIキーの確認: `echo $COINCHECK_API_KEY`
- ネットワーク接続確認: `curl -I https://coincheck.com`
- レート制限チェック: 短時間に複数リクエストしていないか
   export TEST_FLOW_QTY=0.001  # 小量
#### 注文エラー (続)

```text
InvalidOrder: Minimum quantity not met
```

**解決策:**

- 最小注文数量確認: 取引所仕様を確認
- 数量計算の確認: `TEST_FLOW_QTY` でテスト
---
#### トレーニングが遅い (続)

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
- **学習基盤**: PPO トレーナ、1M ステップ前提のチェックポイント運用（async + zstd 圧縮）
#### ログが大きくなる (続)


**解決策:**

- ログローテーション設定
- 定期的なログ整理: `find logs/ -name "*.log" -mtime +7 -delete`
# **安全**: Circuit Breakers、Kill-file、冪等な注文状態機械、ドライラン

#### トレーニング状態確認 (続)

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

|---------------|----------|----------|

#### エラーログ解析 (続)

```bash
# エラーパターン検索
grep -r "ERROR" logs/ | tail -10

# メモリ使用量トレンド
grep "Memory usage" logs/*.log | tail -20
# ```

| cost_estimator.py | ztb/ops/artifacts/cost_estimator.py | artifacts/reports |

#### 設定検証 (続)


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

### Live Trading

### 準備

1. **APIキー設定** (オプション - 設定しない場合はデモモード)

```bash
export COINCHECK_API_KEY="your_api_key"
export COINCHECK_API_SECRET="your_api_secret"
export DISCORD_WEBHOOK="your_webhook_url"  # 通知用（オプション）
```

1. **実行**

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
```

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

```text
RuntimeError: CUDA out of memory
```
**解決策:**

- バッチサイズを小さくする: `--batch-size 32` または `--batch-size 16`
- ストリーミングを有効化: `--enable-streaming --stream-batch-size 64`
- CPU専用モード: `export CUDA_VISIBLE_DEVICES=""`

#### データ読み込みエラー

```text
FileNotFoundError: ml-dataset-enhanced.csv
```
**解決策:**

- データファイルの存在を確認: `ls -la ml-dataset-enhanced.csv`
- パスが正しいか確認: プロジェクトルートから実行するか確認
- データ生成: `python generate_enhanced_training_data.py`

#### 特徴量計算エラー

```text
ValueError: Input contains NaN
```
**解決策:**

- データの欠損値チェック: `python -c "import pandas as pd; df = pd.read_csv('ml-dataset-enhanced.csv'); print(df.isnull().sum())"`
- 欠損値補完: スクリプト内でfillnaを使用

#### GPU関連エラー

```text
CUDA error: no kernel image is available
```
**解決策:**

- GPU互換性確認: `nvidia-smi`
- PyTorchバージョン確認: `python -c "import torch; print(torch.version.cuda)"`
- CPUフォールバック: `export CUDA_VISIBLE_DEVICES=""`

### ライブトレーディングの問題

#### API接続エラー

```text
ConnectionError: HTTPSConnectionPool
```
**解決策:**

- APIキーの確認: `echo $COINCHECK_API_KEY`
- ネットワーク接続確認: `curl -I https://coincheck.com`
- レート制限チェック: 短時間に複数リクエストしていないか

#### 注文エラー

```text
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
