# Zaif Trade Bot (exchange-agnostic)

長時間学習・実運用を見据えた、**強化学習 + バックテスト + ペーパートレード**一体型の取引基盤。
リポジトリ名に"Zaif"を含みますが、**取引所非依存（Zaif/Coincheck 等を切替可能）**な設計です。

> ⚠️ 免責: 本ソフトウェアは研究・検証目的です。実取引は自己責任で行ってください。大きな損失が発生する可能性があります。
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
# 全テスト実行
make test

# ユニットテストのみ
make unit

# 統合テストのみ
make integration

# コード品質チェック
make check

# セキュリティ監査
make audit
```

### よく使うユーティリティ

- 監視: python -m ztb.training.watch_1m --correlation-id `<ID>` --run-once
- 要約: python -m ztb.training.rollup_artifacts --correlation-id `<ID>`
- Canary: linux_canary.sh / ps_canary.ps1（同等のフロー・出力）

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

## Development & Contribution Guide

- セットアップ/テスト: [docs/contributing/setup.md](./docs/contributing/setup.md), [docs/contributing/testing.md](./docs/contributing/testing.md)
- アーキ概要: [docs/contributing/architecture.md](./docs/contributing/architecture.md)
- ルール: PULL_REQUEST_TEMPLATE.md, CODEOWNERS, LICENSE, DISCLAIMER.md

---

## License

本リポジトリは MIT ライセンスです。詳細は [LICENSE](./LICENSE) を参照してください。
