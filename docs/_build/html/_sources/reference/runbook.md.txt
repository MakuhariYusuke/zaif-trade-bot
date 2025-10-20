# Zaif Trade Bot Runbook

## 概要

このドキュメントは、Zaif Trade Botの運用、監視、トラブルシューティングの手順をまとめたものです。

## 🚀 実験開始・停止手順

### 実験開始

1. **環境準備**
   ```bash
   # 依存関係のインストール
   npm install
   pip install -r requirements.txt -r requirements-dev.txt

   # 設定ファイルの準備
   cp .env.example .env
   # .envファイルを編集してAPIキーなどを設定
   ```

2. **事前チェック**
   ```bash
   # コード品質チェック
   npm run pre-commit

   # ユニットテスト実行
   npm run test:unit

   # 型チェック
   npm run type-check
   npm run type-check:py
   ```

3. **1M Training 開始 (Makeターゲット使用)**
   ```bash
   # Correlation ID生成
   export CORR=$(date -u +%Y%m%dT%H%M%SZ)

   # トレーニング開始
   make 1m-start CORR=$CORR

   # 別ターミナルで監視
   make 1m-watch CORR=$CORR

   # トレーニング完了後、アーティファクト生成
   make 1m-rollup CORR=$CORR

   # 停止が必要な場合
   make 1m-stop
   ```

4. **Paper Trading 開始**
   ```bash
   # 設定確認
   export EXCHANGE=paper
   export DRY_RUN=1
   export TEST_MODE=1

   # シナリオ実行
   npm run mock:scenario
   ```

### 再現性ポリシー

トレーニングの再現性を確保するため、以下のポリシーを遵守してください：

1. **Seed管理**: すべての実験で固定seedを使用
   ```bash
   # 同一seedで再現可能な実行
   python -m ztb.training.run_1m --correlation-id test_001 --seed 42
   ```

2. **環境一貫性**: 同一のPythonバージョン、ライブラリバージョンを使用
   ```bash
   # 環境確認
   python --version
   pip freeze | grep -E "(torch|numpy|stable-baselines3)"
   ```

3. **決定論的動作**: PyTorchの決定論的アルゴリズムを有効化
   - `torch.backends.cudnn.deterministic = True`
   - `torch.backends.cudnn.benchmark = False`

4. **Seed由来の派生**: 同一ベースseedから派生seedを生成
   ```python
   from ztb.utils.seed_manager import get_seed_manager
   manager = get_seed_manager()
   manager.set_seed(42)
   env_seed = manager.fork_seed("environment")
   eval_seed = manager.fork_seed("evaluation")
   ```

5. **結果検証**: 同一seedでの複数実行でreward系列が一致することを確認

6. **Live Trading 移行**
   ```bash
   # 設定変更
   export EXCHANGE=zaif
   export DRY_RUN=0
   export TEST_MODE=0

   # 本番開始（慎重に！）
   npm run trade:live
   ```

### 実験停止

1. **Graceful Shutdown**
   ```bash
   # SIGTERM送信
   pkill -TERM -f "trade-live"

   # または直接停止
   npm run trade:live:dry  # DRY_RUNモードで安全停止
   ```

2. **Kill Switch による緊急停止**
   ```bash
   # Kill switch ファイル作成（デフォルト: /tmp/ztb.stop）
   touch /tmp/ztb.stop

   # または環境変数で強制停止
   export ZTB_KILL=1

   # またはHTTPエンドポイント（実装されている場合）
   curl -X POST http://localhost:8080/kill
   ```

3. **強制停止（緊急時）**
   ```bash
   # SIGKILL送信
   pkill -KILL -f "trade-live"
   ```

4. **インシデント対応手順**
   - Kill switch が作動したら、まず原因をログで確認
   - 設定されたトリガー（損失率、レイテンシ、エラーレート）をチェック
   - 問題解決後、kill switch をリセット: `rm /tmp/ztb.stop`
   - 再開前にリスクプロファイルを conservative に変更

5. **クリーンアップ**
   ```bash
   # テンポラリファイル削除
   npm run clean:tmp

   # ログ保存
   cp logs/ logs_backup_$(date +%Y%m%d_%H%M%S) -r
   ```

## 📊 監視方法

### Prometheus Metrics

主要なメトリクス：

- `ztb_data_drift_score`: データドリフト検出スコア
- `ztb_model_performance_drift`: モデル性能ドリフト
- `ztb_quality_gates_passed_total`: 品質ゲート合格数
- `ztb_active_jobs`: 実行中のジョブ数
- `ztb_portfolio_balance`: ポートフォリオ残高
- `ztb_portfolio_pnl`: ポートフォリオP&L

### Grafana Dashboard

1. **アクセス**: `http://localhost:3000`
2. **主要パネル**:
   - Portfolio Performance
   - Data Quality Metrics
   - System Health
   - Drift Detection

### Discord通知

以下のイベントで通知：

- ✅ **成功通知**: 取引実行、チェックポイント保存
- ⚠️ **警告**: ドリフト検出、スリッページ超過
- 🚫 **エラー**: API障害、品質ゲート失敗
- 🔴 **緊急**: サーキットブレーカー発動

## 🔧 トラブルシューティング

### API障害時

**症状**: API接続エラー、レートリミット超過

**対応手順**:
1. **ログ確認**
   ```bash
   tail -f logs/trade-bot.log | grep -i error
   ```

2. **Watchdog状態確認**
   ```bash
   # Bridge接続状態確認
   curl http://localhost:8000/metrics | grep ztb_bridge
   ```

3. **APIエンドポイントテスト**
   ```bash
   # Zaif API疎通確認
   npm run test:zaif
   ```

4. **自動復旧待機**
   - Watchdogが自動的に再接続を試行（最大5回）
   - 10分ごとに再試行

5. **手動介入（復旧しない場合）**
   ```bash
   # プロセス再起動
   npm run trade:live:dry  # 安全モードで再起動
   ```

### メモリ不足

**症状**: OOM Killer発動、メモリ使用率95%以上

**対応手順**:
1. **メモリ使用状況確認**
   ```bash
   # プロセスメモリ確認
   ps aux | grep trade-live | head -1

   # システムメモリ確認
   free -h
   ```

2. **設定調整**
   ```bash
   # バッチサイズ削減
   export ML_BATCH_SIZE=32  # デフォルト64から削減

   # ワーカー数削減
   export ML_MAX_WORKERS=2  # デフォルト4から削減
   ```

3. **プロセス再起動**
   ```bash
   npm run trade:live:dry
   ```

### NaN発生時

**症状**: 特徴量計算でNaN、無限大値

**対応手順**:
1. **NaN検出**
   ```bash
   # 特徴量ログ確認
   tail -f logs/features.log | grep -i nan
   ```

2. **データ品質チェック**
   ```bash
   # 欠損値確認
   python -c "
   import pandas as pd
   df = pd.read_csv('data/price_data.csv')
   print('NaN counts:', df.isna().sum().sum())
   print('Inf counts:', (df == float('inf')).sum().sum())
   "
   ```

3. **特徴量再計算**
   ```bash
   # キャッシュクリア
   rm -rf __pycache__/ ztb/features/__pycache__/

   # 強制再計算
   export FEATURE_CACHE_ENABLED=0
   npm run trade:live:dry
   ```

### ドリフト検出

**症状**: データドリフト警告、モデル性能低下

**対応手順**:
1. **ドリフト分析**
   ```bash
   # ドリフトレポート生成
   python python/drift_analysis.py
   ```

2. **ベースライン更新**
   ```bash
   # 新しい基準値で更新
   python -c "
   from ztb.utils.drift import DriftMonitor
   monitor = DriftMonitor()
   # 現在のデータを基準値に設定
   monitor.update_baseline(current_features_df, current_pnl)
   "
   ```

3. **モデル再訓練**
   ```bash
   # MLパイプライン実行
   npm run ml:search
   npm run ml:export
   ```

## 🔄 チェックポイント復旧手順

### 自動復旧

システムは自動的に以下のチェックポイントを保存：

- `checkpoints/model_checkpoint.pkl`: モデル状態
- `trade-state.json`: 取引状態
- `positions.json`: ポジション状態

### 手動復旧

1. **プロセス状態確認**
   ```bash
   # 実行中プロセス確認
   ps aux | grep trade-live
   ```

2. **チェックポイント確認**
   ```bash
   # 最新チェックポイント確認
   ls -la checkpoints/ | tail -5
   ```

3. **状態復元**
   ```bash
   # 設定ファイル確認
   cat trade-state.json | jq .phase

   # ポジション確認
   cat positions.json | jq '.positions | length'
   ```

4. **部分復旧**
   ```bash
   # 特定の状態のみ復旧
   export RESTORE_FROM_CHECKPOINT=1
   export CHECKPOINT_PATH=checkpoints/model_checkpoint.pkl
   npm run trade:live:dry
   ```

## 🛑 緊急停止（サーキットブレーカー）

### 自動停止条件

- **日次損失**: 2%超過
- **最大ドローダウン**: 5%超過
- **連続損失**: 5回連続
- **価格変動**: ±10%（1分以内）

### 手動緊急停止

1. **即時停止**
   ```bash
   # 全プロセス強制終了
   pkill -9 -f "trade-live"
   pkill -9 -f "node.*trade"
   ```

2. **ポジション決済**
   ```bash
   # 手動決済（必要な場合）
   npm run balance  # 残高確認
   # 必要に応じて手動決済
   ```

3. **システム隔離**
   ```bash
   # APIキー無効化
   unset ZAIF_API_KEY
   unset ZAIF_API_SECRET

   # 設定ファイルバックアップ
   cp .env .env.emergency_backup
   ```

### 復旧前の確認事項

- [ ] API接続正常性
- [ ] 残高確認
- [ ] ポジション状態確認
- [ ] ログ分析（エラー原因特定）
- [ ] 設定ファイル検証

## 📈 パフォーマンス最適化

### メモリ使用量削減

```bash
# 設定調整
export ML_BATCH_SIZE=16
export FEATURE_CACHE_SIZE=1000
export MAX_POSITIONS=3
```

### CPU使用量最適化

```bash
# 並列処理調整
export ML_MAX_WORKERS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

### ディスクI/O削減

```bash
# ログレベル調整
export LOG_LEVEL=WARNING  # DEBUGからWARNINGに

# キャッシュ有効化
export FEATURE_CACHE_ENABLED=1
export PRICE_CACHE_ENABLED=1
```

## 🔍 ログ分析

### 主要ログファイル

- `logs/trade-bot.log`: メイン取引ログ
- `logs/features.log`: 特徴量計算ログ
- `logs/errors.log`: エラーログ
- `logs/metrics.log`: メトリクスログ

### ログ分析コマンド

```bash
# エラー集計
grep -c "ERROR" logs/*.log

# 取引成功率
grep "order executed" logs/trade-bot.log | wc -l

# パフォーマンス統計
grep "PnL:" logs/trade-bot.log | tail -10
```

## 📞 サポート

### 緊急連絡先

- **Discord**: 自動通知チャンネル
- **ログ**: 詳細は `logs/` ディレクトリ
- **メトリクス**: `http://localhost:8000/metrics`

### エスカレーション

1. **Level 1**: 自動復旧（Watchdog）
2. **Level 2**: 手動再起動（オペレーター）
3. **Level 3**: コード修正（開発者）

---

## ✅ Go/No-Go Checklist

### 事前チェックリスト

本番移行前に以下の項目をすべて確認してください：

#### 🔧 システム準備

- [ ] **CI/CD パイプライン**: すべてのテストが通過 (`npm run test:unit && npm run test:int-fast`)
- [ ] **依存関係**: すべての依存パッケージがインストール済み
- [ ] **設定ファイル**: `.env` ファイルが正しく設定（APIキー、シークレット）
- [ ] **データ品質**: 価格データが最新で欠損なし
- [ ] **ネットワーク**: APIエンドポイントへの接続確認

#### 📊 パフォーマンス基準

- [ ] **メモリ使用量**: < 500MB（ピーク時）
- [ ] **CPU使用率**: < 80%（平均）
- [ ] **レスポンスタイム**: API呼び出し < 2秒
- [ ] **エラー率**: < 1%（直近1時間）
- [ ] **スリッページ**: 平均 < 0.5%

#### 💰 リスク管理

- [ ] **テスト取引**: Paper trading で最低10回成功
- [ ] **損失制限**: Kill switch が正しく設定（日次損失2%）
- [ ] **ポジション制限**: 最大ポジション数 = 3
- [ ] **ドローダウン**: 最大許容5%
- [ ] **バックテスト**: 過去データで安定したパフォーマンス

#### 🔍 品質ゲート

- [ ] **コードカバレッジ**: > 80%
- [ ] **型チェック**: mypy エラーなし
- [ ] **セキュリティ**: detect-secrets クリア
- [ ] **ドリフト検出**: データ/モデルドリフトなし
- [ ] **ログ**: 構造化ログが正しく出力

### 自動評価スクリプト

Go/No-Go 判定を自動化するために以下のスクリプトを使用：

```bash
# 自動チェック実行
python scripts/go_nogo_check.py

# 詳細レポート生成
python scripts/go_nogo_check.py --report
```

#### 判定基準

- **GO**: すべてのチェックが✅
- **NO-GO**: 1つ以上のチェックが❌
- **CONDITIONAL**: 警告あり、判断が必要

### 移行手順

1. **GO判定後**:

   ```bash
   # 設定変更
   export EXCHANGE=zaif
   export DRY_RUN=0
   export TEST_MODE=0

   # 本番開始
   npm run trade:live
   ```

2. **監視開始**:
   - Grafanaダッシュボード監視
   - Discord通知確認
   - ログ監視継続

3. **ロールバック準備**:
   - Kill switch を即時使用可能に
   - バックアップ設定確認
   - 緊急連絡先確認

---

## 📊 データ保持ポリシー

### ログ保持期間

- **取引ログ**: 90日間保持（コンプライアンス要件）
- **システムログ**: 30日間保持
- **デバッグログ**: 7日間保持
- **監査ログ**: 1年間保持

### バックアップ保持

- **デイリーバックアップ**: 30日間保持
- **ウィークリーバックアップ**: 12週間保持
- **マンスリーバックアップ**: 12ヶ月保持
- **年次バックアップ**: 7年間保持

### データ削除手順

```bash
# 古いログの削除
find logs/ -name "*.log" -mtime +30 -delete

# 古いバックアップの削除
find backups/ -name "daily_*" -mtime +30 -delete
find backups/ -name "weekly_*" -mtime +90 -delete
```

### アーカイブストレージ

- 長期保存が必要なデータは圧縮してアーカイブ
- アーカイブは暗号化して保存
- アクセスログを保持して監査対応

### コンプライアンス

- 金融取引関連データは法定保持期間に従う
- 個人情報はGDPR等に従い適切に処理
- データ削除時には完全消去を実施

---

**最終更新**: 2025-01-27
**バージョン**: 2.3.0

## 📢 通知・清掃・インデックス化

### アラート通知
- `scripts/alert_notifier.py` で監視ログからWARN/FAILをWebhook送信
- 環境変数 `ZTB_ALERT_WEBHOOK` でSlack/Discord等連携

### TensorBoard スクレイピング
- `scripts/tb_scrape.py` でスカラー値をCSV出力・metrics.json統合
- TensorBoard未インストール時はスキップ

### アーティファクト清掃
- `scripts/artifacts_janitor.py` で古い実行削除・ログローテーション
- `--dry-run` で安全確認

### セッションインデックス
- `scripts/index_sessions.py` で全セッションのインデックス生成
- `artifacts/index.json` にステータス・最新ステップ等記録

### 進捗推定
- `scripts/progress_eta.py` でmetrics.json/logsからsteps/sec推定・ETA算出
- `summary.json` に進捗情報追記（存在時のみ）

### エラー収集
- `scripts/collect_last_errors.py` でlogs/*.logとwatch_log.jsonlから最新ERROR/FAIL抽出
- `artifacts/<ID>/reports/last_errors.txt` 生成

### アーティファクトバンドル
- `scripts/bundle_artifacts.py` でartifacts/<ID>/をZIP化・SHA256ハッシュ生成
- `--exclude-logs` でログ除外オプション

### 監視プロセス一括起動
- `scripts/launch_monitoring.py` でwatch/tb_scrape/alert_notifierをサブプロセス管理
- `--dry-run` でコマンド表示、`--execute` で実行、Ctrl+Cで全子プロセス終了

### ディスク容量・I/O 健康監視
- `scripts/disk_health.py` で空き容量・inode使用率・I/Oレイテンシをチェック
- 閾値超過でJSONアラートをstdoutとops_alerts.jsonlに出力
- `--check-io` でI/Oテスト実行

### 反復実行ランチャ
- `scripts/cronish.py` でジッター付き定期コマンド実行
- ztb.stopファイルで停止、`--fail-fast` で初回失敗時exit

### アーティファクト整合性チェッカー
- `scripts/validate_artifacts.py` でセッション成果物の検証
- schema/artifacts_expectations.json で期待値定義、`--strict` で厳格チェック

### 運用スクリプト統合ラッパー
- `scripts/ops.py` で既存スクリプトを統一インターフェースで実行
- サブコマンド: eta/bundle/errors/health/cronish/validate/launch
- 各スクリプトの引数を透過的に渡す

### ステータススナップショット集約
- `scripts/status_snapshot.py` で複数ソースからステータス情報を集約・Markdownレポート生成
- index.json/gates.json/tb_summary.json/disk_health.json/last_errors.json を統合
- `--correlation-id` で特定セッション指定、`--output` でレポートファイル指定

### 設定ドリフト検出
- `scripts/config_diff.py` で2つのeffective-config JSONを比較
- キー欠落/値差分をhuman-readable形式で表示
- 重大キー差分（model/policy/learning_rate等）でexit code 3、それ以外差分で2

### JSONLログ区切り圧縮
- `scripts/compact_jsonl.py` でwatch_log.jsonl等をtimestampの日付ごとにYYYY-MM-DD.jsonl.gzに分割
- `--apply` で実行、元ファイル.backup退避、`--dry-run`（デフォルト）で対象一覧表示

## ヘルスチェック

### 会場APIヘルスチェック

- `scripts/check_venue_health.py` でCoincheck公開APIの接続性・レイテンシ・レート制限余裕を計測
- オフライン環境ではgraceful degrade（exit 0 / SKIP表示）、接続時はJSONレポート出力

- 使用例:

  ```bash
  # BTC/JPYのヘルスチェック（5秒タイムアウト）
  python scripts/check_venue_health.py --venue coincheck --symbol BTC_JPY --timeout 5
  ```

- 出力例:

  ```json
  {
    "venue": "coincheck",
    "symbol": "BTC_JPY",
    "timestamp": 1640995200.123,
    "connectivity": {
      "internet": true,
      "rest_api": true,
      "websocket": true
    },
    "latency": {
      "rest_ms": 145.67,
      "ws_connect_ms": 234.56
    },
    "rate_limits": {
      "remaining": 299,
      "reset_time": 1640995200
    },
    "status": "healthy",
    "errors": []
  }
  ```

- ステータス: `healthy` (全API正常), `degraded` (一部異常), `unhealthy` (全API異常), `offline` (インターネット未接続)
- exit code: 0 (healthy/offline), 1 (degraded), 2 (unhealthy)

## 保持ポリシー

### アーティファクトクリーンアップ

- `scripts/retention_policy.py` で保持日数・世代数・サイズ上限に基づく削除候補を提案/適用
- デフォルトは提案のみ（--dry-run）、--apply で実削除実行
- パラメータ: --keep-days 14（日数）、--keep-best 3（best候補数）、--max-size-gb 50（総サイズ上限GB）
- best.marker付きセッションは保護、サイズ超過時は古いものから削除

## 薄いラッパースクリプト

### トレーニング開始ラッパー
- `scripts/training_start.py` でトレーニングセッションを開始
- `--correlation-id` でID指定（未指定時は自動生成）、`--dry-run` でコマンド表示のみ

### 監視開始ラッパー
- `scripts/monitoring_start.py` で監視プロセスを開始
- `--correlation-id` でセッション指定、`--dry-run` でコマンド表示のみ

### ステータスチェックラッパー
- `scripts/status_check.py` でシステムステータスをチェック
- `--correlation-id` でセッション指定、`--output` でレポートファイル指定

### アーティファクトクリーンアップラッパー
- `scripts/cleanup_artifacts.py` で古いアーティファクトをクリーンアップ
- `--days` で保持日数指定（デフォルト30日）、`--dry-run` でプレビュー

## 型安全と例外方針

### 型安全強化

Zaif Trade Botでは、Pythonの型システムを活用してコードの信頼性を向上させています：

- **mypy strictモード**: `ztb/**` 配下の全モジュールでstrictモードを有効化
- **型アノテーション**: すべての関数とメソッドに適切な型アノテーションを付与
- **TypedDict/Protocol**: 複雑なデータ構造にはTypedDict、インターフェースにはProtocolを使用
- **Decimal使用**: 金額・サイズ計算ではDecimalクラスを使用し、浮動小数点誤差を回避

```python
# 例: 型安全な関数定義
from typing import Optional, TypedDict
from decimal import Decimal

class TradeConfig(TypedDict):
    symbol: str
    amount: Decimal
    price: Optional[Decimal]

def calculate_order_size(config: TradeConfig) -> Decimal:
    """注文サイズを計算（型安全）"""
    base_amount = config['amount']
    # Decimal演算で精度を保証
    return base_amount * Decimal('0.01')
```

### 例外処理方針

bare `except:` 句の使用を禁止し、構造化された例外処理を実装：

- **禁止**: `except:` （何でもキャッチ）
- **推奨**: `except Exception as e:` （具体的な例外クラス）
- **統一例外クラス**: `ztb/utils/errors.py` のクラスを使用
- **構造化ログ**: エラー発生時に `err_type`, `err_msg`, `correlation_id`, `component` をログ出力

```python
# 例: 適切な例外処理
from ztb.utils.errors import NetworkError, DatabaseError
import logging

def fetch_market_data(symbol: str) -> dict:
    try:
        # API呼び出し
        response = api_call(symbol)
        return response.json()
    except requests.RequestException as e:
        # ドメイン固有のエラーに変換
        raise NetworkError(
            f"Failed to fetch {symbol}",
            details={'symbol': symbol, 'original_error': str(e)}
        ) from e
    except Exception as e:
        # 予期せぬエラーもログ出力
        logging.error(
            "Unexpected error in fetch_market_data",
            extra={
                'err_type': type(e).__name__,
                'err_msg': str(e),
                'correlation_id': get_correlation_id(),
                'component': 'market_data'
            }
        )
        raise
```

## 常駐ランナーの名称統一

### サービスランナーの統一

24/7取引サービスのランナーを `service_runner.py` に統一：

- **正規モジュール**: `ztb/live/service_runner.py`
- **後方互換**: `ztb/scripts/trading_service.py` （非推奨警告付き）
- **設定ファイル**: systemd/Windowsサービス設定を更新

```bash
# 新しい方法（推奨）
python -m ztb.live.service_runner --config config/production.yaml

# 古い方法（非推奨、警告表示）
python ztb/scripts/trading_service.py --config config/production.yaml
```

### サービス設定

**Linux (systemd)**:
```ini
[Service]
ExecStart=/opt/zaif-trade-bot/venv/bin/python -m ztb.live.service_runner --config /opt/zaif-trade-bot/config/production.yaml
```

**Windows (バッチファイル)**:
```bat
python -m ztb.live.service_runner --config config\production.yaml
```

### 移行時の注意

- 既存の `trading_service.py` 参照は自動的に `service_runner.py` にリダイレクト
- 非推奨警告が表示されるため、徐々に新しい名前への移行を推奨
- Makeターゲット（`make 1m-start/1m-watch/...`）は自動的に正規モジュールを使用

## 🔒 注文送信の安全性 (PR-Core-26)

### 概要

注文送信時に発生する可能性のあるエッジケース（精度誤差、競合送信、重複注文）を防ぐための包括的な安全機構を実装。

### 安全機能

- **シンボル正規化**: 取引所固有のシンボル表記（BTC/JPY, BTC_JPY 等）を統一
- **精度量子化**: 価格/数量を取引所の最小ティック/ステップサイズに丸め
- **最小注文額検証**: 取引所固有の最小注文額を満たすかチェック
- **冪等性トークン**: 同一注文の重複送信を防ぐためのクライアントID管理
- **自動ID生成**: クライアントID未指定時はUUIDv4を自動生成

### 使用方法

```python
from ztb.live.order_submission import OrderPreparer

preparer = OrderPreparer()
order = preparer.prepare_order(
    venue='coincheck',
    symbol='BTC/JPY',
    side='BUY',
    quantity=Decimal('0.1'),
    price=Decimal('5000000'),
    client_order_id='optional-custom-id'  # 省略時は自動生成
)

# order は PreparedOrder オブジェクト
# - venue: 正規化された取引所名
# - normalized_symbol: 正規化されたシンボル (BTC_JPY)
# - side: BUY/SELL
# - quantity: 量子化された数量
# - price: 量子化された価格 (Noneで成行)
# - client_order_id: 冪等性トークン
```

### テスト

```bash
# ユニットテスト実行
python -m pytest ztb/tests/unit/live/test_order_submission.py -v

# 並行性テストを含む
python -m pytest ztb/tests/unit/live/test_order_submission.py::TestOrderPreparerConcurrency -v
```

### 設定

精度ポリシーは `ztb/live/precision_policy.py` で取引所ごとに定義。デフォルト値:
- 価格ティック: 0.01 (JPYペア)
- 数量ステップ: 0.0001 (BTC)
- 最小数量: 未設定 (取引所依存)

### トラブルシューティング

- **ValidationError**: 無効なvenue/symbol/数量/価格
- **IdempotencyError**: 同一client_order_idの重複注文
- **精度警告**: 量子化により価格/数量が変更された場合のログ出力


