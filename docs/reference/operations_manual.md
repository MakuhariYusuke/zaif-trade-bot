# 運用手順マニュアル

このマニュアルは、Zaif Trade Botのトレーニング実行からライブトレーディング開始までの完全な運用手順を記載しています。

## 目次

1. [環境準備](#環境準備)
2. [トレーニング実行](#トレーニング実行)
3. [モデル評価](#モデル評価)
4. [ライブトレーディング準備](#ライブトレーディング準備)
5. [ライブトレーディング実行](#ライブトレーディング実行)
6. [監視と運用](#監視と運用)
7. [エラー対処法](#エラー対処法)
8. [バックアップと復旧](#バックアップと復旧)

## 環境準備

### 1. システム要件確認

```bash
# Pythonバージョン確認
python --version  # 3.11以上推奨

# メモリ確認
python -c "import psutil; print(f'メモリ: {psutil.virtual_memory().total / 1024**3:.1f}GB')"

# GPU確認（オプション）
nvidia-smi
```

### 2. 依存関係インストール

```bash
# 仮想環境作成
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存関係インストール
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 開発ツールインストール
pre-commit install
npm install
```

### 3. データ準備

```bash
# データファイル存在確認
ls -la ml-dataset-enhanced.csv

# データ品質チェック
python -c "
import pandas as pd
df = pd.read_csv('ml-dataset-enhanced.csv')
print(f'データ行数: {len(df)}')
print(f'特徴量数: {len(df.columns)}')
print(f'欠損値: {df.isnull().sum().sum()}')
"
```

### 4. 設定ファイル確認

```bash
# 設定ファイル構文チェック
python -c "import json; json.load(open('unified_training_config.json')); print('設定OK')"

# 必須ディレクトリ作成
mkdir -p checkpoints logs models reports
```

## トレーニング実行

### 1. 事前検証

```bash
# ドライラン実行
python -m ztb.training.unified_trainer --config unified_training_config.json --dry-run

# 短時間テスト実行
python -m ztb.training.unified_trainer --config unified_training_config.json --total-timesteps 1000
```

### 2. 本トレーニング実行

```bash
# 基本実行（推奨）
CORR_ID=training_$(date +%Y%m%d_%H%M%S)
python -m ztb.training.unified_trainer \
  --config unified_training_config.json \
  --correlation-id $CORR_ID
```

### 3. トレーニング監視

```bash
# リアルタイム監視（別ターミナル）
python -m ztb.training.watch_1m --correlation-id $CORR_ID

# 定期レポート生成
python -m ztb.training.rollup_artifacts --correlation-id $CORR_ID --interval-minutes 5
```

### 4. トレーニング中断と再開

```bash
# 安全停止
touch ztb.stop

# 再開実行
python -m ztb.training.unified_trainer \
  --config unified_training_config.json \
  --correlation-id $CORR_ID \
  --resume-from latest
```

## モデル評価

### 1. 評価実行

```bash
# バックテスト評価
python -m ztb.training.evaluate_model \
  --model-path models/scalping_model.zip \
  --data-path ml-dataset-enhanced.csv \
  --output-dir reports/evaluation
```

### 2. パフォーマンス分析

```bash
# Sharpe Ratio計算
python -c "
import pandas as pd
results = pd.read_csv('reports/evaluation/results.csv')
returns = results['returns']
sharpe = returns.mean() / returns.std() * (252**0.5)  # 年率化
print(f'Sharpe Ratio: {sharpe:.3f}')
"
```

### 3. ウォークフォワード分析

```bash
# 複数期間での評価
python -m ztb.training.walk_forward_analysis \
  --model-path models/scalping_model.zip \
  --data-path ml-dataset-enhanced.csv \
  --windows 5 \
  --output-dir reports/walk_forward
```

## ライブトレーディング準備

### 1. API設定

```bash
# 環境変数設定
export COINCHECK_API_KEY="your_api_key"
export COINCHECK_API_SECRET="your_api_secret"
export DISCORD_WEBHOOK="your_webhook_url"  # 通知用
```

### 2. リスク管理設定

```bash
# リスクパラメータ確認
python -c "
import json
config = json.load(open('scalping-config.json'))
print('リスク設定:')
print(f'  最大ポジションサイズ: {config[\"environment\"][\"max_position_size\"]}')
print(f'  取引コスト: {config[\"environment\"][\"transaction_cost\"]}')
print(f'  報酬クリップ: {config[\"environment\"][\"reward_clip_value\"]}')
"
```

### 3. ヘルスチェック

```bash
# 取引所接続テスト
python -m ztb.ops.check_venue_health --venue coincheck --symbol BTC_JPY

# API制限確認
python -c "
import time
import requests
start = time.time()
response = requests.get('https://coincheck.com/api/ticker')
end = time.time()
print(f'API応答時間: {(end-start)*1000:.1f}ms')
"
```

## ライブトレーディング実行

### 1. 最小検証実行

```bash
# DRY RUNモード
export DRY_RUN=1
python -m ztb.live.service_runner --config scalping-config.json

# 最小ライブモード（注文後即キャンセル）
export LIVE_MINIMAL=1
export TEST_FLOW_QTY=0.001
python -m ztb.live.service_runner --config scalping-config.json
```

### 2. 本番実行

```bash
# 本番モード開始
unset DRY_RUN LIVE_MINIMAL TEST_FLOW_QTY
python -m ztb.live.service_runner --config scalping-config.json
```

### 3. 段階的スケールアップ

```bash
# Phase 1: 小量テスト
export MAX_POSITION_SIZE=0.01  # 1%のみ
python -m ztb.live.service_runner --config scalping-config.json

# Phase 2: 中量運用
export MAX_POSITION_SIZE=0.05  # 5%
python -m ztb.live.service_runner --config scalping-config.json

# Phase 3: フル運用
unset MAX_POSITION_SIZE
python -m ztb.live.service_runner --config scalping-config.json
```

## 監視と運用

### 1. リアルタイム監視

```bash
# システムリソース監視
watch -n 30 'ps aux | grep unified_trainer'

# ログ監視
tail -f logs/live_trading_*.log

# パフォーマンス監視
python -m ztb.ops.health_monitor
```

### 2. アラート設定

```bash
# Discord通知設定確認
curl -X POST $DISCORD_WEBHOOK \
  -H "Content-Type: application/json" \
  -d '{"content": "Bot起動通知"}'
```

### 3. 定期レポート

```bash
# 日次レポート生成
python -m ztb.ops.generate_daily_report

# パフォーマンス分析
python -m ztb.ops.analyze_performance --days 7
```

### 4. ポジション管理

```bash
# 現在ポジション確認
python -c "
import json
try:
    with open('trade-state.json', 'r') as f:
        state = json.load(f)
        print(f'現在ポジション: {state.get(\"position_size\", 0)}')
        print(f'未実現損益: {state.get(\"unrealized_pnl\", 0)}')
except FileNotFoundError:
    print('ポジションなし')
"
```

## エラー対処法

### トレーニングエラー

#### メモリ不足

```
RuntimeError: CUDA out of memory
```
**対処:**
```bash
# GPUメモリ削減
export PYTORCH_CUDA_ALLOC_CONF=1
export CUDA_VISIBLE_DEVICES=""

# バッチサイズ削減
sed -i 's/"batch_size": 64/"batch_size": 32/' scalping-config.json

# ストリーミング有効化
python -m ztb.training.unified_trainer --enable-streaming --stream-batch-size 32
```

#### データ読み込みエラー

```
FileNotFoundError: ml-dataset-enhanced.csv
```
**対処:**
```bash
# データファイル確認
ls -la ml-dataset-enhanced.csv

# 再生成
python generate_enhanced_training_data.py
```

### ライブトレーディングエラー

#### API接続エラー

```
ConnectionError: HTTPSConnectionPool
```
**対処:**
```bash
# ネットワーク確認
ping coincheck.com

# APIキー確認
echo $COINCHECK_API_KEY | wc -c  # 長さ確認

# レート制限チェック
python -m ztb.ops.check_rate_limits
```

#### 注文エラー

```
InvalidOrder: Minimum quantity not met
```
**対処:**
```bash
# 最小注文数量確認
python -c "
import requests
response = requests.get('https://coincheck.com/api/exchange/orders/rate?pair=btc_jpy')
data = response.json()
print(f'最小数量: {data[\"min_amount\"]}')
"

# 数量調整
export MIN_ORDER_SIZE=0.001
```

### システムエラー

#### プロセス停止

**対処:**
```bash
# プロセス再起動
python -m ztb.live.service_runner --config scalping-config.json --restart

# ログ分析
grep "ERROR\|CRITICAL" logs/*.log | tail -10
```

#### ディスク容量不足

**対処:**
```bash
# 容量確認
df -h

# 古いログ削除
find logs/ -name "*.log" -mtime +30 -delete

# チェックポイント整理
python -m ztb.ops.cleanup_checkpoints --keep-last 5
```

## バックアップと復旧

### 1. 定期バックアップ

```bash
# 自動バックアップスクリプト
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$DATE"

mkdir -p $BACKUP_DIR

# モデルと設定のバックアップ
cp -r models/ $BACKUP_DIR/
cp *.json $BACKUP_DIR/
cp -r checkpoints/ $BACKUP_DIR/

# 圧縮
tar -czf ${BACKUP_DIR}.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR

echo "バックアップ完了: ${BACKUP_DIR}.tar.gz"
EOF

chmod +x backup.sh
```

### 2. 復旧手順

```bash
# バックアップからの復旧
tar -xzf backups/20241201_120000.tar.gz
cp -r backups/20241201_120000/models/* models/
cp backups/20241201_120000/*.json .

# トレーニング再開
python -m ztb.training.unified_trainer --resume-from latest
```

### 3. 緊急停止手順

```bash
# 即時停止
touch ztb.stop

# ポジション確認
python -c "
# 現在のポジションをログ
"

# 手動決済（必要な場合）
python -m ztb.ops.manual_close_positions
```

## メンテナンス

### 1. 定期メンテナンス

```bash
# 週次メンテナンススクリプト
cat > weekly_maintenance.sh << 'EOF'
#!/bin/bash
echo "=== 週次メンテナンス開始 ==="

# ログ整理
find logs/ -name "*.log" -mtime +7 -delete

# チェックポイント整理
python -m ztb.ops.cleanup_checkpoints --keep-last 10

# データ更新
python generate_enhanced_training_data.py

# システムチェック
python -m ztb.ops.smoke_tests

echo "=== メンテナンス完了 ==="
EOF
```

### 2. パフォーマンス監視

```bash
# パフォーマンスレポート生成
python -m ztb.ops.performance_report --period 30d

# 異常検知
python -m ztb.ops.anomaly_detection --metrics pnl,sharpe,drawdown
```

このマニュアルは運用中に更新してください。新しい問題や改善点が見つかったら、適宜更新を。
