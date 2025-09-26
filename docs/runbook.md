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

3. **Paper Trading 開始**
   ```bash
   # 設定確認
   export EXCHANGE=paper
   export DRY_RUN=1
   export TEST_MODE=1

   # シナリオ実行
   npm run mock:scenario
   ```

4. **Live Trading 移行**
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

2. **強制停止（緊急時）**
   ```bash
   # SIGKILL送信
   pkill -KILL -f "trade-live"
   ```

3. **クリーンアップ**
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

**最終更新**: 2025-01-27
**バージョン**: 2.3.0