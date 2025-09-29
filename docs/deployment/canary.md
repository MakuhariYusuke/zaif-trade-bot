# Canary Deployment Guide

## 概要

Canary deployment は、本番環境への安全な移行を目的とした段階的デプロイメント手法です。Zaif Trade Bot では、以下の段階で移行を行います：

1. **Replay Mode**: 過去データを用いたシミュレーション
2. **Live-Lite Mode**: 実際の市場データを使用したテスト（取引なし）
3. **Live Mode**: 本番取引開始

## 自動 Canary スクリプト

`scripts/run_canary.sh` を使用して、自動化された canary deployment を実行します。

### 使用方法

```bash
# 基本実行
./scripts/run_canary.sh

# カスタム設定
./scripts/run_canary.sh --duration 30 --policy custom_policy
```bash

### スクリプトの動作

1. **Phase 1: Replay Mode (10分)**
   - 過去データを用いたバックテスト実行
   - パフォーマンス指標のベースライン収集
   - Kill switch の動作確認

2. **Phase 2: Live-Lite Mode (20分)**
   - リアルタイム市場データを使用
   - 取引シグナル生成のみ（注文なし）
   - システム安定性の検証

3. **Phase 3: Kill/Resume Exercise (5分)**
   - Kill switch の自動トリガー
   - システム停止と再開の検証
   - 状態復旧の確認

4. **Phase 4: Artifact Collection**
   - ログ、アーティファクトの収集
   - パフォーマンスレポート生成
   - ZIP アーカイブ作成

## 設定パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--duration` | 30 | 各フェーズの実行時間（分） |
| `--policy` | sma_fast_slow | 使用する取引ポリシー |
| `--kill-threshold` | 0.01 | Kill switch トリガー閾値（1%） |
| `--connectivity-skip` | false | 接続性テストをスキップ（CI環境用） |
| `--output-dir` | artifacts/canary_$(date +%Y%m%d_%H%M%S) | 出力ディレクトリ |

## 成功基準

### Phase 1: Replay Mode
- [ ] バックテスト正常完了
- [ ] エラー率 < 1%
- [ ] メモリ使用量 < 500MB
- [ ] CPU 使用率 < 80%

### Phase 2: Live-Lite Mode
- [ ] 市場データ受信正常
- [ ] シグナル生成正常
- [ ] API レートリミット遵守
- [ ] ログ構造化出力

### Phase 3: Kill/Resume
- [ ] Kill switch 即時反応
- [ ] プロセス正常停止
- [ ] 状態復旧成功
- [ ] 再開後安定動作

## 監視ポイント

### メトリクス監視
- **システムメトリクス**: CPU/メモリ/ディスク使用率
- **アプリケーションメトリクス**: レスポンスタイム、エラー率
- **ビジネスメトリクス**: シグナル品質、スリッページ

### ログ監視
```bash
# リアルタイムログ監視
tail -f artifacts/canary_*/logs/trade-bot.log

# エラー集計
grep "ERROR" artifacts/canary_*/logs/*.log | wc -l
```bash

## トラブルシューティング

### Replay Mode 失敗
**症状**: バックテストが異常終了
**対応**:
1. ログ確認: `artifacts/canary_*/logs/errors.log`
2. データ品質チェック: `python -c "import pandas as pd; pd.read_csv('data/price_data.csv').info()"`
3. 設定確認: `.env` ファイルの有効性

### Live-Lite Mode 失敗
**症状**: 市場データ受信エラー
**対応**:
1. ネットワーク接続確認: `curl -I https://api.zaif.jp/api/1/ticker/btc_jpy`
2. API キー確認: `.env` ファイルの ZAIF_API_KEY
3. レートリミット確認: ログの "rate limit" エントリ
4. CI環境では `--connectivity-skip` オプションを使用

### Kill Switch 失敗
**症状**: Kill switch が反応しない
**対応**:
1. Kill file 存在確認: `ls -la /tmp/ztb.stop`
2. プロセス状態確認: `ps aux | grep trade-live`
3. 設定確認: `grep KILL_SWITCH .env`

## アーティファクト構造

```

artifacts/canary_20250127_143000/
├── logs/
│   ├── trade-bot.log
│   ├── features.log
│   └── errors.log
├── metrics/
│   ├── system_metrics.json
│   └── performance_metrics.json
├── reports/
│   ├── replay_report.json
│   ├── livelite_report.json
│   └── kill_resume_report.json
├── config/
│   ├── .env.backup
│   └── trade-config.json
└── canary_report.zip

```text

## ロールバック手順

Canary deployment で問題が発生した場合：

1. **即時停止**:
   ```bash
   touch /tmp/ztb.stop
   pkill -TERM -f "trade-live"
   ```

1. **設定ロールバック**:

   ```bash
   cp artifacts/canary_*/config/.env.backup .env
   ```

2. **ログ保存**:

   ```bash
   cp artifacts/canary_* logs/canary_backup_$(date +%Y%m%d_%H%M%S) -r
   ```

## 次のステップ

Canary deployment が成功したら：

1. **Go/No-Go レビュー**: `python scripts/go_nogo_check.py --report`
2. **本番移行**: `export EXCHANGE=zaif; export DRY_RUN=0`
3. **監視強化**: Grafana ダッシュボード有効化

## Fault Injection Testing

Canary harness supports automated fault injection to test system resilience. Faults are defined in `ztb/tests/integration/canary_cases.json` and executed via `ztb/utils/fault_injection.py`.

### Supported Fault Types

- **ws_disconnect**: WebSocket connection drops
- **network_delay**: Network latency spikes
- **data_gap**: Missing market data
- **duplicate_ticks**: Duplicate price ticks
- **slow_disk**: Slow checkpoint I/O
- **cpu_pause**: CPU throttling simulation
- **corrupted_checkpoint**: Checkpoint corruption
- **stream_throttle**: Data stream throttling

### Running Fault Injection Tests

```bash
# Run all canary cases
python -m pytest ztb/tests/integration/test_canary_harness.py -v

# Run specific fault type
python -c "
from ztb.utils.fault_injection import get_fault_injector, inject_fault
import asyncio

async def test():
    injector = get_fault_injector()
    async with await injector.inject_fault({
        'name': 'test_fault',
        'type': 'network_delay',
        'duration_s': 5.0,
        'severity': 0.8,
        'expected_action': 'pause'
    }):
        print('Fault injected')
        
asyncio.run(test())
"
```python

### Fault Injection Observability

All fault injections are logged with correlation IDs:

```

FAULT_INJECTION_START: ws_disconnect_transient (type=ws_disconnect, duration=2.0s, severity=0.3) correlation_id=fault_ws_disconnect_transient_1643123456
FAULT_INJECTION_END: ws_disconnect_transient correlation_id=fault_ws_disconnect_transient_1643123456

```text

---

**最終更新**: 2025-01-27
**バージョン**: 2.3.0
