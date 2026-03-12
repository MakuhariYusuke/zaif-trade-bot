# V433 Phase 5 運用ガイド

## 概要

V433 Phase 5 Production Migration Systemの運用手順とベストプラクティスを説明します。

## 日常運用

### 起動と停止

#### アプリケーション起動

```bash
# 仮想環境有効化
source venv_prod/bin/activate

# アプリケーション起動
python -m uvicorn ztb.app:app --host 0.0.0.0 --port 8000

# または本番用
gunicorn \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    ztb.app:app
```

#### Dockerでの起動

```bash
# イメージビルド
docker build -t zaif/trading:v433-phase5 .

# コンテナ起動
docker run -d \
    --name zaif-trading \
    -p 8000:8000 \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/config:/app/config \
    zaif/trading:v433-phase5
```

#### 停止

```bash
# プロセス停止
pkill -f uvicorn
# または
kill $(cat app.pid)

# Docker停止
docker stop zaif-trading
```

### 監視

#### ヘルスチェック

```bash
# APIヘルスチェック
curl http://localhost:8000/health

# 詳細ヘルスチェック
curl http://localhost:8000/health/detailed
```

#### ログ監視

```bash
# リアルタイムログ監視
tail -f logs/production.log

# エラーログ確認
grep ERROR logs/production.log | tail -20

# ログローテーション確認
ls -la logs/
```

#### メトリクス監視

```bash
# Prometheusメトリクス
curl http://localhost:9090/metrics

# カスタムメトリクス
curl http://localhost:8000/metrics
```

### パフォーマンス監視

#### 主要メトリクス

- **取引メトリクス**
  - 取引量（1分/1時間/1日）
  - 成功率
  - 平均応答時間
  - エラー率

- **システムメトリクス**
  - CPU使用率
  - メモリ使用率
  - ディスク使用率
  - ネットワークI/O

- **ビジネスメトリクス**
  - Sharpe Ratio
  - 最大ドローダウン
  - 取引回数
  - PNL（損益）

#### アラート設定

```python
# アラート閾値設定例
alert_thresholds = {
    'cpu_usage': 80.0,      # CPU使用率80%以上
    'memory_usage': 85.0,   # メモリ使用率85%以上
    'error_rate': 0.05,     # エラー率5%以上
    'response_time': 5.0,   # 応答時間5秒以上
    'success_rate': 0.95    # 成功率95%未満
}
```

## Phase 5移行運用

### Phase 1: Paper Trading（仮想取引）

#### 目的
- 新システムの動作確認
- パフォーマンス検証
- リスク評価

#### 運用手順

1. **環境セットアップ**
   ```bash
   # Paper Trading環境起動
   python scripts/deploy/start_paper_trading.py
   ```

2. **テスト取引実行**
   ```python
   from ztb.trading.production.paper_trading_manager import PaperTradingManager

   manager = PaperTradingManager(market_data_provider=None)
   session_id = await manager.start_session("paper_test_001")

   # テスト注文実行
   orders = generate_test_orders()
   for order in orders:
       result = await manager.submit_order(order)
       print(f"Order {order.order_id}: {'SUCCESS' if result else 'FAILED'}")
   ```

3. **パフォーマンス評価**
   ```python
   # パフォーマンスレポート生成
   report = await manager.generate_performance_report(session_id)
   print(f"Sharpe Ratio: {report.sharpe_ratio}")
   print(f"Max Drawdown: {report.max_drawdown}")
   print(f"Win Rate: {report.win_rate}")
   ```

#### 移行判定基準
- Sharpe Ratio > 0.5
- 最大ドローダウン < 10%
- 成功率 > 95%

### Phase 2: Parallel Running（並列実行）

#### 目的
- レガシーシステムとの並列運用
- トラフィック分散
- 結果比較

#### 運用手順

1. **システム設定**
   ```python
   from ztb.trading.production.traffic_distributor import TrafficDistributor, SystemEndpoint

   distributor = TrafficDistributor()

   # レガシーシステムエンドポイント
   legacy = SystemEndpoint(
       system_id='legacy',
       name='Legacy System',
       capacity=1000
   )

   # V433システムエンドポイント
   v433 = SystemEndpoint(
       system_id='v433',
       name='V433 System',
       capacity=1000
   )

   distributor.add_endpoint(legacy)
   distributor.add_endpoint(v433)
   ```

2. **トラフィック分散開始**
   ```python
   # 初期: 95% レガシー, 5% V433
   distributor.set_traffic_distribution({
       'legacy': 95.0,
       'v433': 5.0
   })
   ```

3. **結果比較**
   ```python
   from ztb.trading.production.result_comparator import ResultComparator

   comparator = ResultComparator()
   comparison = await comparator.perform_comparison('legacy', 'v433', 24)

   if comparison:
       print(f"V433 Performance: {comparison.v433_performance}")
       print(f"Legacy Performance: {comparison.legacy_performance}")
       print(f"Recommendation: {comparison.recommendation}")
   ```

#### 移行判定基準
- V433のパフォーマンスがレガシー以上
- 安定性（エラー率 < 1%）
- トラフィック処理能力

### Phase 3: Gradual Rollout（段階的ロールアウト）

#### 目的
- リスクベースのトラフィック増加
- リアルタイム監視
- 自動ロールバック準備

#### 運用手順

1. **リスクベース配分設定**
   ```python
   from ztb.trading.production.risk_based_allocator import RiskBasedAllocator, RiskMetrics

   allocator = RiskBasedAllocator()

   # リスク評価
   risk_metrics = RiskMetrics(
       volatility=0.2,
       max_drawdown=Decimal('0.05'),
       sharpe_ratio=1.2,
       value_at_risk=Decimal('0.02'),
       correlation=0.1,
       concentration_risk=0.1
   )

   # 配分決定
   decision = await allocator.evaluate_allocation('v433', risk_metrics)
   print(f"Allocated percentage: {decision.proposed_percentage}%")
   ```

2. **段階的増加**
   ```python
   # Phase 1: 5%
   await allocator.execute_allocation_decision(decision)

   # 監視期間（1日）
   await asyncio.sleep(86400)

   # Phase 2: 15%（リスク評価に基づく）
   # ... 自動または手動で増加
   ```

3. **ロールバック準備**
   ```python
   from ztb.trading.production.rollback_manager import RollbackManager

   rollback = RollbackManager()

   # ロールバックポイント設定
   checkpoint = await rollback.create_checkpoint('phase3_15percent')

   # 異常検知時のロールバック
   if error_detected:
       await rollback.rollback_to_checkpoint(checkpoint)
   ```

#### 移行判定基準
- 各段階で安定動作確認
- パフォーマンス劣化なし
- 顧客影響最小限

### Phase 4: Production Monitoring（本番監視）

#### 目的
- 継続的なシステム監視
- アラート管理
- ヘルスチェック

#### 運用手順

1. **監視システム起動**
   ```python
   from ztb.trading.production.performance_monitor import PerformanceMonitor
   from ztb.trading.production.alert_system import AlertSystem
   from ztb.trading.production.health_checker import HealthChecker

   # パフォーマンス監視
   monitor = PerformanceMonitor()
   monitor.start_monitoring()

   # アラートシステム
   alert_system = AlertSystem()
   alert_system.start_monitoring()

   # ヘルスチェック
   health_checker = HealthChecker()
   health_checker.start_monitoring()
   ```

2. **アラート設定**
   ```python
   # アラートルール設定
   alert_system.add_rule({
       'name': 'high_error_rate',
       'condition': 'error_rate > 0.05',
       'severity': 'CRITICAL',
       'action': 'notify_team'
   })

   alert_system.add_rule({
       'name': 'performance_degradation',
       'condition': 'response_time > 3.0',
       'severity': 'WARNING',
       'action': 'scale_up'
   })
   ```

3. **定期レポート**
   ```python
   # 日次レポート生成
   daily_report = await monitor.generate_daily_report()
   print(f"Daily Performance: {daily_report}")

   # 週次レポート生成
   weekly_report = await monitor.generate_weekly_report()
   print(f"Weekly Trends: {weekly_report}")
   ```

### Phase 5: Emergency Control（緊急制御）

#### 目的
- 異常時の自動保護
- 緊急停止機能
- 復旧システム

#### 運用手順

1. **回路ブレーカー設定**
   ```python
   from ztb.trading.production.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

   config = CircuitBreakerConfig(
       failure_threshold=5,
       recovery_timeout_seconds=60,
       success_threshold=3,
       timeout_seconds=30,
       monitoring_window_seconds=300,
       name='trading_system'
   )

   circuit_breaker = CircuitBreaker(config)
   ```

2. **緊急停止設定**
   ```python
   from ztb.trading.production.emergency_stop import EmergencyStop

   emergency_stop = EmergencyStop()

   # 停止コールバック設定
   async def emergency_callback(event):
       print(f"Emergency stop triggered: {event.level} - {event.reason}")
       # 通知、ログ記録、クリーンアップなど

   emergency_stop.add_stop_callback(emergency_callback)
   ```

3. **復旧システム設定**
   ```python
   from ztb.trading.production.recovery_system import RecoverySystem

   recovery = RecoverySystem()

   # 復旧プラン設定
   recovery.add_recovery_plan('service_restart', {
       'description': 'サービス再起動',
       'steps': [
           'stop_application',
           'restart_database',
           'start_application',
           'health_check'
       ]
   })
   ```

## 異常時対応

### 自動対応フロー

1. **監視システム検知**
   - パフォーマンス監視が異常を検知

2. **アラート発報**
   - AlertSystemがチームに通知

3. **自動保護**
   - CircuitBreakerが自動的に保護動作

4. **緊急停止（必要時）**
   - EmergencyStopがシステムを停止

5. **自動復旧**
   - RecoverySystemが自動復旧を試行

### 手動対応フロー

1. **状況確認**
   ```bash
   # システム状態確認
   curl http://localhost:8000/health/detailed

   # ログ確認
   tail -100 logs/production.log
   ```

2. **緊急停止**
   ```bash
   # API経由
   curl -X POST http://localhost:8000/emergency/stop \
        -H "Content-Type: application/json" \
        -d '{"level": "CRITICAL", "reason": "Manual emergency stop"}'
   ```

3. **手動復旧**
   ```bash
   # 復旧実行
   curl -X POST http://localhost:8000/recovery/initiate \
        -H "Content-Type: application/json" \
        -d '{"description": "Manual recovery after emergency stop"}'
   ```

## メンテナンス

### 日次メンテナンス

```bash
#!/bin/bash
# scripts/maintenance/daily.sh

# ログローテーション
logrotate -f /etc/logrotate.d/zaif-trading

# ヘルスチェック
curl -f http://localhost:8000/health

# ディスク容量確認
df -h | grep -E "(Filesystem|/)"

# プロセス確認
ps aux | grep zaif-trading
```

### 週次メンテナンス

```bash
#!/bin/bash
# scripts/maintenance/weekly.sh

# セキュリティアップデート
apt update && apt upgrade -y

# バックアップ
bash scripts/backup/backup.sh

# パフォーマンスレポート
python scripts/reports/performance_report.py

# ログ分析
python scripts/analysis/log_analysis.py
```

### 月次メンテナンス

```bash
#!/bin/bash
# scripts/maintenance/monthly.sh

# 包括的なテスト実行
python tests/run_integration_tests.py

# 依存関係更新
pip list --outdated
pip install --upgrade -r requirements_temp.txt

# データベース最適化
python scripts/maintenance/db_optimize.py
```

## レポートと分析

### パフォーマンスレポート

```python
# scripts/reports/performance_report.py
from ztb.trading.production.performance_monitor import PerformanceMonitor

async def generate_performance_report():
    monitor = PerformanceMonitor()

    # 期間指定
    report = await monitor.generate_performance_report(
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )

    print("=== Performance Report ===")
    print(f"Period: {report.period_start} to {report.period_end}")
    print(f"Total Trades: {report.total_trades}")
    print(f"Success Rate: {report.success_rate:.2%}")
    print(f"Average Response Time: {report.avg_response_time:.3f}s")
    print(f"Sharpe Ratio: {report.sharpe_ratio:.3f}")
    print(f"Max Drawdown: {report.max_drawdown:.2%}")

    return report
```

### 異常検知レポート

```python
# scripts/reports/anomaly_report.py
from ztb.trading.production.alert_system import AlertSystem

async def generate_anomaly_report():
    alert_system = AlertSystem()

    # 異常パターン分析
    anomalies = await alert_system.analyze_anomaly_patterns(
        days=7
    )

    print("=== Anomaly Report ===")
    for anomaly in anomalies:
        print(f"Time: {anomaly.timestamp}")
        print(f"Type: {anomaly.anomaly_type}")
        print(f"Severity: {anomaly.severity}")
        print(f"Description: {anomaly.description}")
        print("---")

    return anomalies
```

## トレーニングとドキュメント

### オペレータトレーニング

1. **基本操作トレーニング**
   - システム起動/停止手順
   - 監視コンソール使用方法
   - アラート対応手順

2. **異常時対応トレーニング**
   - 緊急停止手順
   - 復旧手順
   - ロールバック手順

3. **メンテナンストレーニング**
   - 定期メンテナンス手順
   - バックアップ/復旧手順
   - 更新手順

### ドキュメント更新

- **運用手順書の更新**: 新しい手順や変更点を反映
- **トラブルシューティングガイド**: 新しい問題と解決方法を追加
- **FAQ**: よくある質問と回答を更新

## 監査とコンプライアンス

### 運用ログ

- すべての操作をログに記録
- 変更履歴を追跡
- 監査ログを定期的に確認

### セキュリティ監査

- 定期的なセキュリティスキャン
- アクセス権限の確認
- 脆弱性パッチの適用

---

この運用ガイドはV433 Phase 5の安全で効率的な運用を支援するためのものです。実際の運用では、組織のポリシーと規制要件に従って適宜カスタマイズしてください。