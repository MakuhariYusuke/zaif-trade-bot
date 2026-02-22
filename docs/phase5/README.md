# V433 Phase 5: Production Migration System

## 概要

V433 Phase 5は、V433取引システムの本番移行を安全かつ段階的に行うための包括的なシステムです。Paper Trading、Parallel Running、Gradual Rollout、Production Monitoring、Emergency Controlの5つのレイヤーで構成されています。

## アーキテクチャ

### 5つのレイヤー

#### 1. Paper Trading Layer (仮想取引レイヤー)
- **VirtualPortfolioManager**: 仮想ポートフォリオの管理
- **MarketDataSimulator**: 市場データシミュレーション
- **PerformanceValidator**: パフォーマンス検証

#### 2. Parallel Running Layer (並列実行レイヤー)
- **TrafficDistributor**: トラフィック分散
- **SystemSwitcher**: システム切り替え
- **ResultComparator**: 結果比較

#### 3. Gradual Rollout Layer (段階的ロールアウトレイヤー)
- **RiskBasedAllocator**: リスクベース配分
- **PerformanceMonitor**: パフォーマンス監視
- **RollbackManager**: ロールバック管理

#### 4. Production Monitoring Layer (本番監視レイヤー)
- **RealTimeMetrics**: リアルタイムメトリクス
- **AlertSystem**: アラートシステム
- **HealthChecker**: ヘルスチェック

#### 5. Emergency Control Layer (緊急制御レイヤー)
- **CircuitBreaker**: 回路ブレーカー
- **EmergencyStop**: 緊急停止
- **RecoverySystem**: 復旧システム

## インストールとセットアップ

### 必要条件
- Python 3.11+
- pip
- 仮想環境（推奨）

### インストール

```bash
# 仮想環境作成
python -m venv venv311
venv311\Scripts\activate  # Windows
# source venv311/bin/activate  # Linux/Mac

# 依存関係インストール
pip install -r requirements_temp.txt
```

## 使用方法

### 統合テスト実行

```bash
# 全テスト実行
python tests/run_integration_tests.py

# 特定のテスト実行
python tests/run_integration_tests.py test_paper_trading_integration
```

### 個別コンポーネント使用

```python
from ztb.trading.production.paper_trading_manager import PaperTradingManager

# Paper Trading Manager 初期化
manager = PaperTradingManager(market_data_provider=None)

# セッション開始
session_id = await manager.start_session("test_session")

# 注文実行
order = Order(
    order_id="test_001",
    symbol="BTC/JPY",
    side=OrderSide.BUY,
    quantity=Decimal("0.001"),
    price=Decimal("5000000"),
    order_type=OrderType.MARKET
)
result = await manager.submit_order(order)

# セッション終了
await manager.stop_session()
```

## 設定

### 環境変数

```bash
# ログレベル
export LOG_LEVEL=INFO

# データベース接続
export DATABASE_URL=postgresql://user:pass@localhost:5432/zaif_trading

# Redis接続（オプション）
export REDIS_URL=redis://localhost:6379
```

### 設定ファイル

各コンポーネントは設定クラスを通じて設定可能です：

```python
from ztb.trading.production.paper_trading_manager import PaperTradingConfig

config = PaperTradingConfig(
    initial_balance=Decimal("1000000"),
    commission_rate=Decimal("0.001"),
    max_position_size=Decimal("10000")
)
```

## テスト

### 統合テスト

Phase 5の全コンポーネント統合テスト：

```bash
# 全テスト実行
python tests/run_integration_tests.py

# 結果例:
# ✓ Paper Trading Integration: PASSED
# ✓ Parallel Running Integration: PASSED
# ✓ Gradual Rollout Integration: PASSED
# ✓ Monitoring Integration: PASSED
# ✓ Emergency Control Integration: PASSED
# ✓ Failure Recovery Integration: PASSED
# ✓ Performance Under Load: PASSED
# ✓ Full System Integration: PASSED
#
# SUMMARY: 8/8 tests passed
# Success Rate: 100.0%
# 🎉 INTEGRATION TESTS SUCCESSFUL!
```

### 単体テスト

各コンポーネントの単体テスト：

```bash
python -m pytest tests/ -v
```

## 運用ガイド

### 本番移行手順

1. **Paper Tradingフェーズ**
   - 仮想環境でシステム動作確認
   - パフォーマンス検証
   - リスク評価

2. **Parallel Runningフェーズ**
   - レガシーシステムと並列実行
   - トラフィック分散（1%開始）
   - 結果比較と検証

3. **Gradual Rolloutフェーズ**
   - リスクベースでトラフィック増加
   - リアルタイム監視
   - 自動ロールバック準備

4. **Production Monitoringフェーズ**
   - 継続的な監視
   - アラート設定
   - ヘルスチェック

5. **Emergency Controlフェーズ**
   - 回路ブレーカー有効化
   - 緊急停止手順確認
   - 復旧計画準備

### 監視とアラート

システムは以下のメトリクスを監視：

- CPU使用率
- メモリ使用率
- 応答時間
- エラー率
- 取引成功率
- Sharpe Ratio

### 緊急時対応

1. **自動対応**: 回路ブレーカーが異常を検知し自動停止
2. **手動対応**: 緊急停止ボタンで即時停止
3. **復旧**: 自動または手動でシステム復旧

## トラブルシューティング

### よくある問題

#### テスト実行時のエラー

**問題**: `ImportError` や `ModuleNotFoundError`

**解決**:
```bash
# パス確認
python -c "import sys; print(sys.path)"

# 仮想環境確認
which python
pip list | grep ztb
```

#### パフォーマンス問題

**問題**: テスト実行が遅い

**解決**:
- 並列実行数を減らす
- タイムアウト時間を延ばす
- リソース使用量を確認

#### メモリ不足

**問題**: 大規模テストでメモリ不足

**解決**:
- テストデータを減らす
- バッチサイズを小さくする
- メモリ監視を有効化

### ログ確認

```bash
# ログファイル確認
tail -f logs/trading_system.log

# エラーログ確認
grep ERROR logs/*.log
```

## API リファレンス

### PaperTradingManager

```python
class PaperTradingManager:
    async def start_session(self, session_id: str) -> str
    async def submit_order(self, order: Order) -> bool
    async def stop_session(self) -> None
```

### TrafficDistributor

```python
class TrafficDistributor:
    def add_endpoint(self, endpoint: SystemEndpoint) -> None
    async def distribute_order(self, order: Order) -> Optional[str]
```

### その他のコンポーネント

詳細は各コンポーネントのソースコードを参照してください。

## 貢献ガイド

### コードスタイル

- Blackでフォーマット
- isortでインポート整理
- mypyで型チェック
- pytestでテスト

### コミットメッセージ

```
feat: 新機能追加
fix: バグ修正
docs: ドキュメント更新
test: テスト追加・修正
refactor: リファクタリング
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## サポート

問題が発生した場合：

1. ログファイルを確認
2. 統合テストを実行
3. GitHub Issuesで報告

---

**注意**: このシステムは金融取引を扱うため、本番環境での使用前に十分なテストと検証を行ってください。