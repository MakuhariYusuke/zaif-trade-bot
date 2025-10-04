# Advanced Auto-Stop System

高度な自動停止システムは、ライブ取引におけるリスク管理を強化するための包括的なソリューションです。複数の停止条件を監視し、市場の変動や取引パフォーマンスに基づいて自動的に取引を停止します。

## 特徴

- **多層リスク管理**: ボラティリティ、ドローダウン、時間、パフォーマンス、市場状況、連続損失などの複数の停止条件
- **リアルタイム監視**: 市場データと取引結果を継続的に分析
- **クールダウン期間**: 停止後の自動再開を防ぐための待機期間
- **Discord通知**: 停止イベントの即時通知
- **設定可能な閾値**: リスク許容度に応じたカスタマイズ可能なパラメータ

## 停止条件

### 1. ボラティリティ停止 (Volatility Stop)

- **目的**: 市場の異常な変動時に取引を停止
- **トリガー**: 設定されたボラティリティ閾値を超える
- **デフォルト閾値**: 3% (本番設定)
- **クールダウン**: 15分

### 2. ドローダウン停止 (Drawdown Stop)

- **目的**: 過度の損失が発生した場合に取引を停止
- **トリガー**: 設定されたドローダウン閾値を超える
- **デフォルト閾値**: 5% (本番設定)
- **クールダウン**: 30分

### 3. 時間ベース停止 (Time-based Stop)

- **目的**: 長時間の取引セッションによる疲労や市場変化を防ぐ
- **トリガー**: 設定された時間制限を超える
- **デフォルト閾値**: 6時間
- **クールダウン**: 1時間

### 4. パフォーマンス停止 (Performance Stop)

- **目的**: パフォーマンスの継続的な低下を検知
- **トリガー**: 最近の取引で設定された損失率を超える
- **デフォルト閾値**: -2%
- **クールダウン**: 15分

### 5. 市場状況停止 (Market Condition Stop)

- **目的**: 異常な市場状況での取引を防ぐ
- **トリガー**: ボラティリティの異常な上昇
- **デフォルト閾値**: 通常ボラティリティの1.5倍
- **クールダウン**: 30分

### 6. 連続損失停止 (Consecutive Losses Stop)

- **目的**: 連敗時のさらなる損失を防ぐ
- **トリガー**: 設定された連続損失回数を超える
- **デフォルト閾値**: 3回
- **クールダウン**: 15分

## インストールと使用方法

### 基本的な使用方法

```python
from ztb.risk.advanced_auto_stop import create_production_auto_stop

# 本番設定で自動停止システムを作成
auto_stop = create_production_auto_stop()

# 市場データを更新
from datetime import datetime
auto_stop.update_market_data(datetime.now(), current_price)

# 取引結果を更新
auto_stop.update_trade_result(pnl, trade_info)

# 停止条件をチェック
should_stop, reason, message = auto_stop.check_stop_conditions()
if should_stop:
    print(f"Trading stopped: {reason} - {message}")
    # 取引を停止する処理
```

### LiveTraderとの統合

自動停止システムは`live_trade.py`に自動的に統合されています：

```bash
# 通常通りライブ取引を実行
python live_trade.py --model-path models/my_model.zip --duration-hours 1
```

システムは自動的に：

- 各取引ループで停止条件をチェック
- 市場データを継続的に更新
- 取引結果をリスク分析に使用
- 停止イベントをDiscordで通知

### カスタム設定

```python
from ztb.risk.advanced_auto_stop import AdvancedAutoStop

# カスタム設定
config = {
    "volatility_stop": {
        "enabled": True,
        "threshold": 0.02,  # 2% ボラティリティ
        "window_size": 30,   # 30分ウィンドウ
        "cooldown_period": 600,  # 10分クールダウン
        "severity": "warning"
    },
    "drawdown_stop": {
        "enabled": True,
        "threshold": 0.03,  # 3% ドローダウン
        "window_size": 720,  # 12時間ウィンドウ
        "cooldown_period": 1800,  # 30分クールダウン
        "severity": "critical"
    }
}

auto_stop = AdvancedAutoStop(config)
```

## 設定パラメータ

### 各停止条件の設定項目

- `enabled`: 条件を有効にするかどうか
- `threshold`: 停止をトリガーする閾値
- `window_size`: 分析ウィンドウのサイズ（分単位）
- `cooldown_period`: 停止後のクールダウン期間（秒単位）
- `severity`: 重要度（"warning", "critical", "emergency"）

### 本番推奨設定

```python
# 保守的な本番設定
PRODUCTION_CONFIG = {
    "volatility_stop": {
        "enabled": True,
        "threshold": 0.03,  # 3%
        "window_size": 30,
        "cooldown_period": 900,  # 15分
        "severity": "warning"
    },
    "drawdown_stop": {
        "enabled": True,
        "threshold": 0.05,  # 5%
        "window_size": 720,  # 12時間
        "cooldown_period": 1800,  # 30分
        "severity": "critical"
    },
    "time_stop": {
        "enabled": True,
        "threshold": 21600,  # 6時間
        "cooldown_period": 3600,  # 1時間
        "severity": "warning"
    },
    "performance_stop": {
        "enabled": True,
        "threshold": -0.02,  # -2%
        "window_size": 120,  # 2時間
        "cooldown_period": 900,  # 15分
        "severity": "warning"
    },
    "market_condition_stop": {
        "enabled": True,
        "volatility_multiplier": 1.5,
        "trend_strength_threshold": 0.8,
        "cooldown_period": 1800,  # 30分
        "severity": "warning"
    },
    "consecutive_losses_stop": {
        "enabled": True,
        "threshold": 3,  # 3回
        "cooldown_period": 900,  # 15分
        "severity": "critical"
    }
}
```

## 監視とログ

### ステータス確認

```python
# 現在のシステムステータスを取得
status = auto_stop.get_status()
print(f"Active: {status['is_active']}")
print(f"Drawdown: {status['current_drawdown']:.2%}")
print(f"Volatility: {status['volatility']:.2%}")
print(f"Consecutive Losses: {status['consecutive_losses']}")
```

### Discord通知

システムは以下のイベントでDiscord通知を送信します：

- **停止トリガー**: 停止条件が満たされたとき
- **取引完了**: 自動停止ステータスを含む完了通知
- **エラー**: システムエラー発生時

### ログ出力

システムは以下のログを記録します：

```text
INFO: Advanced Auto-Stop system initialized
WARNING: Trading stopped: volatility_spike - Volatility 4.2% exceeds threshold 3.0%
INFO: Cooldown until: 2024-01-15 14:30:00
INFO: Trading resumed manually
```
```

## テスト

### ユニットテスト実行

```bash
# 自動停止システムのテスト
python -m pytest ztb/risk/tests/test_advanced_auto_stop.py -v

# 統合テスト
python -m pytest ztb/risk/tests/test_auto_stop_integration.py -v
```

### テストカバレッジ

テストは以下の機能をカバーします：

- システム初期化
- 市場データ更新
- 取引結果処理
- 各停止条件の評価
- クールダウン動作
- 手動再開機能
- LiveTrader統合

## トラブルシューティング

### 一般的な問題

1. **停止条件が作動しない**
   - 設定が正しく読み込まれているか確認
   - `enabled`フラグが`True`になっているか確認
   - 閾値が適切か確認

2. **過度な停止**
   - 閾値を保守的に調整
   - ウィンドウサイズを拡大
   - クールダウン期間を延長

3. **通知が届かない**
   - Discord webhook URLが正しく設定されているか確認
   - ネットワーク接続を確認

### デバッグモード

詳細なログを出力するには：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## アーキテクチャ

### クラス構造

- `AdvancedAutoStop`: メインの自動停止システムクラス
- `StopReason`: 停止理由の列挙型
- `StopCondition`: 個別の停止条件設定

### データフロー

1. **市場データ更新**: `update_market_data()` で価格データを蓄積
2. **リスク指標計算**: ボラティリティ、ドローダウンなどを計算
3. **条件評価**: 各停止条件を評価
4. **停止判定**: 条件が満たされた場合に停止をトリガー
5. **通知**: Discordで停止イベントを通知

### スレッド安全性

システムは単一スレッドで動作することを前提としています。マルチスレッド環境では適切な同期処理を追加してください。

## 拡張性

### 新しい停止条件の追加

```python
def _check_custom_condition(self, condition: StopCondition) -> Tuple[bool, Optional[StopReason], str]:
    """カスタム停止条件の実装例"""
    # カスタムロジックを実装
    if custom_metric > condition.threshold:
        return True, StopReason.CUSTOM_CONDITION, f"Custom metric {custom_metric} exceeds threshold"
    return False, None, "Custom condition normal"
```

### 設定の動的変更

```python
# 実行中に設定を変更
auto_stop.stop_conditions["volatility_stop"].threshold = 0.04
auto_stop.stop_conditions["volatility_stop"].enabled = False
```

## ライセンス

このシステムはZaif Trade Botプロジェクトの一部として配布されます。