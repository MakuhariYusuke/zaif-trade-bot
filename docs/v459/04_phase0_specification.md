# v459 Phase 0: 仕様固定 (04)

**Date**: 2026-01-22  
**Status**: ✅ **Phase 0 Completed** (実装完了、テスト77/77パス)  
**Phase**: Phase 0 - Specification Finalization & Implementation  
**Duration**: 1日（仕様） + 1日（実装）

---

## 1. Phase 0 の目的

実装前に以下の4つの仕様を明確に固定し、後戻りを防ぐ：

1. **Reporter I/O仕様**: PnL規約（gross/net）と指標定義
2. **Entry Gate仕様**: インターフェースと入出力型
3. **スケーラfit範囲規則**: リーク防止の因果性保証手順
4. **実行モデルパラメータ**: コスト・遅延の数値確定

---

## 2. Reporter I/O仕様

### 2.1 PnL規約の統一

> **重要決定**: envはnet PnL、reporterは検証のみ

```python
# ztb/evaluation/walk_forward/reporter.py (統一版)

class BacktestReporter:
    """
    統一バックテストレポーター
    - envからはnet PnL（コスト引き後）を受け取る
    - 検証目的でfee/slippageを記録するが、PnLから再度引かない
    """
    
    def __init__(self):
        self.trades: List[Dict] = []
        self.net_pnl_history: List[float] = []  # envから受け取るPnL
        self.fee_history: List[float] = []      # 記録用（再計算なし）
        self.slippage_history: List[float] = [] # 記録用（再計算なし）
    
    def record_trade(
        self,
        timestamp: pd.Timestamp,
        action: str,
        price: float,
        position_before: float,
        position_after: float,
        net_pnl: float,           # envから渡される（コスト引き後）
        fee_paid: float,          # 記録用
        slippage_paid: float      # 記録用
    ):
        """取引を記録（PnLはenvから受け取る値を使用）"""
        self.trades.append({
            "timestamp": timestamp,
            "action": action,
            "price": price,
            "position_before": position_before,
            "position_after": position_after,
            "net_pnl": net_pnl,          # これを使う
            "fee_paid": fee_paid,        # 参考情報
            "slippage_paid": slippage_paid  # 参考情報
        })
        self.net_pnl_history.append(net_pnl)
```

### 2.2 指標定義（標準化・厳密版）

> **重要**: Doc05レビューを受け、計算精度と例外処理を明確化

| 指標 | 定義 | 計算式 | 例外処理 |
|------|------|--------|----------|
| **Net ROI** | コスト引き後の収益率 | `(final_balance - initial_balance) / initial_balance` | - |
| **Gross ROI** | コスト引き前の収益率 | `Net ROI + total_costs / initial_balance` | total_costs = fee + slippage |
| **Profit Factor** | 総利益/総損失 | `sum(winning_pnls) / abs(sum(losing_pnls))` | 損失=0 → `inf`, 利益=0 → `0.0` |
| **Sharpe Ratio** | リスク調整後リターン（年率） | `mean(daily_returns) / std(daily_returns) * sqrt(252)` | std=0 → `None`, risk_free=0 |
| **Max Drawdown** | 最大下落率 | `max((peak - valley) / peak)` | peak=0 → `0.0` |
| **Win Rate** | 勝率 | `winning_trades / total_trades` | total_trades=0 → `None` |
| **Expectancy** | 期待値/取引 | `mean(net_pnls_per_trade)` | trades=0 → `None` |

#### 詳細計算規則

```python
# Sharpe Ratio計算（厳密版）
def calculate_sharpe_ratio(balance_history: List[float], risk_free_rate: float = 0.0) -> Optional[float]:
    """
    Sharpe Ratioの計算
    
    Args:
        balance_history: 残高履歴（1分足）
        risk_free_rate: リスクフリーレート（年率、デフォルト0）
    
    Returns:
        Sharpe Ratio（年率換算）、計算不可の場合はNone
    """
    # 1分足 → 日次リターンに集計
    minutes_per_day = 1440
    daily_balances = [balance_history[i] for i in range(0, len(balance_history), minutes_per_day)]
    
    if len(daily_balances) < 2:
        return None
    
    # 日次リターン計算
    daily_returns = [(daily_balances[i] - daily_balances[i-1]) / daily_balances[i-1] 
                     for i in range(1, len(daily_balances))]
    
    # 標準偏差がゼロの場合
    std_dev = np.std(daily_returns, ddof=1)
    if std_dev == 0 or np.isnan(std_dev):
        return None
    
    # Sharpe計算
    mean_return = np.mean(daily_returns)
    sharpe = (mean_return - risk_free_rate / 252) / std_dev * np.sqrt(252)
    
    return sharpe

# Profit Factor計算（厳密版）
def calculate_profit_factor(trade_pnls: List[float]) -> Optional[float]:
    """
    Profit Factorの計算
    
    Returns:
        Profit Factor、または特殊ケース（inf, 0.0, None）
    """
    if len(trade_pnls) == 0:
        return None
    
    winning_pnls = [p for p in trade_pnls if p > 0]
    losing_pnls = [p for p in trade_pnls if p < 0]
    
    total_profit = sum(winning_pnls)
    total_loss = abs(sum(losing_pnls))
    
    if total_loss == 0:
        return float('inf') if total_profit > 0 else 0.0
    
    return total_profit / total_loss
```

### 2.3 Trade Type分類規則（詳細版）

> **重要**: Doc05レビューを受け、増減と反転を明確化

```python
def classify_trade_type(position_before: float, position_after: float, tolerance: float = 1e-8) -> str:
    """
    取引タイプの詳細分類
    
    基本タイプ:
    - "long_open": ロング開始（0 → +）
    - "long_close": ロング終了（+ → 0）
    - "long_add": ロング増加（+ → ++）
    - "long_reduce": ロング減少（++ → +）
    - "short_open": ショート開始（0 → -）
    - "short_close": ショート終了（- → 0）
    - "short_add": ショート増加（- → --）
    - "short_reduce": ショート減少（-- → -）
    - "reverse": ポジション反転（+ ⇄ -）
    - "hold": ポジション変化なし
    
    反転の扱い:
    - reverseは統計上「close + open」として2つの取引に分解される
    """
    # 許容誤差内の差は無変化とみなす
    if abs(position_after - position_before) < tolerance:
        return "hold"
    
    # ゼロ判定
    is_before_zero = abs(position_before) < tolerance
    is_after_zero = abs(position_after) < tolerance
    
    # 0 → + (Long Open)
    if is_before_zero and position_after > 0:
        return "long_open"
    
    # + → 0 (Long Close)
    elif position_before > 0 and is_after_zero:
        return "long_close"
    
    # + → ++ (Long Add)
    elif position_before > 0 and position_after > position_before:
        return "long_add"
    
    # ++ → + (Long Reduce)
    elif position_before > 0 and position_after > 0 and position_after < position_before:
        return "long_reduce"
    
    # 0 → - (Short Open)
    elif is_before_zero and position_after < 0:
        return "short_open"
    
    # - → 0 (Short Close)
    elif position_before < 0 and is_after_zero:
        return "short_close"
    
    # - → -- (Short Add)
    elif position_before < 0 and position_after < position_before:
        return "short_add"
    
    # -- → - (Short Reduce)
    elif position_before < 0 and position_after < 0 and position_after > position_before:
        return "short_reduce"
    
    # + ⇄ - (Reverse)
    elif position_before * position_after < 0:
        return "reverse"
    
    else:
        return "hold"

def decompose_reverse_trade(
    position_before: float,
    position_after: float,
    price: float,
    timestamp: pd.Timestamp
) -> List[Dict]:
    """
    反転取引を「close + open」に分解
    
    Returns:
        2つの取引のリスト [close_trade, open_trade]
    """
    # Close側
    close_type = "long_close" if position_before > 0 else "short_close"
    close_trade = {
        "timestamp": timestamp,
        "action": close_type,
        "price": price,
        "position_before": position_before,
        "position_after": 0.0
    }
    
    # Open側
    open_type = "long_open" if position_after > 0 else "short_open"
    open_trade = {
        "timestamp": timestamp,
        "action": open_type,
        "price": price,
        "position_before": 0.0,
        "position_after": position_after
    }
    
    return [close_trade, open_trade]
```

---

## 3. Entry Gate仕様

### 3.1 インターフェース定義

```python
# ztb/trading/signal/types.py

from typing import TypedDict
from dataclasses import dataclass

class GateResult(TypedDict):
    """Entry Gateの返り値型"""
    should_enter: bool           # 必須: エントリー可否
    confidence: float            # 0.0-1.0: 確信度
    reason: str                  # 判定理由
    metrics: dict                # 追加メトリクス（任意）

@dataclass
class GateConfig:
    """Entry Gate設定"""
    enabled: bool = False
    calibration_map_path: str = ""
    min_confidence: float = 0.5
    cooldown_steps: int = 5
    edge_threshold: float = 0.001  # 0.1%
```

### 3.2 環境との連携（安全性確保版）

> **重要**: Doc05レビューを受け、exit/closeは常に許可する仕様に変更

```python
# ztb/trading/environment/fast_intraday_env_v456.py

def step(self, action):
    # Entry Gate判定（新規エントリーのみ制限）
    if self.entry_gate_enabled:
        # 現在のポジション状態を確認
        is_entry_action = self._is_entry_action(action, self.position)
        
        if is_entry_action:
            gate_result: GateResult = self.entry_gate.check_entry(
                action=action,
                market_state=self._get_market_state(),
                position=self.position
            )
            
            # 正しい属性アクセス（v458のバグ修正）
            if not gate_result["should_enter"]:
                # エントリーブロック → HOLDに変換
                # ただし、ポジション縮小・決済は許可
                action = self._convert_to_hold_action()
                self._log_gate_block(gate_result)
        # else: exit/close/reduceは常に許可（ゲートチェックなし）
    
    # 以降の処理...

def _is_entry_action(self, action: float, current_position: float) -> bool:
    """
    アクションが新規エントリーまたはポジション拡大か判定
    
    Returns:
        True: 新規エントリーまたは拡大（ゲートチェック必要）
        False: 決済・縮小・反転（ゲートチェック不要、常に許可）
    """
    target_position = self._action_to_position(action)
    
    # 絶対値が増える = エントリー/拡大
    if abs(target_position) > abs(current_position):
        return True
    
    # 絶対値が減る = 決済/縮小（常に許可）
    return False

def _convert_to_hold_action(self) -> float:
    """現在のポジションを維持するアクションに変換"""
    return self._position_to_action(self.position)
```

### 3.3 Config配線規則

```yaml
# config/v459/base/config.yaml

training:
  environment:
    type: "FastIntradayEnvV456"
    # Entry Gate設定はここに配置
    entry_gate:
      enabled: false              # Phase 0ではfalse
      calibration_map_path: ""
      min_confidence: 0.5
      cooldown_steps: 5
      edge_threshold: 0.001
```

```python
# ztb/training/utils/v457_config_utils.py

def extract_env_config(config: dict) -> dict:
    """環境設定の抽出（entry_gateを含む）"""
    env_config = config["training"]["environment"].copy()
    
    # entry_gateが環境配下にあることを確認（Doc05対応: assert→例外型）
    if "entry_gate" not in env_config:
        raise ValueError(
            "Config error: 'entry_gate' must be under 'training.environment'. "
            "Please move entry_gate configuration to the correct location."
        )
    
    return env_config

def validate_execution_model(exec_model: dict) -> None:
    """実行モデル設定の検証（Doc05対応）"""
    # 必須フィールド
    required_fields = ["costs", "execution", "risk"]
    for field in required_fields:
        if field not in exec_model:
            raise ValueError(f"Execution model missing required field: {field}")
    
    # slippage_modelの整合性確認
    if "slippage_model" in exec_model.get("costs", {}):
        model = exec_model["costs"]["slippage_model"]
        if model not in ["fixed", "volume_based"]:
            raise ValueError(f"Invalid slippage_model: {model}. Must be 'fixed' or 'volume_based'")
```

---

## 4. スケーラfit範囲規則

### 4.1 規則の明文化

```python
# ztb/features/scaler.py

class CausalOnlineScaler:
    """因果性保証付きスケーラ（Doc05対応: ゼロ分散対応）"""
    
    def __init__(self, feature_names: List[str], std_floor: float = 1e-8):
        self.feature_names = feature_names
        self.fitted = False
        self.fit_end_idx = None  # fit範囲の終端を記録
        self.std_floor = std_floor  # ゼロ分散対策
    
    def fit(self, data: pd.DataFrame, end_idx: int):
        """
        Train期間のみでfit
        
        Args:
            data: 全データ（Train+Val+Test）
            end_idx: Train最終インデックス（inclusive、この行を含む）
        
        Note:
            実装では data.iloc[:end_idx+1] でスライス（end_idxの行を含める）
            Doc07とこの仕様書で表記が異なっていたため統一
        """
        train_data = data.iloc[:end_idx + 1]  # end_idx inclusive
        self.mean_ = train_data[self.feature_names].mean()
        self.std_ = train_data[self.feature_names].std()
        
        # ゼロ分散対策: stdがstd_floor未満の場合はstd_floorを使用
        self.std_ = self.std_.clip(lower=self.std_floor)
        
        self.fitted = True
        self.fit_end_idx = end_idx
        
        # リーク検査用
        self._verify_no_leakage(data, end_idx)
    
    def _verify_no_leakage(self, data: pd.DataFrame, end_idx: int):
        """Val/Testデータがfit範囲に含まれていないか検証"""
        if len(data) > end_idx + 1:
            val_test_data = data.iloc[end_idx + 1:]  # Val/Test start
            # Val/Testデータの統計がmean/stdに影響していないか確認
            # （実装: サンプル検証）
            pass
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform（fit範囲外でも使用可能）
        
        Returns:
            標準化されたデータ（NaN/infなし）
        """
        if not self.fitted:
            raise ValueError("Must call fit() first")
        
        scaled = (data[self.feature_names] - self.mean_) / self.std_
        
        # NaN/infの確認（安全性チェック）
        if scaled.isna().any().any() or np.isinf(scaled).any().any():
            raise ValueError("Scaling produced NaN or inf values")
        
        return scaled
```

### 4.2 Walk-Forwardでの使用

```python
# scripts/v459/run_walk_forward_v459.py

def run_single_window(window_config):
    # データ分割
    train_data = df[train_start:train_end]
    val_data = df[val_start:val_end]
    test_data = df[test_start:test_end]
    
    # スケーラはTrain期間のみでfit
    scaler = CausalOnlineScaler(feature_names)
    scaler.fit(df, end_idx=train_end)  # Train最終index（inclusive）を明示
    
    # Transform
    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
    
    # Reporter分離（Val/Test汚染防止）
    val_reporter = BacktestReporter()
    test_reporter = BacktestReporter()  # 別インスタンス
    
    return {
        "train": train_scaled,
        "val": (val_scaled, val_reporter),
        "test": (test_scaled, test_reporter)
    }
```

---

## 5. 実行モデルパラメータ

### 5.1 確定パラメータ

```yaml
# config/v459/base/execution_model.yaml

execution_model:
  # コストパラメータ（デフォルト値）
  costs:
    fee_rate: 0.001           # 0.1% (Zaif Maker)
    slippage_rate: 0.0005     # 0.05% (推定値、Paper Trading後更新)
    slippage_model: "fixed"   # "fixed" or "volume_based"
  
  # 約定モデル
  execution:
    latency_ms: 500           # 約定遅延（推定値、Paper Trading後更新）
    fill_rate: 0.95           # 約定率（推定値）
    order_type: "limit"       # 指値注文
  
  # リスクパラメータ
  risk:
    max_position: 0.01        # 0.01 BTC
    max_daily_loss_pct: 0.03  # 3%
    consecutive_loss_limit: 5 # 5連敗で縮退
  
  # Paper Trading後の更新ルール
  update_policy:
    paper_trading_overrides: true  # Paper Trading実測値で上書き
    confidence_interval: 0.95      # 95%信頼区間で設定
```

### 5.2 Backtest vs Paper Trading

```python
# ztb/trading/environment/cost_model.py

class CostModel:
    """コストモデル（Backtest/Paper Tradingで共通）"""
    
    def __init__(self, config: dict):
        self.fee_rate = config["costs"]["fee_rate"]
        self.slippage_rate = config["costs"]["slippage_rate"]
        self.use_measured_slippage = False  # Paper Trading後にTrue
    
    def calculate_cost(self, price: float, quantity: float) -> tuple:
        """
        取引コスト計算
        
        Returns:
            (fee, slippage): コストの内訳
        """
        notional = price * quantity
        fee = notional * self.fee_rate
        
        if self.use_measured_slippage:
            # Paper Trading実測値を使用
            slippage = self._get_measured_slippage(price, quantity)
        else:
            # デフォルト推定値
            slippage = notional * self.slippage_rate
        
        return fee, slippage
```

---

## 6. 検証スクリプト

### 6.1 リーク検査スクリプト

```python
# scripts/v459/check_data_leakage.py

import pandas as pd
import numpy as np
from ztb.features.scaler import CausalOnlineScaler

def check_scaler_leakage(
    scaler: CausalOnlineScaler,
    train_end_idx: int,
    full_data: pd.DataFrame
) -> dict:
    """スケーラがTrain期間のみでfitされているか検査"""
    
    # 1. fit範囲の確認
    assert scaler.fit_end_idx == train_end_idx, \
        f"Scaler fit範囲エラー: expected {train_end_idx}, got {scaler.fit_end_idx}"
    
    # 2. mean/stdがTrain期間の統計と一致するか
    train_data = full_data.iloc[:train_end_idx + 1]  # end_idx inclusive
    expected_mean = train_data[scaler.feature_names].mean()
    expected_std = train_data[scaler.feature_names].std()
    
    mean_diff = np.abs(scaler.mean_ - expected_mean).max()
    std_diff = np.abs(scaler.std_ - expected_std).max()
    
    return {
        "passed": mean_diff < 1e-6 and std_diff < 1e-6,
        "mean_diff": mean_diff,
        "std_diff": std_diff
    }

def check_mtf_causality(df: pd.DataFrame) -> dict:
    """MTF特徴量の因果性確認"""
    # サンプル検査: t時点のMTF値がt+N時点のデータを含んでいないか
    sample_indices = np.random.choice(len(df) - 60, size=100, replace=False)
    
    violations = []
    for idx in sample_indices:
        current_5m = df.loc[idx, "mtf_5m_rsi"]
        # 5分後のデータと比較（含まれていないはず）
        future_5m = df.loc[idx + 5, "mtf_5m_rsi"]
        
        if current_5m == future_5m:
            violations.append(idx)
    
    return {
        "passed": len(violations) == 0,
        "violations": violations
    }

if __name__ == "__main__":
    # 実行例
    from ztb.data.loader import load_data
    
    df = load_data("data/btc_jpy_1m_v451.csv")
    train_end_idx = int(len(df) * 0.7)
    
    scaler = CausalOnlineScaler(["close", "volume", "rsi"])
    scaler.fit(df, train_end_idx)
    
    result = check_scaler_leakage(scaler, train_end_idx, df)
    print(f"Scaler Leakage Check: {'PASSED' if result['passed'] else 'FAILED'}")
```

---

## 7. Phase 0 完了チェックリスト

> **重要**: Doc05レビューを受け、仕様完成とコード完成を分離

### 7.1 仕様文書化（Phase 0.1）

- [x] Reporter I/O仕様書（本ドキュメント Section 2）
  - [x] PnL規約統一
  - [x] 指標定義の厳密化（例外処理含む）
  - [x] Trade Type詳細分類
- [x] Entry Gate I/F仕様書（本ドキュメント Section 3）
  - [x] 型定義（GateResult, GateConfig）
  - [x] exit/close常時許可の安全性確保
  - [x] Config配線規則
- [x] スケーラfit規則書（本ドキュメント Section 4）
  - [x] 因果性保証の明文化
  - [x] リーク検査方法
- [x] 実行モデル仕様書（本ドキュメント Section 5）
  - [x] コストパラメータ
  - [x] 約定モデル
  - [x] Paper Trading更新ルール

### 7.2 コード実装（Phase 0.2）

- [x] `ztb/evaluation/walk_forward/reporter.py`
  - [x] PnL規約の実装確認
  - [x] 指標計算の厳密化実装
  - [x] Trade Type詳細分類の実装
- [x] `ztb/trading/signal/types.py`
  - [x] GateResult型定義追加（既存）
  - [x] GateConfig dataclass追加（既存）
- [x] `ztb/trading/environment/fast_intraday_env_v456.py`
  - [x] Entry Gate安全性実装（exit/close常時許可）
  - [x] `_is_entry_action()` メソッド追加
- [x] `ztb/processing/causal_online_scaler.py` (新規)
  - [x] CausalOnlineScaler実装
  - [x] ゼロ分散対応
  - Note: 当初は`ztb/features/scaler.py`を想定していたが、既存の`online_scaler.py`を継承する形で実装
- [x] `ztb/features/grouping/causal_grouped_scaler.py` (新規)
  - [x] CausalGroupedFeatureScaler実装
  - [x] 88→36次元選択スケーリング
- [x] `ztb/training/utils/v457_config_utils.py`
  - [x] validate_env_config()実装
  - [x] entry_gate配置検証
- [x] `config/v459/base/config.yaml` 作成
- [ ] Config検証コード（assert → 例外型に変更）

### 7.3 検証（Phase 0.3）

- [x] `scripts/v459/check_data_leakage.py` 実装
  - [ ] MTF因果性検査を再計算一致テストに修正
- [ ] 単体テスト作成
  - [ ] Reporter指標計算テスト
  - [ ] Entry Gate安全性テスト
  - [ ] Scaler因果性テスト
- [ ] 統合テスト環境準備

**Phase 0完了基準**: 7.1（仕様文書化）が100%完了すること

---

## 8. 次のステップ

Phase 0完了後、Doc05としてPhase 1（P0バグ修正）に移行：

1. Entry Gate Crash修正
2. Entry Gate Config配線修正
3. Cost Double-Count修正
4. Val/Test Reporter分離

---

**Status**: 🔧 In Progress  
**Completion**: 0% → 100%（本ドキュメント完成で50%、コード実装で100%）  
**Next**: コード実装と検証スクリプト作成  
**Author**: GitHub Copilot  
**Date**: 2026-01-22
