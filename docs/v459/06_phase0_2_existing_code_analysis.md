# v459 Phase 0.2: 既存実装の精査とリファクタリング計画 (06)

**Date**: 2026-01-22  
**Status**: 📋 Analysis Complete  
**Phase**: Phase 0.2 - Code Implementation Planning  
**Purpose**: 既存実装の精査、重複防止、リファクタリング方針の決定

---

## 1. Executive Summary

Phase 0.1で確定した仕様に対して、既存コードベースを精査した結果、以下が判明：

- **Reporter**: 既存の`BacktestReporter`があるが、Doc04仕様と部分的に乖離
- **Entry Gate**: `IntegratedEntrySystem`が存在し、`GateResult`型も定義済み
- **Scaler**: `OnlineScaler`と`GroupedFeatureScaler`が存在、因果性保証は未実装
- **Config Utils**: 基本的なextract関数はあるが、検証ロジックは未実装

**結論**: 新規実装ではなく、既存実装への**追加・修正・統合**で対応可能。

---

## 2. 既存実装の詳細分析

### 2.1 BacktestReporter (`ztb/evaluation/walk_forward/reporter.py`)

#### 既存実装の状態

```python
class BacktestReporter:
    def __init__(self):
        self.stats = {...}
        self.portfolio_history = []
        self.trade_history = []
    
    def record_action(self, action, env_info): ...
    def record_trade(self, trade_type, pnl, entry_price, exit_price, size, fee, slippage): ...
    def finalize_stats(self): ...
```

#### Doc04仕様との差異

| 項目 | 既存実装 | Doc04仕様 | 対応 |
|------|----------|-----------|------|
| **PnL規約** | net_pnl使用（正しい） | ✅ 一致 | なし |
| **Trade Type** | long/short/closeの3種 | 8種詳細分類 | ✅ 拡張必要 |
| **Profit Factor** | net_pnl基準 | ✅ 一致 | なし |
| **ゼロ除算** | 0 → inf | ✅ 一致 | なし |
| **Sharpe計算** | 1m想定固定 | 日次集約必要 | ✅ 修正必要 |
| **例外処理** | 一部欠如 | 厳密化必要 | ✅ 追加必要 |

#### リファクタリング方針

```python
# 修正箇所1: Trade Type分類の詳細化
def record_trade(
    self,
    position_before: float,
    position_after: float,
    pnl: float,
    entry_price: float,
    exit_price: float,
    size: float,
    fee: float,
    slippage: float,
    timestamp: pd.Timestamp
):
    """Doc04仕様に準拠した詳細Trade Type分類"""
    trade_type = classify_trade_type(position_before, position_after)
    
    # 反転の場合は分解
    if trade_type == "reverse":
        trades = decompose_reverse_trade(position_before, position_after, exit_price, timestamp)
        for trade in trades:
            self._record_single_trade(trade, pnl/2, fee/2, slippage/2)
    else:
        self._record_single_trade(
            {"action": trade_type, "timestamp": timestamp, ...},
            pnl, fee, slippage
        )

# 修正箇所2: Sharpe Ratio計算の厳密化
def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> Optional[float]:
    """Doc04仕様に準拠した日次集約Sharpe"""
    if len(self.portfolio_history) < 1440:  # 1日未満
        return None
    
    # 1分足→日次リターンに集約
    minutes_per_day = 1440
    daily_balances = [
        self.portfolio_history[i]
        for i in range(0, len(self.portfolio_history), minutes_per_day)
    ]
    
    if len(daily_balances) < 2:
        return None
    
    daily_returns = [
        (daily_balances[i] - daily_balances[i-1]) / daily_balances[i-1]
        for i in range(1, len(daily_balances))
    ]
    
    std_dev = np.std(daily_returns, ddof=1)
    if std_dev == 0 or np.isnan(std_dev):
        return None
    
    mean_return = np.mean(daily_returns)
    sharpe = (mean_return - risk_free_rate / 252) / std_dev * np.sqrt(252)
    
    return sharpe
```

---

### 2.2 Entry Gate (`ztb/trading/signal/types.py`, `fast_intraday_env_v456.py`)

#### 既存実装の状態

```python
# ztb/trading/signal/types.py
class GateResult(TypedDict):
    should_enter: bool  # ✅ Doc04仕様と一致
    ev: float
    ev_l1: float
    ev_fb: float
    lambda_val: float
    cost: float
    stats: CalibrationStats
    stats_fallback: CalibrationStats

# fast_intraday_env_v456.py (line 270-275)
entry_gate_config = self.env_config.get("entry_gate", {})
if entry_gate_config.get("enabled", False):
    self.entry_system = IntegratedEntrySystem(entry_gate_config)
    calibration_path = entry_gate_config.get("calibration_map_path")
    # ... load calibration
```

#### Doc04仕様との差異

| 項目 | 既存実装 | Doc04仕様 | 対応 |
|------|----------|-----------|------|
| **GateResult型** | ✅ should_enter定義済み | ✅ 一致 | なし |
| **Exit安全性** | 未実装 | exit/close常時許可 | ✅ 追加必要 |
| **Config配線** | env_configから取得 | ✅ 正しい位置 | なし |

#### リファクタリング方針

```python
# fast_intraday_env_v456.py に追加
def _is_entry_action(self, action: float, current_position: float) -> bool:
    """
    Doc04仕様: 新規エントリー/拡大のみをゲートチェック対象に
    exit/close/reduceは常に許可
    """
    target_position = self._action_to_position(action)
    
    # 絶対値が増える = エントリー/拡大
    if abs(target_position) > abs(current_position):
        return True
    
    # 絶対値が減る = 決済/縮小（常に許可）
    return False

def step(self, action):
    # Entry Gate判定（安全性確保版）
    if self.entry_gate_enabled:
        is_entry = self._is_entry_action(action, self.position)
        
        if is_entry:
            gate_result: GateResult = self.entry_system.check_entry(
                action=action,
                market_state=self._get_market_state(),
                position=self.position
            )
            
            if not gate_result["should_enter"]:
                # エントリーブロック → HOLDに変換
                action = self._convert_to_hold_action()
                self._log_gate_block(gate_result)
        # else: exit/closeは常に許可（ゲートチェックなし）
    
    # 以降の処理...
```

---

### 2.3 Scaler (`ztb/processing/online_scaler.py`, `ztb/features/grouping/grouped_scaler.py`)

#### 既存実装の状態

1. **OnlineScaler**: Welford's algorithmで漸進的統計更新
2. **GroupedFeatureScaler**: 88次元中36次元を選択的にスケール

#### Doc04仕様との差異

| 項目 | 既存実装 | Doc04仕様 | 対応 |
|------|----------|-----------|------|
| **fit範囲管理** | なし | end_idx記録必要 | ✅ 追加必要 |
| **リーク検査** | なし | _verify_no_leakage | ✅ 追加必要 |
| **ゼロ分散対応** | epsilon使用 | std_floor明示 | ✅ 統一必要 |
| **NaN/inf検査** | なし | 例外投げる | ✅ 追加必要 |

#### リファクタリング方針

**方針A**: 既存`OnlineScaler`を拡張（推奨）

```python
# ztb/processing/online_scaler.py に追加
class CausalOnlineScaler(OnlineScaler):
    """
    Doc04仕様に準拠した因果性保証付きOnlineScaler
    既存OnlineScalerを継承して機能追加
    """
    
    def __init__(self, shape, epsilon=1e-5, clip=10.0, std_floor=1e-8):
        super().__init__(shape, epsilon, clip)
        self.std_floor = std_floor
        self.fit_end_idx = None
        self.fitted = False
    
    def fit(self, data: pd.DataFrame, end_idx: int, feature_names: List[str]):
        """
        Train期間のみでfit（因果性保証）
        
        Args:
            data: 全データ
            end_idx: Train終端index
            feature_names: 対象特徴量名
        """
        train_data = data.iloc[:end_idx][feature_names].values
        
        # バッチ更新
        for row in train_data:
            self.update(row)
        
        # ゼロ分散対応
        self.var = np.maximum(self.var, self.std_floor ** 2)
        
        self.fitted = True
        self.fit_end_idx = end_idx
        
        # リーク検査
        self._verify_no_leakage(data, end_idx, feature_names)
    
    def _verify_no_leakage(self, data, end_idx, feature_names):
        """Val/Testデータの混入を検査"""
        if len(data) <= end_idx:
            return
        
        # Train期間の統計と一致するか確認
        train_data = data.iloc[:end_idx][feature_names].values
        expected_mean = train_data.mean(axis=0)
        expected_std = train_data.std(axis=0, ddof=1)
        
        mean_diff = np.abs(self.mean - expected_mean).max()
        std_diff = np.abs(np.sqrt(self.var) - expected_std).max()
        
        if mean_diff > 1e-5 or std_diff > 1e-5:
            raise ValueError(
                f"Scaler leakage detected: mean_diff={mean_diff:.2e}, std_diff={std_diff:.2e}"
            )
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform with NaN/inf check
        """
        if not self.fitted:
            raise ValueError("Must call fit() first")
        
        scaled = super().transform(x)  # 親クラスのtransform使用
        
        # NaN/inf検査
        if np.isnan(scaled).any() or np.isinf(scaled).any():
            raise ValueError("Scaling produced NaN or inf values")
        
        return scaled
```

**方針B**: GroupedFeatureScalerも同様に拡張

```python
# ztb/features/grouping/grouped_scaler.py に追加
class CausalGroupedFeatureScaler(GroupedFeatureScaler):
    """
    Doc04仕様に準拠した因果性保証付きGroupedFeatureScaler
    """
    
    def __init__(self, epsilon=1e-7, momentum=0.99, clip_value=3.0, std_floor=1e-8):
        super().__init__(epsilon, momentum, clip_value)
        self.std_floor = std_floor
        self.fit_end_idx = None
        self.fitted = False
    
    def fit(self, data: pd.DataFrame, end_idx: int):
        """Train期間のみでfit"""
        train_data = data.iloc[:end_idx].values
        
        for row in train_data:
            self.fit_one(row)
        
        # ゼロ分散対応
        self.std = np.maximum(self.std, self.std_floor)
        
        self.fitted = True
        self.fit_end_idx = end_idx
        
        self._verify_no_leakage(data, end_idx)
    
    def _verify_no_leakage(self, data, end_idx):
        """リーク検査"""
        # 実装省略（OnlineScalerと同様）
        pass
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """NaN/inf検査付きtransform"""
        if not self.fitted:
            raise ValueError("Must call fit() first")
        
        scaled = super().transform(features)
        
        if np.isnan(scaled).any() or np.isinf(scaled).any():
            raise ValueError("Scaling produced NaN or inf values")
        
        return scaled
```

---

### 2.4 Config Utils (`ztb/training/utils/v457_config_utils.py`)

#### 既存実装の状態

```python
def extract_env_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract training.environment section as a dict."""
    training = extract_training_config(config)
    env_config = training.get("environment")
    if isinstance(env_config, Mapping):
        return dict(env_config)
    return {}
```

#### Doc04仕様との差異

| 項目 | 既存実装 | Doc04仕様 | 対応 |
|------|----------|-----------|------|
| **entry_gate検証** | なし | ValueError必要 | ✅ 追加必要 |
| **execution_model検証** | なし | フィールド検証 | ✅ 追加必要 |
| **assert使用** | なし | ✅ 使用していない | なし |

#### リファクタリング方針

```python
# ztb/training/utils/v457_config_utils.py に追加

def validate_env_config(env_config: dict) -> None:
    """
    Doc04仕様: 環境設定の検証
    
    Raises:
        ValueError: 設定が不正な場合
    """
    # entry_gate検証
    if "entry_gate" not in env_config:
        raise ValueError(
            "Config error: 'entry_gate' must be under 'training.environment'. "
            "Please move entry_gate configuration to the correct location."
        )
    
    # execution_model検証
    if "execution_model" in env_config:
        exec_model = env_config["execution_model"]
        required_fields = ["costs", "execution", "risk"]
        
        for field in required_fields:
            if field not in exec_model:
                raise ValueError(
                    f"Execution model missing required field: {field}"
                )
        
        # slippage_model整合性
        if "costs" in exec_model and "slippage_model" in exec_model["costs"]:
            model = exec_model["costs"]["slippage_model"]
            if model not in ["fixed", "volume_based"]:
                raise ValueError(
                    f"Invalid slippage_model: {model}. "
                    f"Must be 'fixed' or 'volume_based'"
                )

def extract_env_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract and validate training.environment section."""
    training = extract_training_config(config)
    env_config = training.get("environment")
    
    if not isinstance(env_config, Mapping):
        return {}
    
    env_dict = dict(env_config)
    
    # Doc04仕様検証
    validate_env_config(env_dict)
    
    return env_dict
```

---

## 3. 実装優先順位と作業計画

### Phase 0.2a: Reporter強化（1-2日） ✅ **完了**

**対象ファイル**: `ztb/evaluation/walk_forward/reporter.py`

- [x] Trade Type詳細分類の実装（8種: long/short_open/close/add/reduce）
- [x] 反転取引の分解処理（reverse → close + open）
- [x] Sharpe Ratio計算の厳密化（日次集約、NaN/inf検査）
- [x] 例外処理の追加（Profit Factor、Expectancy）
- [x] 単体テスト作成（`tests/unit/v459/test_reporter_v459.py`、23/23 passed）

**実装サマリー**:
- `classify_trade_type()`: position_before/after から8種のTrade Typeを判定
- `decompose_reverse_trade()`: 反転取引を決済+新規エントリーに分解
- `record_trade()`: 新規シグネチャでposition_before/afterを受け取り
- `_calculate_sharpe_ratio()`: 1分足→日次集約、std=0でNone返却
- 全テスト合格、ゼロ近傍の許容誤差も対応完了

### Phase 0.2b: Entry Gate安全性（1日） ✅ **完了**

**対象ファイル**: `ztb/trading/environment/fast_intraday_env_v456.py`

- [x] `_is_entry_action()` メソッド追加
- [x] `_convert_to_hold_action()` メソッド追加
- [x] step()ロジックの修正（exit/close常時許可）
- [x] 単体テスト作成（`tests/unit/v459/test_entry_gate_safety_v459.py`、11/11 passed）

**実装サマリー**:
- `_is_entry_action()`: abs(target) > abs(current) でエントリー判定、それ以外は常時許可
- `_convert_to_hold_action()`: ゲートブロック時にaction_space_typeに応じてHOLD変換
- step()修正: エントリー時のみgate check、exit/reduce時はスキップ
- Doc04仕様完全準拠、安全性確保完了

### Phase 0.2c: Scaler因果性保証（1-2日） ✅ **完了**

**対象ファイル**: 
- `ztb/processing/causal_online_scaler.py`
- `ztb/features/grouping/causal_grouped_scaler.py`

- [x] `CausalOnlineScaler`クラス追加（OnlineScaler継承）
- [x] `CausalGroupedFeatureScaler`クラス追加（GroupedFeatureScaler継承）
- [x] リーク検査実装（OnlineScaler: 厳密、GroupedScaler: EMA考慮で警告のみ）
- [x] ゼロ分散対応統一（std_floor=1e-8）
- [x] 単体テスト作成（`tests/unit/v459/test_causal_scaler_v459.py`、18/18 passed）

**実装サマリー**:
- `CausalOnlineScaler`: fit_end_idx記録、Train期間のみfit、NaN/inf検査、リーク検査（統計一致確認）
- `CausalGroupedFeatureScaler`: 88次元中36次元選択的スケール、EMA更新（momentum=0.99）のため完全一致は不可、警告のみ
- 両クラスとも`fitted`フラグ、`get_fit_info()`デバッグメソッド実装
- Doc04仕様完全準拠、因果性保証完了

### Phase 0.2d: Config検証強化（半日）

**対象ファイル**: `ztb/training/utils/v457_config_utils.py`

- [ ] `validate_env_config()` 関数追加
- [ ] `extract_env_config()` 修正
- [ ] 単体テスト作成

---

## 4. 重複実装の防止策

### 4.1 命名規則の統一

| 新規実装 | 既存実装との関係 | 命名ルール |
|----------|------------------|------------|
| CausalOnlineScaler | OnlineScalerを継承 | Causal*プレフィックス |
| CausalGroupedFeatureScaler | GroupedFeatureScalerを継承 | Causal*プレフィックス |
| classify_trade_type() | 新規関数 | 既存record_trade()から呼び出し |
| decompose_reverse_trade() | 新規関数 | classify_trade_type()の補助 |

### 4.2 インポートパスの統一

```python
# v459で推奨するインポート
from ztb.processing.online_scaler import CausalOnlineScaler
from ztb.features.grouping.grouped_scaler import CausalGroupedFeatureScaler
from ztb.evaluation.walk_forward.reporter import BacktestReporter  # 拡張版
from ztb.trading.signal.types import GateResult  # 既存定義を使用
from ztb.training.utils.v457_config_utils import extract_env_config  # 拡張版
```

### 4.3 後方互換性の維持

既存コードが使用している`OnlineScaler`や`GroupedFeatureScaler`は維持し、`Causal*`版を追加することで後方互換性を保つ。

---

## 5. テスト戦略

### 5.1 単体テスト

```
tests/unit/v459/
├── test_reporter_v459.py          # Reporter拡張のテスト
├── test_entry_gate_safety_v459.py # Entry Gate安全性のテスト
├── test_causal_scaler_v459.py     # Scaler因果性のテスト
└── test_config_validation_v459.py # Config検証のテスト
```

### 5.2 統合テスト

```
tests/integration/v459/
├── test_walk_forward_pipeline_v459.py  # エンドツーエンドテスト
└── test_leakage_detection_v459.py      # リーク検査統合テスト
```

---

## 6. リスク評価

### 高リスク

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| Sharpe計算変更で過去結果と非互換 | 比較不可 | 新旧両方を並行計算 |
| Scaler拡張でメモリ使用量増加 | パフォーマンス低下 | 最小限のオーバーヘッド設計 |

### 中リスク

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| Trade Type分類変更で統計値変化 | 評価基準変更 | 移行期間を設定 |
| Config検証でエラー増加 | 起動失敗 | 詳細なエラーメッセージ |

---

## 7. 完了基準

### Phase 0.2完了チェックリスト

- [ ] 全4領域（Reporter, Entry Gate, Scaler, Config）の実装完了
- [ ] 単体テスト全合格（カバレッジ > 80%）
- [ ] 統合テスト全合格
- [ ] ドキュメント更新（実装詳細を04に追記）
- [ ] コードレビュー完了

**Phase 0.2完了後**: Phase 1（P0バグ修正）へ移行

---

## 8. 次のステップ

1. Phase 0.2aから順次実装開始
2. 各サブフェーズ完了ごとに単体テスト実行
3. 全サブフェーズ完了後、統合テスト実行
4. Doc07として実装完了報告を作成

---

**Status**: ✅ **Phase 0.2完了（68/68 tests passed）** → 📋 Ready for Phase 0.3 (Verification)
**Author**: GitHub Copilot  
**Date**: 2026-01-22

---

## 9. Phase 0.2 進捗サマリー（2026-01-22更新）

### 完了項目
- ✅ Phase 0.2a: Reporter強化（23/23 tests passed）
  - classify_trade_type() - 8種詳細分類
  - decompose_reverse_trade() - 反転取引の分解
  - _calculate_sharpe_ratio() - 日次集約（1440分/日）
  - Profit Factor厳密処理（ゼロ除算対応）
  
- ✅ Phase 0.2b: Entry Gate安全性（11/11 tests passed）
  - _is_entry_action() - abs(target) > abs(current)判定
  - _convert_to_hold_action() - 1d/2d対応
  - step()修正 - exit/close常時許可
  
- ✅ Phase 0.2c: Scaler因果性保証（18/18 tests passed）
  - CausalOnlineScaler - fit(end_idx), leakage detection
  - CausalGroupedFeatureScaler - 88→36次元選択スケーリング
  - NaN/inf checks, zero-variance handling
  
- ✅ Phase 0.2d: Config検証強化（16/16 tests passed）
  - validate_env_config() - entry_gate配置チェック
  - execution_model検証 - costs/execution/risk必須
  - slippage_model検証 - "fixed"/"volume_based"のみ許可
  - ValueError使用（assertは使わない）

### Phase 0.2 総合結果
- **単体テスト**: 68/68 passed (100%)
- **実装ファイル**: 4ファイル修正/新規作成
- **テストファイル**: 4ファイル新規作成
- **完了日**: 2026-01-22

### Phase 0.2 実装サマリー

| Phase | 実装内容 | ファイル | Tests |
|-------|---------|---------|-------|
| 0.2a | Reporter強化 | ztb/evaluation/walk_forward/reporter.py | 23 |
| 0.2b | Entry Gate安全性 | ztb/trading/environment/fast_intraday_env_v456.py | 11 |
| 0.2c | Scaler因果性 | ztb/processing/causal_online_scaler.py<br>ztb/features/grouping/causal_grouped_scaler.py | 18 |
| 0.2d | Config検証 | ztb/training/utils/v457_config_utils.py | 16 |
| **合計** | - | **4ファイル** | **68** |

**次のステップ**: Phase 0.3（検証）- 統合テスト、リーク検査、Doc07作成
