# v456 Phase-Based Implementation Roadmap

**Status**: Active  
**Date**: 2026-01-14  
**Based on**: Code Review Response (26_code_review_response.md)  
**Target**: Restore learning capability + establish evaluation validity

---

## Executive Summary

3つの致命的な問題を段階的に修正：
1. **ランダム特徴量** → 学習信号完全崩壊
2. **reward/balance 混在** → 環境設計不在
3. **Train/Eval データリーク** → 評価無効化

修正後: **Phase 1完了時点で正常な訓練が可能になる**

---

## Phase 1: Critical Foundation Fixes (This Week)

### 目標
環境の基本動作を修正し、学習信号が有意に存在することを確認

### 1.1 ランダム特徴量を撤廃 → Explicit Error

**ファイル**: `ztb/trading/environment/fast_intraday_env_v456.py`

**変更内容**:
```python
# Before (現在 - 破壊的)
for col in col_list:
    if col not in df.columns:
        df[col] = np.random.randn(len(df))  # ← ランダムノイズ投入

# After (修正版 - 安全)
for col in col_list:
    if col not in df.columns:
        raise ValueError(
            f"Missing feature: {col}. "
            f"Available columns: {df.columns.tolist()}"
        )
```

**影響**: 
- 破壊的デバッグ情報から前進への転換
- 欠損データの場所を明示的に指摘

**実装工数**: 低（5 min）

---

### 1.2 MTF/Regime 特徴量の計算実装

**ファイル**: `scripts/v456/feature_calculator_v456.py` （新規作成）

**責務**:
- OHLCV から実計算で MTF (Multi-Timeframe) 特徴量を生成
- Market Regime を VIX-like volatility から算出
- Base 30次元は既存維持

**特徴量例**:
```
MTF Features (27):
  - RSI(14), MACD, MACD Signal, MACD Hist (4)
  - Bollinger Bands: Upper, Middle, Lower, %B, Bandwidth (5)
  - ATR(14), NATR (2)
  - ADX(14), +DI(14), -DI(14) (3)
  - Volatility (3)
  - Volume Profile (8)

Regime Features (13):
  - Volatility Regime: Low/Mid/High flags (3)
  - Trend Direction: Up/Down/Sideways (3)
  - Volume Regime (3)
  - Support/Resistance Detection (4)
```

**実装工数**: 中（2-3 hours）

**参照**: TA-Lib または talib_wrapper.py の活用

---

### 1.3 reward と balance を分離

**ファイル**: `ztb/trading/environment/fast_intraday_env_v456.py` の `step()` メソッド

**設計変更**:

```python
# 現在（破壊的）
reward = -(fee + slippage)                  # -0.216 × step
balance -= reward                           # 毎 step -0.216
# 結果: 2ステップで drawdown_limit 触れる

# 修正版（分離）
# 1. 実PnLを計算（入場・出場ベース）
if self.position != prev_position:  # ポジション変化あり
    # Entry/Exit の確定損益計算
    pnl = self._calculate_pnl(entry_price, exit_price, quantity)
else:
    # Holding時は未決済（mark-to-market）
    pnl = self._mark_to_market(current_price)

# 2. 資金を PnL で更新（1エピソードに1回または日次）
self.balance += pnl  # 実損益のみ

# 3. 報酬を正規化（学習用）
normalized_reward = pnl / self.initial_balance  # スケーリング
normalized_reward = np.clip(normalized_reward, -0.1, 0.1)  # 正規化範囲
reward = normalized_reward
```

**Key Changes**:
- Fee/Slippage は PnL に含める（balance に直接反映）
- reward は**学習用スケーリング後**の値
- balance は**日次確定**または**トレード確定**で更新（毎ステップではない）

**実装工数**: 中（2 hours）

---

### 1.4 設定統一 → Single Source of Truth

**ファイル**: `ztb/config/environment_config.py` （新規作成）

```python
# environment_config.py
class TrainingConfig:
    INITIAL_BALANCE = 100000  # JPY（学習用スケール）
    MAX_POSITION = 0.01       # BTC
    DRAWDOWN_LIMIT = 0.30     # 30%
    MAX_STEPS = 500
    FEE_RATE = 0.001
    SLIPPAGE_RATE = 0.0005
    
class LiveConfig:
    INITIAL_BALANCE = 1000000  # JPY（実運用）
    MAX_POSITION = 0.1
    DRAWDOWN_LIMIT = 0.10
    # 他は同じ
```

**参照方法**:
```python
from ztb.config import TrainingConfig as CONFIG

env = FastIntradayEnvV456(
    initial_balance=CONFIG.INITIAL_BALANCE,
    max_position=CONFIG.MAX_POSITION,
    drawdown_limit=CONFIG.DRAWDOWN_LIMIT,
    max_steps=CONFIG.MAX_STEPS,
)
```

**実装工数**: 低（1 hour）

---

### Phase 1 テスト

**Smoke Test**: `tests/v456/test_phase1_fixes.py`

```python
def test_missing_feature_raises_error():
    """不足特徴量は ValueError を raise する"""
    df_incomplete = pd.DataFrame({'base_0': [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing feature"):
        env = FastIntradayEnvV456(df=df_incomplete, ...)

def test_reward_balance_separation():
    """reward と balance が独立している"""
    env.reset()
    obs, reward, done, info = env.step(action=0.5)
    
    # reward は [-0.1, 0.1] 範囲で正規化
    assert -0.1 <= reward <= 0.1
    
    # balance は初期値から ±5% 程度の変化（不自然ではない）
    assert 95000 <= env.balance <= 105000

def test_episode_length_varies():
    """エピソード長が不変でない"""
    lengths = []
    for _ in range(10):
        env.reset()
        steps = 0
        done = False
        while not done:
            obs, reward, done, info = env.step(env.action_space.sample())
            steps += 1
            if steps > 1000:
                break
        lengths.append(steps)
    
    # 有意な変動がある
    assert np.std(lengths) > 0
```

**実装工数**: 中（1 hour）

---

### Phase 1 完了の定義

- ✅ ランダム特徴量が ValueError を raise
- ✅ MTF/Regime 特徴量が計算される
- ✅ reward が [-0.1, 0.1] に正規化
- ✅ balance が PnL ベースで更新
- ✅ エピソード長が可変（早期終了が発動）
- ✅ スモークテスト全パス

---

## Phase 2: Evaluation Pipeline Reconstruction (Week 2)

### 目標
Train/Test データリークを除去し、評価結果を有効化

### 2.1 時系列 Split 実装

**ファイル**: `ztb/data/timeseries_split.py` （新規作成）

```python
def time_series_split_with_embargo(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    embargo_days: int = 7
) -> tuple:
    """
    Train / Validation / Test に時系列順で分割
    Embargo: 訓練データ直後 N日間をスキップ（forward-looking bias 防止）
    """
    n_total = len(df)
    n_embargo = embargo_days * 24 * 60  # 1m bar 基準
    
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_end = n_train
    val_start = train_end + n_embargo
    val_end = val_start + n_val
    test_start = val_end + n_embargo
    
    return (
        df.iloc[:train_end],
        df.iloc[val_start:val_end],
        df.iloc[test_start:]
    )
```

**実装工数**: 低（1 hour）

---

### 2.2 Walk-Forward Validation フレームワーク

**ファイル**: `scripts/v456/validate_walkforward.py` （新規作成）

```python
class WalkForwardValidator:
    def __init__(self, df, train_window=90*24*60, test_window=30*24*60):
        self.df = df
        self.train_window = train_window
        self.test_window = test_window
    
    def run(self, model, n_folds=10):
        """90日訓練 / 30日テストを rolling で実行"""
        results = []
        
        for fold in range(n_folds):
            train_start = fold * self.test_window
            train_end = train_start + self.train_window
            test_end = train_end + self.test_window
            
            train_data = self.df.iloc[train_start:train_end]
            test_data = self.df.iloc[train_end:test_end]
            
            # Retrain + Evaluate on OOS
            model.learn(...)
            result = self._evaluate(model, test_data)
            results.append(result)
        
        return pd.DataFrame(results)
```

**実装工数**: 中（2 hours）

---

### 2.3 Evaluation Pipeline 更新

**ファイル**: `scripts/v456/model_evaluation.py` 修正

```python
def evaluate_on_oos_data(model_path, test_data):
    """OOS データでのみ評価"""
    env = FastIntradayEnvV456(df=test_data, ...)
    
    metrics = {
        'pnl': [],
        'sharpe': [],
        'max_dd': [],
        'win_rate': [],
    }
    
    for episode in range(50):
        obs = env.reset()
        done = False
        episode_pnl = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_pnl += info.get('trade_pnl', 0)
        
        metrics['pnl'].append(episode_pnl)
    
    return metrics
```

**実装工数**: 中（1.5 hours）

---

## Phase 3: Baseline & Validation (Week 2 Evening)

### 3.1 Rule-Based Baseline 実装

**ファイル**: `scripts/v456/baseline_strategy.py` （新規作成）

```python
class RSIMACDBaseline:
    """シンプルなルール: RSI + MACD"""
    
    def signal(self, rsi, macd, macd_signal):
        if rsi > 70 and macd < macd_signal:
            return -1.0  # SELL signal
        elif rsi < 30 and macd > macd_signal:
            return 1.0   # BUY signal
        else:
            return 0.0   # HOLD
    
    def evaluate(self, test_data):
        # test_data で実行
        results = ...
        return results
```

**期待値**: RL モデルはこのベースラインを**最低限超える**必要がある

**実装工数**: 低（1 hour）

---

### 3.2 Action Scaling Validator

**ファイル**: `scripts/v456/validate_action_scaling.py` （新規作成）

```python
def validate_action_to_position_mapping():
    """action [-1, 1] が position に正しくマッピングされているか"""
    
    env = FastIntradayEnvV456(...)
    env.reset()
    
    test_actions = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for action in test_actions:
        obs, reward, done, info = env.step(action)
        
        position = info.get('position', 0)
        expected_position = action * env.max_position
        
        assert abs(position - expected_position) < 0.001, \
            f"Action {action} → Position {position} (expected {expected_position})"
```

**実装工数**: 低（0.5 hour）

---

## Phase 4: Re-Training (Week 3)

### 4.1 修正版モデル訓練

**ファイル**: `scripts/v456/train_mlp_v456_phase2.py`

```python
# Phase 1 修正を反映した訓練
model = SAC(
    policy='MlpPolicy',
    env=env,
    learning_rate=3e-4,
    buffer_size=100000,
    learning_starts=1000,
    verbose=1,
    tensorboard_log="logs/v456_phase2"
)

model.learn(total_timesteps=100000)  # 10K → 100K に増加
model.save("models/v456_phase2/sac_corrected")
```

**期待値**: 
- PnL Mean > 0
- Win Rate > 20%
- Sharpe Ratio > -5.0

**実装工数**: 低（スクリプト実行 - 2-3時間の訓練時間）

---

## Implementation Checklist

### Phase 1 (This Week - Critical)
- [ ] 1.1: ランダム特徴量 → ValueError
- [ ] 1.2: MTF/Regime 特徴量 計算実装
- [ ] 1.3: reward/balance 分離
- [ ] 1.4: environment_config.py で統一
- [ ] Test: Phase 1 スモークテスト全パス

### Phase 2 (Week 2 - Validation)
- [ ] 2.1: time_series_split 実装
- [ ] 2.2: WalkForwardValidator 実装
- [ ] 2.3: model_evaluation.py 修正
- [ ] Test: OOS データで評価可能

### Phase 3 (Week 2 Evening - Baseline)
- [ ] 3.1: RSIMACDBaseline 実装
- [ ] 3.2: action_scaling_validator 実装
- [ ] Validate: RL > Baseline 確認

### Phase 4 (Week 3 - Training)
- [ ] 4.1: 修正版モデル訓練実行
- [ ] Validate: PnL > 0 確認
- [ ] Document: 訓練ログ分析

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| MTF計算が遅い | 事前計算してキャッシュ |
| reward スケーリング不適切 | 複数スケール試行 + A/B比較 |
| 訓練がまた失敗 | ベースラインとの差分ログ |

---

## Success Criteria

**Phase 1 完了時点で**:
- ✅ No random features
- ✅ reward in [-0.1, 0.1]
- ✅ balance updates make sense
- ✅ episode_length varies

**Phase 2 完了時点で**:
- ✅ OOS evaluation working
- ✅ Walk-forward validation baseline exists

**Phase 3 完了時点で**:
- ✅ Baseline implemented
- ✅ RL > Baseline on **same OOS data**

**Phase 4 完了時点で**:
- ✅ Retrained model: PnL Mean > 0
- ✅ Win Rate > 20%
- ✅ Reproducible results

---

**Owner**: Development Team  
**Next Milestone**: Phase 1 完了（2026-01-15）
