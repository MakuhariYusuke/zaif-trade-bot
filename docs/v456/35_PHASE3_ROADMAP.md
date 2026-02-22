# Phase 3 実装ロードマップ: OOS評価パイプライン

**目的**: 訓練データ外での性能を体系的に検証  
**期間**: 2-3日 (推定)  
**成果物**: OOS評価フレームワーク + 評価レポート

---

## 🎯 Phase 3 の目標

### 1. Time-Series Split の実装
- **目的**: 訓練データ / 検証データ / テストデータの分割
- **方法**: Time-series に対応した分割 (forward-looking bias を防止)
- **比率**: 70% train / 15% val / 15% test

### 2. Embargo Period の実装
- **目的**: 訓練データが評価データに漏洩するのを防止
- **方法**: 訓練終了後 7日間のデータを評価対象から除外
- **効果**: Forward-looking bias を完全に排除

### 3. Walk-Forward Validation の実装
- **目的**: 複数の時間帯での性能を評価
- **方法**: 90日間の訓練ウィンドウで順次実行
- **期間**: 訓練: 90日 / テスト: 30日
- **ウィンドウ数**: 6-8回 (rolling)

### 4. Rule-Based Baseline の実装
- **目的**: RL モデルの相対的な性能を評価
- **ロジック**:
  - RSI > 70 → SELL (過熱状態)
  - RSI < 30 → BUY (過売状態)
  - MACD Crossover → BUY/SELL シグナル
- **期待**: RL > Baseline であることを確認

### 5. 統計的検定の実装
- **指標**:
  - Win Rate (勝率)
  - Sharpe Ratio (リスク調整後リターン)
  - Maximum Drawdown (最大ドローダウン)
  - Calmar Ratio (ドローダウン調整後リターン)
- **検定**: Paired t-test (RL vs Baseline)

---

## 📋 実装ステップ

### Step 1: Time-Series Split

```python
# ztb/training/time_series_split.py

class TimeSeriesSplitter:
    """時系列データの分割"""
    
    def __init__(self, df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
        self.df = df
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
    
    def split(self):
        """訓練/検証/テストに分割"""
        n = len(self.df)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        
        train_df = self.df.iloc[:train_end]
        val_df = self.df.iloc[train_end:val_end]
        test_df = self.df.iloc[val_end:]
        
        return train_df, val_df, test_df
```

**出力**:
```
訓練: 2024-01-01 ~ 2024-06-01 (18,900行)
検証: 2024-06-02 ~ 2024-08-01 (4,050行)
テスト: 2024-08-02 ~ 2024-09-30 (4,050行)
```

---

### Step 2: Embargo Period

```python
# ztb/training/embargo_period.py

class EmbargoPeriod:
    """Forward-looking bias 防止"""
    
    @staticmethod
    def apply_embargo(
        test_df: pd.DataFrame,
        embargo_days: int = 7
    ) -> pd.DataFrame:
        """訓練終了後 N 日を除外"""
        # 訓練ウィンドウの終了時刻を取得
        train_end_time = train_df.index[-1]
        
        # Embargo 期間を計算
        embargo_start = train_end_time
        embargo_end = train_end_time + pd.Timedelta(days=embargo_days)
        
        # テストデータから Embargo 期間を除外
        test_after_embargo = test_df[test_df.index > embargo_end]
        
        return test_after_embargo
```

**効果**:
```
訓練ウィンドウ: 2024-01-01 ~ 2024-06-01
Embargo期間:   2024-06-02 ~ 2024-06-09 (除外)
テスト対象:    2024-06-10 ~ 2024-08-01 (評価対象)
```

---

### Step 3: Walk-Forward Validation

```python
# ztb/training/walk_forward_validator.py

class WalkForwardValidator:
    """Rolling window validation"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        train_window_days: int = 90,
        test_window_days: int = 30,
        embargo_days: int = 7,
    ):
        self.df = df
        self.train_window = pd.Timedelta(days=train_window_days)
        self.test_window = pd.Timedelta(days=test_window_days)
        self.embargo_period = pd.Timedelta(days=embargo_days)
    
    def generate_windows(self):
        """訓練/テストウィンドウを順次生成"""
        start_idx = 0
        
        while True:
            # 訓練ウィンドウを取得
            train_start = self.df.index[start_idx]
            train_end = train_start + self.train_window
            
            # テストウィンドウを取得 (embargo後)
            test_start = train_end + self.embargo_period
            test_end = test_start + self.test_window
            
            # データが足りなければ終了
            if test_end >= self.df.index[-1]:
                break
            
            train_df = self.df[(self.df.index >= train_start) & 
                             (self.df.index < train_end)]
            test_df = self.df[(self.df.index >= test_start) & 
                            (self.df.index < test_end)]
            
            yield train_df, test_df
            
            # ウィンドウをずらす
            start_idx += len(train_df) // 2  # 50% overlap
```

**実行例**:
```
Fold 1: Train 2024-01-01~06-01 | Embargo 06-02~09 | Test 06-10~08-01
Fold 2: Train 2024-02-15~07-16 | Embargo 07-17~24 | Test 07-25~09-15
Fold 3: Train 2024-04-01~08-31 | Embargo 09-01~08 | Test 09-09~10-31
```

---

### Step 4: Rule-Based Baseline

```python
# ztb/training/baseline_strategy.py

class BaselineStrategy:
    """RSI/MACD ベース戦略"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.rsi = self._calculate_rsi()
        self.macd = self._calculate_macd()
    
    def generate_signals(self):
        """BUY/SELL シグナルを生成"""
        signals = np.zeros(len(self.df))
        
        # RSI シグナル
        signals[self.rsi > 70] = -1  # SELL
        signals[self.rsi < 30] = 1   # BUY
        
        # MACD シグナル
        macd_line = self.macd['macd']
        signal_line = self.macd['signal']
        
        signals[(macd_line > signal_line) & (signals == 0)] = 1   # BUY
        signals[(macd_line < signal_line) & (signals == 0)] = -1  # SELL
        
        return signals
    
    def backtest(self):
        """バックテストを実行"""
        signals = self.generate_signals()
        
        # ポジション計算
        positions = np.zeros(len(self.df))
        for i in range(len(signals)):
            if signals[i] == 1:
                positions[i] = 0.01  # 1% ロング
            elif signals[i] == -1:
                positions[i] = -0.01  # 1% ショート
            else:
                positions[i] = positions[i-1] if i > 0 else 0
        
        # PnL 計算
        returns = self.df['close'].pct_change()
        pnl = positions * returns
        
        return {
            'cumulative_pnl': pnl.cumsum(),
            'total_return': pnl.sum(),
            'win_rate': (pnl > 0).sum() / len(pnl),
        }
```

---

### Step 5: 統計的検定

```python
# ztb/training/statistical_validator.py

class StatisticalValidator:
    """統計的な性能評価"""
    
    @staticmethod
    def calculate_metrics(returns: np.ndarray):
        """性能指標を計算"""
        
        # Win Rate
        win_rate = (returns > 0).sum() / len(returns)
        
        # Sharpe Ratio (年率化, 252営業日)
        excess_return = returns - 0  # リスクフリーレート=0
        sharpe_ratio = np.sqrt(252) * excess_return.mean() / excess_return.std()
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        annual_return = returns.mean() * 252
        calmar_ratio = annual_return / abs(max_drawdown)
        
        return {
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'mean_return': returns.mean(),
            'std_return': returns.std(),
        }
    
    @staticmethod
    def paired_t_test(rl_returns, baseline_returns):
        """RL vs Baseline の有意性検定"""
        from scipy import stats
        
        t_stat, p_value = stats.ttest_rel(rl_returns, baseline_returns)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
        }
```

---

## 📊 期待される出力

### 評価レポート例

```
================================================================================
OOS Evaluation Report: SAC v456 vs Baseline
================================================================================

Dataset: 2024-01-01 to 2024-09-30 (40,000 bars)
Train/Test Split: 70% / 30% with 7-day embargo

================================================================================
SUMMARY METRICS (全期間)
================================================================================
                    SAC v456        Baseline        Difference
Win Rate            28.5%           22.3%           +6.2%
Sharpe Ratio         0.85            0.45           +0.40
Max Drawdown        -12.3%          -18.5%          +6.2%
Calmar Ratio         1.2             0.6            +0.6

================================================================================
WALK-FORWARD RESULTS (6 Folds)
================================================================================
Fold  Train Period         Test Period          Win Rate  Sharpe  PnL
────  ────────────────────  ───────────────────  ────────  ───────  ─────────
1     2024-01-01~04-01     2024-04-09~06-01     25.0%     0.72    +25,000
2     2024-02-15~05-16     2024-05-24~07-15     30.2%     0.95    +35,000
3     2024-04-01~06-31     2024-07-09~09-05     32.1%     1.05    +42,000
4     2024-05-15~08-15     2024-08-23~10-20     26.8%     0.65    +18,000
────────────────────────────────────────────────────────────────────────────
Mean (±Std)        28.5% (±2.5%)  0.85 (±0.15)  +30,000 (±10,000)

================================================================================
STATISTICAL SIGNIFICANCE
================================================================================
Paired t-test (SAC vs Baseline):
  t-statistic: 2.34
  p-value: 0.032
  Result: ✓ SAC significantly better than baseline (p < 0.05)

================================================================================
CONCLUSION
================================================================================
SAC v456 demonstrates superior performance to rule-based baseline across
multiple time periods. The model shows consistent positive returns with
acceptable drawdown levels.

Recommendation: PROCEED TO PHASE 4 (100K timesteps training)

================================================================================
```

---

## 🛠️ 実装チェックリスト

### Time-Series Split
- [ ] TimeSeriesSplitter クラス実装
- [ ] Split 機能テスト
- [ ] データ連続性検証

### Embargo Period
- [ ] EmbargoPeriod クラス実装
- [ ] Forward-looking bias テスト
- [ ] Embargo 期間の検証

### Walk-Forward Validation
- [ ] WalkForwardValidator クラス実装
- [ ] ウィンドウ生成テスト
- [ ] Fold 数の検証

### Rule-Based Baseline
- [ ] BaselineStrategy クラス実装
- [ ] RSI/MACD 計算テスト
- [ ] バックテスト機能テスト

### 統計検定
- [ ] StatisticalValidator クラス実装
- [ ] Sharpe ratio 計算テスト
- [ ] t-test 実装テスト
- [ ] 評価レポート生成

---

## 📝 実装順序 (推奨)

1. **Day 1 AM**: TimeSeriesSplitter + EmbargoPeriod
   - 時系列分割の基礎を実装
   - Forward-looking bias を完全に防止

2. **Day 1 PM**: WalkForwardValidator
   - Rolling window による複数期間評価
   - 6-8 fold の生成を確認

3. **Day 2 AM**: BaselineStrategy
   - RSI/MACD ベース戦略
   - バックテスト機能

4. **Day 2 PM**: StatisticalValidator + レポート生成
   - 統計的検定
   - 最終評価レポート

5. **Day 3**: 統合テストと最適化
   - 全フェーズの統合実行
   - パフォーマンス検証

---

## 🎯 成功基準

### 最低条件
- ✅ SAC Win Rate >= 25%
- ✅ SAC vs Baseline で有意差あり (p < 0.05)
- ✅ Maximum Drawdown <= 20%

### 理想条件
- ✅ SAC Win Rate >= 35%
- ✅ Sharpe Ratio >= 0.8
- ✅ Calmar Ratio >= 1.0

---

## 🚀 次フェーズ (Phase 4)

OOS評価で成功基準を達成した場合、Phase 4 へ進行：

```bash
python scripts/v456/train_mlp_v456_phase2_complete.py --timesteps 100000
```

**期待される改善**:
- Win Rate: 28% → 35%+
- Sharpe Ratio: 0.85 → 1.2+
- Avg PnL: +30,000 → +50,000+ JPY

---

**次のステップ**: Phase 3 の実装に進行

