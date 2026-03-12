# v456 Phase 1→2 移行時の潜在的問題調査レポート

**実行日**: 2026年1月14日  
**目的**: Phase 2進行前に、コードレビューで指摘された潜在的な問題を systematic に確認

---

## 🔍 問題スキャン結果

### 1. アクション変換の複数パス（★重要）

#### 発見内容
コード内にアクション変換のロジックが複数存在：

**パス A: environment/constants.py**
```python
continuous_to_discrete_action():
  action > 0.3333  → BUY (1)
  action < -0.3333 → SELL (-1)
  その他         → HOLD (0)
```

**パス B: trading/constants.py**
```python
normalize_action():
  action >= SAC_CONTINUOUS_THRESHOLD (0.3333) → BUY
  action <= SAC_CONTINUOUS_THRESHOLD_NEG (-0.3333) → SELL
  その他 → HOLD
```

**パス C: backtest/sac_strategy.py**
```python
_convert_action_to_signal():
  action > 0.3 → BUY (異なる閾値!)
  action < -0.3 → SELL
  その他 → HOLD
```

**パス D: scripts/analysis/phase2_real_market_backtest.py**
```python
run_backtest():
  action > 0.05 → BUY (さらに異なる!)
  action < -0.05 → SELL
  その他 → HOLD
```

#### リスク
- **パス不一致**: SAC が 0.3 を出力した場合、A/Bでは HOLD、C/D では BUY になる
- **デバッグ困難**: どの変換が使用されているか不明確
- **本番リスク**: train と live で異なる変換が使用される可能性

#### 推奨対応（Phase 2）
```python
# Single Source of Truth
class ActionConverterV456:
    CONTINUOUS_BUY_THRESHOLD = 0.3333
    CONTINUOUS_SELL_THRESHOLD = -0.3333
    
    @staticmethod
    def continuous_to_discrete(action: float) -> int:
        """統一された変換ロジック"""
        action = float(np.clip(action, -1.0, 1.0))
        if action > ActionConverterV456.CONTINUOUS_BUY_THRESHOLD:
            return ACTION_BUY
        elif action < ActionConverterV456.CONTINUOUS_SELL_THRESHOLD:
            return ACTION_SELL
        else:
            return ACTION_HOLD
```

---

### 2. SafeIntradayEnvWrapper との差異（★重要）

#### 発見内容
train_mlp_v456_fixed.py で `SafeIntradayEnvWrapper` が使用されている：

```python
class SafeIntradayEnvWrapper(gym.Env):
    def __init__(
        self,
        base_env: FastIntradayEnvV456,
        warmup_steps: int = 10,
        initial_drawdown_limit: float = 0.5,  # 初期は 0.5
        final_drawdown_limit: float = 0.3,    # 最終は 0.3
    ):
```

#### 問題
- **訓練時**: drawdown_limit が段階的に 0.5 → 0.3 に変更
- **評価時**: 常に 0.3 で固定（Wrapper なし）
- **本番時**: さらに別の値？

#### train/eval/live パリティの崩壊
```
訓練: 0.5 (初期) → 0.3 (中期)
評価: 0.3 (固定)
本番: ??? (不明)
```

#### 推奨対応（Phase 2）
- Wrapper を除去（Phase 1修正で必要なし）
- TrainingConfig/EvaluationConfig で統一

---

### 3. アクションスケーリングの謎（★中程度）

#### 指摘内容（レビュー）
> Actionスケールの異常疑い：`pos=-100.000` 等の記録がある

#### 調査結果
FastIntradayEnvV456.step() で：

```python
target_pos_fraction = float(np.clip(action[0], -1.0, 1.0))  # [-1, 1]
raw_target_position = target_pos_fraction * self.max_position  # max_position=0.01 BTC
```

**期待値**: raw_target_position は [-0.01, 0.01] (BTC単位)

**報告の `pos=-100.000`**: 
- おそらく JPY 単位で報告されていた可能性
- BTC を JPY に変換時に 100,000 × (-0.01) = -1,000 JPY ではなく
- 誤ってスケーリング倍数を掛けてしまったか？

#### 推奨対応
action_scaling_validator.py で動作確認（Phase 3）

---

### 4. MTF特徴量の計算（★未実装）

#### 発見内容
Phase 1.2「MTF特徴量の計算実装」はまだ TODO：

```python
for col in mtf_cols:
    if col not in df.columns:
        raise ValueError(...)  # ← 欠損を検出するだけ
        # 計算はしていない
```

#### 影響
- 現在、実データ利用時に MTF特徴量が存在しないため ValueError が発生
- train_mlp_v456_fixed.py はダミーデータで動作（ランダム特徴量なし）
- 本実データでの訓練ができない

#### 優先度
**P1 - Phase 2早期に対応必須**

#### 実装内容（案）
```python
class MTFFeatureCalculator:
    @staticmethod
    def calculate_mtf_features(df: pd.DataFrame, base_price: str = 'close') -> pd.DataFrame:
        """
        OHLCV から MTF（Multi-Timeframe）特徴量を計算
        
        5m / 15m / 1h への change indicators
        - RSI, MACD, ATR など
        """
```

---

### 5. 1分足がHFT向けではない（★設計問題）

#### 指摘内容（レビュー）
> 1分足はHFTではない。実質は「短期イントラデイ」

#### 現状
- Project name: "zaif-trade-bot" (HFT志向)
- 実装: 1分足 OHLCV (イントラデイ)
- max_steps: 500 (約8時間)

#### 推奨対応
ドキュメント更新のみで OK：
- "HFT" ではなく "Short Intraday Trading" に改名
- 目的の再定義

---

### 6. 訓練ステップ不足（★既知）

#### 指摘内容（レビュー）
> 30K timesteps は 88D 特徴量 + SAC には不足

#### Phase 1修正後の影響
- Phase 1により、ランダム特徴量が除去された
- 訓練開始時点で学習が成立する条件が整った
- 30K → 100K への増加が有効になる（前は無駄だった）

#### Phase 4推奨
100K timesteps での再訓練を実行

---

## 📋 潜在的問題の優先度一覧

| # | 問題 | 影響度 | 対応予定 | 工数 |
|---|------|--------|---------|------|
| 1 | アクション変換パス | ★★★★★ | Phase 2 | 中 |
| 2 | Wrapper/設定不一致 | ★★★★☆ | Phase 2 | 低 |
| 3 | アクションスケーリング | ★★★☆☆ | Phase 3 | 低 |
| 4 | MTF特徴量未計算 | ★★★★★ | Phase 2早期 | 高 |
| 5 | 1分足/HFT名称 | ★★☆☆☆ | ドキュメント | 低 |
| 6 | 訓練ステップ不足 | ★★★☆☆ | Phase 4 | 低 |

---

## ✅ Phase 2推奨アクションプラン

### 優先順位 1: アクション変換の統一化
- **タスク**: ActionConverterV456 を新規作成
- **ファイル**: `ztb/training/action_converter_v456.py`
- **変更**: 全アクション変換パスをこのクラスに統一
- **工数**: 2-3 hours
- **ブロッカー**: False（ただし精度に影響）

### 優先順位 2: MTF特徴量の計算実装
- **タスク**: feature_calculator_v456.py を完成させる
- **ファイル**: `scripts/v456/feature_calculator_v456.py`
- **内容**: OHLCV → RSI/MACD/ATR/BB などの計算
- **工数**: 3-4 hours
- **ブロッカー**: True（実データ訓練に必須）

### 優先順位 3: SafeIntradayEnvWrapper の除去
- **タスク**: Wrapper を削除し、TrainingConfig で管理
- **ファイル**: `train_mlp_v456_fixed.py`
- **工数**: 0.5 hours
- **ブロッカー**: False（既に Phase 1修正で対応済み）

### 優先順位 4: OOS評価パイプライン
- **タスク**: walk-forward validation 実装
- **ファイル**: `scripts/v456/validate_walkforward.py`
- **工数**: 2-3 hours
- **ブロッカー**: False（ただし評価有効性に必須）

---

## 次フェーズへの条件

### Go / No-Go 判定

**GO: Phase 2へ進行可能** ✅
- Phase 1修正が完了
- 潜在的な問題が顕在化した（対応可能）
- ブロッカーなし

**進行条件**:
1. ✅ Phase 1修正完了
2. ⚠️ MTF特徴量計算を優先実装（1-2日）
3. ✅ アクション変換の統一化（並行実装可能）

---

**推奨**: **MTF特徴量計算 + アクション統一化 を Phase 2 Day 1で実装。その後 OOS 評価に進む。**

