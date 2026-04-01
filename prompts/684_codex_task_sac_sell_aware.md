# Codex Task: SAC Sell-Aware Reward & Observation 改善 (684# Phase A2)

## 目的
SAC sidecar の sell 側損失抑制能力を強化する。4/1 データで sell PnL=-1.68 bps が全損失源であるにもかかわらず、SAC の observation space に sell risk 指標が欠如しており、reward も side-neutral であるため sell の有毒性を学習できていない。

## 背景

### 現状
- SAC sidecar confidence ≈ 0.06 → offset 寄与 max_boost_bps × 0.06 = **0.012 bps**（実質ゼロ）
- Observation space: 17 features（price_velocity, spread_bps, vpin, orderbook_imbalance 等）
- Reward: `pnl * reward_scaling / clip_value`（side 非依存）
- γ=0.95, gradient_steps=2, incremental_timesteps=25000
- 4/1 sell データ: trend_5s>0 の sell で PnL=-3.25 bps, trend_5s>0+obi>0.1 で PnL=-6.01 bps

### 問題の根幹
1. **Observation gap**: `mid_price_trend_5s` が feature に含まれず、5s 方向性リスクを認知できない
2. **Reward blindness**: sell AS fill と buy AS fill に同じ reward → sell 特有の損失パターンの学習不足
3. **Confidence scaling**: `confidence_roi_full=0.002` だと実 ROI が小さい環境ではほぼ 0 に張り付く

## タスク

### Task 1: Observation Space 拡張

**対象ファイル**: `ztb/ml/feature_registry.py` および関連する feature 計算コード

1. `mid_price_trend_5s` が FeatureRegistry に存在するか確認
   - 存在する場合: `configs/v460/experiments/g2_sac_train.yaml` の `features.selected` に追加
   - 存在しない場合: FeatureRegistry に新規登録
     - 定義: 直近 5 秒の mid price 変化率 (bps)
     - 計算: `(mid_price_now - mid_price_5s_ago) / mid_price_5s_ago * 10000`
     - NaN 対策: 5s 分のデータがない場合は 0.0

2. `signed_obi` = `orderbook_imbalance × position_sign` を新規登録
   - position_sign: buy=+1, sell=-1（ポジション方向に合わせた OBI）
   - sell で OBI>0（買い圧力）→ signed_obi < 0 → 「逆方向圧力あり」を表現

3. parquet 生成パイプライン（`scripts/v460/ml/` 配下の feature 生成スクリプト）にも新特徴量を追加

### Task 2: Sell-Side Reward Penalty

**対象ファイル**: `ztb/trading/environment/components/calculators/reward_calculator.py`

`use_simple_reward=true` のパスに sell-conditional penalty を追加:

```python
# RewardSettings に追加
sell_as_penalty_mult: float = 1.5  # sell + adverse_selected 時の罰増幅

# simple reward 計算部に分岐追加
if side == "sell" and adverse_selected:
    reward *= self.settings.sell_as_penalty_mult
```

**注意点**:
- `adverse_selected` フラグの取得方法を確認（FillResult or StepInfo 経由）
- 既存の reward clipping との順序: penalty 適用 → clip の順
- PPO trainer や SkipGate 学習に影響しないことを確認（SAC 専用パス）

### Task 3: Confidence Scaling テスト

**対象ファイル**: `scripts/v460/ml/sac_retrain_scheduler.py`

現状 `confidence_roi_full: 0.002` の場合、ROI が低いと confidence がほぼ 0 に張り付く。

1. `confidence_roi_full: 0.001` への引下げをテスト
   - テストで ROI=0.001 時に confidence=1.0 を出力することを確認
   - ROI=0.0005 時に confidence=0.5 を出力することを確認
2. YAML 変更のみ（コード変更は不要のはず）。コード内のスケーリングロジックを読んで確信した上で YAML を更新

### Task 4: テスト作成

**新規作成**: `tests/unit/v460/test_sac_sell_aware_reward.py`

```python
class TestSellAwareReward:
    def test_sell_as_penalty_applied(self):
        """sell + adverse_selected → reward が sell_as_penalty_mult 倍"""
    
    def test_buy_as_penalty_not_applied(self):
        """buy + adverse_selected → penalty mult は適用されない"""
    
    def test_sell_no_as_no_penalty(self):
        """sell + NOT adverse_selected → penalty mult は適用されない"""
    
    def test_penalty_before_clip(self):
        """penalty 適用後に clip される（clip → penalty ではない）"""

class TestNewFeatures:
    def test_mid_price_trend_5s_registered(self):
        """FeatureRegistry に mid_price_trend_5s が登録されている"""
    
    def test_signed_obi_registered(self):
        """FeatureRegistry に signed_obi が登録されている"""
    
    def test_signed_obi_sell_positive_obi(self):
        """sell + OBI>0 → signed_obi < 0"""
    
    def test_signed_obi_buy_positive_obi(self):
        """buy + OBI>0 → signed_obi > 0"""
    
    def test_sac_train_yaml_has_new_features(self):
        """g2_sac_train.yaml の features.selected に新特徴量が含まれる"""
```

**既存テスト実行**:
```bash
python -m pytest tests/unit/v460/test_sac_retrain_scheduler.py -x --tb=short
python -m pytest tests/ -x --tb=short
```

## 制約
- `git commit --no-verify -m "684# SAC sell-aware reward & observation"` でコミット
- `git add .` 禁止。対象ファイルを個別指定
- Any 型禁止、mypy 準拠
- 既存 PPO / SkipGate テストを壊さない
- parquet 生成スクリプトも更新する（新特徴量が訓練データに含まれるように）

## 成果物
1. FeatureRegistry に `mid_price_trend_5s` と `signed_obi` を追加
2. `reward_calculator.py` に sell AS penalty を追加
3. `g2_sac_train.yaml` に新特徴量と `confidence_roi_full: 0.001` を反映
4. テストファイル `test_sac_sell_aware_reward.py` (全 pass)
5. 全既存テスト pass 確認
