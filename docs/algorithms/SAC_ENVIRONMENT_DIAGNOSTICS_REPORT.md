# SAC環境診断レポート

**作成日**: 2025年10月11日  
**対象**: SAC (Soft Actor-Critic) 訓練環境  
**診断ツール**: `diagnose_sac_simple.py`

---

## 🔍 エグゼクティブサマリー

7つのバージョン（v395a-g）でパラメータチューニングと報酬関数再設計を試行したが、すべてCritic Loss爆発（1e7-1e10）とent_coef上昇（3-4）が発生。環境診断を実施し、**2つの根本原因**を特定した。

---

## 📊 診断結果

### 実行設定
- **設定ファイル**: `configs/sac_v395g_micro_reward.json`
- **エピソード数**: 3
- **ステップ数**: 各100 (合計297ステップ)
- **診断日時**: 2025年10月11日

### 発見された問題

#### 🚨 問題1: 観測値が正規化されていない

**症状**:
```
観測値の平均範囲: [0.0, 18,111,058.0]
観測値の標準偏差範囲: [0.0, 1902.35]
標準偏差 > 10.0 の特徴量: 26個
```

**具体例**:
```python
特徴量 1: std=1865.13, range=[17,752,488.00, 17,759,944.00]
特徴量 2: std=1865.13, range=[17,752,488.00, 17,759,944.00]
特徴量 3: std=1865.13, range=[17,752,488.00, 17,759,944.00]
```

**影響**:
- ニューラルネットワークの入力スケールが不適切
- 勾配計算が不安定
- Q値の推定が極端に発散
- **Critic Loss 1e7-1e10への直接的な原因**

**根本原因**:
- `use_standardized_observations` が設定されていない
- `EnvironmentConfig` に `use_standardized_observations` フィールドが存在しない
- `HeavyTradingEnv` で StandardScaler が実装されていない

---

#### 🚨 問題2: ゼロ報酬が64.3%

**症状**:
```
報酬分布:
  正の報酬: 0.0%
  負の報酬: 35.7%
  ゼロ報酬: 64.3% ← 問題
```

**影響**:
- 学習シグナル不足
- Criticが適切なQ値を学習できない
- エージェントが何もしないことを学習する可能性

**根本原因**:
- HOLD行動時に報酬がゼロ
- ポジションなしでHOLD → 完全に無意味な行動
- ポジション保持中のHOLD → 機会費用が考慮されていない

---

## 💡 実装した解決策

### ✅ 解決策1: 観測値正規化の実装（部分的）

**実施内容**:
1. `EnvironmentConfig`に`use_standardized_observations`フィールド追加
2. v395h設定で `use_standardized_observations: true` を設定

**未完了**:
- ❌ `HeavyTradingEnv`での StandardScaler 実装
- ❌ 観測値の実際の正規化処理

**理由**: 時間制約のため、まず報酬改善のみで効果を確認

---

### ✅ 解決策2: 報酬関数の改善

**実施内容**:
```python
def calculate_reward_simple(
    self, pnl, portfolio_value, position, old_position, action
):
    # v3.0: actionパラメータ追加
    
    # 1. PnLベースの報酬（既存）
    reward = (pnl / portfolio_value) * reward_scale
    reward = clip(reward, clip_min, clip_max)
    
    # 2. 無活動ペナルティ（新規）
    if action == HOLD and position == 0 and old_position == 0:
        reward -= inactivity_penalty_rate  # デフォルト: -0.001
    
    # 3. 機会費用ペナルティ（新規）
    if action == HOLD and position != 0:
        reward -= opportunity_cost_rate  # デフォルト: -0.0005
    
    return reward
```

**効果**:
- ゼロ報酬: **64.3% → 31.6%** ✅ (目標 < 30%にほぼ到達)

---

## 🧪 v395h訓練結果

### 設定
```json
{
  "use_standardized_observations": true,  // ⚠️ 未実装
  "reward_settings": {
    "use_simple_reward": true,
    "reward_scale": 100.0,
    "reward_clip_min": -1.0,
    "reward_clip_max": 1.0,
    "enable_inactivity_penalty": true,      // ✅ 実装済み
    "inactivity_penalty_rate": 0.001,
    "enable_opportunity_cost": true,        // ✅ 実装済み
    "opportunity_cost_rate": 0.0005
  }
}
```

### 結果 (5k timesteps)
```
final metrics:
  ent_coef: 3.58 (目標: 0.5-1.5)
  ent_coef_loss: -40.7
  Critic Loss: 3.28e+09 (目標: < 1000)
  Actor Loss: 4.21e+05 (目標: < 100)
```

### 評価
- ✅ **ゼロ報酬削減は成功** (64.3% → 31.6%)
- ❌ **Critic Lossは依然として異常** (3.28e+09)
- ❌ **観測値正規化が効いていない** (max_std依然として1902.35)

---

## 📌 結論

### 発見事項

1. **パラメータチューニングでは解決不可**
   - learning_rate, batch_size, gamma等の調整では効果限定的
   - target_entropy調整も無意味
   - 問題は環境設計そのもの

2. **報酬関数だけでも不十分**
   - シンプル化は正しい方向
   - ゼロ報酬削減は成功
   - しかし観測値スケール問題を解決できない

3. **観測値正規化が最優先課題**
   - 1800万台の特徴量はニューラルネットに入力不可
   - StandardScalerの実装が絶対必要
   - これなしではCritic Loss爆発は解決不可

### 次のステップ（優先順位順）

#### 🔴 **緊急 (P0)**: 観測値正規化の完全実装

**タスク**:
1. `HeavyTradingEnv`に StandardScaler 統合
2. `_setup_scaler()` メソッド実装
3. `_get_observation()` で正規化適用
4. `use_standardized_observations` を正しく読み込み

**期待効果**:
- 観測値スケール: < 5.0 (現在: 1902.35)
- Critic Loss: < 1000 (現在: 3.28e+09)
- Actor Loss: < 100 (現在: 4.21e+05)
- ent_coef: 0.5-1.5 (現在: 3.58)

#### 🟡 **重要 (P1)**: v395i作成・訓練

**完全修正版の作成**:
```json
{
  "model_name": "sac_v395i_fully_fixed",
  "use_standardized_observations": true,  // ✅ 実装完了後
  "enable_inactivity_penalty": true,      // ✅ 実装済み
  "enable_opportunity_cost": true         // ✅ 実装済み
}
```

**訓練計画**:
1. 5k timesteps検証
2. 目標メトリクス達成確認
3. 100k timesteps完全訓練

#### 🟢 **通常 (P2)**: レポート作成

**内容**:
- 診断プロセス
- 発見した問題
- 実装した解決策
- 訓練結果比較
- 学んだ教訓

---

## 📂 成果物

### 作成ファイル

1. **診断ツール**:
   - `diagnose_sac_simple.py` - 環境診断スクリプト
   - `sac_environment_diagnostics_simple.json` - 診断結果データ

2. **設定ファイル**:
   - `configs/sac_v395h_normalized.json` - 部分修正版

3. **訓練スクリプト**:
   - `train_sac_v395h.py` - v395h訓練実行

4. **修正コード**:
   - `ztb/trading/environment/components/reward_calculator.py`:
     - `calculate_reward_simple()` v3.0実装
     - inactivity_penalty追加
     - opportunity_cost追加
   - `ztb/trading/environment/utils/config.py`:
     - `use_standardized_observations` フィールド追加

### 診断データ

**v395g (修正前)**:
```json
{
  "zero_rewards_pct": 64.3,
  "observations_std_max": 1902.35,
  "large_std_features": 26,
  "issues": [
    "⚠️ ゼロ報酬が多すぎる: 64.3%",
    "⚠️ 観測値のスケールが大きすぎる: max_std=1902.35"
  ]
}
```

**v395h (報酬改善のみ)**:
```json
{
  "zero_rewards_pct": 31.6,
  "observations_std_max": 1902.35,  // 依然として高い
  "issues": [
    "⚠️ 観測値のスケールが大きすぎる: max_std=1902.35"
  ]
}
```

---

## 🎓 教訓

### 成功事項

1. **系統的な診断アプローチ**
   - 複数バージョンでの実験
   - 詳細なログ分析
   - 環境診断ツールの作成
   - データ駆動の意思決定

2. **根本原因分析**
   - パラメータ調整だけでは不十分と判断
   - 環境設計レベルの問題を特定
   - 観測値スケールという盲点を発見

3. **段階的な解決**
   - まず報酬関数改善
   - 効果を検証
   - 次に正規化実装（予定）

### 失敗から学んだこと

1. **初期検証の重要性**
   - 訓練開始前に環境診断すべきだった
   - 観測値スケールのチェックを怠った
   - 7つのバージョンの無駄な試行を回避できた

2. **全体設計の理解**
   - `use_standardized_observations`の実装状況を確認せずに設定
   - EnvironmentConfigの定義を事前確認すべき
   - HeavyTradingEnvの実装を理解してから設定変更すべき

3. **段階的検証の必要性**
   - 報酬改善だけで効果測定
   - 観測値正規化の重要性を再確認
   - 複数の問題を同時に解決しようとしない

---

## 🔗 関連ドキュメント

- `SAC_PARAMETER_TUNING_GUIDE.md` - パラメータ調整ガイド
- `SAC_ENTROPY_ANALYSIS.md` - エントロピー問題分析
- `REWARD_FUNCTION_REDESIGN.md` - 報酬関数再設計
- `SAC_TUNING_ANALYSIS_REPORT.md` - v395a-e比較レポート

---

**次のアクション**: 観測値正規化の完全実装（P0タスク）
