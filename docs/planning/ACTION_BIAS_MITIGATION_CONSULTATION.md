# アクションバイアス問題と緩和策 - 第三者相談用サマリー

## 背景

強化学習によるBTC/JPY取引ボット開発プロジェクトにおいて、モデルの**アクションバイアス**（特定のアクション(HOLD/BUY/SELL)への偏り）が深刻な問題として浮上しました。

## 問題の発見経緯

### 初期問題: SELL Bias
- **現象**: 訓練データで SELL rate が 2-3% と極端に低い
- **影響**: モデルがSELLアクションを学習できず、ロング戦略のみに最適化
- **原因**: データ生成時のトレーディングロジックがBUY優位

### SELL Bias緩和策の実装 (2025年10月初旬)

4つの手法を実装:

#### 1. Mirror Augmentation (Data-level)
- **概念**: SELLトレードを反転してBUYトレードとして生成
- **実装**: `scripts/mirror_augment.py`
- **効果**: SELL rate 9.1% → 11.1% (+2.0pp)
- **状態**: ✅ 完全動作確認済み

#### 2. Lagrange Constraint (Loss-level)
- **概念**: 目標SELL rateを制約条件として追加
  - L(θ) = L_PPO(θ) - λ * max(0, r_min - r_sell(θ))
- **実装**: `ztb/training/lagrange_constraint.py`
- **状態**: ⚠️ SB3のCallbackAPIでは損失関数修正不可 → 非機能

#### 3. Gradient Probes (Monitoring)
- **概念**: SELL actionの勾配とadvantageを監視、異常時に早期停止
- **実装**: `ztb/training/grad_probes.py`
- **状態**: ⚠️ Callbackからは勾配アクセス不可 → 統計ログのみ

#### 4. Action Weights (Sampling-level)
- **概念**: アクション不均衡の逆数重みを計算
  - w_a = min(1/freq(a), beta)
- **実装**: `ztb/training/weights.py`
- **状態**: ⚠️ 初期化済みだがサンプリングに未適用

### 新たな問題発覚: BUY Bias (2025年10月6日)

統合テストで**予期せぬBUY bias**が発生:

```
デバッグログ: position=1.0 (ロング) が訓練全体の95%以上を占有
SELL rate: 0.0% (目標 15% に対して)
ep_rew_mean: 1410 (大幅プラス = モデルは学習している)
```

**原因仮説**:
1. **curriculum_stage問題は解決済み** (forced_balance → full に修正完了)
2. **データセット自体にBUY bias存在**
   - Mirror augmentation後もSELL 11.1% vs BUY 24.1%
   - 報酬構造がロング優位の可能性
3. **SELL緩和策がBUY優位を助長**
   - Lagrange/Weights/Probesは全てSELL専用実装
   - BUY/HOLD biasに対する制約なし

## 一般化の必要性

### 現状の限界
- **SELL専用設計**: すべての緩和策がSELLアクション特化
- **バランス制御不能**: 全3アクション(HOLD/BUY/SELL)の均衡を保てない
- **新たなバイアス誘発**: SELL対策がBUY biasを生む可能性

### 一般化対応 (本日実施)

#### Lagrange Constraint → Action Balance Constraint
```python
# Before
LagrangeConstraint(r_min=0.15)  # SELL専用

# After
LagrangeConstraint(
    target_action="SELL",  # or "BUY", "HOLD"
    r_target=0.15,
    tolerance=0.05  # ±5%の許容範囲
)
```

- **柔軟性**: 任意のアクションに対して目標レート設定可能
- **tolerance追加**: 厳密な等値制約→範囲制約に改善
- **deviation計算**: |r_target - r_actual| - tolerance

#### Gradient Probe → Action Gradient Probe
```python
# Before
SELLGradientProbe()  # SELL専用

# After
ActionGradientProbe(target_action="SELL")  # 任意アクション対応
# または
ActionGradientProbe(target_action="BUY")   # BUY監視用
ActionGradientProbe(target_action="HOLD")  # HOLD監視用
```

- **マルチアクション監視**: 複数プローブを同時使用可能
- **後方互換性**: `SELLGradientProbe`エイリアスで既存コード維持

## 現在の技術的課題

### 1. SB3 API制限による実装ギャップ

**問題**:
- Stable-Baselines3のCallbackシステムでは損失関数・勾配に直接アクセス不可
- `BaseCallback`はstep/rollout境界でのフックのみ提供

**影響**:
```python
# ❌ 不可能な実装
class SELLBiasMitigationCallback(BaseCallback):
    def _on_rollout_end(self):
        # SB3は loss computation を公開していない
        constrained_loss = original_loss + lagrange_penalty  # 不可能
```

**現状**:
- Lagrange penalty: 計算のみ、損失に未適用
- Gradient probes: 統計ログのみ、実際の勾配未アクセス
- Action weights: 初期化のみ、サンプリングに未適用

**解決策の選択肢**:
1. **カスタムPPO実装** (~500-1000行)
   - MaskablePPOを継承してtrain()メソッドをオーバーライド
   - 完全制御可能だが、SB3バージョンとの結合度高

2. **データレベルのみに集中**
   - Mirror augmentation + より洗練されたルールベース拡張
   - 実装シンプル、既に動作実績あり

3. **ハイブリッド**
   - Mirror augmentation (主力)
   - サンプリングバイアス (補助、実装比較的容易)
   - Post-training validation & retraining

### 2. データセットの根本的バイアス

**ml-dataset-final.csv** (mirror augmentation後):
```
HOLD: 712 (64.7%)
BUY:  265 (24.1%)
SELL: 123 (11.1%)
```

**問題**:
- Mirror後もBUY >>> SELL
- 元データの取引ロジック自体がBUY優位
- トレンドフォロー戦略の性質上、ロング保有期間が長い

**対策オプション**:
1. **追加データ拡張**
   - ベア相場データの追加収集
   - 合成データ生成 (GANなど)

2. **リワード再設計**
   - Action balanceにボーナス付与
   - Sharp ratio最大化 (方向性中立的)

3. **多環境訓練**
   - 複数の市場regime (bull/bear/sideways) でローテーション訓練

### 3. 評価指標の不足

**現在のメトリクス**:
```python
ep_rew_mean: 1410  # 総報酬のみ
SELL rate: 0.0%    # アクション分布のみ
```

**必要なメトリクス**:
- **リスク調整済みリターン**: Sharpe ratio, Sortino ratio
- **ドローダウン**: Maximum drawdown, drawdown duration
- **Action efficiency**: Win rate per action, profit factor
- **Regime stability**: Performance across market conditions

## 質問と相談事項 (第三者エージェント向け)

### Q1: アーキテクチャ選択
現在の3つのオプションについて、プロジェクトの成熟度と目標を考慮した推奨を教えてください:

**コンテキスト**:
- 目標: 実運用可能な取引ボット (Sharpe >0.5, MDD <30%, 各アクション≥15%)
- 現状: PoC段階、既存データセット1100行のみ
- リソース: 個人開発、計算リソース限定的

**オプション評価軸**:
1. 実装コスト vs 効果
2. メンテナンス負荷
3. 技術的リスク
4. Time-to-market

### Q2: バイアス対策の優先順位
複数のアクションバイアス(SELL不足、BUY過多、HOLD偏重)が混在する場合、どのような順序で対処すべきですか?

**検討ポイント**:
- 同時に複数制約を課す vs 段階的適用
- データレベル vs アルゴリズムレベル どちらを優先
- バイアス間の相互作用 (SELL促進 → BUY抑制の副作用)

### Q3: 評価フレームワーク設計
アクションバランスと収益性の両立を評価する指標体系をどう構築すべきですか?

**要件**:
- Action balance (各≥15%) と Sharpe ratio のトレードオフ
- 過学習検出 (train vs validation performance gap)
- Regime robustness (異なる市場環境での安定性)

**提案例**:
```python
composite_score = (
    0.4 * sharpe_ratio +
    0.3 * action_balance_score +  # How to quantify?
    0.2 * (1 - max_drawdown/100) +
    0.1 * regime_stability
)
```

### Q4: データ拡張戦略
Mirror augmentationの次の一手として、何を推奨しますか?

**候補**:
1. **時系列反転** (Time reversal): 未来→過去の順序で再生
2. **ノイズ注入**: Gaussian noise to features
3. **SMOTE for time-series**: Synthetic minority oversampling
4. **GAN-based**: 条件付き生成モデル
5. **Real data collection**: 追加の履歴データ収集

### Q5: プロダクション移行基準
研究フェーズから実運用への移行判断基準は?

**現在の基準案**:
```python
smoke_test = {
    "timesteps": 50_000,
    "seeds": 3,
    "acceptance": {
        "sharpe": ">0",
        "sell_rate": "≥15%",
        "convergence": "loss < threshold"
    }
}

long_paper = {
    "eval_steps": "≥500",
    "criteria": {
        "sharpe": ">0.5",
        "mdd": "<30%",
        "action_balance": "all ≥15%"
    }
}
```

これで十分か、追加すべき基準は?

## 技術スタック

**Framework**:
- RL: Stable-Baselines3 (sb3-contrib MaskablePPO)
- Environment: Gym 0.26
- Data: pandas, numpy
- Features: TA-Lib (technical indicators)

**制約**:
- Offline training (歴史データのみ)
- Single-asset (BTC/JPY)
- Discrete actions (HOLD=0, BUY=1, SELL=2)
- Position limits: [-1, 1] (short/neutral/long)

## 求める視点

1. **機械学習工学的視点**: アルゴリズム選択、ハイパラチューニング
2. **ソフトウェア工学的視点**: アーキテクチャ設計、技術的負債管理
3. **量的取引的視点**: リスク管理、評価指標、実運用考慮
4. **プロジェクト管理的視点**: 優先順位付け、マイルストーン設定

## 添付情報

- コミット履歴: 9079ef2 (mirror), 78eb910 (BC), 3d8ba46 (final summary)
- 主要ファイル:
  - `ztb/training/lagrange_constraint.py` (280行, 一般化済み)
  - `ztb/training/grad_probes.py` (320行, 一般化済み)
  - `ztb/training/weights.py` (250行, 未統合)
  - `scripts/mirror_augment.py` (220行, 動作確認済み)

- テスト結果:
  - 基本PPO: ep_rew_mean=-71 → 1410 (curriculum修正後)
  - SELL mitigation: SELL rate 0% (依然未達)
  - Integration test: ✅ pass (機能は統合、効果なし)

---

**最終更新**: 2025年10月6日
**プロジェクト状態**: アクションバイアス対策の一般化完了、次のステップ検討中
