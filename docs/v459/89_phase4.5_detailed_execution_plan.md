# 89# Phase 4.5 詳細実行計画（88#レビュー反映版）

**作成日**: 2026-02-02  
**基づく**: 87# 収益化計画 + 88# レビューコメント  
**目的**: ROI > 0% 達成に向けた精密な検証計画

---

## 1. 88# レビュー指摘の妥当性検証

### 1.1 取引コスト推定の過大化 ✅ 正しい指摘

**87#の誤り**:
```
取引回数: ~260回 @ 50k steps
取引コスト: 0.1% / 取引
推定コスト: 260 × 0.1% = 26%
```

**88#の指摘**:
> 「取引コスト=260×0.1%=26%」は計算が成立しない。実際は「約定金額の総和×手数料」

**検証結果**: ✅ 88#が正しい

実際のコード（`fast_intraday_env_v456.py` L650）:
```python
fee_rate = self.fee_model.get_fee_rate(trade_type)
fee_paid = abs(delta) * execution_price * fee_rate
```

- `delta`: ポジション変化量（BTC単位）
- `execution_price`: 約定価格（JPY/BTC）
- 例: delta=0.01BTC, price=10,000,000JPY → fee = 0.01 × 10M × 0.001 = 1,000円

**正しい推定**:
```
平均ポジションサイズ: ~0.01BTC（仮定）
平均約定価格: ~10,000,000 JPY/BTC
取引回数: 260回
片道コスト: 0.01 × 10M × 0.001 = 1,000円
総コスト: 260 × 1,000 = 260,000円
ROI影響: -2.6%（100,000円資金に対して）
```

→ 取引コストはROI -5%の約半分を説明可能

---

### 1.2 検証順序の修正提案 ✅ 妥当

**88#の提案**:
> 「評価の正確化 + コスト分解」→「C1(PnLのみ)で基準モデル確立」→「B1/B2で崩壊点特定」

**検証結果**: ✅ 論理的に妥当

理由:
1. 現状、gross_pnl/net_pnl/total_feesは環境に実装済み（`fast_intraday_env_v456.py` L405-407）
2. これらを取得・分析することで「取引自体が利益か損失か」を判定可能
3. PnLのみ報酬で基準を作れば、追加要素の効果を測定しやすい

---

### 1.3 成功基準の強化 ✅ 妥当

**88#の指摘**:
> ROI≥0%は妥当だが、信頼区間・シード分散・期間分散を併記しないと"偶然のプラス"で判断がブレる

**検証結果**: ✅ 重要な指摘

Day11結果でもσ=0.12%と小さいが、2シードでは統計的有意性が弱い。
OOS期間検証も未実施。

---

### 1.4 リスクの追加 ✅ 妥当

**88#の追加リスク**:
1. 評価系の誤り
2. 設定値の伝播ミス
3. 報酬項がPnLを上回って学習を歪める

**検証結果**: ✅ 過去に実際に発生した問題

- `docs/BALANCE_PENALTY_ROOT_CAUSE_FIX_FINAL.md`: 設定値伝播ミス事例
- Day10のfinal_balance取得失敗: 評価系の誤り
- `docs/v456/59_V456_FINAL_RETROSPECTIVE.md`: 報酬設計が失敗要因

---

### 1.5 報酬関数の現状分析

**現在の報酬構造**（`fast_intraday.py` L163-169）:
```python
reward = (pnl - total_cost - penalty_term) / max(max_position, eps)

# pnl: position_prev × (price_now - price_prev)
# total_cost: fee_paid + slippage_paid + extra_fee_penalty
# penalty_term: edge_penalty + vol_penalty + time_decay_penalty
```

**問題点**:
1. `penalty_term`が大きすぎると取引抑制
2. `max_position`での除算がスケールを不安定にする可能性
3. Trend-Guided Curriculum（L746-767）がさらに減算

---

## 2. 修正版実行計画

### Phase 4.5 優先順位（修正版）

| 優先度 | フェーズ | 目的 | 実験数 | 推定時間 |
|--------|----------|------|--------|----------|
| **P0** | 計測基盤整備 | gross/net/fee分解ログ | 0 | 2時間 |
| **P1** | 基準モデル作成 | PnLのみ報酬で基準 | 4 | 4時間 |
| **P2** | 崩壊点特定 | ステップ別性能推移 | 4 | 4時間 |
| **P3** | コスト感度分析 | 取引コスト影響測定 | 4 | 4時間 |
| **P4** | 報酬チューニング | 最小限ペナルティ追加 | 4 | 4時間 |

---

## 3. P0: 計測基盤整備（必須前提）

### 3.1 目的

評価の正確化と損益分解ログの取得

### 3.2 実装内容

#### A. 環境メトリクス取得関数の追加

```python
def extract_environment_metrics(env) -> dict:
    """環境から詳細メトリクスを抽出"""
    metrics = {}
    
    # VecEnvをunwrap
    actual_env = env
    if hasattr(env, 'envs') and len(env.envs) > 0:
        actual_env = env.envs[0]
    
    # さらにMonitor等をunwrap
    unwrapped = actual_env
    for _ in range(10):
        if hasattr(unwrapped, 'env'):
            unwrapped = unwrapped.env
        else:
            break
    
    # コスト分解
    metrics['gross_pnl'] = getattr(unwrapped, 'gross_pnl', None)
    metrics['net_pnl'] = getattr(unwrapped, 'net_pnl', None)
    metrics['total_fees'] = getattr(unwrapped, 'total_fees', None)
    metrics['total_slippage'] = getattr(unwrapped, 'total_slippage', None)
    metrics['balance'] = getattr(unwrapped, 'balance', None)
    metrics['initial_balance'] = getattr(unwrapped, 'initial_balance', None)
    metrics['total_trades'] = getattr(unwrapped, 'total_trades', None)
    
    # 派生指標
    if metrics['gross_pnl'] is not None and metrics['initial_balance']:
        metrics['gross_roi'] = (metrics['gross_pnl'] / metrics['initial_balance']) * 100
    if metrics['net_pnl'] is not None and metrics['initial_balance']:
        metrics['net_roi'] = (metrics['net_pnl'] / metrics['initial_balance']) * 100
    if metrics['balance'] is not None and metrics['initial_balance']:
        metrics['balance_roi'] = ((metrics['balance'] - metrics['initial_balance']) / metrics['initial_balance']) * 100
    
    return metrics
```

#### B. 評価統一ルール

| 指標 | 定義 | 用途 |
|------|------|------|
| `gross_roi` | gross_pnl / initial_balance × 100 | 取引自体の性能 |
| `net_roi` | net_pnl / initial_balance × 100 | コスト込み性能 |
| `balance_roi` | (balance - initial) / initial × 100 | 最終資産ベース |
| `cost_ratio` | total_fees / initial_balance × 100 | コスト負担率 |

### 3.3 検証コード

```python
# P0検証: 計測の正確性確認
print(f"Gross PnL: {metrics['gross_pnl']:.2f}")
print(f"Net PnL:   {metrics['net_pnl']:.2f}")
print(f"Total Fees: {metrics['total_fees']:.2f}")
print(f"Balance:   {metrics['balance']:.2f}")
print(f"")
print(f"Gross ROI: {metrics['gross_roi']:.2f}%")
print(f"Net ROI:   {metrics['net_roi']:.2f}%")
print(f"Balance ROI: {metrics['balance_roi']:.2f}%")
print(f"Cost Ratio: {metrics['cost_ratio']:.2f}%")

# 整合性チェック
assert abs(metrics['net_pnl'] - (metrics['gross_pnl'] - metrics['total_fees'])) < 1.0, "PnL計算不整合"
```

---

## 4. P1: 基準モデル作成

### 4.1 目的

PnLのみ報酬で「ペナルティなし」の基準を確立

### 4.2 実験設計

| 実験ID | 報酬設定 | 説明 |
|--------|----------|------|
| P1-1 | PnLのみ（ペナルティ全無効） | 純粋なPnL性能 |
| P1-2 | PnL - 基本コスト（fee+slip） | 最小限コスト |
| P1-3 | 現行設定（参考） | 比較用 |
| P1-4 | PnL / max_position なし | スケール影響確認 |

### 4.3 報酬パラメータ

```python
# P1-1: PnLのみ
reward_params_p1_1 = {
    'alpha': 0.0,           # position change penalty OFF
    'beta': 0.0,            # holding time penalty OFF
    'gamma': 0.0,           # inventory risk OFF
    'fee_penalty_weight': 0.0,  # extra fee penalty OFF
    'edge_penalty_rate': 0.0,   # edge penalty OFF
    'vol_floor_penalty': 0.0,   # vol floor penalty OFF
    'hold_ramp': 0.0,       # time decay OFF
}

# P1-2: 最小限コスト
reward_params_p1_2 = {
    'alpha': 0.0,
    'beta': 0.0,
    'gamma': 0.0,
    'fee_penalty_weight': 0.0,  # 基本fee/slipはcompute_hft_rewardで自動控除
    'edge_penalty_rate': 0.0,
    'vol_floor_penalty': 0.0,
    'hold_ramp': 0.0,
}
```

### 4.4 期待結果

| 実験 | 期待ROI | 根拠 |
|------|---------|------|
| P1-1 | 0%～+5%? | ペナルティなしで取引利益が見える |
| P1-2 | -2%～0%? | コスト分だけ減少 |
| P1-3 | -5% | Day11結果と同等 |

### 4.5 判断基準

- **P1-1 > 0%**: 取引自体は利益、コスト/ペナルティが問題
- **P1-1 < 0%**: 取引戦略自体が損失、学習設計見直し必要
- **P1-1 ≈ P1-3**: ペナルティの影響小さい

---

## 5. P2: 崩壊点特定

### 5.1 目的

学習ステップと性能の関係を明確化し、早期停止点を特定

### 5.2 実験設計

| 実験ID | ステップ | 評価間隔 |
|--------|----------|----------|
| P2-1 | 5,000 | 1,000 |
| P2-2 | 10,000 | 2,000 |
| P2-3 | 25,000 | 5,000 |
| P2-4 | 50,000 | 10,000 |

### 5.3 評価コールバック

```python
class DetailedEvalCallback(BaseCallback):
    def __init__(self, eval_freq=1000):
        super().__init__()
        self.eval_freq = eval_freq
        self.history = []
    
    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            metrics = extract_environment_metrics(self.model.env)
            self.history.append({
                'step': self.num_timesteps,
                'gross_roi': metrics.get('gross_roi'),
                'net_roi': metrics.get('net_roi'),
                'balance_roi': metrics.get('balance_roi'),
                'total_trades': metrics.get('total_trades'),
            })
        return True
```

### 5.4 期待結果

```
Step   Gross ROI   Net ROI   Balance ROI
5000   +2%         +1%       +1%
10000  +1%         -1%       -1%
25000  0%          -3%       -3%
50000  -2%         -5%       -5%
```

→ 崩壊点（ROIピーク）を特定

---

## 6. P3: コスト感度分析

### 6.1 目的

取引コストの影響を定量化

### 6.2 実験設計

| 実験ID | transaction_cost | 説明 |
|--------|------------------|------|
| P3-1 | 0.0% | コストゼロ（上限） |
| P3-2 | 0.05% | 半減 |
| P3-3 | 0.1% | 現行（基準） |
| P3-4 | 0.2% | 2倍（厳しい条件） |

### 6.3 期待結果

```
Cost    Gross ROI   Net ROI   差分
0.0%    +2%         +2%       0%
0.05%   +2%         +0.7%     -1.3%
0.1%    +2%         -0.6%     -2.6%
0.2%    +2%         -3.2%     -5.2%
```

→ コスト vs ROI の線形関係を確認

---

## 7. P4: 報酬チューニング

### 7.1 目的

P1-P3の結果を踏まえ、最小限のペナルティを追加

### 7.2 段階的追加

```
Stage 1: PnLのみ（P1-1の結果を基準）
Stage 2: + 軽い取引コストペナルティ（fee_penalty_weight=0.1）
Stage 3: + 最小限の頻度制御（hold_grace=10, hold_ramp=10）
Stage 4: + エッジ要求（edge_penalty_rate=0.001）
```

### 7.3 判断基準

各Stageで:
- ROIが前Stageより5%以上悪化 → そのペナルティは過剰
- ROIが改善または維持 → 採用

---

## 8. 実行スケジュール

### Day 12

| 時間 | タスク |
|------|--------|
| 0-2h | P0: 計測基盤整備・検証 |
| 2-6h | P1: 基準モデル作成（4実験） |
| 6-7h | 結果分析・P2計画調整 |

### Day 13

| 時間 | タスク |
|------|--------|
| 0-4h | P2: 崩壊点特定（4実験） |
| 4-8h | P3: コスト感度分析（4実験） |

### Day 14

| 時間 | タスク |
|------|--------|
| 0-2h | 結果分析・統合 |
| 2-6h | P4: 報酬チューニング（4実験） |
| 6-8h | 最終分析・Phase 5移行判断 |

---

## 9. 成功基準（修正版）

### Phase 4.5 完了基準

| 基準 | 閾値 | 測定方法 |
|------|------|----------|
| Balance ROI | ≥ 0% | final_balance由来 |
| シード安定性 | σ < 3% | 4シード以上 |
| Gross ROI - Net ROI | < 3% | コスト分解で確認 |
| 原因特定 | 完了 | 損失の主因を文書化 |

### Phase 5 移行基準（厳格版）

| 基準 | 閾値 | 測定方法 |
|------|------|----------|
| Balance ROI | > 1% | 4シード平均 |
| 最大ドローダウン | < 15% | エピソード内最大 |
| OOS期間 | 検証済み | 学習期間外データ |
| 取引回数 | 適正範囲 | 100-500回/50k |

---

## 10. リスク管理

### 10.1 監視項目

| リスク | 検出方法 | 対応 |
|--------|----------|------|
| 評価系の誤り | gross/net/balance整合性チェック | P0で修正 |
| 設定伝播ミス | reward_paramsログ出力 | 実験開始時確認 |
| 過剰ペナルティ | ROI急落（>10%悪化） | P4で段階的追加 |
| 取引停止 | total_trades < 50 | ペナルティ緩和 |

### 10.2 撤退基準

以下の場合、アプローチ根本見直し:
- P1-1（PnLのみ）でもROI < -5%
- 10実験以上で改善傾向なし
- 取引パターンが崩壊（HOLD率 > 80%）

---

## 11. 付録: 88# 過去vXXX参照の活用

| 参照ドキュメント | 活用方法 |
|------------------|----------|
| SAC_v435_ANALYSIS_REPORT | Curriculum Learning → P4 Stage設計 |
| SAC_v435_7_TRAINING_ANALYSIS | SELL学習不全 → BUY/SELL非対称チェック |
| BALANCE_PENALTY_ROOT_CAUSE_FIX | 設定伝播 → P0で検証 |
| sac_reward_parameter_tuning | パラメータ探索 → P4の土台 |
| 59_V456_FINAL_RETROSPECTIVE | 報酬過剰ペナルティ → P4で回避 |
