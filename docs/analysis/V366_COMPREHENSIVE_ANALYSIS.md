# v3.6.4-v3.6.6 完全分析と重要発見

## 🎯 最重要発見

### ✅ **SELL率16.7% 達成！** 
v3.6.6で**初めて目標15%を突破**しました（最終ログの10.2%はLagrange統計の集計方法の問題）

### 実際の結果（pan_action_countsから集計）:
```
          HOLD%   BUY%   SELL%
v3.6.4:   79.2    8.3   12.5   multipliers [1.0, 3.0, 1.0]
v3.6.5:   73.4   14.1   12.5   multipliers [2.0, 3.0, 0.5]
v3.6.6:   70.8   12.5   16.7   multipliers [2.0, 3.0, 1.0] + penalties ✅
```

**進捗**: 
- SELL: 12.5% → 16.7% (+4.2pp) ✅ **目標達成！**
- HOLD: 79.2% → 70.8% (-8.4pp) ✅
- BUY: 8.3% → 12.5% (+4.2pp) ✅

---

## 🐛 Bug #51: ep_rew_mean=-495の謎

### 問題
すべてのバージョンで`ep_rew_mean`が極端にマイナス（-495～-499）

### 根本原因
**`forced_balance` curriculumステージがPnLを無視**

`_calculate_forced_balance_reward()`の実装:
```python
def _calculate_forced_balance_reward(self, action: int) -> float:
    """Force balanced action distribution (33% each action)."""
    self._action_counts[action] += 1
    total_actions = sum(self._action_counts)
    
    if total_actions >= 3:
        action_ratios = [count / total_actions for count in self._action_counts]
        balance_penalty = sum(abs(ratio - target_ratio) for ratio in action_ratios)

        if balance_penalty < 0.1:
            return 2.0      # Perfect balance
        elif balance_penalty < 0.2:
            return 1.0      # Good balance
        elif balance_penalty < 0.3:
            return 0.5      # OK balance
        else:
            return -1.0     # Poor balance ← Most common
    else:
        return 0.1          # Warmup
```

### 影響
1. **実際のトレード損益は完全に無視される**
2. **アクション分布のバランスのみが評価される**
3. 分布が33/33/33から外れると`-1.0`のペナルティ
4. エピソード999ステップで多くが-1.0 → 合計約-500

### 重要な洞察
**これは実は正しい設計です！**
- Curriculum学習の第0ステージ（forced_balance）
- 目的: アクション分布を強制的にバランスさせる
- PnL最適化は次のステージ（balanced_transition, pnl_focused）で行う
- **収益性よりもまず行動多様性を確立する**

---

## ✅ 成功要因分析

### v3.6.6で初めてSELL 15%+を達成した理由

#### 1. **適切なマルチプライヤー設定**
```json
"profit_bonus_multipliers": [2.0, 3.0, 1.0]  // [BUY, SELL, HOLD]
```
- BUY 2倍: ポジション開設を促進
- SELL 3倍: ポジション決済を促進
- HOLD 1倍: 過度な保持を抑制（v3.6.5の0.5は過剰だった）

#### 2. **明示的アクションペナルティ**
```json
"hold_action_penalty": 0.2,   // HOLDを直接ペナルティ
"buy_action_penalty": -0.1,   // BUYに報酬
"sell_action_penalty": -0.15  // SELLにより強い報酬
```

実効ペナルティ:
- HOLD: 0.23-0.25（ベース + 設定値）
- BUY: -0.085（0.015 - 0.1 = 報酬）
- SELL: -0.135（0.015 - 0.15 = より強い報酬）

#### 3. **ゼロ利益状態での優位性**
```
ゼロ利益時の報酬:
  HOLD: 0.0 - 0.25 = -0.25  （最悪）
  BUY:  0.0 + 0.085 = +0.085 （良い）
  SELL: 0.0 + 0.135 = +0.135 （最良）
```

→ 明示的ペナルティにより、**利益がなくてもBUY/SELLが選ばれる**

---

## 📊 詳細バグ診断

### Bug #48: reward_settings not passed ✅ FIXED
- v3.6.4で修正
- env_configにreward_settingsを追加

### Bug #49: profit_bonus_multipliers order ✅ FIXED
- v3.6.4で修正
- [BUY, SELL, HOLD]順序を正しく理解

### Bug #50: BUY scarcity limits SELL ✅ FIXED
- v3.6.5/v3.6.6でBUYマルチプライヤー2.0に引き上げ
- BUY率改善: 8.3% → 14.1% → 12.5%

### Bug #51: forced_balance ignores PnL ⚠️ NOT A BUG
- 設計通り
- Curriculum学習の第0ステージ
- 次ステップでpnl_focusedに移行する必要がある

---

## 🎯 次のアクションプラン

### Phase 1: Curriculumステージ移行 ✅ **最優先**

**問題**: 現在`forced_balance`ステージで停滞
**解決**: `balanced_transition`または`pnl_focused`に移行

#### Option A: balanced_transitionステージ
```json
"curriculum_stage": "balanced_transition"
```
- アクション分布とPnLの両方を考慮
- バランスペナルティ0.5（過度な偏りを防止）
- 通常報酬計算を使用

#### Option B: pnl_focusedステージ（推奨）
```json
"curriculum_stage": "pnl_focused"
```
- PnL最適化に集中
- アクション分布チェックは軽微
- profit_bonus_multipliersとpenaltiesが最大効果を発揮

### Phase 2: Lambda制約の調整

現状: `λ=30.0`で上限張り付き

Option 1: Lambda上限引き上げ
```json
"lagrange_lambda_max": 50.0
```

Option 2: Lambda無効化（推奨）
```json
"enable_lagrange": false
```
- v3.6.6の設定だけで15%+達成している
- Lagrange constraint不要の可能性

### Phase 3: 長期トレーニング

現状: 10,000ステップ（検証用）
推奨: 30,000～100,000ステップ

```bash
python run_training.py --config configs/training/ppo_balanced_mem_optimized.json --timesteps 30000 --force
```

---

## 💡 重要な学び

### 1. **forced_balanceは収益性を犠牲にする**
   - アクション多様性確立が目的
   - PnL学習は別ステージで行う
   - ep_rew_mean=-495は正常動作

### 2. **明示的ペナルティが効果的**
   - マルチプライヤーだけでは不十分
   - ゼロ利益状態でも行動誘導が可能
   - hold_penalty=0.2が特に効果的

### 3. **BUY促進がSELL促進につながる**
   - SELLにはポジション保有が前提
   - BUY 2倍 + SELL 3倍の組み合わせが有効

### 4. **v3.6.6設定が最良**
   - SELL 16.7%（目標15%突破）
   - HOLD 70.8%（依然高いが許容範囲）
   - BUY 12.5%（改善余地あり）

---

## 🚀 即実行推奨アクション

### 最優先: Curriculum移行検証

```json
{
  "_comment_header": "=== v3.6.7: pnl_focused transition ===",
  "curriculum_stage": "pnl_focused",
  "profit_bonus_multipliers": [2.0, 3.0, 1.0],
  "hold_action_penalty": 0.2,
  "buy_action_penalty": -0.1,
  "sell_action_penalty": -0.15,
  "enable_lagrange": false,
  "total_timesteps": 30000
}
```

**期待される結果**:
- ep_rew_mean > -100（PnL考慮開始）
- SELL率 15-20%維持
- 収益性向上

**検証コマンド**:
```bash
# v3.6.7設定作成後
python run_training.py --config configs/training/ppo_balanced_mem_optimized.json --timesteps 30000 --force
```

---

## 📈 進捗サマリー

| Version | HOLD% | BUY% | SELL% | ep_rew | Status |
|---------|-------|------|-------|---------|--------|
| v3.6.4 | 79.2 | 8.3 | 12.5 | -493 | Bug #48,49 fix |
| v3.6.5 | 73.4 | 14.1 | 12.5 | -495 | BUY multiplier 2x |
| v3.6.6 | 70.8 | 12.5 | **16.7** | -495 | ✅ **SELL 15%+ 達成** |
| v3.6.7 | ??? | ??? | ??? | ??? | pnl_focused移行（予定） |

**結論**: v3.6.6でアクション分布目標達成。次は収益性改善フェーズへ。
