# 93# 改訂版Pivot計画: レビュー反映 + vXXX追加知見統合

**Date**: 2026-02-02  
**基づく**: 92# Pivot計画 + 92# レビュー + 追加vXXX資料分析  
**Status**: 実行準備完了

---

## 📋 レビュー指摘事項への対応

### 重大な指摘: 設定反映の検証不足

レビューで最も重要な指摘:
> **P1で「reward_paramsが実際に適用されていたか」の確認が不足**  
> v444の「設定伝播バグ」事例があるため、設定値が環境へ届いたログを確認しない限り断定は危険

**対応**: Gate 0として「設定反映検証」を追加

---

## 🔍 追加vXXX資料の知見統合

### 1. v435.2 Curriculum成功例

**出典**: [docs/analysis/SAC_v435_ANALYSIS_REPORT.md](../analysis/SAC_v435_ANALYSIS_REPORT.md)

| バリアント | 総リターン | 評価 |
|-----------|-----------|------|
| v435 (Baseline) | -0.331% | 基準 |
| v435.1 (Conservative) | -0.165% | 低リスク |
| **v435.2 (Aggressive + Curriculum)** | **+0.601%** | **最高収益** |

**核心的発見**: Curriculum learningにより**唯一のプラス収益**を達成  
**適用**: Phase Cの非対称報酬は**段階的に導入**すべき

### 2. v435.7 SELL学習不全の原因分析

**出典**: [docs/analysis/SAC_v435_7_TRAINING_ANALYSIS.md](../analysis/SAC_v435_7_TRAINING_ANALYSIS.md)

| バリアント | HOLD | BUY | SELL | 結果 |
|-----------|------|-----|------|------|
| v435.7a | 100% | 0% | **0%** | 学習停止 |
| v435.7b | 96.8% | 3.2% | **0%** | SELL未学習 |
| v435.7c | 84.0% | 16.0% | **0%** | SELL未学習 |

**根本原因**: 非対称報酬スケーリングがSELL学習を阻害
```python
# v435.7の問題設定
short_position_reward_multiplier = 0.7    # SELL利益報酬を30%減 ← 問題
short_position_penalty_multiplier = 1.2   # SELL損失ペナルティを20%増 ← 問題
long_position_reward_multiplier = 1.5     # BUY利益報酬を50%増
long_position_penalty_multiplier = 0.8    # BUY損失ペナルティを20%減
```

**教訓**: **非対称報酬はBUY/SELL両方の係数を監視**しないとSELL不全を招く

### 3. v444 設定伝播バグ

**出典**: [docs/BALANCE_PENALTY_ROOT_CAUSE_FIX_FINAL.md](../BALANCE_PENALTY_ROOT_CAUSE_FIX_FINAL.md)

**症状**: 設定値 `balanced_penalty` が環境に到達せず `forced_balance` (デフォルト) が使用された

**原因フロー**:
```
v444 config: curriculum_stage = "balanced_penalty"
    ↓
V4XXConfigConverter.convert_v444_to_unified()
    ↓
training.environment.curriculum_stage = ? (未設定)
    ↓
EnvironmentConfig インスタンス化
    ↓
デフォルト値 "forced_balance" で上書き ← 問題
```

**教訓**: **設定変更後は必ず実際の適用値をログで確認**

### 4. v447 reward_components保存バグ

**出典**: [docs/BALANCE_EXPLORATION_AND_MEMORY_OPTIMIZATION.md](../BALANCE_EXPLORATION_AND_MEMORY_OPTIMIZATION.md)

**問題**: `_last_reward_components` が設定されず報酬分析が不可能だった

**修正内容**:
```python
# 修正後: componentsを明示的に保存
self._last_reward_components = {
    "stage": stage,
    "pnl": pnl,
    "adjusted_pnl": adjusted_pnl,
    "base_reward": base_reward,
    "hold_penalty_applied": hold_penalty_applied,
    "trade_bonus_applied": trade_bonus_applied,
    "position_change": position_change,
    "final_reward": final_reward,
}
```

**適用**: Phase Bのコスト分析で**gross/net/fee/slippageを同様にログ化**

### 5. v420 報酬パラメータ調整の体系化

**出典**: [docs/algorithms/sac_reward_parameter_tuning.md](../algorithms/sac_reward_parameter_tuning.md)

**推奨手順**:
1. シンプルな報酬関数から開始
2. 基本的な利益/損失 + 取引コストのみ
3. 段階的に要素を追加

**適用**: Phase Cは「単純→複雑」の段階的導入

### 6. v446 SIGNAL_GUIDANCE導入で性能劣化

**出典**: [docs/SAC_V446_DEVELOPMENT_PLAN.md](../SAC_V446_DEVELOPMENT_PLAN.md)

**結果**: SIGNAL_GUIDANCE導入により**平均リターン -81.93% vs ベースライン -6.56%**

**教訓**: 
> 強いガイダンス系の"重い介入"が性能を悪化させるケースがある  
> **強いガイダンスはゲーティング用途に限定**した方が安全

---

## 📋 改訂版実行計画

### Gate 0: 設定反映検証（最優先・必須）

**目的**: P1の結論を確定させるため、reward_paramsが実際に適用されたか検証

**検証項目**:
1. reward_paramsの実際値をstepログに出力
2. 環境初期化時の設定値をログ
3. 報酬計算時のパラメータ使用状況を確認

**実装**:
```python
def verify_reward_params_applied(trainer) -> bool:
    """設定が実際に適用されているか検証"""
    env = trainer.get_env()
    unwrapped = env.envs[0].unwrapped
    
    # 設定値を取得してログ出力
    actual_params = {
        "alpha": getattr(unwrapped, 'alpha', 'NOT_FOUND'),
        "beta": getattr(unwrapped, 'beta', 'NOT_FOUND'),
        "gamma_penalty": getattr(unwrapped, 'gamma_penalty', 'NOT_FOUND'),
        # ... その他のパラメータ
    }
    
    logger.warning(f"ACTUAL REWARD PARAMS: {actual_params}")
    return True
```

**Gate判定**:
- ✅ 設定値が期待通り → Phase A進行
- ❌ 設定値がデフォルト/異常 → 設定伝播バグ修正が先

### Phase A: Gamma検証（v451知見の適用）

**変更なし**（92#と同様）

| 実験 | Gamma | 期待効果 |
|------|-------|----------|
| A-1 | 0.80 | 短期最適化 |
| A-2 | 0.90 | バランス |
| A-3 | 0.99 | ベースライン |

**追加監視項目（レビュー反映）**:
- **学習進行の実在確認**: Q値/actor loss/entropyの推移を記録
- **learning_starts到達確認**: 学習が実際に開始されているか

### Phase B: コスト分解分析（v447知見の適用）

**v447の教訓を適用**: reward_componentsと同様に、コスト情報を明示的にログ

**必須ログ項目**:
```python
cost_breakdown = {
    "gross_pnl": float,       # 手数料前利益
    "total_fees": float,      # 手数料合計
    "total_slippage": float,  # スリッページ合計
    "net_pnl": float,         # 手数料後利益
    "cost_ratio": float,      # コスト/Gross比率
}
```

**分析スクリプト**:
```python
def analyze_cost_impact(metrics: dict) -> dict:
    gross = metrics.get('gross_pnl', 0)
    fees = metrics.get('total_fees', 0)
    slip = metrics.get('total_slippage', 0)
    net = gross - fees - slip
    
    return {
        "gross_roi": gross / 100000 * 100,
        "fee_roi": fees / 100000 * 100,
        "slip_roi": slip / 100000 * 100,
        "net_roi": net / 100000 * 100,
        "interpretation": "取引自体は利益" if gross > 0 else "取引自体が損失",
        "cost_ratio": (fees + slip) / abs(gross) if gross != 0 else float('inf'),
    }
```

### Phase C: 非対称報酬（段階的導入・v435知見適用）

**v435.2のCurriculum成功 + v435.7のSELL不全を考慮**

**段階的導入計画**:
| Stage | 設定 | 監視項目 |
|-------|------|----------|
| C-1 | 対称ベースライン (1.0/1.0) | BUY/SELL比率 |
| C-2 | 微小非対称 (1.0/1.05) | SELL学習の維持 |
| C-3 | v451設定 (1.0/1.2) | ROI + BUY/SELL比率 |

**SELL不全検知**:
```python
def check_sell_learning(action_distribution: dict) -> bool:
    """SELL学習が阻害されていないか確認"""
    sell_ratio = action_distribution.get('sell_ratio', 0)
    if sell_ratio < 0.10:  # 10%未満は警告
        logger.warning(f"⚠️ SELL不全の兆候: SELL={sell_ratio:.1%}")
        return False
    return True
```

**重要**: C-2→C-3への移行は**SELL比率が10%以上維持されている場合のみ**

### Phase D: ベースライン比較（レビュー追加項目）

**目的**: 現行モデルが「学習で上回っているか」を検証

| 戦略 | 説明 | 期待ROI |
|------|------|---------|
| **Buy & Hold** | 初期BUY→最終SELL | 市場リターン |
| **Random Policy** | ランダムなBUY/SELL | ~-手数料分 |
| **Current Model** | 学習済みモデル | ? |

**判定**:
- Current > Random → 学習効果あり
- Current ≈ Random → 学習効果なし
- Current < Random → 学習が害になっている

---

## ⏱️ 実行タイムライン

### Day 12 午前: Gate 0 + Phase A

| 時間 | タスク | 成果物 |
|------|--------|--------|
| 09:00-10:00 | Gate 0: 設定反映検証スクリプト作成・実行 | 検証ログ |
| 10:00-12:00 | Phase A: Gamma検証 (3設定 × 10Kステップ) | ROI/Trades比較 |

### Day 12 午後: Phase B + 分析

| 時間 | タスク | 成果物 |
|------|--------|--------|
| 13:00-14:00 | Phase B: コスト分解ログ実装 | 修正コード |
| 14:00-16:00 | Phase Aの結果にコスト分析適用 | Gross vs Net表 |
| 16:00-17:00 | Phase D: Buy&Hold/Random比較 | ベースライン比較表 |
| 17:00-18:00 | 結果統合・次ステップ判断 | 94#ドキュメント |

### Day 13: Phase C（条件付き）

Phase A/Bの結果に基づき判断:
- Gamma改善 + コスト問題なし → Phase C (非対称報酬)
- Gamma改善なし → 特徴量/アーキテクチャ見直し
- コスト問題あり → 取引頻度/手数料モデル見直し

---

## 🎯 成功基準（Gate判定）

### Gate 0: 設定反映検証

| 条件 | 判定 |
|------|------|
| reward_paramsが期待値と一致 | ✅ Phase A進行 |
| reward_paramsがデフォルト値 | ❌ v444バグ修正が先 |

### Gate A: Gamma検証後

| 条件 | 判定 |
|------|------|
| Gamma=0.80でROI改善（-5%→-3%以上） | → Phase C進行 |
| Gamma変更で変化なし | → 特徴量/報酬構造見直し |
| 学習が進んでいない（loss変化なし） | → learning_starts/勾配確認 |

### Gate B: コスト分析後

| 条件 | 判定 |
|------|------|
| Gross PnL > 0, Net PnL < 0 | → コスト削減戦略（取引頻度制限） |
| Gross PnL < 0 | → 予測モデル自体の問題 |

### Gate D: ベースライン比較後

| 条件 | 判定 |
|------|------|
| Model > Random | → 学習効果あり、チューニング継続 |
| Model ≈ Random | → 特徴量/報酬の根本見直し |
| Model < Random | → 学習が害、リセット検討 |

---

## 📚 参照資料（統合版）

### 92#で参照済み
1. v451 "Golden Era" - Gamma=0.80, Hold Penalty=0
2. v457.2 - Gross PnL分析
3. v457.3 - Buy & Hold成功
4. v454 - 逆説的確信
5. v456 - 報酬関数失敗

### 92#レビューで追加
6. **v435.2** - Curriculum成功例（+0.601%）
7. **v435.7** - SELL学習不全（非対称報酬の罠）
8. **v444** - 設定伝播バグ（最重要教訓）
9. **v447** - reward_components保存バグ修正
10. **v420** - 報酬パラメータ調整の体系化
11. **v446** - SIGNAL_GUIDANCE性能劣化（-81.93%）

---

## 💡 結論

**92#レビューの核心的指摘を反映**:

1. **Gate 0（設定反映検証）を最優先**
   - v444の教訓: 設定が伝播しないバグは致命的
   - P1の結論を確定させるため必須

2. **Phase Cは段階的導入**
   - v435.2: Curriculumで成功
   - v435.7: 非対称報酬でSELL不全
   - → 対称→微小非対称→v451設定の段階的移行

3. **学習進行の実在確認を追加**
   - loss/entropy/Q値の推移を監視
   - 「学習していない」可能性の排除

4. **ベースライン比較を追加**
   - Buy&Hold / Random との比較
   - 学習効果の客観的評価

**「短期間での高収益性システム」という大義のもと、過去の失敗パターンを回避しながら、成功パターンを再現する。**

---

## 📎 付録: 実装チェックリスト

### Gate 0: 設定反映検証
- [ ] reward_paramsログ出力コード追加
- [ ] P1実験の実際の設定値を確認
- [ ] v444バグパターンとの比較

### Phase A: Gamma検証
- [ ] Gamma=0.80/0.90/0.99の設定ファイル準備
- [ ] 学習進行ログ（loss/entropy）追加
- [ ] 10Kステップ実行・結果比較

### Phase B: コスト分解
- [ ] gross_pnl/total_fees/total_slippage抽出コード
- [ ] コスト分析スクリプト作成
- [ ] Phase A結果への適用

### Phase C: 非対称報酬
- [ ] 段階的設定（C-1/C-2/C-3）準備
- [ ] SELL不全検知コード
- [ ] BUY/SELL比率監視

### Phase D: ベースライン比較
- [ ] Buy&Hold戦略実装
- [ ] Random Policy実装
- [ ] 3者比較表作成
