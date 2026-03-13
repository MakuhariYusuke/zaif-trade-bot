# 401# 深層調査レポート — Reward パスウェイ & 設定伝搬

**前提**: 400# (reward-clean: `194e30a2e`) コミット後、reward-clean 実験を裏で実行しながら構造的課題を発掘。

---

## 実験結果 (reward-clean 20K×4seeds) — 完了

### G2-train: **PASS** ✅

| チェック | 値 | 閾値 | 結果 |
|----------|-----|------|------|
| positive_seed_ratio | 1.0 (4/4) | ≥0.75 | ✅ |
| roi_seed_std | 0.00158 | ≤0.03 | ✅ |
| convergence | 0.0% | ≤5.0% | ✅ |
| worst_seed_roi | +0.332% | >-3.5% | ✅ |

### G3-pnl: **PASS** ✅

| チェック | 値 | 閾値 | 結果 |
|----------|-----|------|------|
| pf_median | 1.145 | >1.05 | ✅ |
| pf_worst | 1.089 | >0.95 | ✅ |
| gross_gt_fee | true | true | ✅ |
| max_drawdown | 0.26% | <15% | ✅ |
| sharpe_annual | 5.70 | >0.8 | ✅ |

### Seed 別 OOS 結果

| Seed | ROI | PF | Sharpe(年率) | MaxDD | r(reward,profit) | 訓練時間 |
|------|------|------|-------|--------|-----|-----|
| 42 | +0.69% | 1.198 | 5.76 | 0.20% | 0.537 | 743s |
| 123 | +0.64% | 1.116 | 6.00 | 0.20% | 0.562 | 783s |
| 456 | +0.33% | 1.089 | 3.02 | 0.26% | **-0.203** | 747s |
| 789 | +0.55% | 1.174 | 5.64 | 0.20% | 0.606 | 838s |

**所見**:
- 全 4 seed で**正の ROI** (G2 100% positive 条件クリア)
- PF は全て >1.0 (利益 > 損失)
- Sharpe 3.0~6.0 は堅実 (ただし cost=0 前提)
- **Seed 456 の reward-profit 相関が負** (-0.203) — 報酬信号と実際の利益方向が逆相関。報酬飽和 (F1) の影響の可能性
- ROI 平均 +0.55% は 20K steps (= ~14日分データ) としては妥当だが、スケーリング (50K+) 時の安定性は未検証

### Checkpoint ROI 推移

| Steps | Seed 42 | Seed 123 | Seed 456 | Seed 789 |
|-------|---------|----------|----------|----------|
| 5K    | 0.0021  | 0.0020   | 0.0026   | 0.0025   |
| 10K   | 0.0018  | 0.0024   | 0.0022   | 0.0017   |
| 15K   | 0.0015  | 0.0027   | 0.0024   | 0.0008   |
| 20K   | 0.0015  | 0.0030   | 0.0024   | 0.0013   |

Seed 42/789 は低下傾向、Seed 123 は上昇傾向 — **seed 間のばらつきが大きい**.

---

## 発見事項一覧

### F1: 報酬飽和 (Reward Saturation) — CRITICAL

**概要**: `reward_scaling=100.0` で PnL を乗算した後に `clip[-1.0, 1.0]` を適用するため、ほぼ全ての非ゼロ報酬が ±1.0 に飽和する。

**フロー**:
```
step_pnl (JPY) → × reward_scaling (100.0) → + penalties (全て0) → clip[-1, 1]
```

**典型値** (BTC/JPY, position=0.01, 1分足):
| 価格変動 | step_pnl | × 100 | clip後 |
|----------|----------|-------|--------|
| 500 JPY  | 5.0 JPY  | 500.0 | **1.0** (飽和) |
| 100 JPY  | 1.0 JPY  | 100.0 | **1.0** (飽和) |
| 1 JPY    | 0.01 JPY | 1.0   | 1.0 (境界) |
| 0.01 JPY | 0.0001   | 0.01  | 0.01 (非飽和) |

**影響**: エージェントは利益の「大きさ」を区別できず、sign(pnl) の二値信号でしか学習できない。勾配信号が0の平坦な clip 領域で殆どの時間を過ごす。

**経路**:
- [reward_calculator.py L1886](ztb/trading/environment/components/calculators/reward_calculator.py#L1886): `pnl * reward_scaling * pnl_reward_multiplier`
- [reward_calculator.py L1217-1219](ztb/trading/environment/components/calculators/reward_calculator.py#L1217): `np.clip(reward, reward_clip_min, reward_clip_max)`

**反証**: v459 E2α (同等設定) で +41.95% ROI を記録した事実がある。ただし 4 seed 中 2 seed が負であり、高分散の原因が飽和にある可能性。

**対策候補**:
1. `reward_scaling` を 1.0 に下げ、`clip` を [-100, 100] にする
2. PnL を ATR で正規化し (atr_normalised を使用)、clip を [-5, 5] 程度にする
3. `tanh(pnl * scale)` のソフトクリップに変更

---

### F2: Double Clip (二重クリッピング) — LOW (現時点無害)

**発見**: 報酬クリップが2箇所で適用される。

1. **reward_calculator.py** L1217-1219: `reward_clip_min/max` (config: [-1, 1])
2. **validation_manager.py** L126-133: `reward_clip_value` (default: 10000.0)

**評価**: `DEFAULT_REWARD_CLIP_VALUE = 10000.0` >> `clip[-1, 1]` なので実質無害。
ただし、3つの異なるクリップ設定 (`reward_clip_min`, `reward_clip_max`, `reward_clip_value`) が存在し、設計上の技術的負債。

**経路**:
- [ppo_config.py L130](ztb/training/config/ppo_config.py#L130): `DEFAULT_REWARD_CLIP_VALUE = 10000.0`
- [config.py L77](ztb/trading/environment/utils/config.py#L77): `reward_clip_value: float = DEFAULT_REWARD_CLIP_VALUE`
- [validation_manager.py L126-133](ztb/trading/environment/heavy_env/components/validation_manager.py#L126)

---

### F3: `balance_penalty_tolerance` 設定無視 — MEDIUM

**発見**: `behavior_optimization.balance_penalty_tolerance` は YAML で設定可能だが、`EnvironmentConfig.from_dict()` のマッピング対象外。

**メカニズム**:
- `from_dict()` は `balance_penalty`, `consistency_penalty`, `balance_shaping_enabled`, `action_entropy_shaping_enabled` のみマッピング
- `balance_penalty_tolerance` は `RewardSettings` にフィールドが存在し、デフォルト 0.05 を持つ
- `_behavior_opt_fallback` は RewardSettings フィールドが既に存在するケースではフォールバックしない
- 結果: YAML で値を変えても常にデフォルト 0.05 が使用される

**影響**: 現行 YAML は偶然 0.05 を設定しているため顕在化していない。

**修正方針**: `from_dict()` に `balance_penalty_tolerance` マッピングを追加。

**ステータス**: ✅ 修正済 — [config.py L467](ztb/trading/environment/utils/config.py#L467) にマッピング追加。テスト追加済。

---

### F4: デフォルト値の不一致 — LOW (明示設定時は無害)

#### F4-1: `balance_penalty` デフォルト不一致
| ソース | デフォルト値 | 場所 |
|--------|-------------|------|
| `RewardSettings.balance_penalty` | **0.1** | [config.py L51](ztb/trading/environment/utils/config.py#L51) |
| `BehavioralPenaltyCalculator._rs_get()` | **1.0** | [behavioral_penalty_calculator.py L240](ztb/trading/environment/components/behavioral_penalty_calculator.py#L240) |
| `RewardCalculator.get_setting_float()` | **1.0** | [reward_calculator.py L171](ztb/trading/environment/components/calculators/reward_calculator.py#L171) |

読み取り順序: RS フィールド (0.1) → `_rs_get` fallback (1.0) — 通常は RS フィールドが優先。

#### F4-2: `consistency_penalty` デフォルト不一致
| ソース | デフォルト値 | 場所 |
|--------|-------------|------|
| `RewardSettings.consistency_penalty` | **0.0** | [config.py L116](ztb/trading/environment/utils/config.py#L116) |
| `BehavioralPenaltyCalculator` (else branch) | **-0.05** | [behavioral_penalty_calculator.py L300](ztb/trading/environment/components/behavioral_penalty_calculator.py#L300) |
| `RewardCalculator.get_setting_float()` | **0.05** | [reward_calculator.py L149](ztb/trading/environment/components/calculators/reward_calculator.py#L149) |

**影響**: 明示的に値を設定しない場合、どの経路で読まれるかにより異なるデフォルトが適用される。reward-clean は全て明示設定済みのため現時点では安全。

---

### F5: タイポ黙殺 (Silent Key Swallowing) — HIGH

**発見**: YAML の `reward_settings` でキー名を誤記した場合、エラーなく `custom_reward_params` に格納されて黙殺される。

**メカニズム**:
- `RewardSettings.from_dict()`: 未知キーは `custom_reward_params` dict に入る
- `EnvironmentConfig.from_dict()`: 未知キーは `logger.debug(f"Skipping config key ...")` でスキップ (DEBUG レベルでは通常非表示)

**リスク**: `hold_penalt_weight` (typo) → 黙殺 → デフォルト 0.01 が適用される → 意図しないペナルティが有効に

**修正方針**: `from_dict()` で未知キーを WARNING ログに変更。

**ステータス**: ✅ 修正済 — [config.py L635](ztb/trading/environment/utils/config.py#L635) `logger.debug` → `logger.warning`。テスト追加済。

---

### F6: OOS Best-Checkpoint 未実装 — HIGH (50K 崩壊対策)

**発見**: `_train_with_checkpoints()` は各チェックポイントで in-sample ROI を記録するが、OOS 評価や best-model 保存は行わない。

**影響**: v459 Day9b で確認された「50K崩壊」問題:
- 25K 時点で最良性能だが、50K まで訓練を続行すると過学習で -30.54% ROI 劣化
- 現在は最終モデルのみ保存 → 最良時点のモデルが失われる

**現行フロー** ([sac_train.py L363-420](scripts/v460/lib/tasks/sac_train.py#L363)):
```
for each checkpoint_interval:
    model.learn(steps)
    roi = _checkpoint_eval_roi(model, train_env)  ← IN-SAMPLE
    log(roi)  ← 記録のみ、保存なし
model.save(final_path)  ← 最終モデルのみ保存
```

**設計案**:
```
best_oos_roi = -inf
for each checkpoint_interval:
    model.learn(steps)
    oos_roi = _checkpoint_eval_roi(model, val_env)  ← OOS
    if oos_roi > best_oos_roi:
        best_oos_roi = oos_roi
        model.save(best_checkpoint_path)  ← Best 保存
model.save(final_path)  ← 最終モデルも保存
```

**注意点**:
- val_env での評価ステップ数 (5K) を考慮すると、各チェックポイントで ~6秒追加
- 20K 訓練で 4 checkpoint → 計 ~24 秒のオーバーヘッド (許容範囲)
- val_env の scaler を train env から転送する必要あり (384# HIGH-2 と同様)

---

### F7: G3 E3 (gross > fee) の形骸化 — INFO

**発見**: `transaction_cost: 0.0` (maker 0% fee) の場合、G3 の E3 チェック (`avg_gross_per_trade > avg_fee_per_trade`) は自明に PASS する。

**影響**: cost=0 実験では G3 E3 は判別力を持たない。将来的にコスト導入時に意味を持つ。現時点では情報としてのみ記録。

---

## 優先度順の対策マトリクス

| # | Finding | 深刻度 | 対策 | 状態 |
|---|---------|--------|------|------|
| F1 | 報酬飽和 | CRITICAL | reward_scaling / clip 再調整 | ⬜ 要検討 (G2/G3 PASS だが seed 456 の r<0 が懸念) |
| F6 | OOS Best-Checkpoint | HIGH | _train_with_checkpoints に OOS 評価 + best 保存 | ⬜ 50K 拡張前に必須 |
| F5 | タイポ黙殺 | HIGH | from_dict に WARNING ログ | ✅ 修正済 |
| F3 | balance_penalty_tolerance 無視 | MEDIUM | from_dict マッピング追加 | ✅ 修正済 |
| F2 | Double Clip | LOW | 将来統合 | ⬜ 現時点無害 |
| F4 | デフォルト不一致 | LOW | デフォルト値統一 | ⬜ 次回整理 |
| F7 | G3 E3 形骸化 | INFO | cost>0 時に再評価 | ⬜ |

---

## 次のアクション

1. **✅ 実験完了**: G2 PASS / G3 PASS — reward-clean (cost=0, 20K) は基礎的成功
2. **F1 対策検討**: Seed 456 の reward-profit 逆相関が飽和の影響か、seed乱数の影響かを切り分け
3. **50K スケールテスト**: F6 (OOS best-checkpoint) を実装してから実施 — 50K 崩壊リスク対策
4. **コスト導入**: cost=0 での G3 PASS を確認できたので、次は実際の取引コスト (coincheck maker 0% / taker 0.1%) で再実験
5. **F3/F5 コミット**: 修正済み、テスト PASSED
