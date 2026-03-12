# 103# Phase Cレビュー — 102実験ログの再検証と次ステップ方針

**Date**: 2026-02-09  
**対象**: `prompts/codex_phase_c_review.md`, `docs/v459/102_phase_c_experiment_log.md`, 関連実装  
**目的**: 「次ステップ」の優先順位を、実装整合性まで含めて再定義する

---

## 0. 結論（先に要点）

1. `102#` の「14実験すべてNO-GO」「Net ROIが-15%近傍で固定」は、観測事実として妥当。  
2. ただし「deterministic評価で常に0取引」は、**方策崩壊の証拠としては不十分**。  
3. 理由は、`run_phase_c.py` の deterministic 評価経路が、学習時と異なる入力分布で動いている可能性が高いため。  
4. よって次ステップは、報酬やアルゴリズム切替の前に、**評価経路の整合性修正（C0'）を最優先**にするのが合理的。

---

## 1. 事実確認（コード照合）

### 1.1 `102#` の主要観測値

- 学習中は約742〜1,080回の取引が発生  
- deterministic評価は14実験すべて0取引  
- Net ROIは約-15%に収束  

この3点は、`run_phase_c.py` / バッチ実行スクリプトの設計とも整合する。

### 1.2 追加で判明した重要点（評価経路の不整合）

#### A. 学習側（`SACTrainer`）
- `ztb/training/unified_trainer/algorithms/sac_trainer.py` で  
  `Monitor -> DummyVecEnv -> VecNormalize` を適用（`normalization.enabled`の既定は有効）。

#### B. `run_phase_c.py` deterministic評価側
- `_deterministic_eval_gate2()` で `resolve_env()` 後に `unwrap_env()` し、**生の環境**を直接 `model.predict(..., deterministic=True)` で回している。

#### C. 含意
- 学習時の観測分布（VecNormalize後）と、評価時の観測分布（生値）がズレる。  
- このズレで action が閾値内（HOLD）に縮退し、0取引を誘発している可能性がある。  
- したがって、現時点で「SACは必ずDo Nothingを学習する」と断定するのは早い。

### 1.3 閾値実験に関する補足

- `ThresholdManager` には `dynamic_threshold_mode` / `z_score_*` ロジックがある。  
- ただし `EnvironmentConfig` にはこれらのフィールドが明示されておらず、現行C0+C1は実質「固定閾値スイープ」になっている可能性が高い。  

---

## 2. 根本原因仮説の再評価

| 仮説 | 判定 | コメント |
|---|---|---|
| 1分足 + 0.1%手数料で期待値が薄い | 妥当 | `102#` の-15%収束はコスト支配と整合 |
| 報酬が弱くHOLD有利 | 妥当 | simple rewardでも取引優位が出にくい |
| deterministic 0取引 = 学習失敗の決定的証拠 | **要修正** | eval経路不整合の影響を除去してから判断すべき |
| 50Kは不足 | 可能性高 | ただし構造問題を残したまま延長は非効率 |

---

## 3. 次ステップ（優先順位の改訂）

### 3.1 最優先 C0'（評価経路整合）

**目的**: 「本当に方策がHOLDなのか」を判定可能にする。

実施項目:
1. deterministic評価を2系統で同時実行  
   - `Eval-A`: 学習時と同じVecEnv/VecNormalize経路  
   - `Eval-B`: 現行のunwrapped生env経路  
2. action分布ログを追加  
   - `mean/std/p10/p50/p90`  
   - `|action| > threshold` の比率  
3. A/Bの取引回数・ROI・PF差分を比較

**Go条件**: A/B差分の有無で、0取引の原因を「評価経路」か「方策本体」かに切り分ける。

### 3.2 C1'（最小因果分離）

C0'完了後、最小グリッドで再検証:
- `reward_scale`: 100 vs 1000  
- `ent_coef`: auto vs 0.01  
- そのほかは固定（まず要因を増やさない）

狙い:
- エントロピー優勢仮説の検証  
- deterministicで閾値越え action が増えるか確認

### 3.3 C2'（分岐）

- C1'で改善が出るなら: 同路線を200Kへ延長  
- 依然0取引なら:  
  1. 離散行動（BUY/SELL/HOLD）パイロット  
  2. `FastIntradayEnv` 系統（`compute_hft_reward`活用可能経路）を並行比較

### 3.4 C3（実用性判定）

短期収益化の観点では、以下を満たせない場合は時間軸見直しを検討:
- 手数料控除後でNet ROIが持続的に負  
- Gross PnL/Fees 比が改善しない  

代替:
- 5分足/15分足  
- シグナル学習（ML）+ 執行はルールベースのハイブリッド

---

## 4. 過去vXXX・既存実装の再利用提案

| 区分 | 再利用対象 | 使い方 |
|---|---|---|
| 評価 | `ztb/metrics/metrics.py` | Gate2算出の単一基盤として継続利用 |
| OOS検証 | `ztb/evaluation/walk_forward/*` | C1'で兆候が出た条件のみ4splitへ拡張 |
| 取引頻度制御 | `ztb/trading/environment/components/threshold_manager.py` | 固定閾値に加え、動的閾値パラメータをConfigに正式配線 |
| 高頻度報酬 | `ztb/trading/rewards/fast_intraday.py` | HeavyEnv直接適用ではなくFast系分岐で評価 |
| ベースライン | `scripts/v459/run_baselines.py` | RL条件と同窓・同コスト比較を継続 |

---

## 5. 実行順（48時間の現実案）

1. **Day 1前半**: C0'（評価整合 + action分布ロギング）  
2. **Day 1後半**: C1'（4条件の軽量再実験）  
3. **Day 2**:  
   - 改善あり: 4seed化または200K延長  
   - 改善なし: 離散行動 or Fast系へ分岐

---

## 6. 最終コメント

`102#` の問題意識（SACが実収益を作れていない）は正しい。  
ただし、次ステップの順序は修正が必要。まず評価経路の整合性を確保しない限り、報酬設計やアルゴリズム比較の結論が歪む。  
Phase Cは「**評価バイアス除去 → 因果分離 → 分岐判断**」の順で進めるべき。

---

## 7. 追加調査: deterministic eval 0取引の根本原因（2026-02-09）

### 7.1 調査結果サマリ

**根本原因は2つ（複合的）:**

| # | 原因 | 重大度 | 影響 |
|---|---|---|---|
| **ROOT-1** | `DrawdownController.is_emergency_stop` が学習中にラッチし、`position_manager.reset()` で解除されない | **致命的** | eval中の全取引をブロック |
| **ROOT-2** | `SACTrainer.train()` の `finally` ブロックが `env.close()` を呼び、`self.df = pd.DataFrame()` で空化 | 中程度 | 観測値が変わるが取引実行自体は阻害しない（cached arrayで動作継続） |

### 7.2 ROOT-1 詳細: DrawdownController emergency_stop ラッチ

**コードパス:**
```
env.step(action)
  → position_manager.execute_action(BUY, ...)
    → open_position(1, ...)
      → risk_manager.calculate_risk_adjusted_position(...)
        → drawdown_controller.update_portfolio_value(100000, step)
          → control_info["emergency_stop"] = self.is_emergency_stop  ← True!
        → if drawdown_info["emergency_stop"]:
            return {"adjusted_position": 0.0, ...}  ← 全取引ブロック
      → actual_position_size = 0.0 < min_trade_size(0.001)
      → return 0.0  (取引中止)
```

**なぜラッチするか:**

1. 学習中にポートフォリオが100,000→84,978に下落（15.02%ドローダウン）
2. `DrawdownController` の `emergency_stop_threshold = 0.15` (15%) を超過
3. `self.is_emergency_stop = True` が設定される
4. **解除条件**: `portfolio_value > self.peak_value` AND `_check_recovery()` = True
5. 学習後の eval で `reset()` → `portfolio_value = 100,000` だが `peak_value ≥ 100,000`
6. `100,000 > 100,000` は **False**（strictly greater required）→ 解除処理に入らない
7. `_apply_risk_controls()` の正常状態 `else` ブランチは `position_reduction_factor` をリセットするが **`is_emergency_stop` はクリアしない**
8. 結果: **50,000ステップ全てで emergency_stop = True が持続**

**未リセット箇所の特定:**
- `PositionManager.reset()` ([position_manager.py](ztb/trading/environment/components/position_manager.py#L695)): position/pnl/tradesのみリセット。`risk_manager.reset()` 未呼出
- `DrawdownController`: `reset()` メソッド自体が存在しない
- `DynamicPositionSizer.reset()`: 存在するが呼ばれない
- `RiskManager`: `reset()` メソッドが存在しない

### 7.3 ROOT-2 詳細: env.close() による DataFrame 空化

**コードパス:**
```
SACTrainer.train() finally:
  → wrapped_env.close()  (VecNormalize)
    → DummyVecEnv.close()
      → env.close()  (HeavyTradingEnv)
        → self.df = pd.DataFrame()  ← DataFrameが空に!
```

`_deterministic_eval_gate2()` は `trainer.train()` **返却後**に呼ばれるため、
eval時には `raw_env.df` が空。ただし:
- `_price_array`, `_atr_array` はキャッシュされており有効
- `data_manager.df` は独立コピーで保持（1,211,605行の元データ）
- observation_builder は data_manager 経由で特徴量を構築可能
- **取引実行自体は _price_array 経由で動作する**ため、ROOT-2単体では0取引にならない

### 7.4 診断結果の裏付け

| テスト | 結果 | 含意 |
|---|---|---|
| 500step学習 + forced action=0.9 (closed env) | **取引成功** (trades 0→1) | close()後のenvは動作する |
| 500step学習 + model.predict (closed env) | 0取引 | model actionが閾値以下（学習不足） |
| 500step学習 drawdown_controller.is_emergency_stop | **False** (DD=6.7% < 15%) | emergency_stop未トリガー → 取引可能 |
| 50Kstep実験 final_balance | 84,978 (DD=15.02% ≥ 15%) | emergency_stop **トリガー済み** |
| 50Kstep実験 eval trades | 0 (52% actions above threshold) | ROOT-1 が全取引をブロック |
| Fresh env + forced action=0.9 | **取引成功** | env自体の取引ロジックに問題なし |

### 7.5 修正方針

**即座に対応すべき修正（2箇所）:**

1. **`PositionManager.reset()` に `risk_manager` リセットを追加**
   - `DrawdownController` に `reset()` メソッドを追加
   - `is_emergency_stop = False`, `peak_value = 0.0`, `current_drawdown = 0.0` 等の初期化
   - `DynamicPositionSizer.reset()` も呼び出し
2. **`_deterministic_eval_gate2()` で eval前に drawdown_controller を明示リセット**
   - `raw_env.position_manager.risk_manager.drawdown_controller` の状態をクリア
   - または `raw_env.position_manager.risk_manager` 全体を再初期化

**補助的修正:**
3. `DrawdownController._apply_risk_controls()` のロジック修正:
   - 正常状態 (`else` ブランチ) で `self.is_emergency_stop = False` を追加
   - ドローダウンが閾値を下回ったら emergency_stop を自動解除
4. `SACTrainer.train()` の `finally` ブロック: eval用envの参照を保持し、close()前にevalを実行するか、eval後にclose()する

### 7.6 影響範囲

- **全14実験のGate2結果が無効**: emergency_stop ラッチにより全てeval_trades=0
- **学習中の取引にも影響**: 学習後半でemergency_stopが発動すると、以降のエピソードでも取引がブロックされる可能性がある（エピソードリセット時にDrawdownControllerがリセットされない）
- **NB**: 学習中は VecEnv 経由で自動reset → HeavyTradingEnv.reset() が呼ばれるが、PositionManager.reset()→RiskManagerリセットが無いため、学習中のエピソード間でもemergency_stopが持続する
