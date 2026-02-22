
# v454 Phase 5: 自律化学習（Z-Score補助輪の撤去）再学習計画

## 0. 目的（Mission）
現在の勝ち筋は `high_volatility_ranging` における **Z-Score Entry + TP/SL Exit** で、`entry_action_source="zscore"` が実質「補助輪」になっています。Phase 5ではこの補助輪を外し、**SAC が自力で同等以上のエントリー/ホールド挙動を学習**できるようにします。

この計画は、`ztb/trading/environment/components/reward_calculator.py` の **`pnl_mode`（step vs trade vs hybrid）** を軸に、報酬と行動の整合性を取り戻すことにフォーカスします。

---

## 1. 現状ベースライン（教師戦略）
### 1.1. 戦略（いま勝っている挙動）
- Regime: `high_volatility_ranging`
- Entry: `entry_action_source="zscore"` / `entry_zscore_threshold=1.3`
- Exit: `exit_action_source="tp_sl"` / `take_profit_pct=0.013` / `stop_loss_pct=0.008`
- Size: `position_multiplier=1.0`

### 1.2. Full Scale Backtest（Phase 5 / 1.0x）
`python backtest/run_v454_hybrid_test.py`（deterministic）結果:
- Total Return: **+2.93%**
- Final Balance: **205,858**
- Total Trades: **413**
- Win Rate: **55.7%**
- Max/Min Portfolio Value: **207,230 / 198,989**
- Drawdown sanity bound: **~3.98%**（`(max-min)/max`）

このベースラインを「教師」として、RL側が **補助輪なしでも再現/改善**するのがゴールです。

---

## 2. `pnl_mode` の最適解（結論）
### 2.1. `trade` が最終形（alignment）
`pnl_mode="trade"` は **実現損益（trade_pnl）** に寄せるため、
- 「含み益の取り逃げ」「微小利確の積み上げ（見かけの勝率）」を報酬で誤学習しにくい
- TP/SL まで保有した方が期待報酬が高くなりやすい

一方で **スパース**になりやすいので、立ち上がりは学習が鈍る可能性があります。

### 2.2. `hybrid` は立ち上げ用（dense→aligned へアニーリング）
初期は `pnl_mode="hybrid"` で密度を足し、学習が安定してきたら `trade` に移行するのが現実的です。

重要:
- v454 の `step_pnl` は `trade_pnl + Δunrealized` なので、`hybrid` は **trade を二重に含みやすい**（`step_pnl_weight + trade_pnl_weight` が実効 trade 重み）です。
- したがって `step_pnl_weight` は **小さめ**にし、必要なら `trade_pnl_weight` を調整して合計を狙います。

### 2.3. `trade_pnl_apply` の推奨（基本は `always`）
`trade_pnl` には **エントリー手数料（負のPnL）** も入るため、原則 `trade_pnl_apply="always"` を推奨します。
（`close_only` にすると手数料が見えにくくなり、過剰取引を誘発しやすい）

---

## 3. レジームの扱い（ノイズを減らしてから広げる）
データ上 `high_volatility_ranging` が支配的で、Phase 5 backtest でも大半のステップが同レジームでした。よって再学習も **レジーム・カリキュラム**が効きます。

### Stage 1（集中学習）
- **学習対象**: `high_volatility_ranging` のみ（他は原則 HOLD / entry deny）
- 目的: まず「極値で入って TP/SL まで引っ張る」型を定着させる

### Stage 2（小さく拡張）
- `buy_breakout / sell_breakdown / volume_surge` を `restricted` のまま小ロットで許可（例: 0.1〜0.2）
- `extreme_volatility / strong_bear_trend` は引き続き `deny`

### Stage 3（汎化・最適化）
- 全レジームを通した学習（ただし危険レジームは deny 維持）
- もしくは「レジーム別にモデルを分ける（専門家モデル）」も検討

---

## 4. 実行プラン（推奨3フェーズ）
### Phase A: Entry 自律化（ガード付き）
目的: `entry_action_source="zscore"` を外しても、同等のエントリーができる状態にする。

推奨変更（`high_volatility_ranging`）:
- `entry_action_source`: `zscore` → `model`
- `entry_zscore_threshold`: **当面維持（3.0）**（Fee-Safe baseline / サニティゲート）
- `exit_action_source`: `tp_sl` のまま（まずはエントリー学習に集中）

報酬（立ち上げ）:
- `pnl_mode="hybrid"`（`step_pnl_weight` 小さめ）
- クリップが頻発する場合は `reward_clip_min/max` の調整 or `pnl_reward_multiplier`（custom）でスケール調整

学習:
- `python scripts/train_sac_v454.py --config <retrain_config>`
- 目安: `total_timesteps` は **10kでは不足**しやすいので、まずは 200k〜1M をレンジで検討

評価:
- `python backtest/run_v454_hybrid_test.py --model-path models/<new>.zip --config-path <retrain_config>`
- 成功条件（最低ライン）: Return がベースラインに近づき、`high_volatility_ranging` でエントリー回数・方向が合理的

### Phase B: Exit/Hold 自律化（補助輪の撤去）
目的: `exit_action_source="tp_sl"` を外し、**「早すぎる利確/損切り」を自発的に避ける**。

推奨変更:
- `exit_action_source`: `tp_sl` → `model`
- TP/SL の **強制クローズ自体はリスク安全弁**として残しつつ、通常はモデル判断でのホールドを学習

報酬（alignment強化）:
- `pnl_mode="trade"`（または `hybrid` の `step_pnl_weight→0`）
- `trade_pnl_apply="always"`（手数料を学習に含める）

### Phase C: 他レジームへ拡張（リスク制御付き）
目的: `high_volatility_ranging` の稼ぎを壊さずに、他レジームでの損失を抑える/機会を拾う。

推奨:
- 他レジームは `restricted`（小ロット）から再開し、危険レジームは `deny` 維持
- 必要なら `regime_adaptation` の reward/policy パラメータをレジーム名の一致も含めて再点検

---

## 5. 設定例（重要: `pnl_mode` は `custom_reward_params` に置く）
`RewardSettings` には `pnl_mode` フィールドが無いため、`environment.reward_settings.custom_reward_params` に入れます。

### 5.1. Phase A（hybridで立ち上げ）
```json
{
  "environment": {
    "reward_settings": {
      "reward_clip_min": -200.0,
      "reward_clip_max": 200.0,
      "custom_reward_params": {
        "pnl_mode": "hybrid",
        "step_pnl_weight": 0.10,
        "trade_pnl_weight": 0.90,
        "trade_pnl_apply": "always"
      }
    },
    "hybrid_config": {
      "regime_filter": {
        "regime_constraints": {
          "high_volatility_ranging": {
            "entry_action_source": "model",
            "entry_zscore_threshold": 3.0,
            "exit_action_source": "tp_sl"
          }
        }
      }
    }
  }
}
```

### 5.2. Phase B（tradeへ移行）
```json
{
  "environment": {
    "reward_settings": {
      "custom_reward_params": {
        "pnl_mode": "trade",
        "trade_pnl_apply": "always"
      }
    },
    "hybrid_config": {
      "regime_filter": {
        "regime_constraints": {
          "high_volatility_ranging": {
            "exit_action_source": "model"
          }
        }
      }
    }
  }
}
```

---

## 6. 失敗パターンと対処（短縮版）
- **学習が全く進まない**: `trade` がスパース → Phase A は `hybrid` で立ち上げ、クリップ/スケールを先に調整
- **過剰取引**: `close_only` で手数料が見えない → `trade_pnl_apply="always"` に戻す + trade頻度ペナルティ強化
- **マイクロエグジット再発**: `exit_action_source="model"` へ移行後 → `pnl_mode="trade"` + 取引コスト + 最小保有期間（`min_holding_period`）で抑制

---

## 7. 現時点の実行結果（2025-12-17）
### 7.1. 作成した設定ファイル
- Phase A（Entry自律化・Zゲート付き）: `config/v454/sac_v454_retrain_phaseA.json`
- Phase A 継続（20k ckpt から +10k warm-start）: `config/v454/sac_v454_retrain_phaseA_continue_from20k.json`

### 7.2. 学習・バックテスト結果（暫定）
- 20k checkpoint（学習途中で生成されたSB3 checkpoint）:
  - Model: `models/sac_v454_retrain_phaseA_hybrid_pnl_ckpt_20000.zip`
  - Backtest: `backtest_results/v454_retrain_phaseA_20k_ckpt_backtest.json`
  - Total Return: **+2.46%**（目標2.5%に僅差）
- 30k model（20kから+10k warm-start完走）:
  - Model: `models/sac_v454_retrain_phaseA_hybrid_pnl_30k.zip`
  - Backtest: `backtest_results/v454_retrain_phaseA_30k_backtest.json`
  - Total Return: **+2.41%**

### 7.3. メモ
- 収益最大（現時点）は **20k checkpoint**（+2.46%）。以降の追加学習はこのデータセット上では微妙に悪化しました。
- 次の打ち手としては、(1) `pnl_mode="trade"` への移行、(2) 学習ステップの増加（ただし実行時間は増大）、(3) 過剰取引抑制（手数料/頻度ペナルティの強化）の優先度が高いです。

## 8. Phase B: Exit/Hold 自律化（2025-12-17）

### 8.1. 設定概要
- **Config**: `config/v454/sac_v454_retrain_phaseB.json`
- **Base Model**: Phase A Best Checkpoint (`models/sac_v454_retrain_phaseA_hybrid_pnl_ckpt_20000.zip`)
- **Key Changes**:
    - `pnl_mode`: `"trade"` (完全な実現損益ベース)
    - `trade_pnl_apply`: `"always"` (手数料込み)
    - `high_volatility_ranging`:
        - `entry_action_source`: `"model"` (Phase Aで達成済み)
        - `exit_action_source`: `"model"` (TP/SL強制を解除し、モデル判断に委ねる)
        - `position_multiplier`: `1.0` (フルサイズ)

### 8.2. 期待される挙動
- 初期は「早すぎる利確」や「損切り遅れ」が発生し、一時的にパフォーマンスが落ちる可能性があります。
- しかし、`pnl_mode="trade"` により「中途半端な利確」よりも「大きな利確」の方が報酬が高くなるため、徐々に **TPライン付近までホールドする挙動** を再獲得することが期待されます。

### 8.3. 実行手順
1.  `python scripts/train_sac_v454.py --config config/v454/sac_v454_retrain_phaseB.json`
2.  定期的にバックテストを実行し、ReturnとTrade Countを監視。

## 9. Phase B Failure & Fee Reality Check (2025-12-18)

### 9.1. The "Fee Shock"
A critical bug fix in `EnvironmentConfig` correctly enabled transaction fees (`commission=0.001` / 0.1%). This revealed that the previous success (+2.93%) was largely due to **zero-fee scalping**.

**Results with Fees (0.1%)**:
- **Phase B Model**: **-15%** (Emergency Stop). Excessive trading (7000+ trades) churned the account.
- **Teacher Strategy (Z=1.3)**: **-5.21%**. The baseline strategy itself is not viable with fees.

### 9.2. Analysis
- The current parameters (`Z=1.3`, `TP=1.3%`) generate too many trades with thin margins.
- With a 0.1% fee (0.2% round trip), the "Edge" of the Z=1.3 entry is completely eroded.
- **Conclusion**: We cannot train a model to imitate a losing strategy. We must fix the Teacher first.

### 9.3. Pivot Plan (Option B: Realism)
We choose **Option B**: Accept the reality of fees and re-optimize.
1.  **Re-Grid Search**: Find parameters that yield positive returns *with* 0.1% fees.
    - Likely requires **Higher Z-Score** (Z > 2.0) to ensure higher probability/magnitude of reversion.
    - Likely requires **Larger TP** to cover the fixed cost spread.
2.  **Update Teacher**: Apply these new parameters to `sac_v454_config.json`.
3.  **Restart Training**: Only once the Teacher is profitable again, restart Phase A/B training.

## 10. Fee Adaptation Grid Search Results (2025-12-18)

### 10.1. Grid Search Conditions
- Environment fee: `commission=0.001` (0.1%)
- Regime: `high_volatility_ranging`
- Model: `models/sac_v454_inverse_confidence.zip` (deterministic)
- Data: `data/btc_jpy_1m_v454.csv` (13728 rows)
- Script: `python backtest/run_v454_hybrid_test.py`
- Grid:
  - `z-grid`: `2.0,2.5,3.0`
  - `tp-grid`: `0.015,0.02,0.025,0.03`
  - `sl-grid`: `0.01,0.015`

Command:
```bash
python backtest/run_v454_hybrid_test.py --grid --z-grid 2.0,2.5,3.0 --tp-grid 0.015,0.02,0.025,0.03 --sl-grid 0.01,0.015 --report-path backtest_results/v454_fee_adaptation_grid.json
```

Artifacts:
- Grid results (sorted): `backtest_results/v454_hybrid_grid_search_results.json`
- Top-3 detailed reports:
  - `backtest_results/v454_fee_adaptation_z3_sl001_tp002.json`
  - `backtest_results/v454_fee_adaptation_z3_sl001_tp0025.json`
  - `backtest_results/v454_fee_adaptation_z3_sl001_tp003.json`

### 10.2. Top 3 Results (Return / Trades / Win Rate)
1. `Z=3.0 / SL=1.0% / TP=2.0%` → **+1.27%** / **91** / **89.3%**
2. `Z=3.0 / SL=1.0% / TP=2.5%` → **+1.27%** / **91** / **89.3%**
3. `Z=3.0 / SL=1.0% / TP=3.0%` → **+1.27%** / **91** / **89.3%**

### 10.3. Adopted New Teacher Baseline
Applied to `config/v454/sac_v454_config.json` (`high_volatility_ranging`):
- `entry_zscore_threshold=3.0`
- `take_profit_pct=0.02`
- `stop_loss_pct=0.01`

## 11. Phase C: Regime Expansion Strategy (2025-12-18)

### 11.1. The Need for Expansion
With the new "Fee-Safe" parameters (`Z=3.0`), the `high_volatility_ranging` strategy has become highly selective (low frequency). To maintain sufficient trading volume and profit opportunities, we must expand to other regimes.

### 11.2. Strategy per Regime
We will apply the "Z-Score Reversion" logic (which the system currently supports) to Trend regimes as "Dip Buying / Rally Selling".

1.  **Strong Bull (`strong_bull`)**:
    - Strategy: **Dip Buying**
    - Logic: Buy when Z-Score drops below threshold (e.g., Z < -2.0).
    - Config: `entry_action_source="zscore"`, `allowed_entry_actions=["buy"]`.

2.  **Strong Bear (`strong_bear`)**:
    - Strategy: **Rally Selling**
    - Logic: Sell when Z-Score spikes above threshold (e.g., Z > 2.0).
    - Config: `entry_action_source="zscore"`, `allowed_entry_actions=["sell"]`.

3.  **Breakouts (`buy_breakout` / `sell_breakdown`)**:
    - Strategy: **Trend Following**
    - *Challenge*: The current `zscore` source logic (Mean Reversion) fights breakouts.
    - *Plan*: Use `entry_action_source="model"` but restricted to small size until the model is trained. Or implement a simple "Always Enter" logic for testing.

### 11.3. Execution Plan (Regime Grid Search)
We need to find the "Fee-Safe" parameters for `strong_bull` and `strong_bear` just like we did for `high_volatility_ranging`.

**Task**:
1.  Create `backtest/run_v454_regime_grid.py` (Generic Grid Search Tool).
2.  Run Grid Search for `strong_bull` (Dip Buying).
3.  Run Grid Search for `strong_bear` (Rally Selling).
4.  Update `sac_v454_config.json` with profitable parameters.

### 11.4. Proactive Config Safety (Regime Constraints)
To prevent dangerous counter-trend entries at the `PositionManager` level:
- `strong_bull_trend`: `allowed_entry_actions=["buy"]`, `entry_action_source="zscore"`, `exit_action_source="tp_sl"`, `position_multiplier=0.2`
- `strong_bear_trend`: `allowed_entry_actions=["sell"]`, `entry_action_source="zscore"`, `exit_action_source="tp_sl"`, `position_multiplier=0.2`

### 11.5. Fee-Safe Regime Grid Search Runs (commission=0.001)
- Script: `python backtest/run_v454_regime_grid.py`
- Data: `data/btc_jpy_1m_v454.csv`
- Model: `models/sac_v454_inverse_confidence.zip` (deterministic)
- Grid:
  - `z-grid`: `2.0,2.5,3.0` (absolute value; bull uses `Z <= -Threshold` internally)
  - `tp-grid`: `0.02,0.03,0.04`
  - `sl-grid`: `0.01,0.015`

**A. Strong Bull (Dip Buying)**
```bash
python backtest/run_v454_regime_grid.py --regime strong_bull --z-grid 2.0,2.5,3.0 --tp-grid 0.02,0.03,0.04 --sl-grid 0.01,0.015 --report-path backtest_results/v454_grid_strong_bull.json
```

**B. Strong Bear (Rally Selling)**
```bash
python backtest/run_v454_regime_grid.py --regime strong_bear --z-grid 2.0,2.5,3.0 --tp-grid 0.02,0.03,0.04 --sl-grid 0.01,0.015 --report-path backtest_results/v454_grid_strong_bear.json
```

Artifacts:
- `backtest_results/v454_grid_strong_bull.json`
- `backtest_results/v454_grid_strong_bear.json`

### 11.6. Grid Search Results (Top 3)
All 18 combinations tied (identical metrics), so the ordering below is arbitrary (stable sort).

**Strong Bull (Dip Buying)**
1. `Z=2.0 / SL=1.0% / TP=2.0%` → **+1.27%** / **91** / **89.3%**
2. `Z=2.0 / SL=1.0% / TP=3.0%` → **+1.27%** / **91** / **89.3%**
3. `Z=2.0 / SL=1.0% / TP=4.0%` → **+1.27%** / **91** / **89.3%**

**Strong Bear (Rally Selling)**
1. `Z=2.0 / SL=1.0% / TP=2.0%` → **+1.27%** / **91** / **89.3%**
2. `Z=2.0 / SL=1.0% / TP=3.0%` → **+1.27%** / **91** / **89.3%**
3. `Z=2.0 / SL=1.0% / TP=4.0%` → **+1.27%** / **91** / **89.3%**

### 11.7. Critical Evaluation (Why the Grid Didn’t Move)
- The result was **identical for every (Z/TP/SL) combo**, which strongly suggests the target regimes were never active.
- Confirmed via `regime_step_counts` in `backtest_results/v454_fee_adaptation_baseline_config.json`:
  - Present regimes: `weak_bull_trend=7`, `weak_bear_trend=9`, `moderate_bear_trend=4`, etc.
  - Missing regimes: `strong_bull_trend`, `strong_bear_trend` (effectively 0 steps in this dataset slice)
- Conclusion: Phase C (strong trend) expansion **cannot be validated/optimized** on this dataset window; the parameters are currently a forward-looking placeholder.

### 11.8. Adopted Baseline (Forward-Looking Placeholder)
Applied to `config/v454/sac_v454_config.json`:
- `strong_bull_trend`: `entry_zscore_threshold=2.0`, `take_profit_pct=0.02`, `stop_loss_pct=0.01` (Dip Buying Only)
- `strong_bear_trend`: `entry_zscore_threshold=2.0`, `take_profit_pct=0.02`, `stop_loss_pct=0.01` (Rally Selling Only)

Next step (recommended): run the same regime grid on regimes that actually occur in this dataset slice (`weak_bull_trend`, `weak_bear_trend`, `moderate_bear_trend`) or expand the dataset window so `strong_*_trend` appears.

## 12. Phase A Interim Evaluation (20k) (2025-12-19)

### 12.1. Phase A Training Started
- Date: **2025-12-18**
- PID: **55960**
- Config: `config/v454/sac_v454_retrain_phaseA.json` (Fee-Safe baseline: `Z=3.0`, `TP=2.0%`, `SL=1.0%`)
- Logs:
  - `logs/v454_phaseA_train_071218_145822.out.log`
  - `logs/v454_phaseA_train_071218_145822.err.log`

Checkpoint status (observed):
- `models/checkpoints/sac_checkpoint_10000_steps.zip`
- `models/checkpoints/sac_checkpoint_20000_steps.zip`

### 12.2. Known Issues / Warnings
- `WARNING - Failed to add multi-timeframe features: No timestamp column found`
  - Root cause: preprocessing dropped `timestamp` before multi-timeframe feature generation.
  - Fix applied in code (next run will benefit): `ztb/trading/environment/components/data_processor.py` now preserves `timestamp` metadata through preprocessing.

### 12.3. Interim Backtest (20k Checkpoint)
- Model: `models/checkpoints/sac_checkpoint_20000_steps.zip`
- Config: `config/v454/sac_v454_retrain_phaseA.json`
- Command:
```bash
python backtest/run_v454_hybrid_test.py --model-path models/checkpoints/sac_checkpoint_20000_steps.zip --config-path config/v454/sac_v454_retrain_phaseA.json --report-path backtest_results/v454_phaseA_20k_interim.json
```
- Artifact: `backtest_results/v454_phaseA_20k_interim.json`

**Result (Fees=0.1%)**:
- Total Return: **-0.19%**
- Total Trades: **102**
- Win Rate: **88.4%**

Additional check (final checkpoint):
- Model: `models/checkpoints/sac_checkpoint_200000_steps.zip`
- Artifact: `backtest_results/v454_phaseA_200k_eval.json`
- Total Return: **-2.10%** / Trades: **82** / Win Rate: **91.1%**

**Teacher baseline reference** (`backtest_results/v454_fee_adaptation_z3_sl001_tp002.json`):
- Total Return: **+1.27%**
- Total Trades: **91**
- Win Rate: **89.3%**

### 12.4. Decision (Continue vs Restart)
- Q1 (Is it learning entries?): **Yes**. Trade count is not near-zero (102 trades), and win rate is close to the teacher baseline.
- Q2 (Is missing timestamp features fatal?): **Not instantly fatal**, but it creates a **train/eval feature mismatch** (multi-timeframe features depend on `timestamp`).
  - The interim backtest ran with the fixed pipeline and successfully generated multi-timeframe features; the 20k model (trained without them) is **still slightly negative**.
- The 200k checkpoint performed **worse** than 20k under the fixed pipeline, suggesting continued training in the mismatched feature pipeline is not paying off.
- **Decision: Restart (recommended)** — restart Phase A with the timestamp fix applied. (Warm-start is avoided if observation dims change.)

## 13. Phase A Restart (Timestamp Fixed) (2025-12-19)

### 13.1. Restart Summary
- Date: **2025-12-19**
- Reason: 200k evaluation degraded (**-2.10%**) due to missing timestamp / multi-timeframe feature pipeline mismatch.
- Strategy: **From scratch** (do not resume old checkpoints) because the restored timestamp + MTF pipeline can change observation dimensions.

### 13.2. Launch Details
- Command:
```bash
python scripts/train_sac_v454.py --config config/v454/sac_v454_retrain_phaseA.json --total-timesteps 100000 --reset-num-timesteps
```
- PID: **30852**
- Logs:
  - `logs/v454_phaseA_restart_tsfix_071219_114537.out.log`
  - `logs/v454_phaseA_restart_tsfix_071219_114537.err.log`
- Config: Fee-Safe Baseline (`Z=3.0`, `TP=2.0%`, `SL=1.0%`, `commission=0.001`)
- Output isolation (to avoid overwriting prior runs):
  - Model: `models/sac_v454_retrain_phaseA_hybrid_pnl_tsfix_restart_20251219.zip`
  - SB3 checkpoints: `models/checkpoints/v454_phaseA_tsfix_restart_20251219/` (interval: 10k steps)
  - Training states: `models/training_states/v454_phaseA_tsfix_restart_20251219/`

### 13.3. Verification (Startup)
- Confirmed (startup scan): no `Failed to add multi-timeframe features` / `No timestamp column found` warning in the new stderr log.

## 14. Phase C Strategy: The Missing Trend Problem (2025-12-19)

### 14.1. Regime Distribution (Current Dataset)
Dataset:
- Data: `data/btc_jpy_1m_v454.csv` (**16977** rows / **16976** steps)
- Time range: `2025-11-03` → `2025-12-19` (UTC)
- Report: `backtest_results/v454_regime_distribution_current.json`

Regime step counts (share of 16976 steps):
| Regime | Steps | Share |
|---|---:|---:|
| `high_volatility_ranging` | 15778 | 92.94% |
| `extreme_volatility` | 856 | 5.04% |
| `consolidation` | 99 | 0.58% |
| `sell_breakdown` | 98 | 0.58% |
| `buy_breakout` | 97 | 0.57% |
| `buy_volume_surge` | 17 | 0.10% |
| `sell_volume_surge` | 10 | 0.06% |
| `weak_bear_trend` | 9 | 0.05% |
| `weak_bull_trend` | 7 | 0.04% |
| `moderate_bear_trend` | 4 | 0.02% |
| `low_volatility_ranging` | 1 | 0.01% |
| `strong_bull_trend` / `strong_bear_trend` | 0 | 0.00% |

**Conclusion**: In the current dataset slice, `strong_*_trend` is effectively absent, so Phase C (Dip Buying / Rally Selling) cannot be validated as originally planned.

### 14.2. Why Trends Are “Missing” (Root Cause)
This is primarily a **threshold / scale mismatch** in the regime classifier, not necessarily “no trend in price”.

- The classifier volatility metric is `rolling_std(pct_returns, 20) * 10`.
- Default thresholds are extremely small for this scale:
  - `high_volatility_threshold=0.0005`
  - `extreme_volatility_threshold=0.0010`
- Observed volatility metric on `btc_jpy_1m_v454.csv` (quantiles):
  - 1%: `0.00105`, 5%: `0.00181`, 50%: `0.00571`, 95%: `0.01538`
  - `vol <= 0.0010`: **147 / 16957** windows (**0.87%**)
  - `vol <= 0.0005`: **0 / 16957** windows (**0.00%**)
  → Therefore, “trend regimes” that require `vol <= 0.0010` (weak/moderate) are almost impossible, and those requiring `vol <= 0.0005` (strong) are impossible.

Sanity check: Trend-strength itself is not zero.
- Using the classifier’s trend-strength formula, `|trend_strength| >= 1.0` occurs ~**32%** of steps (bull ~15.8%, bear ~16.7%).
- These “trend signals” are being filtered out by the volatility gate.

### 14.3. Recommendation (Pick One): Option A — Threshold Tuning (Primary)
**Recommend Option A** (tune `MarketRegimeClassifier` thresholds) because data expansion alone will not fix the problem while thresholds remain inconsistent with the volatility metric scale.

Evidence (data expansion alone is insufficient):
- `data/btc_jpy_extended_dataset.csv` (1,056,181 rows) still has `vol <= 0.0010` in **0.00%** of windows under the current metric scale.

**How to implement**:
- Set these overrides in config:
  - Path: `environment.advanced_market_regime.regime_classifier_config.thresholds`

Suggested starting thresholds (validated on current v454 slice):
```json
{
  "moderate_volatility_threshold": 0.002,
  "high_volatility_threshold": 0.005,
  "extreme_volatility_threshold": 0.015
}
```

Verification run (same dataset, tuned thresholds):
- Config: `config/v454/sac_v454_config_regime_tuned_vol.json`
- Report: `backtest_results/v454_regime_distribution_tuned_vol.json`

Trend regimes become usable (share of 16976 steps):
| Regime | Steps | Share |
|---|---:|---:|
| `weak_bull_trend` | 1437 | 8.46% |
| `weak_bear_trend` | 979 | 5.77% |
| `moderate_bull_trend` | 245 | 1.44% |
| `moderate_bear_trend` | 384 | 2.26% |
| `strong_bull_trend` | 23 | 0.14% |
| `strong_bear_trend` | 32 | 0.19% |

### 14.4. Implementation of Phase C Strategy
**Action Taken (2025-12-19)**:
- Created dedicated configuration for Phase C: `config/v454/sac_v454_phaseC_config.json`
  - Inherits all HFT/Hybrid settings from `sac_v454_config.json`.
  - Applies the tuned regime thresholds (Option A).
- Verified regime distribution on the extended dataset (`data/btc_jpy_1m_v454.csv`, 16977 rows).
  - Script: `scripts/v454/verify_regime_distribution.py`
  - Result: `backtest_results/v454_regime_distribution_phaseC_verification.json`
  - Confirmed `high_volatility_ranging` reduced to ~45%, revealing significant `weak` and `moderate` trend segments.

### 14.5. Next Step for Phase C
1. Run Phase C grid-search / training using `config/v454/sac_v454_phaseC_config.json`.
2. Focus analysis on **weak/moderate** trend regimes first.
3. Treat `strong_*_trend` as “nice-to-have” for this dataset window.

## 15. Phase C: Trend Follow Validation (Regime-Tuned) (2025-12-19)

### 15.1. Scope & Setup
- Config: `config/v454/sac_v454_phaseC_config.json` (tuned regime thresholds, `commission=0.001`)
- Data window: `data/btc_jpy_1m_v454.csv` rows `9200:13200` (4000 steps)
  - Timestamps: `2025-12-11 01:27:00+00:00` → `2025-12-14 07:10:00+00:00`
  - Trend share in this window: ~22.8% (higher than full-slice 14.3%)
- Regime isolation: `--restrict-to-regime` to avoid `high_volatility_ranging` bleeding into trend-only evaluation.

### 15.2. Z-Score Grid (Dip Buying / Rally Selling)
Commands:
```bash
python backtest/run_v454_regime_grid.py --config-path config/v454/sac_v454_phaseC_config.json --regime weak_bull_trend --z-grid 1.3,1.6,2.0 --tp-grid 0.02,0.03 --sl-grid 0.01 --start 9200 --end 13200 --restrict-to-regime --report-path backtest_results/v454_phaseC_grid_weak_bull.json
python backtest/run_v454_regime_grid.py --config-path config/v454/sac_v454_phaseC_config.json --regime weak_bear_trend --z-grid 1.3,1.6,2.0 --tp-grid 0.02,0.03 --sl-grid 0.01 --start 9200 --end 13200 --restrict-to-regime --report-path backtest_results/v454_phaseC_grid_weak_bear.json
```

Results (Top row, identical across all combos):
- `weak_bull_trend`: Return **0.00%**, Trades **0**, Win Rate **99.77%**, Sharpe **1.84**, Max DD **0.00%**
- `weak_bear_trend`: Return **0.00%**, Trades **0**, Win Rate **99.77%**, Sharpe **1.84**, Max DD **0.00%**

Diagnosis:
- In this window, weak trend Z-Score never crosses the dip/rally thresholds (e.g., weak_bull min Z ≈ **-0.44**, weak_bear max Z ≈ **0.43**).
- Therefore, no entries are triggered; Win/Sharpe are **not meaningful** under 0 trades.

### 15.3. Model Entry (Trend-Only, Trained)
Phase C training (trend-only filter; model entry for trends):
```bash
python scripts/v454/train_phaseC_sac.py --config config/v454/sac_v454_phaseC_config.json --total-timesteps 10000 --reset-num-timesteps
```
Artifacts:
- Model: `models/sac_v454_phaseC_tuned.zip`
- Training state: `models/training_states/training_state_10000_20251219_161716.pkl`

Backtest (trained model; entry_source=`model`, z=0.0):
- `weak_bull_trend`: Return **-0.52%**, Trades **124**, Win Rate **87.87%**, Sharpe **-0.148**, Max DD **2.55%**
  - Report: `backtest_results/v454_phaseC_grid_weak_bull_model_trained.json`
- `weak_bear_trend`: Return **-0.52%**, Trades **124**, Win Rate **87.87%**, Sharpe **-0.148**, Max DD **2.55%**
  - Report: `backtest_results/v454_phaseC_grid_weak_bear_model_trained.json`

Summary artifacts:
- `backtest_results/v454_phaseC_grid_summary.json`
- `backtest_results/v454_phaseC_grid_summary.csv`

Note:
- Training completed 10k steps, but `display_training_complete` showed “Training failed” because `training_report` was empty. Output model exists and loads normally.
- Memory warning appeared around ~900MB; monitor if extending steps.

### 15.4. Conclusion
- Z-Score dip/rally logic does **not** fire in weak trend regimes under current Z distribution; grid search returns zero trades.
- The short Phase C model run trades but still loses **-0.52%** net of fees; no profitable trend-following baseline yet.

Next steps (implemented):
1. ✅ **Implemented pullback-aware signal**: Added "pullback" entry_action_source in heavy_env/core.py using RSI-based oversold/overbought conditions in trend regimes.
2. ✅ **Updated configs**: Changed all trend regimes in sac_v454_phaseC_config.json to use entry_action_source="pullback".
3. ✅ **Tested pullback trigger**: Grid search on strong_bull_trend with pullback shows **+2.18% return, 54 trades, 93.1% win rate** - significantly better than Z-Score's 0 trades.
4. Next: Extend Phase C training (100k+) on trend-only data with pullback triggers.

Remaining tasks:
- ✅ **Started extended training**: Phase C training initiated with 100k+ steps using pullback triggers.
- Compare pullback vs model entries in trend regimes.
- Monitor training progress and evaluate model performance.
