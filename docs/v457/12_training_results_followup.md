# 12. Training Results Follow-up（追記）

## 1. vXXXシリーズの参考点（張り付き対策）
1) **v431（対称閾値とHold Bonus）**  
   - 「値張り付き」対策として **対称アクション閾値** を導入。  
   - HOLDを罰するより、適度なHOLDの**正のボーナス**化で偏りを緩和。  
   - 参照: `docs/v431/sac_v431_readme.md`

2) **v3.6.6（明示的アクションペナルティ）**  
   - `hold_action_penalty=0.2` など、**PnLがゼロでもHOLDが不利**になる設計。  
   - 多様性の強制フェーズ（balanced/pnl_focused）の有効性を示唆。  
   - 参照: `docs/analysis/V366_COMPREHENSIVE_ANALYSIS.md`

3) **v394（ent_coefでは解決しない）**  
   - エントロピー強化ではHOLD偏重が崩れず、**報酬強度の不足が原因**という結論。  
   - hold_penalty_weight / consecutive_hold_penalty の増強が重要。  
   - 参照: `docs/training/v394_final_report.md`

4) **v4.0.1（環境設定の不一致がHOLD偏重を生む）**  
   - 訓練時の環境設定が推論側で無視され、**取引不可→HOLD固定**に。  
   - v457でも「train/backtestの設定不一致」は再発リスク。  
   - 参照: `docs/bugs/CRITICAL_FIX_HOLD_BIAS_v4.0.1.md`

## 2. コード面の改善点（v457）
1) **報酬スケールの不整合**  
   - `docs/v457/11_training_results_analysis.md` では `1/100,000` と記載。  
   - 実際は `reward / 100` ＋ `[-0.1, 0.1]` クリップ。  
   - 参照: `ztb/trading/environment/fast_intraday_env_v456.py`  
   → 分析前提がずれている可能性が高い。

2) **ペナルティが“微小ポジション逃げ”を許す**  
   - `compute_hft_reward` は `abs(position)/max_position` でペナルティ縮小。  
   - 0.01 BTCでの「居座り」が最適解化する構造。  
   - 参照: `ztb/trading/rewards/fast_intraday.py`

3) **Backtestでreward_paramsが未適用**  
   - 学習時は `reward_settings` を注入しているが、  
     `scripts/v457/backtest.py` は同注入なし。  
   - 学習と評価で報酬条件が異なる可能性。  

4) **Backtestのデータ参照バグ**  
   - `FastIntradayEnvV456` は `self.df = None` にしている。  
   - `scripts/v457/backtest.py` では `env.df` を参照しており、  
     実データ出力が破綻する。  

5) **ADX/DIがランダム生成**  
   - `scripts/v457/train.py` / `scripts/v457/backtest.py` で  
     ADX/DIが `np.random` により生成されている。  
   - “同じデータでも学習挙動が変動” → freeze検証が困難。  

## 3. 具体的な追加対策（vXXX由来）
1) **対称閾値の導入（v431）**  
   - `action_threshold` を設け、微小ポジションは強制ゼロに。  
   - freezeが「微小保有」なら即効性が高い。  

2) **HOLD罰則の“固定額化”**  
   - v3.6.6系と同様に、**保有している事実**に罰則を課す。  
   - `max_position` スケール依存を減らす。  

3) **報酬強度をPnLスケールに合わせる**  
   - v394の示唆通り、ent_coefではなく報酬強度を再設計。  
   - クリップ幅・除数の再設定が必要。

4) **train/backtestの設定整合性確認**  
   - v4.0.1の教訓通り、環境設定は必ず共有する。  
   - `reward_settings`, `max_position`, `transaction_cost` の一致を必須化。

## 4. 最小の確認タスク（提案）
1) `reward_info` を学習ログに出し、penaltyが実際に効いているか確認。  
2) Backtestにも `reward_settings` を注入し、train/backtest差を排除。  
3) ADX/DIの乱数を廃止し、再現性を確保。  

---

以上を踏まえると、v457のfreezeは「ロジック不足」よりも  
**(a) スケール設計の不一致**と**(b) 小ポジション逃げ道**の影響が大きいと考えられます。
