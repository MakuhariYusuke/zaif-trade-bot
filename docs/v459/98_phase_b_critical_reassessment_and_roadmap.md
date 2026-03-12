# 98# Phase B 批判的再分析と収益化再設計ロードマップ

**Date**: 2026-02-08  
**対象**: `00`, `66`, `95`, `96`, `97` + 実コード/結果JSON照合  
**目的**: 「短期間で高収益」の大義に対し、事実誤認を除去し、次の一手をNo-Go基準込みで再設計する

---

## 0. 結論（先に要点）

1. **現時点でPhase 5移行はNo-Go**。  
2. `97`の主要数値（Gross/Fees/Net）はJSON整合だが、**解釈の一部に構造的な誤り**がある。  
3. 収益性ボトルネックは「手数料」だけでなく、**計測指標の不正確さ・報酬純粋性の未達・評価設計不足**の複合。  
4. 先にやるべきは最適化ではなく、**測定基盤の是正（Gate C0）**。  
5. その上で、**真のPnL基準再実験 → コスト圧縮 → ランダム超過証明**の順で進める。

---

## 1. 97#の事実検証（JSON/コード照合）

## 1.1 数値整合（`results/phase45_p1_baseline/p1_results_20260206_150903.json`）

| 指標 | P1-1 | P1-3 | 差分 |
|---|---:|---:|---:|
| Gross PnL mean | +389 JPY | -306 JPY | +695 JPY |
| Fees mean | 15,394 JPY | 14,708 JPY | +686 JPY |
| Net PnL mean | -15,005 JPY | -15,014 JPY | +9 JPY |
| Net ROI mean | -15.0047% | -15.0144% | +0.0097pt |
| Trades mean | 978.5 | 919.0 | +59.5 |

補足:
- Gross差分の有意性は弱い（4seed/群では強い主張不可）。  
- 参考として全組合せ置換検定（2群4サンプル）では、両側 `p = 0.286`。

## 1.2 重大修正A: BUY:SELL完全対称は「実測」ではない

`HeavyTradingEnv` の `buy_count` / `sell_count` は、`PositionManager`側に当該属性がなければ `trades_count * 0.5` を返すフォールバック実装。  
よって `97`での「BUY=SELL完全一致」は、**実行結果ではなく推定値**の可能性が高い。

- 参照: `ztb/trading/environment/heavy_env/core.py:1618`, `ztb/trading/environment/heavy_env/core.py:1626`
- 参照: `ztb/trading/environment/components/position_manager.py`（`buy_count`/`sell_count`属性なし）

## 1.3 重大修正B: P1-1は「純粋PnL」ではない

`P1-1`設定で `hold_penalty_multiplier=0.0` が指定されているため、`ACTION_HOLD`時報酬が0に潰れる。  
これは「ペナルティ無効」ではなく、**HOLDのPnL情報消去**に近い。

- 参照: `scripts/v459/run_phase45_p1.py:82`
- 参照: `ztb/trading/environment/components/calculators/reward_calculator.py:1280`

さらに `calculate_reward_simple()` には以下が残る:
- `position_change > 0.1` で `-0.1`  
- `dynamic_reward_shaper.shape_reward(...)`  
- `signal_integrator`（enabled時）  
- `asymmetric_reward_scaler.scale_reward(...)`

- 参照: `ztb/trading/environment/components/calculators/reward_calculator.py:1289`

## 1.4 重大修正C: 「HOLD 96%」推定は根拠が弱い

取引回数から直接HOLD比率を推定するのは不適切。理由:

1. `PositionManager` は「すでにLongでBUY」等を**no-op**として処理する。  
2. no-opは取引回数に計上されない。  
3. したがって「取引が少ない=HOLDが多い」とは限らない。

- 参照: `ztb/trading/environment/components/position_manager.py:254`, `ztb/trading/environment/components/position_manager.py:287`

## 1.5 文書整合の軽微不備

`97`に記載の以下パスは現状不一致:
- `scripts/v459/run_phase45_p1_subprocess.ps1`（実体は `.py`）  
- `tests/v459/test_gate05_reward_purity.py`（実体は `tests/unit/trading/components/...`）

---

## 2. 多角的な根因分析（批判的評価）

## 2.1 経済性: エッジがコストを全く超えていない

P1-1平均で:
- 必要粗利/取引（損益分岐）: 約 `15.74 JPY`  
- 実際粗利/取引: 約 `0.35 JPY`  
- **不足倍率: 約44倍**

つまり「コスト削減だけで勝つ」より前に、**取引1回あたりの情報優位を桁で改善**する必要がある。

## 2.2 学習設計: 行動→約定の写像が粗く、報酬帰属が劣化

- 連続行動を閾値離散化し、さらにno-opが多発。  
- 学習信号（reward）と実際の約定結果の対応が崩れやすい。  
- 「学習しているのに約定は改善しない」状態を作りやすい。

## 2.3 評価設計: OOS証拠不足

- `run_phase45_p1.py` で `walk_forward.enabled=False`。  
- `SACTrainer` は評価時も同一df利用がデフォルト（分割しない限りリークリスク）。  
- `train_end_index`未指定は過去ログでも警告済み。

- 参照: `scripts/v459/run_phase45_p1.py:142`  
- 参照: `ztb/training/unified_trainer/algorithms/sac_trainer.py:911`

## 2.4 統計設計: シード数は最低ライン、結論はまだ弱い

- 4seedは初期判断には妥当だが、優劣断定には不足。  
- 「P1-1が良さそう」は仮説として保持し、**確証扱いは不可**。

## 2.5 観測性: サブプロセス実行で詳細ログが消える

`run_phase45_p1_subprocess.py` は `capture_output=True` で子プロセスログを保持せず、JSON末尾のみ抽出。  
これにより、異常時の因果追跡が困難。

- 参照: `scripts/v459/run_phase45_p1_subprocess.py:36`

---

## 3. vXXX教訓の再評価（採用/保留/破棄）

| 系譜 | 再評価 | 今回の適用判断 |
|---|---|---|
| v444 設定伝播バグ | 依然クリティカル | **採用**（Gate 0継続） |
| v456 ペナルティ積層失敗 | 再現性高い教訓 | **採用**（報酬は最小構成から） |
| v457.2 Tiny Edge vs Cost | 現結果と整合 | **採用**（Gross/Net分解必須） |
| v457.3 TTL固定成功 | 強いが市場依存疑い | **条件付き採用**（行動空間簡略化の検証は継続） |
| v451 γ=0.80成功 | 価値ありだが環境差大 | **保留**（神格化しない） |
| v454 逆説的確信 | 再現検証が必要 | **保留**（補助仮説） |
| v435系単発成功談 | 取引回数不足例あり | **過信禁止** |

---

## 4. 再設計ロードマップ（Phase C+）

## Gate C0（最優先: 測定の正しさ）

1. `buy_count/sell_count` を推定値ではなく実測に変更。  
2. 必須KPI追加: `turnover`, `fee_rate_effective`, `executed_trade_rate`, `no_op_rate`, `avg_holding_steps`。  
3. サブプロセス実行で seed別のstderr/stdoutを保存。  
4. `train_end_index` 未指定時は fail-fast。

**完了条件**: 「数値が何を意味するか」をコードレベルで説明できる状態。

## Gate C1（真のPnL基準を再構築）

1. P1-1再定義:
   - `hold_penalty_multiplier=1.0`  
   - `use_simple_reward=True`  
   - dynamic/signal/asymmetricの影響ゼロをテストで証明
2. 必要なら strictパス追加:
   - `reward = pnl * reward_scale` 以外を通さないモード

**完了条件**: 「純粋PnL」と呼べる経路をテストで担保。

## Gate C2（コスト圧縮実験）

軸を最小化して因果分離:
- `continuous threshold`: 0.333 / 0.5 / 0.7  
- `min_holding_period`: 0 / 15 / 30  
- `allow_reverse`: True / False

評価は「ROI」だけでなく:
- 粗利/取引  
- 手数料/取引  
- 取引回数  
- turnover比率

**狙い**: 950取引/50k → 150〜300取引/50k帯へ圧縮し、粗利効率が維持・改善するか確認。

## Gate C3（ランダム超過の証明）

必須ベースライン:
1. Random（同等約定頻度制約付き）  
2. Buy&Hold  
3. シンプルモメンタム

最低設計:
- 4seed × 4window  
- Mann-Whitney + Holm補正 + Cliff’s delta

## Gate C4（改善が出ない場合のピボット）

1. 足種変更（1m中心を縮小し、5m/15m中心へ）  
2. RLの役割縮小（方向予測は教師あり、RLはサイズ/エグジット）  
3. ルール+学習のハイブリッド（まず手数料に勝つ頻度へ）

---

## 5. Go/No-Go基準（Phase 5移行判定）

Phase 5へ進むための最低条件:

1. **測定健全性**: KPIの定義・計測経路が監査可能。  
2. **純粋PnL検証**: PnL-only経路がテストで保証済み。  
3. **ランダム超過**: OOSでRandomを統計的に上回る。  
4. **経済性**: 粗利/取引が手数料/取引に現実的に接近。  
5. **再現性**: seed/windowを変えても結論が維持される。

1つでも未達なら **No-Go（Phase 4.5継続）**。

---

## 6. 直近48時間の実行順（提案）

1. Gate C0実装（実測KPI/ログ保存/train_end_index必須化）。  
2. Gate C1再実験（真PnL-only vs default、4seed）。  
3. Gate C2最小AB（threshold×holding_period、各2seedで粗選別）。  
4. 勝ち筋2条件のみ4seed×4windowへ拡張。  
5. Random/Buy&Hold/Momentum比較を同一評価器で実行。

---

## 最終コメント

`97`は「手数料支配」の問題提起としては有益だが、現状は**判断に使う計器がまだ歪んでいる**。  
まず計器を直す。その後に最適化する。順番を逆にすると、vXXXで繰り返した誤判定を再発する。

