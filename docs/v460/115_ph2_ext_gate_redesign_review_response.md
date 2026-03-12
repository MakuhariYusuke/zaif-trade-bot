# 115# G1.1 二段階ゲート再設計 — 外部レビュー回答

> 作成日: 2026-02-19  
> 対象: `114_ph2_ext_gate_redesign_review.md`  
> レビュア: GPT-5.3-Codex  
> 結論: **二段階化は妥当。ただし現案は「SkipGate有効性未検証のまま attempted 指標へ依存」しており、Gate設計としては未完成。**

---

## 0. 先に結論（実務提案）

- **Q10.1 二段階化**: 妥当（採用推奨）。ただし 72h Kill は「単一p値判定」ではなく **効果量条件を併設** すること。
- **Q10.2 閾値**: `fill_rate 90→70` は「attempted限定なら」概ね妥当。`AS 20→35` は緩すぎるため **30%推奨**。
- **Q10.3 attempted分母**: 条件付き妥当。**SkipGate有効性ゲート(S0)** を先に満たす場合のみ許可。
- **Q10.4 72h回復解釈**: 「Killしなくて正解」の断定は不可。**適応効果とレジーム変化の混在**。
- **Q10.5 互換性**: `000#`, `009#`, `014#`, `gate_thresholds.yaml`, `fill_quality.py`, `run_fill_test.py` をセット改訂する。
- **Q10.6 見落とし**: **分母定義の不一致（raw/clean/attempted）** と **SkipGate有効性の監査欠落** が最大リスク。

---

## 1. 事実確認（ログ/実データクロスチェック）

### 1.1 114記載値の再現性

`results/v460/fill_test/fill_records_*.jsonl` の時系列先頭 1057 件で再計算すると、主要値は概ね再現。

- total=1057, filled=714, fill_rate=67.55%
- skip=86 (8.14%), attempted_fill_rate=73.53%
- attempted_cancel_ratio=26.47%
- queue_wait_median=12.83s
- pnl30_mean=-0.1957 bps
- AS_ratio=31.79%

### 1.2 現行公式集計との差分（重要）

`run_fill_test.py --results-only` では quarantine 適用後の **clean=907/1072** で判定され、現行閾値は既に `min_fill_rate_p90=0.85`。

- G1.1 結果: FAIL（E1, E5）
- E1=0.623（threshold=0.85）
- E5=0.279（threshold=0.20）

> 示唆: 114の「90%基準」は文書起点として正しいが、実運用の config は既に 85%。
> 今回の再設計は、**仕様書だけでなく実閾値ファイルとの整合** を同時に取る必要がある。

---

## 2. Q10.x への回答

## Q10.1 [Critical] 二段階分割の設計原理

**判断: 妥当（採用）**

**根拠**
- 72h 時点で明確な失敗を早期排除する設計思想は、資本・時間効率に合致。
- 168h を完全廃止せず qualification を残しており、週末/曜日バイアスにも対応可能。
- `000# §3.8` の枝番規則 (`.1`, `.2`) と整合。

**要修正点**
- 72h判定は p値のみだと不安定。**p値 + 効果量（平均/CI）** の二条件化が必要。

**代替案**
- 3段階(24/72/168)は運用複雑度が上がる割に便益が限定的。現時点では **2段階が最適**。

---

## Q10.2 [Critical] 閾値設定の妥当性

### (A) fill_rate 90→70
**判断: 要修正（方向性は妥当）**

- attempted基準なら 70% は妥当域。
- ただし attempted のみだと粉飾余地があるため、**overall_fill_rate 下限を併設**すべき。

**推奨**
- Kill: `attempted_fill_rate >= 60%`（維持）
- Full: `attempted_fill_rate >= 70%`（維持）
- 追加: `overall_fill_rate >= 62%`（SkipGate過剰回避）

### (B) AS_ratio 20→35
**判断: 要修正（35は緩すぎ）**

- 現況31%付近をそのまま通す設計は、改善駆動が弱い。
- SkipGate導入戦略としては **30%以下** を目標に置く方が妥当。

**推奨**
- Full: `AS_ratio <= 30%`

### (C) PnL30 α=0.01（Kill）
**判断: 要修正（単独条件としては不十分）**

- 誤棄却低減には寄与するが、真の悪化戦略を見逃す可能性がある。

**推奨（Kill）**
- `p < 0.02` **かつ** `mean_pnl30 <= -0.8 bps` の同時成立で FAIL
- もしくは `95%CI 上限 < -0.3 bps` で FAIL

---

## Q10.3 [High] attempted ベースの統計的妥当性

**判断: 条件付き妥当**

SkipGate有効性が未確認のまま attempted 分母へ全面移行するのは危険。従って **S0: SkipGate有効性ゲート** を前置きすべき。

### S0（新設）提案
次のいずれかを満たす期間のみ attempted 指標を正規採用:

- OOT AUC ≥ 0.55（AS分類）
- または Top-decile lift ≥ 1.20
- かつ skip_gate_ratio が 5%〜20% に収まる

S0未達時は、判定分母を overall に戻す（もしくは overall と attempted の両方を併記し conservative に判定）。

---

## Q10.4 [High] 72h遡及の解釈

**判断: 「Killしなくて正解」断定は不可（要修正）**

72h→141h 回復は、
- param_adapter 適応
- レジーム変化
- 平均回帰
の混合で説明可能。単一事例で α=0.01 を固定するとカーブフィット化しやすい。

**推奨運用**
- 72h時点は「Kill判定」+「Watch判定（黄信号）」を分ける。
- Watch条件（例）: `p < 0.05` かつ `mean < -0.3bps` → 継続だがパラメータ凍結/監視強化。

---

## Q10.5 [Medium] 000#改訂時の互換性

**判断: 009/014 も同時改訂が必要（archiveのみは不可）**

### 必須改訂対象
1. `docs/v460/000_ph0_plan_project_proposal.md`
   - `§3.3` を `G1.1-quick` / `G1.2-full` に更新
   - `§3.9` の中止条件を新ゲート文言へ整合
2. `docs/v460/009_ph2_plan_g1_1_exec.md`
   - 実行計画を二段階運用に改訂（旧E1-E5の単段前提を解消）
3. `docs/v460/014_ph2_plan_completion_and_transition.md`
   - 移行条件を `G1 + G1.2 PASS` に更新（G1.1 は中止/継続判定ゲートへ）
4. `configs/v460/gate_thresholds.yaml`
   - `g1_1_quick_exec`, `g1_2_full_exec` を新設
5. `ztb/metrics/fill_quality.py`
   - `g1_1_judgment()` を後方互換ラッパにし、
     - `g1_1_quick_judgment(...)`
     - `g1_2_full_judgment(...)`
     を分離
6. `scripts/v460/run_fill_test.py`
   - `--results-only` 出力で quick/full を両方返す

---

## Q10.6 [Medium] 見落とし観点

**判断: 見落としあり（要補強）**

1. **分母の多重化問題**
   - overall / clean / attempted が混在。Gate議論がぶれる。
   - 対応: 判定時に必ず3系列を同時出力。

2. **SkipGate監査不足**
   - 「どれだけスキップしたか」だけでなく「スキップ品質（識別力）」を監査する必要。

3. **Cancel理由の内訳管理不足**
   - timeout / status_unknown / postonly_reject の混合は運用改善の優先度を誤らせる。

4. **HFT一般論への過度依存は非推奨**
   - Coincheck BTC/JPY は板特性が独特。閾値は他市場移植でなく実測回帰で管理すべき。

---

## 3. 推奨する最終ゲート仕様（改訂版）

## 3.1 G1.1-quick (72h Kill)

判定: `72h` または `n_attempted >= 300` の早い方

- K1 attempted_fill_rate >= 60%
- K2 attempted_cancel_ratio <= 40%
- K3 queue_wait_median <= 120s
- K4 PnL30: `p < 0.02` かつ `mean <= -0.8bps` で FAIL
- K5 累積実損 < 10,000 JPY
- K6 skip_gate_ratio <= 25%（緊急ブレーキ）

## 3.2 G1.2-full (168h Qualification)

- F1 attempted_fill_rate >= 70%
- F1b overall_fill_rate >= 62%（新設）
- F2 attempted_cancel_ratio <= 30%
- F3 queue_wait_median <= 60s
- F4 PnL30: 有意な負を示さない（現行維持）
- F5 AS_ratio <= 30%（35→30 推奨）
- F6 skip_gate_ratio <= 20%
- F7 calendar_coverage >= 7 days
- F8 n_attempted >= 500
- F9 S0 (SkipGate有効性) PASS

---

## 4. 実装順（低リスク）

1. `gate_thresholds.yaml` に quick/full/S0 を追加（既存キーは残す）
2. `fill_quality.py` に quick/full 判定関数を追加（互換維持）
3. `run_fill_test.py --results-only` を拡張し、旧 `g1_1_judgment` も出力
4. `000/009/014` 文書を同コミットで同期
5. 72hリプレイ検証（過去jsonlで回帰）→ 168h本番

---

## 5. 最終判定サマリ（依頼形式）

- Q10.1: **妥当（採用）** / 二段階は正しい。p値単独判定は要修正。  
- Q10.2: **要修正** / fill 70は条件付き妥当、AS 35は緩すぎ、α=0.01単独は不十分。  
- Q10.3: **要修正** / attempted採用にはS0前提が必須。  
- Q10.4: **要修正** / 72h回復の断定解釈は不可。Watch層を追加。  
- Q10.5: **妥当（ただし同時改訂必須）** / 000だけでなく009/014とコード更新が必要。  
- Q10.6: **要補強** / 分母統一・SkipGate監査・cancel内訳分離を追加。

---

## Appendix: 本レビューで参照した実体

- `docs/v460/114_ph2_ext_gate_redesign_review.md`
- `docs/v460/000_ph0_plan_project_proposal.md`
- `docs/v460/009_ph2_plan_g1_1_exec.md`
- `docs/v460/014_ph2_plan_completion_and_transition.md`
- `docs/v460/095_ph2_codex_review_v3.md`
- `configs/v460/gate_thresholds.yaml`
- `ztb/metrics/fill_quality.py`
- `results/v460/fill_test/fill_records_*.jsonl`
- `results/v460/fill_test/logs/fill_test.log`
