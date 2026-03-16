# 437# 432#-436# 横断レビュー: 独立検証・val_ratio 感度曲線・実装反映

| 項目 | 内容 |
|---|---|
| 番号 | 437# |
| 分類 | ph4_rev (Phase4 Review) |
| 対象 | 432# / 433# / 434# / 435# / 436# |
| 前提 | 426# P0-P4 実装完了 (27d58c7db), S2 結果取得済, S3 結果取得済 |
| 目的 | 5文書のクロスレビュー + 実装済P4の反映確認 + val_ratio感度曲線の構築 |

---

## §0 Executive Summary

432#-436# の5文書は、異なる視点・異なるAIから概ね同一方向を指している:

> **SAC 単体の強化は ROI が低い。収益ボトルネックは toxic participation。次の一手は Toxicity Veto。**

この結論に対して本レビューは **概ね同意** するが、5文書を精読した結果、以下の修正・補足が必要である。

| # | 指摘 | 種別 |
|---|---|---|
| F1 | S2 (val_ratio=0.05) が G3 PASS (pf=1.150)、S3 (val_ratio=0.10) が G3 FAIL (pf=1.029) — 崩壊は 42-84日の間 | 新事実 |
| F2 | P4 (val_ratio compliance E7) を実装済。S2 は E7 未適用で走ったため G3 PASS は無効 | 実装反映 |
| F3 | 436# の toxicity grading 実装場所が誤記 (cycle_gate_aggregator → orchestrator_mid_cycle) | 事実修正 |
| F4 | 435# が全面同意しかしていない — second opinion の体をなしていない | 構造的懸念 |
| F5 | 5文書共通の盲点: toxicity veto の false positive コスト（参加率低下）が未評価 | 分析不足 |
| F6 | 432# P0 ceiling の正誤表は 436# で修正済だが、432# 本文は未修正 | 文書衛生 |

---

## §1 val_ratio 感度曲線 — S2 結果による更新

### §1.1 完成した4点データ

S2 (`v460_g2train_seed42_20260314_224332.json`) および S3 (`v460_g2train_seed42_20260315_001238.json`) の結果 JSON を検証した。

| 実験 | val_ratio | OOS日数 | pf_median | sharpe | roi_seed_std | G3判定 | E7 (≥0.10) |
|---|---|---|---|---|---|---|---|
| Original | 0.02 | ~17 | 1.145 | 5.72 | 0.0016 | PASS | **不適合** |
| S2 | 0.05 | ~42 | 1.150 | 7.16 | N/A | PASS | **不適合** |
| **S3** | **0.10** | **~84** | **1.029** | **1.23** | **seed間極大** | **FAIL** | **適合 ✓** |
| S1 | 0.20 | ~169 | 1.049 | 1.01 | 0.0360 | FAIL (marginal) | 適合 |

### §1.2 S2 結果の解釈

S2 の pf_median=1.150 は Original (val_ratio=0.02, pf=1.145) と**ほぼ同値**。これは重要な示唆を含む:

1. **val_ratio=0.02→0.05 で劣化がない**: OOS を 17 日→42 日に倍以上延長しても pf が下がらない。つまり最初の 42 日間は SAC の学習が有効に機能している
2. **崩壊は 42日以降に始まる**: S1 (val_ratio=0.20, OOS 169日) の pf 急落 (1.150→1.049) は、42日～169日の区間で崩壊が集中していることを示す
3. **S3 (val_ratio=0.10, OOS ~84日) が分水嶺** → §1.3 で確認

### §1.3 S3 結果: 崩壊は 42-84日の間に発生

S3 (val_ratio=0.10, OOS ~84日) の結果:

- **G3 FAIL** (pf_median=1.029, threshold=1.05)
- **pf_worst=0.856** — S1 (pf_worst=0.95 付近) より悪い
- **val_ratio_compliance (E7): PASS** — P4 実装後に走った初の実験で E7 が正常動作

**seed 別データ**:

| seed | pf | sharpe | max_dd | gross_roi | 評価 |
|---|---|---|---|---|---|
| 42 | 1.121 | 5.42 | 0.004 | +1.74% | ◎ 健全 |
| 123 | 1.446 | 12.66 | 0.006 | +7.13% | ◎ 最良 |
| 456 | 0.856 | -8.07 | 0.031 | -2.40% | ★ 壊滅 |
| 789 | 0.938 | -2.96 | 0.016 | -1.17% | ★ 不良 |

**seed 間分散が極大**: seed 123 (pf=1.446) と seed 456 (pf=0.856) の差は 0.59。20K step の SAC は初期重みに対して**極めて不安定**。half の seed が壊滅的に失敗する。

### §1.4 val_ratio 感度曲線の全貌

```
pf_median
1.15 ┤ ●─────────●
     │ (0.02)    (0.05)
1.10 ┤
     │
1.05 ┤- - - - - - - - - - - - - G3 threshold - -
     │                     ●         ●
1.03 ┤                    (0.10)    (0.20)
     │
     └──────────────────────────────────────── OOS日数
      17        42        84       169
```

**崩壊の cliff edge は 42-84日の間に存在**する。pf は 0.05→0.10 で 1.150→1.029 に急落（-0.121）。426# が報告した mid 期崩壊（~57-113日）と完全に整合。

### §1.5 P4 val_ratio compliance の影響

本セッションで実装した G3 E7 チェック (val_ratio >= min_val_ratio=0.10):

- S2 は P4 コード変更**前**に走ったため、結果 JSON に `val_ratio_compliance` チェックがない
- 仮に E7 が適用されていた場合、S2 (val_ratio=0.05) は **G3 FAIL** となる
- **これは意図通りの動作**: 000# §3.5 の規定に従い、val_ratio < 0.10 の G3 PASS は公式に認めない

### §1.6 retrain 間隔の上界推定

感度曲線から、SAC モデルの有効期間は **最大 ~42日**。安全マージンを含めると:

- **推奨 retrain 間隔: 21-30日** (有効期間の 50-70%)
- retrain_scheduler.py のデフォルト retrain_interval_sec=7200 (2時間) は大幅に短い — これは sidecar 用の incremental retrain であり、full retrain とは用途が異なる
- 436# Phase 2 のretrain有効化は S3 結果に基づき「30日以内の定期 retrain」として具体化すべき

S2 の PASS は「42日以内では SAC が機能する」という情報として有用だが、G3 Gate としては不適合。

---

## §2 各文書の個別レビュー

### §2.1 432# (Fill Records Deep Dive)

**評価: 有益。データに裏付けられた分析で、行動可能な洞察が豊富。**

| 判定 | 項目 |
|---|---|
| ✅ 強い | Skip Gate r=-0.010 — 事実上機能していない |
| ✅ 強い | buy+ranging PF=0.766 — 最大ボリュームかつ最悪 PF |
| ✅ 強い | AS=True → win%=0% — AS 回避が収益支配 |
| ✅ 強い | reprice 悪化 (0回 -0.221 → 1回 -0.728 → 2回 -2.213) |
| ✅ 有益 | regime confidence=1.0 が最悪 (AS=37%) — overfitting 示唆 |
| ⚠️ 要修正 | P0: ceiling=0.15 一律 → 現行 buy=0.20, sell=0.50 (436# で正誤表済) |
| ⚠️ 要修正 | §1.3 の ceiling 論は buy 側のみ有効。sell 側は ceiling 制約なし |

**未検証の論点**: §11 の decision_path 分析で `unknown` パスが 2,440 fills（全体の69%）を占め PnL=-0.344。この `unknown` の正体（パス分類のバグ？ログ欠落？）が特定されていない。

### §2.2 433# (Advanced Microstructure Edge Ideas)

**評価: 発想は良いが、実装コスト・副作用の見積もりが欠如。**

| アイデア | 評価 | 根拠 |
|---|---|---|
| Toxicity Sidecar | ✅ 最有力 | as_classifier.py 328行 production-ready、fill_records で即学習可能 |
| BitFlyer lead-lag | ⚠️ 二軍 | 434# 指摘通り arb ではなく lead-lag。REST 遅延・stale fusion リスク |
| Queue management | ⚠️ 二軍 | Public L2 では真の queue position は不可知。heuristic に留まる |
| Committee (ensemble) | ⚠️ 要再設計 | Hard AND は参加率崩壊リスク。434# の weighted veto が正しい |

**434# が指摘した制御文字問題**: 独立検証の結果、可視的な制御文字は確認できず。ゼロ幅Unicode文字の可能性は排除できないが、内容の読解には支障なし。

### §2.3 434# (横断レビュー by Codex)

**評価: 5文書中最も批判的で最も有用。fact-check が堅実。**

| 判定 | 項目 |
|---|---|
| ✅ 優れている | 432# ceiling config の fact-check (0.15→buy=0.20,sell=0.50) |
| ✅ 優れている | 426# の「optimal val_ratio 0.05-0.10」は未実証という指摘 |
| ✅ 優れている | walk-forward を val_ratio 探しより優先すべきという提言 |
| ✅ 優れている | Committee は hard AND ではなく weighted veto という設計指針 |
| ⚠️ 軽微な誤り | §1.3 で `cycle_gate_aggregator.py` に toxicity grading ありとするが、実際は `orchestrator_mid_cycle.py` L129-130 の `_assess_buy_toxicity()` / `_assess_sell_toxicity()` |
| ⚠️ 補強が必要 | §2.3 の walk-forward 資産言及は存在確認だけ。SAC retrain への適用可能性は未評価 |

### §2.4 435# (Gemini Second Opinion)

**評価: 方向性は正しいが、second opinion としての独立性に疑問。**

435# は 434# の全結論に「完全同意」しており、独自の反論・補足・代替案がない。「full agreement」は second opinion の機能を果たしていない。有用な second opinion には少なくとも以下が必要:

- 同意する点と不同意の点を分けて述べる
- 独自の evidence を提示する
- 434# が見落としている論点を指摘する

435# が唯一独自に追加したのは「Ceiling Observability の急務性」の強調だが、これは 431# で既に議論済み。

**結論**: 435# は「追認」としては価値があるが、「second opinion」としての批判的機能は不十分。

### §2.5 436# (統合検証 + SAC 責務再定義)

**評価: 最も包括的。独立検証と具体的ロードマップが優秀。**

| 判定 | 項目 |
|---|---|
| ✅ 優れている | 全7資産の production-ready 検証 (LOC + 参照数 + 機能確認) |
| ✅ 優れている | §5 正誤表で 432# の ceiling 誤記を明示的に修正 |
| ✅ 優れている | §4 Phase 1-3 ロードマップの具体性 |
| ✅ 優れている | §2.1 損失の因果構造図 |
| ⚠️ 軽微な誤り | §1.2 で toxicity grading が cycle_gate_aggregator にあるとする (実際は orchestrator_mid_cycle) |
| ⚠️ 欠落 | S2/S3 実験結果への言及がない (436# 執筆時にはまだ走っていなかったため) |
| ⚠️ 欠落 | P4 (val_ratio compliance E7) 実装への言及がない |

---

## §3 5文書共通の盲点

### §3.1 Toxicity Veto の False Positive コスト

全文書が toxicity veto を最優先としているが、**skip 率上昇による参加率低下のコスト** を定量的に評価していない。

432# §11 によると、既に以下の cancel が発生:
- skip_gate: 1,375件 (18.0%)
- sell_dynamic_kill: 1,103件 (14.4%)
- buy_dynamic_kill: 648件 (8.5%)
- balance_forced_skip: 377件 (4.9%)

fill率は全体で約31% (3,535 fills / 11,136 records)。Toxicity Veto を追加した場合:

- prob_toxic > 0.8 で skip → AS=True の 960件の一部を回避
- しかし false positive で AS=False の taking も skip → 利益機会の損失

**最低限必要な検証**: as_classifier の ROC 曲線上で、「skip top-20% による PnL 改善量」が「skip による参加率低下の損失」を上回ることの確認。as_classifier.py にはこの skip simulation が内蔵されている (L161-183) が、現在の fill_records で実際に回したエビデンスがない。

### §3.2 Walk-forward と Single-split の比較実験がない

434# / 435# / 436# 全てが walk-forward retrain を推奨しているが、**SAC を walk-forward で retrain した場合に G3 相当の指標が single-split より改善するか** は未検証。

既存資産:
- `ztb/evaluation/walk_forward/splitter.py` — 220行、expanding window + embargo
- `scripts/v460/ml/walk_forward_as.py` — 323行、AS classifier 用

しかしこれらは **XGBoost/AS classifier 用** であり、**SAC の retrain に直接適用可能かは未確認**。SAC の retrain は `sac_retrain_scheduler.py` (880行) が担当するが、これは walk-forward splitter とは統合されていない。

### §3.3 sim-live gap の定量化手段がない

434# §5.1 が指摘した通り、sim-live gap はまだ閉じ切れていない。だが5文書のいずれも、**gap の大きさを計測する指標** を定義していない。

提案: sim-live gap index = |G3_pf_median - live_PF_30d|。これを定期的に計測し、gap が閾値を超えたら retrain トリガーとする。

---

## §4 436# への補足・修正事項

436# は最も完成度が高いが、以下の点で更新が望ましい:

### §4.1 事実修正

| 箇所 | 436# の記述 | 修正 |
|---|---|---|
| §1.2 資産検証表 | "cycle_gate_aggregator: toxicity grading (240#/241#)" | toxicity 評価は `orchestrator_mid_cycle.py` L129-130 (`_assess_buy_toxicity()`, `_assess_sell_toxicity()`)。cycle_gate_aggregator は v2 proportional boost (374#) のみ |

### §4.2 追加すべき情報

| 項目 | 内容 |
|---|---|
| S2 結果 | val_ratio=0.05, pf_median=1.150, G3 PASS (但し E7 不適合 → 公式 PASS 無効) |
| P4 実装 | gate_judgment_core.py に E7 (val_ratio_compliance) チェック実装済 (27d58c7db) |
| P2 実装 | reward_tuned configs 3件を archived/ に移動済 (27d58c7db) |
| P1 実装 | retrain_scheduler ops スクリプト + hot_swap 連携実装済 (27d58c7db) |

### §4.3 ロードマップの修正

436# §4 Phase 2 「SAC Retrain」について: retrain_scheduler.py (880行) は完全実装済みだが、ops スクリプト (`ops/windows/retrain_scheduler.ps1`) と hot_swap 連携 (`ops/windows/hot_swap_restart.ps1`) は **本セッションで実装完了** (27d58c7db)。「未起動の既存資産」から「起動準備完了」にステータスが変わった。

---

## §5 434# レビュー指摘への対応状況

426# §7.2 のアクション項目に対する 434# レビューの指摘と、実装状況の対照:

| 426# 項目 | 434# の評価 | 実装状況 | 備考 |
|---|---|---|---|
| P0: val_ratio=0.05/0.10 実験 | 「未実証を認めつつ方向性は妥当」 | ✅ S2完了 (PASS), S3完了 (FAIL) | S2は E7適用前。S3でE7初PASS |
| P1: retrain_scheduler 起動 | 「walk-forward を val_ratio 探しより優先」と強化 | ✅ ops スクリプト完了 | 実際の起動は Phase 2 |
| P2: reward_tuned 終了 | 全文書で同意 | ✅ 3 configs archived | 27d58c7db |
| P3: clamp 条件付き OOS | 「sim-live gap 閉じるべき」と追認 | ⏳ 未着手 | 依存: clamp observability |
| P4: G3 val_ratio 標準化 | 「比較可能性は重要」 | ✅ E7チェック実装、テスト6件 | 27d58c7db |

---

## §6 合意事項と残存分岐

### §6.1 5文書で合意が取れている事項

| # | 合意内容 | 支持文書 |
|---|---|---|
| A1 | SAC 単体の End-to-End 強化は短期 ROI が低い | 434#, 435#, 436# |
| A2 | Toxicity Veto (as_classifier ベース) が最優先施策 | 433#, 434#, 435#, 436# |
| A3 | reward_clean > reward_tuned は確定 | 426#, 434#, 436# |
| A4 | Skip Gate は現状機能していない (r=-0.010) | 432#, 434#, 436# |
| A5 | Walk-forward retrain が single-split より構造的に優位 | 434#, 435#, 436# |
| A6 | Committee は hard AND ではなく weighted veto | 434#, 436# |

### §6.2 残存する未解決分岐

| # | 分岐点 | 立場A | 立場B | 解決手段 |
|---|---|---|---|---|
| D1 | val_ratio=0.10 は正しい標準か | 436#: val_ratio探索は止めるべき | 426#: 0.05-0.10が適正 | **S3 FAIL で決着**: val_ratio=0.10 は G3 FAIL。E7 標準としては正しいが、モデルの有効 OOS は ~42日が限界 |
| D2 | SAC の step 増加は完全に無意味か | 436#/434#: 無意味 | 426# §5: 「step増加は解にならない」は強すぎる | 100K+val_ratio=0.10 実験（優先度低） |
| D3 | ceiling 動的化 vs toxicity veto どちらが先か | 432#: ceiling動的化がP0 | 434#/436#: toxicity vetoがP0 | **toxicity veto が先** (理由: ceiling変更はAS検知精度に依存するため、先に検知器を作るべき) |
| D4 | 1D action space は主犯か | 426# §5: plausible | 434# §2.2: 未証明 | 検証コスト高、優先度低 |

---

## §7 Profit-First 優先順位（更新版）

436# §4 のロードマップを、本レビューの知見で更新する:

### Phase 0: 完了
1. ✅ **S3 結果取得** → val_ratio 感度曲線完成 → retrain 間隔上界 ~42日、推奨 21-30日
2. **432# 正誤表の反映**: 本文に 436# §5 の修正を追記（or 本文書で代替）

### Phase 1: Toxicity Veto 構築 (1-2週)
1. `as_classifier.py` に fill_records データを投入、walk-forward CV で学習
2. **skip simulation 必須**: top-20%/10% skip の PnL 改善量を定量確認（§3.1 の false positive コスト評価）
3. `sidecar_signal_io.py` 経由で prob_toxic を配信
4. `cycle_gate_aggregator.py` or `orchestrator_mid_cycle.py` で veto gate 評価
5. **受け入れ基準**: skip simulation で PnL 改善 ≥ 1.0 bps/trade かつ 参加率低下 ≤ 10%

### Phase 2: SAC Retrain 有効化
1. `retrain_scheduler.ps1` で retrain_scheduler を起動
2. retrain 間隔: **21-30日** (§1.6 推定。S3 で有効 OOS ~42日が確定)
3. Sidecar v2 proportional boost で SAC bias を注入

### Phase 3: 評価改善
1. G3 multi-slice 評価の必須化
2. sim-live gap index の定期計測
3. clamp 条件付き OOS 評価

---

## §8 結論

5文書のクロスレビューにより、プロジェクトの方向性は明確。

**合意は堅い**: 全文書が「SAC 強化より toxic participation 回避」を指している。これは 432# のデータ、426# の実験結果、434# の理論的整理、436# の資産検証で多角的に支持されている。

**不足しているのは実行**: 資産は揃っている (7/7 production-ready)、分析も揃っている、レビューも3巡した。

**S3 結果により確定した事項**:
- val_ratio 感度曲線は4点で完成。崩壊の cliff edge は **42-84日の間**
- 推奨 retrain 間隔は **21-30日**
- val_ratio=0.10 の E7 標準は正しいが、G3 PASS には至らない（seed 間分散が極大）
- S3 の seed 間分散 (pf 0.856-1.446) は、84日 OOS での SAC の不安定性を決定的に示す

次の一手は **Phase 1: Toxicity Veto** の skip simulation による定量的有効性確認へ着手すべき。

---

*本文書のデータ検証は以下に基づく: fill_records 31ファイル (2026-02-13〜03-15), result JSON 6件, 現行 config (configs/v460/fill_test.yaml), ソースコード独立検証 (as_classifier.py 328行, walk_forward_as.py 323行, orchestrator_mid_cycle.py L129-130, cycle_gate_aggregator.py L368-410, sidecar_signal_io.py, gate_judgment_core.py)。P4 実装は commit 27d58c7db で検証済。*
