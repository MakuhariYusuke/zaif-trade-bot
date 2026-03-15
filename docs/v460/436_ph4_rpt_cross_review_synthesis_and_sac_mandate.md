# 436# 横断レビュー統合・検証結果 および SAC 責務再定義

| 項目 | 内容 |
|---|---|
| 番号 | 436# |
| 分類 | ph4_rpt (Phase4 Report) |
| 対象 | 426# / 432# / 434# / 435# の統合検証 |
| 前提 | 434# (Codex横断レビュー), 435# (Gemini Second Opinion) |
| 目的 | レビュー指摘の事実検証 + SAC担当AIへの方針伝達 |

---

## §0 Executive Summary

426#（SAC実験結果）、432#（fill_records深堀り分析）、434#（横断レビュー）、435#（Gemini追認）の4文書を独立検証した結果、以下の結論を確認した。

| 結論 | 検証結果 | 根拠 |
|---|---|---|
| Skip Gate は死んでいる (r=-0.010) | **確認** | fill_records 11,356件の再集計で再現 |
| buy+ranging が損失の主エンジン (PF=0.766) | **確認** | 最大ボリューム (n=1,310) で最悪の PF |
| 434# ceiling config 修正 (buy=0.20, sell=0.50) | **確認+補足** | config値は正だが、**レコードlevelでの乖離を新発見** |
| AS回避が収益を支配している | **確認** | AS=True の win%=0%, PnL=-6.479 bps |
| Toxicity Sidecar が最優先 | **同意** | 全資産が production-ready (7/7 完備) |
| SAC 単体は中期 OOS に脆弱 | **確認** | 426# S1/S1' 両方 FAIL, roi_seed_std 22-42倍爆発 |

**本文書の最重要メッセージ**: SAC の次の仕事は「もっと賢く方向を当てる」ことではなく、**toxicity veto と協調して「毒のある取引に参加しない」仕組みの一部として機能する**ことである。

---

## §1 434#/435# レビュー指摘の独立検証

### 1.1 Ceiling config の検証結果

434# は「432# が ceiling=0.15 一律と書いているのは stale config」と指摘した。検証結果:

**Config値（現行）**:
```yaml
# configs/v460/fill_test.yaml:580-582
offset_ceiling_ratio: 0.15        # 共通デフォルト
offset_ceiling_ratio_buy: 0.20    # 381# で 0.15→0.20
offset_ceiling_ratio_sell: 0.50   # 320# で 0.15→0.50
```

→ 434# の指摘は**正確**。config は side 別に分かれている。

**しかし432# が参照した値も完全な誤りではない。以下を新発見した:**

`offset_stages["ceiling"]` フィールドは **clamp 発火時にのみ記録される**。実レコード 31ファイル (2026-02-13〜03-15) の検証結果:

| side | 記録件数 | ceiling値分布 | 解釈 |
|---|---|---|---|
| buy | 445 | 0.15 (3/6-3/10), 0.20 (3/12-3/15) | 381# hot-swap (3/11) で切り替わった |
| sell | 75 | 全件 0.15 (3/6-3/7のみ) | 以降は ceiling=0.50 でオフセットが超過しないため**記録なし** |

**重要**: `offset_stages` の記録は 2026-03-06 以降にしか存在しない（306# で導入）。全 11,356 件中 1,598 件のみ。

**結論**: 432# の「ceiling がAS回避を妨害」という洞察は **buy 側については完全に有効**。sell 側は ceiling=0.50 がほぼ制約にならないため、sell 側の losses は ceiling 以外の要因（フロー毒性・regime不一致）が主犯。

### 1.2 AS classifier / Sidecar 資産の検証

434#/435# が「既存資産で sidecar 化すべき」と推奨した資産の実在と完成度を検証した:

| 資産 | パス | LOC | 状態 | 参照数 |
|---|---|---|---|---|
| AS classifier | `scripts/v460/ml/as_classifier.py` | 315 | **完全** | 2 |
| Sidecar I/O | `scripts/v460/lib/sidecar_signal_io.py` | 230 | **完全** | 7 |
| Mid-cycle Orchestrator | `scripts/v460/lib/orchestrator_mid_cycle.py` | 550 | **完全** | fill_loop 統合済 |
| Gate Aggregator | `scripts/v460/lib/cycle_gate_aggregator.py` | 780 | **完全** | 20+ |
| Walk-forward AS | `scripts/v460/ml/walk_forward_as.py` | 280 | **完全** | 7 |
| WF Splitter | `ztb/evaluation/walk_forward/splitter.py` | 220 | **完全** | 20+ |
| Sidecar Types | `scripts/v460/lib/sidecar_types.py` | 380 | **完全** | 12 |

**全7資産が production-ready**。scaffold ではなく、実働パイプライン。

特記事項:
- `sidecar_signal_io.py`: atomic tmp→rename write, mtime-based TTL (7800s)
- `cycle_gate_aggregator.py`: v2 proportional boost (374#) 実装済
- `orchestrator_mid_cycle.py`: sidecar inject point (L136) + toxicity grading `_assess_buy/sell_toxicity()` (L129-130, 240#/241#) 配線済
- `as_classifier.py`: TimeSeriesSplit CV, Imputer→Scaler→Model, skip simulation 内蔵

### 1.3 426# 実験結果の確認

434# が「426# の数値は概ね再現できる」と述べた点を結果 JSON で裏付け確認:

| 条件 | pf_median | 判定 | roi_seed_std |
|---|---|---|---|
| Original clean (val=0.02) | 1.145 | PASS | 0.0016 |
| Original tuned (val=0.02) | 1.006 | FAIL | 0.0025 |
| S1 (clean × val=0.20) | 1.049 | FAIL (marginal) | 0.0360 (×22) |
| S1' (tuned × val=0.20) | 1.031 | FAIL (structural) | 0.0679 (×42) |

**確認済**: val_ratio=0.02 の「安定」は OOS が短すぎて分散蓄積前に評価終了しただけ。

---

## §2 全文書から導かれる Live 損失の構造

### 2.1 損失の因果構造図

```
[market flow]
    │
    ├─ toxic flow (AS率 27.2%, PnL=-6.479 bps, win=0%)
    │   ├─ buy+ranging: PF=0.766, n=1310 (最大ボリューム, 最悪パフォーマンス)
    │   ├─ sell+trending_up: AS率 35.5%, PnL=-0.832
    │   └─ regime_mismatch: PnL=-2.294 (注文～fill間に状態が変わる)
    │
    ├─ Skip Gate が機能していない (r=-0.010)
    │   └─ score=5.0 で AS率 52% → 最も自信のあるスコアが最も危険
    │
    ├─ Ceiling が buy 側の AS 回避を阻害
    │   └─ pipeline が +0.027 offset を足すも ceiling=0.20 で切断
    │       (clamp excess mean=0.080, max=0.300)
    │
    └─ reprice が悪化要因 (PnL: 0→-0.22→-0.73→-2.21)
        └─ 受動 maker → 能動 taker に近づくほど情報劣位で不利
```

### 2.2 何が収益を支配しているか

432# の単変量相関分析では、全 feature の |r| < 0.05。つまり **単一の特徴量で PnL を予測することはできない**。

利益を決めているのは interaction:
- `side × regime × toxicity_flag`
- `decision_path × balance_forced × queue_wait`
- `reprice_count × spread × velocity`

**最大の収益ドライバーは「方向を当てること」ではなく「毒のあるフローに参加しないこと」である。**

---

## §3 SAC への提言: 責務の再定義

### 3.1 SAC が今やるべきでないこと

| やるべきでないこと | 理由 |
|---|---|
| val_ratio の最適値探索 | 0.02 と 0.20 の2点比較のみ。間の探索は実験コスト高・得られるエッジ小 |
| ステップ数の大幅増加 | 100K でも G3 FAIL。step 増加は必要条件かもしれないが十分条件ではない |
| reward_tuned の改良 | reward_tuned は reward_clean より構造的に脆弱 (pf_median 1.031 vs 1.049) |
| 1D action space の拡張 | 仮説としては plausible だが、未実証かつ他の改善が先行 |
| End-to-End で全部を解決 | live fill の毒性は policy 改善だけでは対処できない |

### 3.2 SAC がやるべきこと

**方針**: SAC は「万能 driver」ではなく、**toxicity veto 層と協調する directional hint 提供者**として機能する。

| 優先度 | 施策 | 内容 |
|---|---|---|
| **P0** | Walk-forward retrain 標準化 | 固定 val_ratio 探しから、定期再学習パイプラインへ移行。既存 `walk_forward_as.py` + `splitter.py` を活用 |
| **P1** | reward_clean 維持 | reward_tuned より一貫して頑健。補助損失の追加は逆効果の可能性 |
| **P2** | Sidecar 出力に限定 | SAC の出力は directional_bias として sidecar 経由で注入。veto 権は toxicity classifier に委譲 |
| **P3** | multi-slice 評価の標準化 | early/mid/late のスライス別崩壊検出を G3 の必須出力にする |

### 3.3 SAC × Toxicity Veto の協調設計

434# が推奨し 435# が追認した委員会構造:

```
[CycleGateAggregator]
    ├─ Toxicity Veto (as_classifier) ─── veto 権あり (prob_toxic > 閾値 → Hard Skip)
    ├─ SAC Sidecar ─────────────────── 方向ヒント (directional_bias ∈ [-1, +1])
    ├─ (将来) BitFlyer lead-lag ────── suppress / boost 補助票
    └─ (将来) Queue heuristic ──────── retreat 補助票
```

**Hard AND ではなく weighted score + veto**: toxicity だけが拒否権を持ち、他は boost/suppress の投票。SAC はこの構造の中で「方向のヒントを出す」役割に限定される。

---

## §4 Toxicity Veto 構築の具体ロードマップ

434#/435# が推奨し、本検証で全資産の ready 状態を確認した実装方針:

### Phase 1: 即時 (clamp observability 完了後)

1. **fill_records から Training Data 生成**
   - 正例: `filled=True AND adverse_selected=True` (960件/30日)
   - 負例: `filled=True AND adverse_selected=False` (2,575件/30日)
   - 特徴量: `spread_bps, orderbook_imbalance, price_velocity_bps, ev_score_pretrade, regime, regime_confidence, vg_triggered, vg_boost_factor, side, macro_trend, microprice_bias_bps`

2. **as_classifier.py で Walk-forward 学習**
   - `walk_forward_as.py` の expanding window + embargo
   - 評価: ROC-AUC, PR-AUC, skip simulation (top-20%/10% skip の PnL 改善量)

3. **Sidecar として配線**
   - `sidecar_signal_io.py` で prob_toxic を書き出し
   - `orchestrator_mid_cycle.py:136` で読み込み
   - `cycle_gate_aggregator.py` で veto gate として評価
   - 閾値超過 → Hard Skip (参加見送り)

### Phase 2: SAC Retrain

1. Walk-forward retrain scheduler の有効化
2. reward_clean を基本とし、retrain 間隔を regime 安定性で動的調整
3. SAC 出力は `directional_bias` として sidecar injection (v2 proportional boost)

### Phase 3: 評価改善

1. G3 ゲートに multi-slice (early/mid/late) 評価を必須化
2. Walk-forward split を single-split の代替として標準化
3. sim-live gap の閉ループ: fill_records → retrain → deploy → fill_records

---

## §5 432# 誤記の正誤表

432# は本分析の基盤として有益だが、434# の指摘に基づき以下を修正する。

| 箇所 | 432# の記述 | 正しい事実 |
|---|---|---|
| §1.3 ceiling値 | "ceiling=0.150 で均一に切断" | config は buy=0.20, sell=0.50。ただし buy 側で 0.15→0.20 の変遷あり (381#) |
| §1.3 ceiling論拠 | "AS/非ASで ceiling ほぼ同じ (0.178)" | これは clamp 発火レコードの条件付き平均。sell 側は ceiling=0.50 で発火しないため未記録 |
| §9 P0 | "ceiling を 0.15→0.20 に" | buy は既に 0.20。sell は既に 0.50。「ceiling 動的化」が正しい方向 |

**432# のコア洞察は影響を受けない**: buy 側 clamp excess mean=0.080 (max=0.300) は、buy ceiling=0.20 でも pipeline が要求する offset を 切断していることを意味する。sell 側は ceiling 以外の要因（フロー毒性、regime 状態）が支配的。

---

## §6 全体結論

4文書 (426#/432#/434#/435#) を独立検証した結論:

1. **SAC 単体の強化は短期的に ROI が低い**。val_ratio 探索、step 増加、reward 改造は marginally useful だが、live 損失の主因に届かない。

2. **収益のボトルネックは toxic participation**。AS=True の 100% loss rate, buy+ranging の PF=0.766 が全体 PnL を引き下げている。これは policy の方向予測能力とは別次元の問題。

3. **Toxicity Veto の既存資産は全て揃っている**。as_classifier.py, sidecar_signal_io.py, cycle_gate_aggregator.py, walk_forward_as.py — 全て production-ready。新規開発ではなく配線の問題。

4. **SAC の役割は directional hint に限定**。万能 driver ではなく、toxicity veto 層と協調する sidecar の一票として機能する。委員会は hard AND ではなく weighted veto で構成する。

5. **Walk-forward retrain を標準化**。固定 val_ratio の最適値探しは止め、定期再学習で regime 変化に追随する設計に移行する。

**次の一手**: Toxicity Veto の構築 (§4 Phase 1) → SAC Walk-forward retrain (§4 Phase 2) → 評価改善 (§4 Phase 3)。

---

*本文書は 426#, 432#, 434#, 435# の横断検証に基づく。データ検証は fill_records 31ファイル (2026-02-13〜03-15, 11,356件) および現行 config (configs/v460/fill_test.yaml) に対して実施。*
