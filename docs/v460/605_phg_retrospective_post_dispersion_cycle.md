# 605# 風水渙 総決算 — 536#-604# 断捨離サイクルの回顧と現況俯瞰

- **日付**: 2026-03-25
- **目的**: 536# で立筮した「風水渙」から始まる断捨離・安定化サイクルが 604# で一巡したことを受け、到達点の客観的評価、残存課題の再分類、次の一手の策定を行う
- **入力**: 000#, 536#, 537#, 524#-527#, 553#, 556#, 581#, 592#, 596#-604#, 現行コード・設定値・稼働ログ

---

## §0 「渙」の回顧 — 何が溶け、何が残ったか

536# は風水渙の三義を掲げた:

> 1. **氷を溶かす** — 硬直した固定値をマイクロストラクチャの動的指標に変える
> 2. **大川を渉る** — 小手先のパッチではなくアーキテクチャの骨格に手を入れる
> 3. **王、有廟に至る** — 統制の核（廟）は強固に残す

604# 時点で、この三義はどこまで実現されたか。

### §0.1 溶けた氷

| 凍結物 | 施策 | 結果 |
|--------|------|------|
| 滞留注文による両側膠着 | 602# open order recovery, 603# age_cap cancel | ✅ 二層防御で根本原因 + フェイルセーフ確立 |
| Death spiral (BTC=0 無限 loop) | 596# primary_max_consecutive_skip | ✅ 12h 取引停止インシデント解消 |
| SAC retrain の成功信号消滅 | 600# conditional neutral fallback | ✅ 非 neutral 信号の 24h 保持 |
| SAC retrain の同一データ過学習 | 600# data window guard | ✅ `_last_deployed_val_ts_max` でループ防止 |
| OHLCV データ 12 日間未更新 | 553# auto-update pipeline | ✅ retrain 前に自動更新 |
| ログの不透明性（locked 残高・order_id 不明） | 604# 可観測性改善 | ✅ preflight_skip/age_cap ログ詳細化 |
| Git 追跡ファイル 5,002→105 激減 | 037 session 事故 → 修復 | ✅ 復旧済み（AGENTS.md に教訓記載） |
| silent except（例外握り潰し） | 527# exc_info 追加 | ✅ config_hot_reload, skip_gate 可視化 |
| データ幻覚 3 件（592#） | ev_offset 原因説/macro_strong_down/CV 数値の否定 | ✅ 是正済み、真の残課題を再定義 |

### §0.2 まだ凍っている氷

| 凍結物 | 出典 | 状態 |
|--------|------|------|
| Pipeline ceiling 100% 飽和 | 531#/536#/537# | ⏳ 0.25→0.35 未投入 |
| 乗算膨張（9 段無制限累積） | 537# §3.2/P4 | ⏳ Stage Max Mult 未実装 |
| Sidecar active 率 ~8%（TTL stale） | 537# P5/430# | ⏳ TTL 修正待ち |
| sell_dynamic_kill の遅行性 | 532#/536# §2 | ⏳ Toxicity Budget 統合待ち |
| CV Lead-Lag Widen buy 側 -3.21bps | 592# 提案 C | ⏳ 未修正 |
| EV toxic fill tail loss | 592# 提案 A | ⏳ skip threshold 未設定 |
| entry_gate / SAD / MCB 無効 | 600# P0 | ⏳ YAML 有効化待ち |
| RMS additive pipeline | 581# | ⏳ A/B テスト未実施 |
| one_sided_balance 配線欠落 | 596#/190# | ⏳ 設計のみ、実装なし |

### §0.3 廟（みたまや）に祭られているもの

537# §7.1 が再定義した「不動の三原則」:

| 原則 | 実装 | 健全性 |
|------|------|--------|
| **Spread > AS Cost** | min_spread_jpy (535#: 700→500) + offset pipeline | ⚠️ pipeline が ceiling で潰れ、精度不明 |
| **Inventory Mean-Reversion** | inventory_skewing (226# P5) | ✅ 機能中 |
| **Catastrophic Loss Prevention** | Final Clamp (421#) + kill_switch + SAFE_STOP | ✅ 602#/603# で強化済み |

---

## §1 Gate 進捗 — 0# Phase 定義との照合

### §1.1 通過済み Gate

| Gate | 通過 | 根拠 | 品質評価 |
|------|------|------|---------|
| **G0-data** | ✅ | データハッシュ一致、NaN≤1%、manifest 記録 | 堅牢 |
| **G1-info** | ✅ | XGBoost OOS Spearman IC > 0.02 確認 | 堅牢 |
| **G1.1-exec** (K1-K6) | ✅ 330# | 72h Kill Gate 全条件 PASS | 堅牢 |
| **G1.2-full** (F1-F8) | ✅ 330# | 168h Qualification Gate PASS | 堅牢 |
| **G2-train** | ✅ 387# | 4 seed × 指定 steps、seed 間σ条件 PASS | 堅牢 |
| **G3-pnl** | ✅ 409# | PF=1.145 (median)、Sharpe>0.8、MaxDD<15% | 条件付き ⚠️ |

### §1.2 G3 の「条件付き」について

G3 PASS (409#) は **20K step 学習 + val_ratio=0.10** での結果。422#-426# の追加実験で:

- **100K step G3**: FAIL (val_ratio 交絡で楽観バイアス露呈)
- **8 seed 中 6 seed で OOS mid 期崩壊** (426# §3.2)
- 422# A1: best_model と final_model の乖離

→ G3 PASS 自体は有効だが、**SAC 単体の汎化限界が確認済み** (0# §0.1)。Sidecar + retrain アーキテクチャが SAC の「助言者」化により対処する設計。

### §1.3 未通過 Gate

| Gate | 状態 | ブロッカー |
|------|------|-----------|
| **G3.1-stress** | ⏳ 未実施 | slippage/miss テスト未実行。G3 PASS 済みモデルに stress パラメータを注入して再評価が必要 |
| **G4-live** | ⏳ 未到達 | G3.1 通過 + 7 日連続稼働 + Circuit Breaker テストが前提条件 |

### §1.4 現在の Phase 位置

```
ph0 ──── ph1 ──── ph2 ──── ph3 ──── ph3.1 ──── ph4 ──── ph4.1 ──── ph5
 ✅       ✅      ✅ G1.1   ✅ G2   ⚠️ 部分     ✅ G3   ⏳ G3.1   ⏳ G4
                   PASS     PASS    稼働        PASS
                                                        ← 現在地
```

**ph3.1** (Sidecar 統合) が「部分稼働」なのは:
- retrain_scheduler: ✅ 600#/601# で修正し稼働中
- Sidecar v2 信号: ⚠️ active 率 ~8% (TTL stale)
- SAC 実効影響: ⚠️ ±0.15bps（pipeline 全体の数%未満）

0# §0.1 が定義する Alpha/Execution/Safety 三層分離の実装は完了しているが、**Alpha 層（SAC）の実効的な寄与がほぼゼロ**の状態。これは設計の欠陥ではなく、SAC の汎化限界 (426#) を反映した運用的結果。

---

## §2 536#-604# で浮上した構造的知見

### §2.1 パッチ当ての限界と「渙」の正当性

536# が批判した「IF 文による条件縛り」の病根は、604# までの過程で具体的に立証された:

| インシデント | 病根 | 本質 |
|-------------|------|------|
| 524# stale order 膠着 | `_cancel_stale_orders` が startup のみ | 「起動時」という固定条件に縛られた防御 |
| 596# death spiral | `ev_as_offset_enabled=true` が skip counter を常時リセット | フラグ分岐の組合せ爆発による設計の盲点 |
| 603# age_cap cancel 漏れ | break 後のクリーンアップが未実装 | 「break したら終わり」という暗黙の前提 |
| 531# ceiling 100% 飽和 | 9 段乗算が ceiling 0.25 を常に超過 | 「0.25 で収まるだろう」という静的見積もり |

これらは全て「固定的な前提条件が市場の状態変化に追従できない」ことの表れであり、536# の「氷を溶かす」提案の正当性を裏付ける。

### §2.2 「廟」の再確認 — 何が壊れてはならないか

602#-604# のインシデント対応過程で、システムの生存に不可欠な「廟」が具体化した:

1. **注文ライフサイクルの完全性**: place → monitor → cancel|fill のいずれかで必ず終了する（603# で violation を修正）
2. **資本の非拘束性**: `btc_reserved` + `jpy_free` が長期間ゼロに固着しない（602# で recovery を追加）
3. **自己診断能力**: 膠着状態を検知し、人間の介入なしに回復を試みる（604# diagnose_deadlock + 602# recovery）

### §2.3 データ幻覚問題 (592#) の教訓

592# で否定された 3 つの仮説（ev_offset 原因説、macro_strong_down 防御、CV 数値）は、**Codex（AI アシスタント）が過去の分析結果を記憶違いしたまま提案を生成した**ケース。

**教訓**:
- AI 提案の数値は**必ず実データで検証**してから実装に進む
- 「前回の分析でこうだった」という記述には再現可能なスクリプト（SHA + コマンド）を要求する
- 592# のような是正文書を適時作成し、誤った前提の上に建設された提案を遡及的に無効化する

---

## §3 残存課題の再分類 — 「渙かすべきもの」と「祭るべきもの」

### §3.1 Tier 0: 即効施策（YAML 変更のみ、リスク極低）

| # | 施策 | 出典 | 変更量 | 期待効果 |
|---|------|------|--------|---------|
| T0-1 | `offset_ceiling_ratio_*: 0.25→0.35` | 536#/537# | YAML 2 行 | pipeline 上位 ~30% のロジックが取引価格に反映 |
| T0-2 | `composite_risk_enabled: true` | 537# P2/600# | YAML 1 行 | Soft Gate deadlock 緩和 |
| T0-3 | entry_gate 有効化 | 600# P0 | YAML | 不利エントリ回避 |
| T0-4 | spread_anomaly_detector 有効化 | 600# P0 | YAML | 急変時防御 |
| T0-5 | micro_circuit_breaker 有効化 | 600# P0 | YAML | 連続逆行停止 |

**投入方針**: T0-1〜T0-5 を一括投入し、24h fill_rate/AS-PnL を計測。問題があれば個別にロールバック。

### §3.2 Tier 1: 短期実装（コード 1-30 行、1-3 日）

| # | 施策 | 出典 | 効果の根拠 |
|---|------|------|-----------|
| T1-1 | Stage Max Mult (各段 cap 1.5) | 537# P4 | 9 段全 max でも 0.05×1.5⁸=1.28。ceiling 0.50 で自然に収まる |
| T1-2 | CV Lead-Lag Widen 廃止 or 片側撤退化 | 592# 提案 C | buy 側 -3.21bps の直接止血 |
| T1-3 | EV toxic skip threshold (-5.0) | 592# 提案 A | tail loss 帯域の事前回避 |
| T1-4 | Sidecar TTL 修正 | 537# P5 | active 率 8%→~90%（retrain_interval=7200s に合わせる） |
| T1-5 | sell_dynamic_kill max_duration: 1800→600 | 537# Phase 0 | kill 状態の長期化防止 |

### §3.3 Tier 2: 中期施策（構造変更、1-2 週）

| # | 施策 | 出典 | 前提条件 |
|---|------|------|---------|
| T2-1 | OFI-Lite (cycle OB 差分 → Toxicity 入力) | 537# P3 | OB snapshot が cycle 毎に取得可能 (✅ 既存) |
| T2-2 | RMS additive pipeline A/B テスト | 581# | fill_rate 比較の統計的検出力確保 (n≥300) |
| T2-3 | min_spread_jpy の ATR 連動化 | 537# §2.1 | Parkinson σ が既に実装済み (305#) |
| T2-4 | analysis scripts batch 統一 (A→C→B) | 598# | output/CLI contract の標準化 |
| T2-5 | **G3.1-stress テスト実施** | 0# §3.5.1 | G3 PASS 済みモデル + stress パラメータ注入 |

### §3.4 Tier 3: 長期施策（設計判断を要する）

| # | 施策 | 出典 | 判断ポイント |
|---|------|------|-------------|
| T3-1 | A-S 最適スプレッド参照値導入 | 537# P6 | κ（注文到着強度）の推定精度に依存 |
| T3-2 | sell_dynamic_kill → Toxicity Budget 完全統合 | 537# Phase 2 | graduated response の calibration |
| T3-3 | SAC action 幅拡大 (±0.15→±5.0bps) | 537# P5 | SAC 学習安定性の再検証が必須 |
| T3-4 | eDRC パラメータ再推定 (α=β=0 無効化中) | 600# P3 | 576# インシデント後の安全な再有効化手順 |
| T3-5 | lib→ztb 統合 (106# R5) | 556# | mypy 二重名前空間問題の根本解決 |
| T3-6 | one_sided_balance 配線復旧 | 596# | 190# 設計の再評価（offset 分岐で実質迂回中）|
| T3-7 | walk-forward SAC 再訓練 | 422#/426# | 100K G3 FAIL の根因（val_ratio 交絡 vs 表現力限界）の切り分け |

---

## §4 アーキテクチャの現状図

```
                    ┌─────────────────────────────────┐
                    │         Alpha 層 (SAC)          │
                    │  retrain_scheduler (600#修正済) │
                    │  Sidecar v2 (active ~8% ⚠️)    │
                    │  directional_bias [-1,1]        │
                    └──────────┬──────────────────────┘
                               │ ±0.15bps (実効ほぼ0)
                    ┌──────────▼──────────────────────┐
                    │       Execution 層              │
                    │  9段 offset pipeline:           │
                    │   EV→Vel→Trend→Tox→VG→Macro    │
                    │   →Alert→Sidecar→Clamp         │
                    │  ──────────────────────         │
                    │  乗算膨張: 0.05×8.99=0.449      │
                    │  ceiling 0.25 で 100% 飽和 ⚠️   │
                    │  ──────────────────────         │
                    │  RMS additive (581#) A/B待ち    │
                    │  Final Clamp (421#) ✅           │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │        Safety 層                │
                    │  SkipGate ML         ✅ 稼働中  │
                    │  entry_gate          ⏳ disabled │
                    │  spread_anomaly_det  ⏳ disabled │
                    │  micro_circuit_brk   ⏳ disabled │
                    │  composite_risk      ⏳ disabled │
                    │  sell_dynamic_kill   ✅ 遅行型   │
                    │  ──────────────────────         │
                    │  Runtime防御:                    │
                    │   602# open order recovery  ✅  │
                    │   603# age_cap cancel       ✅  │
                    │   604# 膠着診断             ✅  │
                    │   596# primary skip guard   ✅  │
                    └─────────────────────────────────┘
```

**構造的ボトルネック 2 点**:
1. **Pipeline ceiling 飽和**: 9 段の演算結果が取引価格に反映されていない（531# 実証）
2. **SAC active 率 ~8%**: Alpha 層の出力が TTL stale で 92% 失効

この 2 点が「G3 PASS (PF=1.145) の実環境での再現」を阻む最大要因。

---

## §5 「渙」の第二段階へ — 推奨アクション

### §5.1 即時投入 (Week 1 Day 1-2)

**Tier 0 YAML 一括**:
```yaml
# T0-1: ceiling 段階引き上げ
offset_ceiling_ratio_buy: 0.35
offset_ceiling_ratio_sell: 0.35

# T0-2: composite risk 有効化
composite_risk_enabled: true

# T0-3/4/5: P0 mechanisms 有効化
entry_gate_enabled: true
spread_anomaly_detector_enabled: true
micro_circuit_breaker_enabled: true
```

24h 後に fill_rate / AS-PnL / clamp_rate を計測。問題なければ:

### §5.2 止血施策 (Week 1 Day 3-5)

1. **T1-2: CV Widen 廃止** — buy 側 -3.21bps の直接止血
2. **T1-3: EV toxic skip threshold** — tail loss 帯域の事前回避
3. **T1-5: sell_dynamic_kill max_duration 短縮** — kill 長期化防止

### §5.3 構造改善 (Week 2)

4. **T1-1: Stage Max Mult** — 乗算膨張の構造的解消（コード 1 行）
5. **T1-4: Sidecar TTL 修正** — SAC 信号 active 率 8%→~90%
6. **T2-5: G3.1-stress テスト** — ph4.1 Gate 通過へ

### §5.4 Assessment Gate (Week 2 末)

Week 2 終了時に以下を評価:
- fill_rate: T0 投入前後の変化
- AS-PnL (bps): 逆選択コストの変化
- pipeline clamp_rate: 飽和率の変化
- SAC active_rate: Sidecar 信号の有効利用率
- G3.1 条件: S1-S5 の PASS/FAIL

結果に応じて T2 / T3 の優先度を再編成。

---

## §6 風水渙 再考 — 第三爻変の成就

> **「其の躬（み）を渙（ち）らす。悔いなし。」**

536# がこの爻辞を「開発者の過学習への執着を手放せ」と解釈したのは正しかった。602#-604# の過程で、以下の「執着」が実際に手放された:

- **「startup で cancel すれば十分」** という前提 → runtime recovery の追加 (602#)
- **「break すればクリーンアップ不要」** という暗黙知 → 明示的 cancel (603#)
- **「ログは INFO で十分」** という慣行 → locked 残高・order_id・exc_info の詳細化 (604#)
- **「Codex の分析は正しい」** という信頼 → 実データ検証の義務化 (592#)

一方、まだ手放されていない執着がある:

- **ceiling 0.25 への保守** — 「低い方が安全」という思い込み（実際は全段無効化で逆効果）
- **P0 mechanisms の disabled 維持** — 「有効化するとバグが出る」という恐れ（600# で実装品質は確認済み）
- **SAC の ±0.15bps 制限** — 「ML を信頼しない」という原則（過度に適用すると Alpha 層が無意味化）

これらを次の「渙」で溶かすことが、536# シナリオ A（抜本的ハブ化）への漸進的接近となる。

---

## §7 定量的現況サマリ

| 指標 | 値 | 出典 | 評価 |
|------|-----|------|------|
| G3 PF (median) | 1.145 | 409# | ✅ PASS (>1.05) |
| G3 Sharpe (annualized) | >0.8 | 409# | ✅ PASS |
| G3 MaxDD | <15% | 409# | ✅ PASS |
| Pipeline clamp rate | 100% | 531# | ❌ 全段無効化 |
| SAC active rate | ~8% | 537# | ❌ TTL stale |
| CV Widen buy 被弾 | -3.21 bps | 592# | ❌ 未修正 |
| Runtime deadlock | 602#/603# 防御済み | 604# | ✅ 多層防御 |
| Death spiral | 596# 修正済み | 596# | ✅ 解消 |
| fill_test 稼働 SHA | `487c80808` (604#) | — | ✅ 最新 |
| retrain_scheduler | 600#/601# 修正済み | — | ✅ 稼働中 |
| 総ドキュメント数 | 604 本 | index.md | — |

---

*風水渙。亨。王格有廟。利渉大川。*
*氷は半ば溶け、廟は立ち、大川を渉る準備は整った。*
*以上*
