# 655# 振り返り — 536# 渙から654#まで: 未達事項整理と新規課題

- **日付**: 2026-03-30
- **目的**: 536# で提示された「風水渙」の三原則を座標軸に、653#/654# で見えてきた課題を含む **全未達事項の棚卸し** と **新規構造問題の発見** を行う。
- **入力**: 536#, 561#-565#, 605#-607#, 649#-654#
- **方法**: 文書横断照合 + コードベース実地検証

---

## §0 536# 渙の三原則 — 再掲と現在地

536# で立筮から導いた三原則を改めて引き、654# 時点の達成度を測る。

| # | 原則 | 536# 原文 | 654# 時点 |
|---|------|----------|----------|
| **渙-1** | 氷を溶かす | 固定閾値・遅行Kill → 動的市場指標へ | ⚠️ **方向は正しいが、氷が残っている** |
| **渙-2** | 大川を渉る | 小手先パッチでなくアーキテクチャ骨格改修 | ❌ **まだ渡れていない** |
| **渙-3** | 王、有廟に至る | 全散防止。コア原則を「廟」として堅持 | ✅ **MCB/SAD/ceiling は廟として機能** |

### 渙-1 の進捗: 溶けた氷と残った氷

| 氷 | 出典 | 状態 | 備考 |
|----|------|------|------|
| ceiling 100% 飽和 | 531#/536# | ⚠️ **sell 92% に緩和** | 0.25→0.40 引上げ (565#)。しかし依然飽和 |
| 乗算膨張 (9段無制限) | 536# §2 | ✅ **溶けた** | EV max_mult 1.5、各段に個別cap |
| sell_dynamic_kill 遅行 | 536# §3A | ❌ **凍ったまま** | `enabled: true`。536# は廃止を提案したが存続 |
| inv_skew trending 無効化 | 565# 盲点8 | ❌ **凍ったまま** | `regime_gate_enabled: true` が残存 |
| Sidecar active率 ~8% | 605# | ❌ **凍ったまま** | 650# で 93% stale。649# 分離修正も効果限定的 |
| 時間帯固定ルール | 536# §0 | ⚠️ **部分溶解** | hour_lot_scale → hour_offset 化。だが固定表形式は残存 |
| entry_gate 閉鎖 | 606# | 🟡 **observe モード** | Phase2 移行条件が未明文化 |

### 渙-2 の遅延: なぜまだ渡れていないか

536# のシナリオ A（抜本的ハブ化: OFI ベースの予測型防衛への一元化）は、605# で方針として承認されたが、**650#-654# までの実際の進捗はシナリオ C（枯死機能の剪定）の域を出ていない**。

理由:
1. **計測基盤の先行が必要だった** — 565# の「理論より計測」に従い、650# で RT 分析を先行。計測なき大改修は 576# インシデントの轍
2. **止血が優先された** — 645# (sell model停止)、648# (σ stale fix)、649# (data freshness) は全て事故対応
3. **サンプル不足** — 13 RT では統計的根拠が弱く、大川を渉る勇気を裏付けるデータがない

→ これは **正しい慎重さ** だが、**永続的に渉れない口実** にもなりうる。どこかで踏み切る判断基準を明文化すべき。

### 渙-3 の「廟」の健全性

| 廟 (コア原則) | 実装状況 | 健全性 |
|--------------|---------|--------|
| Spread > AS Cost | min_spread_bps + narrow_spread_pause | ⚠️ Q1 帯で原則違反 (avg -0.86bps) |
| Inventory Mean-Reversion | inv_skew | ❌ 三重無効化 (654# §1.1) |
| Catastrophic Loss Prevention | MCB + Final Clamp + age_cap | ✅ 稼働中。ただし MCB+position 問題あり |
| 逆選択コスト内部化 | VG + OFI + VPIN | ⚠️ VG は ceiling 吸収で形骸化 |

---

## §1 全未達事項マトリクス

653# (561-562) と 654# (651-652) の統合棚卸しに、コード検証で判明した新規項目を加える。

### 1.1 Tier 0: 完了済み

| # | 施策 | 出典 | 完了コミット |
|---|------|------|------------|
| T0-1 | inv_skew neutral_band 0.10→0.05 | 650#/651# | 654# (d335e93) |
| T0-2 | inv_skew decay_tau 1800→3600 | 650#/651# | 654# (d335e93) |
| T0-3 | Toxic sell veto (compound guard) | 651#/652# | 654# (d335e93) |
| — | ceiling sell 0.25→0.40, buy→0.35 | 562#/565# | 565# |
| — | CV tighten sell 無効化 | 562# P-A | 565# |
| — | stage max_mult (EV 1.5) | 562#/537# | 実装済み |
| — | SAD/MCB 有効化 | 605#/606# | 606# |
| — | σ stale feedback loop 修正 | 648# | 648# |
| — | data freshness check 分離 | 649# | 649# |

### 1.2 Tier 1: 短期未達（コード 1-50行、測定）

| # | 施策 | 出典 | 阻害要因 | 備考 |
|---|------|------|---------|------|
| **T1-1** | MCB HALT 時 open position 警告ログ | 650# I2 | なし | micro_circuit_breaker.py に追加。リスクゼロ |
| **T1-2** | RT 主 KPI 化 (pnl30 → RT PnL) | 651# P1 | 分析基盤の整理 | analyze_fill_logs.py 側は 650# で着手済み |
| **T1-3** | Regime 遷移 AS 分析セクション | 565# 盲点3 | section_regime_transition_as() 未実装 | analyze_fill_logs に追加 |
| **T1-4** | AS burst 自己相関分析 | 565# 盲点4 | section_as_burst() 未実装 | φ₁ 算出で AS 時系列構造を明示 |
| **T1-5** | PnL 計測窓正規化 (buy=30s, sell=90s) | 565# 盲点1 | 命名と実態の乖離 | pnl_measurer.py の window 設計再確認 |
| **T1-6** | ★**NEW** max_skip_rate と toxic_sell_veto の干渉対策 | 655# §2.1 | バジェット枯渇リスク | 下記 §2.1 参照 |

### 1.3 Tier 2: 中期未達（構造変更・検証）

| # | 施策 | 出典 | 前提条件 | 備考 |
|---|------|------|---------|------|
| **T2-1** | eDRC α/β 再推定・有効化 | 561#/562# P-G | 576# 安全性確認 + ceiling飽和データ | α=β=0 のまま放置中 |
| **T2-2** | Sidecar retrain 成功率調査 | 650#/562# B4 | retrain_scheduler ログ分析 | 93% stale は未解消 |
| **T2-3** | sell ceiling → VG 連動動的拡張 | 650# I4 / 562# §4 | VG boost>1.5 時の ceiling 緩和 | 562# 三層打消し問題の直接対策 |
| **T2-4** | 曜日効果分析 | 565# 盲点5 | 4 週分データ蓄積 | 最低 n=100 RT 必要 |
| **T2-5** | Asymmetric RT exit tolerance | 652# P1-X | RT position tracking 基盤 | sell-entry close 早期化 |
| **T2-6** | Regime-drift exit for long hold | 652# P0-3 | RT tracking + regime 遷移検知 | micro_timeout ≠ RT hold |
| **T2-7** | ★**NEW** regime_gate_enabled 条件付き緩和 | 655# §2.2 | extreme 在庫偏重データ | 下記 §2.2 参照 |
| **T2-8** | ★**NEW** sell_dynamic_kill 存廃判断 | 655# §2.3 | kill 発動率 vs PnL 相関 | 下記 §2.3 参照 |
| **T2-9** | ★**NEW** preflight_insufficient 動的バッファ | 655# §2.4 | balance_checker 改修 | 下記 §2.4 参照 |

### 1.4 Tier 3: 長期未達（アーキテクチャ）

| # | 施策 | 出典 | 判断ポイント |
|---|------|------|-------------|
| **T3-1** | AS Risk Score 統合 (RMS/max) | 562# §4.3 | pre_clamp 分布データ蓄積後 |
| **T3-2** | MCB HALT 前 position pre-close | 650# I6 | fill_cycle_executor 連携設計 |
| **T3-3** | Kelly lot_sizing 実運用化 | 565# 盲点6 | A/B テスト設計 |
| **T3-4** | inv_skew 条件付き trending 復活 | 565# 盲点8 | extreme 在庫限定 (偏り>40pp) |
| **T3-5** | Sell model 再構築条件明文化 | 645# | retrain pipeline 健全化後 |
| **T3-6** | entry_gate Phase2 移行基準 | 606# | 移行条件の明文化が前提 |
| **T3-7** | 536# シナリオA OFI ハブ化 | 536# §3A | 全上位 Tier 完了後の最終形 |

---

## §2 新規発見 — 654# コミット後に検出された構造問題

### 2.1 ★ max_skip_rate バジェット枯渇問題

**発見経緯**: 654# P0-2 (toxic_sell_veto) のテスト実装中にコードパスを追跡して発見。

**問題**: `max_skip_rate: 0.4` は **side 別** の直近20サイクルのスキップ率上限。toxic_sell_veto は ML スキップと **同一バジェット** を消費する。

| sell_skip_budget (20cycle) | 配分 |
|---|---|
| ML model skip | 従来の主力 |
| velocity_sell_skip | 速度ベース |
| toxic_sell_veto (654# NEW) | **新規追加** |
| **合計** ≤ 8/20 (40%) | これを超えると **全理由が強制PASS** |

**リスクシナリオ**: 低スプレッド+高ボラ局面で toxic_sell_veto が連続発動 → skip 率 40% 到達 → 次の ML 判定「really_bad」が **強制 fill に転換** → 回避すべき損失 fill が通る。

**対策案**:
- (a) rule_based skip を max_skip_rate の計算対象外にする
- (b) toxic_sell_veto 専用の rate limit を別枠で設ける (e.g., `toxic_veto_max_rate: 0.3`)
- (c) max_skip_rate を 0.4→0.5 に引き上げる（他の影響を要検証）

**評価**: (a) が最もクリーン。rule-based の安全弁は ML の統計的 rate limit とは性質が異なる。

### 2.2 ★ regime_gate_enabled が 654# P0-1 を部分無効化

**発見経緯**: 654# で neutral_band/decay_tau を調整したが、trending 中は regime_gate で全停止されたまま。

**問題**: 650# の 13 RT のうち trending 期間の sell-entry が最も損失が大きい (RT#2: -14.27bps)。にもかかわらず **trending 時の inv_skew がゼロ** のまま。654# P0-1 は ranging 時のみ効果を発揮する。

**データ**: 650# deep dive より
- RT#2: trending_up → sell-entry → MCB HALT → -14.27bps (inv_skew = 0.0)
- RT#3: trending_up → sell-entry → -5.52bps (inv_skew = 0.0)
- trending 中の sell-entry は **全敗** だが、inv_skew は regime_gate で無効

**対策案**:
- (a) `regime_gate_enabled: false` に変更（全 regime で inv_skew 有効化）
- (b) extreme 在庫偏重時のみ regime_gate を無視する閾値追加 (e.g., abs(imbalance) > 0.3)
- (c) trending 時の inv_skew は max_factor を下げて弱める (e.g., max_factor 0.2 vs ranging 0.4)

**評価**: (b) が安全。249# で regime_gate を入れた理由（trending 中の skew 過剰反応）は valid だが、**extreme 偏重を放置する理由にはならない**。

### 2.3 ★ sell_dynamic_kill 存続の構造的矛盾

**発見経緯**: 536# シナリオ A で廃止提案されたが、現在も `enabled: true`。

**矛盾**:
1. 536# は sell_dynamic_kill を「損失を被ってから発動する遅行サンドバッグ」と批判
2. 以後 519#/540#/649# で window/duration/threshold を漸次チューニングしてきた
3. **しかし本質的な遅行性は何も変わっていない** — PnL が threshold を下回ってから kill → その間の損失は食らう
4. 654# P0-2 (toxic_sell_veto) は **事前** 遮断。sell_dynamic_kill は **事後** 遮断。両方が sell を止める → **二重遮断**

**計測が必要な問い**:
- sell_dynamic_kill が発動した後の fill PnL は実際に改善しているか？ (kill→resume のサイクルで、resume 後に改善が見られるか)
- toxic_sell_veto が十分に機能すれば、sell_dynamic_kill の役割は縮小するか？

**評価**: いきなり disable ではなく、**まず発動頻度・効果の計測** → データに基づく存廃判断。536# が正しかったのか、チューニングで救えたのかを実測で決着させる。

### 2.4 ★ preflight_insufficient の構造性

**発見経緯**: 650# で 46.2% が最大 cancel 原因と判明。コード検証で balance check が純粋な静的チェックであることを確認。

**構造問題**:
- `balance_checker.last_jpy_free` は **キャッシュされた最終残高** を参照
- 注文→約定の間に残高が変動しても反映されない
- regime_mult による動的調整はあるが、**発注頻度・pending 注文数との連動がない**
- 連続 buy 注文で JPY を消費していくと、3巡目以降が恒常的に insufficient → **buy の構造的遮断**

**影響**: buy fill が構造的にブロックされると、buy-entry RT のサンプル数が永続的に不足し、buy 側の戦略評価ができない (653# §7.3 が指摘した問題)。

**対策案**:
- (a) pending order の推定消費額を balance check に織り込む
- (b) buy insufficient 連続時に lot_size を動的縮小 (e.g., 75%→50%)
- (c) balance 閾値の regime 別最適化（ranging では余裕を持たせる）

---

## §3 562# 分析系タスクの棚卸し更新

653# §6 で整理した A1-A6 の 650#/654# 時点更新。

| # | タスク | 653# 時点 | 655# 時点 |
|---|--------|----------|----------|
| **A1** | 現行設定値棚卸し | 部分解消 | ✅ **本文書 §1 で完了** |
| **A2** | OFI-Lite boost 効果計測 | 未実施 | ⏳ OFI boost ON/OFF 比較は長期課題 |
| **A3** | SAC dead 期間明示 | 解消 | ✅ 650# で 93% stale 文書化 |
| **A4** | Composite Risk 効果分析 | 未実施 | ⏳ block 率の定量化に RT データ蓄積が必要 |
| **A5** | unclamped 反実仮想 PnL | 部分解消 | ⏳ ceiling=[0.40,0.45,0.50] 別の反事実は未実施 |
| **A6** | CV tighten sell 無効化検証 | 解消 | ✅ 565# で実装済み |

---

## §4 565# 盲点の最新ステータス

| # | 盲点 | 653# 判定 | 655# 判定 | 変化 |
|---|------|----------|----------|------|
| 1 | PnL計測窓非対称 | PARTIAL | PARTIAL | 命名問題残存。計測精度への影響は未定量化 |
| 2 | spread_capture 未活用 | RESOLVED | ✅ | 650# section_spread_decomposition で活用 |
| 3 | Regime遷移AS 6分 | UNRESOLVED | ❌ | section_regime_transition_as() 未実装 |
| 4 | AS burst 自己相関 | UNRESOLVED | ❌ | section_as_burst() 未実装 |
| 5 | 曜日効果 | UNRESOLVED | ❌ | データ蓄積待ち |
| 6 | Kelly/lot_sizing 矛盾 | RESOLVED (矛盾のみ) | ✅ | 両方false。実運用は T3 |
| 7 | Pre-clamp offset 分布 | RESOLVED | ✅ | section_clamp_saturation で出力 |
| 8 | inv_skew trending無効化 | UNRESOLVED | ⚠️ **深刻化** | 654# で ranging 改善も trending は regime_gate で不変 |

---

## §5 渙の旅路 — タイムラインと因果鎖

```
536# 渙の提示 (3/22)
  ├→ 561# DRC提案 → 562# 統合レビュー → 563# 第三意見
  │    └→ 565# errata (盲点8つ発見)
  │         └→ 576# eDRC α/β インシデント (教訓: 実測なき理論投入は危険)
  │
  ├→ 605# 渙 第一段階: 凍結棚卸し (3/26)
  │    └→ 606# SAD/MCB 解凍
  │         └→ 607# hot-reload 対応
  │
  ├→ 641#- 止血フェーズ
  │    ├→ 645# sell model 停止
  │    ├→ 648# σ stale fix
  │    └→ 649# data freshness 分離
  │
  ├→ 650# RT分析 (3/29-30): 「理論→計測」の実現
  │    └→ 651#/652# 外部レビュー: 利益保全型提案
  │         └→ 653# 561-562 残存検証
  │              └→ 654# P0-1 inv_skew + P0-2 toxic sell veto (3/30)
  │                   └→ 655# 本文書: 棚卸し + 新規4課題
  │
  └→ [次] 計測→判断サイクルの本格化
```

**536# から 654# まで 118番**: 渙の第三爻「其の躬を渙らす。悔いなし」の実践過程。固定ルールへの執着は段階的に手放してきたが、sell_dynamic_kill・regime_gate_enabled という **最も古い氷** がまだ残っている。

---

## §6 批判的視点

### 6.1 渙 の成果への過大評価リスク

- ceiling 引上げ、SAD/MCB 解凍、σ stale fix — いずれも「壊れていたものを直した」に過ぎない
- **新たな価値を生み出す** 施策（536# シナリオ A の OFI ハブ化、eDRC の有効化）は全て未着手
- 「溶かした」と思っているものの多くは **部分溶解** であり、sell ceiling 92% 飽和、sidecar 93% stale という数字がそれを物語る

### 6.2 計測至上主義の陥穽

- 565# の「理論より計測」は 576# インシデントへの正当な反省だが、**計測データが十分に揃うまで永遠に動けない** 状態を正当化する口実にもなる
- 13 RT で曜日効果は分析できない。では何 RT あれば「十分」か？ → **その基準自体が明文化されていない**
- 提案: **最低 n=50 RT、少なくとも5営業日のデータ** を Phase2 判断の統計的閾値とする

### 6.3 文書量と実装量の乖離

- 536#→655# で120番の文書が生成されたが、**実際のコード変更行数は限定的**
- 654# の実装は config 2値変更 + skip_gate_evaluator 45行 — 文書120番分の知見に対して投入量が少ない
- これは慎重さの表れでもあるが、**分析麻痺 (Analysis Paralysis)** の兆候でもある
- 次フェーズでは **文書1本あたりの実装成果物** を意識すべき

### 6.4 653# が見せた新たな視角

- 653# の 561-562 レビュー検証は、**過去の分析を現在のデータで再検証する** という方法論を確立
- これにより 562# が「正しかった / 過小評価した」を empirical に判定できた
- この方法論は今後の RT データ蓄積により、651#/652# の提案にも再適用可能
- **但し**: 検証のための検証文書が増殖するリスク。653#→654# のように、検証と実装を同一サイクルで完結させることが望ましい

---

## §7 次ステップの優先順位

### 即時 (654# コミット後)
1. **T1-6**: max_skip_rate と toxic_sell_veto の干渉を評価。rule-based skip を rate limit 計算から除外する改修を検討
2. **24h 観測**: inv_skew 発動率、toxic_sell_veto 発動率、sell skip 率の変化を計測

### 短期 (1週間)
3. **T1-1**: MCB HALT + open position 警告ログ追加
4. **T2-7**: regime_gate_enabled の条件付き緩和（extreme 在庫時）
5. **T1-3/T1-4**: regime 遷移 AS + AS burst 分析セクション追加

### 中期 (データ蓄積後)
6. **T2-8**: sell_dynamic_kill 発動頻度・効果の計測 → 存廃判断
7. **T2-1**: eDRC α/β 再推定（n≥50 RT 蓄積後）
8. **T2-3**: ceiling → VG 連動動的拡張

### Phase2 判断基準 (明文化)
- **最低条件**: n≥50 RT、5営業日以上のデータ、SHA 統一期間
- **判断対象**: entry_gate Phase2 移行、sell model 再構築着手、Kelly lot_sizing A/B テスト開始

---

*以上*
