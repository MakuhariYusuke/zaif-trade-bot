# 316# fix: セルフレビュー + 317# 観測比較実験結果 + 先行施策

> **日付**: 2026-03-07  
> **種別**: fix (セルフレビュー修正 + 観測実験 + 先行施策設計)  
> **コミット**: `40db4b2ac` (315# self-review), `72f763d6f` (317# 文書整理+実験)  
> **対象**: [311# 分析スクリプト](../../analysis/311_observational_rerun.py) + [315# 調査報告](315_ph2_rpt_ceiling_ratio_semantics.md)  
> **フェーズ注記**: 000# §2 の ph2 (G1.1-exec) 内作業。314# §6 の「Phase 0/1/2」はタスク分類番号であり、000# のフェーズ体系とは独立。

---

## 目次

- [§1 セルフレビュー修正一覧](#1-セルフレビュー修正一覧)
- [§2 317# 観測比較実験結果](#2-317-観測比較実験結果)
- [§3 構造的課題の整理](#3-構造的課題の整理)
- [§4 先行実施可能な施策](#4-先行実施可能な施策)
- [§5 dcc3064 蓄積後の評価計画](#5-dcc3064-蓄積後の評価計画)

---

## §1 セルフレビュー修正一覧

314# T0〜T2 の実装完了後に実施したセルフレビュー。`40db4b2ac` でコミット済。

### §1.1 バグ修正 (3件)

| ID | 対象 | 問題 | 修正 |
|----|------|------|------|
| BUG-1 | `_stats()` `mean_offset` | `None` 値を `or 0.0` で 0.0 に変換 → 平均を過小評価 | list comprehension で None を除外してから計算 |
| BUG-2 | `derive_improvement_proposals()` | SC/AS 両方が負のとき「AS > SC」と偽陽性レポート | SC < 0 を独立条件として分離。「spread capture が負（検出レイテンシバイアス）」と正確に報告 |
| BUG-3 | JSON 出力キー | `offset_quintiles` — 314# T2 で fill_price/mid ベースに変更済みなのにキー名が旧い | `mid_distance_quintiles` にリネーム |

### §1.2 品質改善 (3件)

| ID | 対象 | 問題 | 修正 |
|----|------|------|------|
| Q-1 | imports | `_block_bootstrap_mean_diff` 等 4 件が未使用 | 削除 |
| Q-2 | §4 AS rate 書式 | 小数表記 (`0.3333`) と百分率 (`33.3%`) が混在 | `{:.1%}` に統一 |
| Q-3 | `decision_path_analysis` | AS rate 計算が O(n²) — 各 decision_path で毎回全レコード走査 | per-path の AS dict を事前計算 → O(n) |

### §1.3 ドキュメント修正 (3件)

| ID | 対象ドキュメント | 問題 | 修正 |
|----|-----------------|------|------|
| DOC-1 | 315# §6 旧値テーブル | 「8.3 bps 過大評価」は **倍率** であって bps 値ではない。旧式出力は +0.862 bps | テーブルの旧値を実際の出力値に修正。倍率の由来を注記追加 |
| DOC-2 | 315# 日付 | `2025-07-05` — 1年以上前の日付が残存 | `2026-03-07` に修正 |
| DOC-3 | 315# §4 B1 見解 | 314# B1（maker_price pipeline の方向性問題）への見解が欠落 | 「概念的に正しいが ceiling + fill_cycle_executor 補償で実害なし → NOT RECOMMENDED」を追加 |

### §1.4 追加発見: sell_offset_floor > offset_ceiling の矛盾

```yaml
sell_guard:
  offset_floor: 0.30           # ratio ≥ 0.30 保証（攻撃的下限）
offset_ceiling_ratio: 0.15     # ratio ≤ 0.15 制限（保守的上限）
```

パイプライン順序: floor → ... → ceiling のため、ceiling (0.15) が常に floor (0.30) に勝ち、**floor は死んだ設定**。

- **実害**: なし（ceiling が勝つ = より保守的 = AS 保護）
- **リスク**: 設定の意図が矛盾しており将来の混乱源
- **対処**: §4 S-5 で検討

### §1.5 317# 追加修正

| ID | 問題 | 修正 |
|----|------|------|
| BUG-4 | `matched_mean_diff` が None のとき `f"{None:+.4f}"` で `TypeError` | `matched_n_pairs` **かつ** `matched_mean_diff` が not None のときのみ出力 |

---

## §2 317# 観測比較実験結果

### §2.1 実験条件

| 項目 | 値 |
|------|------|
| 実行スクリプト | `analysis/311_observational_rerun.py` (314# T0 修正済) |
| 全 SHA データ | 7,254 records / 2,575 filled (22日間, 75+ SHA) |
| dcc3064 (310#) データ | 89 records / 16 filled (3.3h, 蓄積不足) |
| ボット PID | 58008 (dcc3064a8, 2026-03-06 23:44 起動) |
| テスト状態 | 674 passed, 0 failed (coverage threshold のみ exit=1) |
| プロダクションコード | 変更なし |

### §2.2 全 SHA 結果サマリ

#### §2.2.1 AB 判定

| 条件 | overall | fill_rate | avg_pnl30 | downside_p10 |
|------|---------|-----------|-----------|--------------|
| None 除外 | **fail** | ✅ 40.2% vs 39.8% | ✅ -0.33 vs -0.32 | ❌ -6.87 vs -5.67 |
| None 含有 | **fail** | ❌ 33.9% vs 39.1% | ✅ -0.38 vs -0.31 | ❌ -6.85 vs -5.67 |

- Bootstrap p=0.98 (None 除外), p=0.76 (None 含有) — PnL 差は統計的に有意でない
- **downside_p10 が全条件で閾値 -5.0 bps を割れ** → テール改善が最優先課題

#### §2.2.2 Regime 別ワーストケース

| Regime | fill_rate | avg_pnl30 | downside_p10 | 判定 |
|--------|-----------|-----------|--------------|------|
| none | ❌ 14.0% | ✅ -0.80 | ❌ -5.86 | fail |
| ranging | ✅ 45.9% | ✅ -0.17 | ❌ -6.73 | fail |
| trending | ❌ 32.8% | ✅ -0.66 | ❌ -6.56 | fail |
| trending_up | ❌ 18.6% | ❌ -1.16 | ❌ -9.86 | **全指標 fail** |
| trending_down | ✅ 40.6% | ✅ -0.59 | ❌ -7.90 | fail |

**trending_up sell** が全 regime 中最悪: fill_rate 18.6% / avg_pnl -1.16 bps / p10 -9.86 bps。

#### §2.2.3 Spread Capture / AS 分解

| Side | spread_capture | realized_pnl | AS cost | efficiency |
|------|---------------|-------------|---------|------------|
| Sell | **-0.502 bps** | -0.379 bps | -0.124 bps | 75.4% |
| Buy | **-0.487 bps** | -0.306 bps | -0.182 bps | 62.7% |

**両サイドとも spread capture が負** = entry 時点で mid より不利な価格で約定。315# §4.4 で分析した通り `mid_at_fill` は fill 検出時の mid（発注時ではない）であり、検出レイテンシが systematic bias を生む。

#### §2.2.4 Sell Hour Boost

| 区分 | n | PnL | p10 | AS率 |
|------|---|-----|-----|------|
| Boost (UTC 8,13,14,16) | 138 | **-2.75** | -11.89 | **49.3%** |
| 非 Boost | 1,138 | -0.09 | -6.37 | 28.0% |

Boost 時間帯は PnL が大幅に悪い。offset boost が AS 低減に寄与していない可能性。ただし **相関 ≠ 因果** — boost なしではさらに悪化する可能性もあり、post-310# データで対照実験が必要。

#### §2.2.5 Decision Path

| Side | Path | n | PnL | AS率 |
|------|------|---|-----|------|
| Sell | ev_offset | 66 | -0.004 | 31.8% |
| Sell | unknown | 1,210 | -0.399 | 30.2% |
| Buy | ev_offset | 69 | -0.583 | 20.3% |
| Buy | unknown | 1,230 | -0.290 | 28.3% |

- Sell ev_offset は PnL 改善効果あり (-0.004 vs -0.399)
- Buy ev_offset は PnL 悪化 (-0.583 vs -0.290) — EV 判定が buy 側で逆に AS を誘引

### §2.3 dcc3064 単独結果

16 fills のみ。全セクション `insufficient` (min=50 未達)。
310# 改善効果の統計的評価には **追加 ~34 fills (~12h) が必要**。

---

## §3 構造的課題の整理

317# 実験結果 + セルフレビュー知見から、以下の構造的課題を特定。

### §3.1 downside_p10 全面 fail（最重要）

全 regime で downside_p10 が -5.0 bps 閾値を割れている。これは **テール損失** の問題であり、平均 PnL よりも大きな損失イベントの頻度・深度が課題。

**根本原因仮説**:
1. **検出レイテンシ** — fill 検出が遅れ、不利な mid で PnL 計算される
2. **AS 集中時間帯** — UTC 8/13/14/16 の AS 率 49-63% が p10 を押し下げ
3. **trending_up sell** — 逆張り売りが大幅不利値動きに直撃

### §3.2 trending_up sell の三重苦

fill_rate 18.6% / pnl -1.16 / p10 -9.86 は全 regime×side 最悪。

- fill_rate 低い = 逆張り sell が約定しにくい（市場上昇中に高値で売ろうとしている）
- 約定した場合 = さらに上昇して AS 被弾 → p10 が -9.86 bps
- 現行設定: `trending_up_sell_offset_boost: 1.8` — offset 拡大で保守化しているが不十分

### §3.3 sell_offset_floor の死亡

§1.4 の通り。安全面では問題ないが、設定の整合性が損なわれている。

### §3.4 buy ev_offset の PnL 悪化

EV offset は sell では効果的 (PnL 改善) だが buy では逆効果。EV スコアが buy 側の AS リスクを正しく反映していない可能性。

---

## §4 先行実施可能な施策

dcc3064 データ蓄積を待つ間に実施可能な施策を優先度順に整理。

### S-1: trending_up sell の追加防御 [P0 — 即時実施可能]

**根拠**: §3.2 三重苦。全指標最悪。

**選択肢**:
- **(a)** `trending_up_sell_offset_boost: 1.8 → 2.5` — さらなる offset 拡大で AS 被弾軽減
- **(b)** trending_up regime 時の sell を完全スキップ (`skip_sell_trending: true` は既に trending 全体に適用中 → trending_up のみ強化)
- **(c)** trending_up 検出時の sell cycle interval を延長して参加頻度低減

**推奨**: **(a)** から開始。trending_up sell n=83 のうち p10=-9.86 bps は 1.8 倍の offset でも防御不足。2.5 倍で offset を mid から大幅に離し、「約定しないなら損もしない」を徹底。

**実装**: YAML 1行変更。

```yaml
# Before
trending_up_sell_offset_boost: 1.8
# After
trending_up_sell_offset_boost: 2.5  # 316# S-1: p10=-9.86bps 防御強化
```

### S-2: Sell Hour Boost 時間帯の見直し [P1 — 即時実施可能]

**根拠**: §2.2.4 — Boost 時間帯 PnL -2.75 bps / AS 49.3%。

本来 AS 防御のための offset boost が、現行の乗数では効果不足の疑い。ただし **post-310# データがゼロ** のため因果推定が不可能。

**選択肢**:
- **(a)** 乗数引き上げ: UTC 8/16 を 1.5 → 2.0
- **(b)** データ蓄積待ち（24h 後に post-310# 比較）
- **(c)** 最悪時間帯 (UTC 8: AS 63%) のみスキップ

**推奨**: **(b)** 保留。post-310# データが必要。UTC 8 は n=27 と少なく、現時点での判断は早計。ただし S-1 の trending_up boost とは独立に適用可能。

### S-3: fill record に `mid_at_order` フィールド追加 [P1 — プロダクションコード変更]

**根拠**: §2.2.3 spread capture が両サイドで負。315# §4.4 の検出レイテンシバイアス。

注文発行時の mid を記録することで、spread capture 計算から検出遅延の影響を排除。

**実装箇所**:
- `fill_cycle_executor.py`: `_run_single_cycle_v3` 内で注文発行時の `mid_price` を fill record に追加
- `pnl_measurer.py`: 既存の `mid_at_fill` と別フィールドとして保存

**影響範囲**:
- 後方互換性: 旧レコードには `mid_at_order` がない → 分析スクリプトで `r.get("mid_at_order", r.get("mid_at_fill"))` フォールバック
- 追加データ量: float 1件/record — 無視できる

### S-4: None regime sell fill_rate 改善 [P1 — YAML + ロジック検討]

**根拠**: §2.2.2 — none regime sell fill_rate 14.0% (absolute min 30% を大幅に下回る)。

none regime = レジーム検出器がまだ安定していない期間。`passive_mm_enabled: true` (306#) で参加しているが sell 側の約定率が極端に低い。

**選択肢**:
- **(a)** none regime 時の sell offset を緩和（offset boost を 0.8 等に設定）
- **(b)** none regime 時は sell をスキップ
- **(c)** レジーム検出器の warmup 期間短縮

**推奨**: **(b)** を検討。none regime の sell pnl=-0.80 bps は全 regime 中ワースト、かつ 14% しか約定しない。参加のリスクリターンが悪い。ただし **buy は fill_rate 33.7% / pnl -0.15 bps** で許容範囲内のため、sell のみスキップが合理的。

### S-5: sell_offset_floor の整合性修正 [P2 — YAML]

**根拠**: §1.4 の矛盾。

**選択肢**:
- **(a)** floor を ceiling 以下に引き下げ: `offset_floor: 0.10` (ceiling 0.15 の範囲内に収まる)
- **(b)** YAML コメントで「ceiling が優先するため floor は無効」と明記し、将来の ceiling 引き上げ時に自動的に有効化されるよう温存
- **(c)** floor を削除

**推奨**: **(b)**。floor の意図（246# Glosten-Milgrom: sell AS premium）は将来的に有用。ceiling を引き上げる場合に自動復活するため温存が合理的。

```yaml
# 316# S-5: ceiling (0.15) が floor (0.30) に優先するため現在 floor は無効
# ceiling を 0.30 以上に引き上げた場合に自動的に floor が有効化される
offset_floor: 0.30
```

### S-6: buy ev_offset の挙動調査 [P2 — 分析]

**根拠**: §2.2.5 — buy ev_offset の PnL -0.583 vs unknown -0.290。

EV スコアに基づくアグレッシブ化が buy 側では逆効果の疑い。

**実施内容**:
1. ev_score_pretrade の分布を buy/sell 別に調査
2. EV 高スコア buy の fill 条件（regime, 時間帯, velocity）を特定
3. buy 側のみ ev_offset を無効化する対照実験の設計

**推奨**: 311# 分析スクリプトに EV score × side 別の cross tabulation を追加。dcc3064 データ蓄積と並行して実施可能。

### S-7: downside_p10 改善のためのテール損失分析 [P1 — 分析]

**根拠**: §3.1 — downside_p10 が全 regime で fail。

**実施内容**:
1. p10 以下（テール）の fill records を抽出
2. 共通特徴の特定: regime, 時間帯, velocity, spread_at_order, decision_path
3. テール事象の条件付き回避ルール設計

**ゴール**: テール損失の 30% を回避する conditional skip rule の設計。p10 を -5.0 bps 以内に収める。

---

## §5 dcc3064 蓄積後の評価計画

### §5.1 データ蓄積見込み

| 項目 | 値 | 根拠 |
|------|------|------|
| 現在の fill rate | ~4.6 fills/h | 16 fills / 3.3h |
| min=50 fills 到達 | **+7.5h** (累計 ~11h) | 50/4.6 ≈ 10.9h |
| 目標 n=100 fills | +18h | 統計的安定性のため |

### §5.2 dcc3064 評価で確認すべき項目

1. **sell_hour_boost の pre/post 比較** — 310# で追加された boost の AS 低減効果
2. **none regime passive_mm の効果** — 306# 有効化による fill rate / PnL 変化
3. **decision_path 分布** — ev_offset の適用率と PnL 影響
4. **全体 AB 判定** — 310# コードが sell side を改善したか
5. **downside_p10** — テール特性の変化

### §5.3 次回実行コマンド

```powershell
# 50+ fills 蓄積後に実行
python analysis/311_observational_rerun.py --git-sha dcc3064

# 全データ（ベースライン更新）
python analysis/311_observational_rerun.py
```

---

## §6 施策優先度サマリ

| ID | 施策 | 優先度 | 種別 | 前提条件 | 影響 |
|----|------|--------|------|----------|------|
| S-1 | trending_up_sell boost 2.5 | **P0** | YAML | なし | p10 改善 |
| S-7 | テール損失分析 | **P1** | 分析 | なし | p10 改善計画 |
| S-3 | mid_at_order フィールド追加 | **P1** | コード | テスト必要 | SC 精度向上 |
| S-4 | none sell スキップ検討 | **P1** | YAML+ロジック | なし | fill_rate 整理 |
| S-2 | hour boost 見直し | **P1** | YAML | post-310# データ | AS 低減 |
| S-5 | floor/ceiling YAML 整合 | P2 | YAML | なし | 設定衛生 |
| S-6 | buy ev_offset 調査 | P2 | 分析 | なし | EV 最適化 |

**即座に実施可能**: S-1 (YAML 1行) + S-5 (コメント追加) + S-7 (分析スクリプト拡張)

---

## §7 関連ドキュメント

| # | 文書 | 関係 |
|---|------|------|
| 314 | [314# resp: 312/313 review response plan](314_ph2_resp_312_313_review_response_plan.md) | 元の実行計画 (T0/T1/T2 タスク定義) |
| 315 | [315# rpt: Ceiling/Ratio Semantics](315_ph2_rpt_ceiling_ratio_semantics.md) | T1 調査結果 (ceiling 正常動作確認) |
| 317 | 観測実験出力: `analysis_results/317_observation_full.txt`, `317_observation_dcc3064.txt` | §2 の根拠データ |
| 310 | [310# impl: design improvements](310_ph2_impl_design_improvements.md) | 現在稼働中の改善コード (dcc3064) |
| 246 | [246# fix: DD cooldown + sell defence](246_ph2_fix_dd_cooldown_release_sell_defense.md) | sell_offset_floor 0.30 の由来 |
| 306 | [306# impl: 6 proposals + observational redesign](306_ph2_impl_six_proposals_observational_redesign.md) | offset_ceiling 0.15 / passive_mm / sell_hour_boost の由来 |
