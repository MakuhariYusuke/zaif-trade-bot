# 708# 706/707 セカンドオピニオン深堀り: 盲点検証と実装判断

## 概要

[706#](706_cplt_post705_profitability_reality_check.md) と [707#](707_cplt_second_opinion_post706_academic_pipeline_evolution.md) の二者からセカンドオピニオンを受けた。
本稿では両者の主張をデータで裏取りし、**両者が見落としている盲点を7つ特定**した上で、即実行項目・自身タスク・Codex委託を仕分ける。

---

## 1. 706/707 の主張に対するデータ検証

### 1.1 検証結果サマリ

| # | 主張 | 出典 | データ検証結果 | 判定 |
|---|------|------|---------------|------|
| V1 | trend_5s_sell_guard は sell にとって有害 | 706# §5 | counterfactual +136.8 bps, WR=75%, n=32 | **完全支持** |
| V2 | final_clamp_hard_skip は利益機会を喪失 | 706# §5 | counterfactual +150.7 bps, WR=52%, n=40 | **方向は支持** (WR低い) |
| V3 | entry_gate は 100% suppressed の dead layer | 706# §3 | buy=242/242, sell=234/234 (100.0%) | **完全支持** |
| V4 | buy base_offset 引き上げで止血 | 706# §8 P3, 707# §3 P0 | buy+ranging n=211, avg=-0.350 bps (主損失源) | **支持** |
| V5 | OBI deadband 0.1→0.25 で防御 | 707# §3 P2 | corr(OBI,buy_pnl)=-0.007, **U字型関係** | **不支持** (後述) |
| V6 | skip_gate は selector 不全 | 706# §4 | bypass_mode=true, 80.5% bypass/forced_pass | **完全支持** |
| V7 | SAG は定数税 | 706# §3 | 0.500 bps × 474 fills = **237.0 bps 税** | **完全支持** |
| V8 | sell veto 群は過防御 | 706# §5 | trend_5s + final_clamp 合計 **287.5 bps** 機会損失 | **支持** |

### 1.2 706# 数値の乖離

706# は `final_clamp_hard_skip` の counterfactual を `+5.455 bps` と報告。
本検証では `order_price` ベースの future mid 再計算で `+3.768 bps`。
方向は同じだが約 30% 低い。706# の計算手法（future mid の取得方法）が不明であり、数値を額面通り受け取るべきではない。
ただし結論（有害なブロック）は変わらない。

### 1.3 707# OBI 提案の反証

707# は `ranging_obi_asymmetry_factor` を 0.3→0.6, `ranging_obi_threshold` を 0.1→0.25 に引き上げることを提案。
前提: OBI が sell 方向に偏っているとき buy は危険 → offset を上げる。

**データが示す現実**:

| OBI 帯 | n | avg buy PnL (bps) |
|---------|---|-------------------|
| [-1.0, -0.25) sell_heavy | 55 | **-0.545** |
| [-0.25, 0.0) mild_sell | 52 | **-0.059** |
| [0.0, 0.25) mild_buy | 70 | **-0.096** |
| [0.25, 1.0) buy_heavy | 65 | **-0.602** |

OBI-PnL 関係は **U 字型**。sell_heavy だけでなく **buy_heavy も等しく悪い**。
707# の線形モデル（`offset += OBI * factor`）は、buy_heavy 時に offset を**下げる**方向に作用し、最も危険な帯域でさらに防御を薄くする。

**結論**: 707# の OBI deadband 提案はそのまま採用不可。U字型を考慮した absolute value ベースか、OBI 中立帯のみ活用する設計が必要。

---

## 2. 706# と 707# の盲点 — 7つの発見

### 盲点①: AS trailing gate (700#) が完全不活性

`as_trailing_gate_enabled: true` にもかかわらず:

```
Actions: boost=4, veto=0 (476 fills)
AS trailing rate: mean=0.098, P90=0.174, max=0.333
soft_threshold=0.30 → 到達率 1.2%
hard_threshold=0.45 → 到達率 0.0%
```

**700# で導入した AS trailing gate は実質 non-operational**。
max AS rate が 0.333 で soft_threshold=0.30 とほぼ一致するが、1.2% しか到達しない。

706# は entry_gate と skip_gate の不全を指摘したが、**もう一つの gate が同様に死んでいる**ことを見落としている。
707# も触れていない。

**対処**: soft_threshold を 0.20 に下げるか、観測モードに明示的に切り替える。

### 盲点②: sell+trending_down が 704# 後にむしろ悪化

704# は `sell_trending_down_offset` を追加した。しかし:

```
sell+trending_down PRE:  n=60, avg=-0.728 bps
sell+trending_down POST: n=16, avg=-1.072 bps (悪化)
```

n=16 と小サンプルだが、704# の sell_trending_down_offset が**利益化に寄与していない可能性**がある。
全体 sell が +9.9 bps に改善したのは `sell_hour_offset_boost` と `sell_trending_down_offset` の複合効果だが、trending_down 単体では効果が見えない。

706# は sell 修正を全体として肯定したが、**セグメント別の分解検証を行っていない**。
707# は sell 側の分析をほぼ行っていない。

### 盲点③: SAG penalty が定数設計（0.500 bps 一律）

706# は SAG を「定数税」と正しく指摘した。しかし **なぜ一律なのか** を掘っていない。

```
spread_as_guard:
  ev_penalty_bps: 0.5     # ← flat constant
  spread_threshold_bps: 15.0  # ← Coincheck平均spread 2bpsで常時発動
```

- 市場 spread が 2 bps のとき penalty 0.5 bps → 実質の 25% 課税
- 市場 spread が 4 bps のとき penalty 0.5 bps → 実質の 12.5% 課税

本来は「spread が狭いほど逆選択リスクが高い」ため、penalty は spread に反比例すべき（spread が広いときは penalty 不要）。
現在の threshold=15bps は Coincheck の実態（spread median ≈ 2bps）と乖離しすぎており、100% 発動する。

**対処案**: `ev_penalty_bps` をスプレッド比例型に変更するか、threshold を現実的な値（例: 4 bps）に下げて発動頻度を制御する。

### 盲点④: 30s→120s reversion rate 33% — 計測窓の妥当性

705# deep dive で発見した「30s で利益→120s で損失」の reversion rate 33%:

```
Buy reversion (30s+→120s-): 17/52 (32.7%)
```

30s PnL で最適化しているが、**1/3 のケースで 120s 後にはマイナスに転落**する。
706# も 707# も optimization target (30s vs 120s) の妥当性を議論していない。
市場メイカーの真の利益は「ポジション解消時」であり、30s は intermediate metric に過ぎない。

**対処**: 計測窓の変更は大きな設計判断。現時点では記録のみ。

### 盲点⑤: 706# P0「一度に一つだけ変更」と P1-P4 の矛盾

706# は P0 で「一度に guard を複数触らない」と述べつつ、P1-P4 を並列提案している。
これを文字通り実行すると、**P1 だけで最低 3 回の A/B テスト**（trend_5s, final_clamp, sell_dynamic_kill）が必要。
各テストに 24-48h × 3 回 = **3-6 日**。P3 (buy offset) まで到達するのに 1-2 週間。

短期高収益が大義（0#）である本プロジェクトでは、「clean causal inference」と「speed of iteration」のトレードオフが不可避。

**本稿の方針**: 706# P0 の精神を尊重しつつ、**データで因果が明確なもの（trend_5s: protocol 695 で否定）は即変更**。不確実なもの（final_clamp: WR=52%）は観測を続ける。

### 盲点⑥: spread_offset_ratio フィールドの実態

検証 V3 で `offset_ratio` が fill record 上 0.0 と出た。
実際には `spread_offset_ratio: 0.17250` として final offset が記録されているが、**base_offset 成分の内訳がログ上分離されていない**。
base_offset 変更の効果を A/B で測定するには、**offset stage breakdown** のログ出力が必須。
`offset_stage_recording_enabled: true` は既に有効だが、V3 でフィールドが取れなかったことは記録方法の問題。

### 盲点⑦: skip_gate score の「符号崩れ」は model drift か threshold drift か

706# は skip_gate score avg が `+0.33→-0.31` に変化したことを指摘し、「selector として機能していない」と結論。
しかし **score の変化が model drift（学習データの陳腐化）か、入力特徴量の分布シフトか** を分離していない。

fill record のサンプル:
```
skip_gate_score: 0.7397
skip_gate_threshold_used: 0.1
skip_gate_reason: pass
```

score=0.74 はかなり高い（> threshold 0.1）。
705# deep dive の平均 score 推移（+0.33→-0.31）と矛盾 — 個体差が大きい。
score 分布の二峰性（一部で高い score、他で負）が疑われる。

---

## 3. 即実行した変更（708# コミット）

### 変更 A: trend_5s_sell_guard 無効化

```yaml
trend_5s_sell_guard:
  enabled: false    # 708# 706#P1: veto→無効化
```

**根拠**: protocol 695 の counterfactual: net -361 bps。本検証: n=32, avg=+4.276 bps, WR=75%。
sell guard の中で最も明確に有害。

### 変更 B: buy base_offset 引き上げ

```yaml
side_offset:
  buy: 0.08    # 708# 706#P3/707#P0: 0.05→0.08
```

**根拠**: buy+ranging n=211, avg=-0.350 bps が主損失源。706#/707# 合意。即効性の止血。

---

## 4. 今後のタスク仕分け

### 4.1 自分でやるタスク（短期、YAML/設定レベル）

| # | タスク | 根拠 | 優先度 |
|---|--------|------|--------|
| T1 | final_clamp_hard_skip_mult の sell 側緩和検討 | counterfactual +150.7 bps, WR=52% — WR が低いため観察継続後 | P2 |
| T2 | AS trailing gate の threshold 再調整 | soft=0.30 で 1.2% しか到達しない | P2 |
| T3 | SAG threshold 現実化 (15→4 bps) | 常時発動 → 選択的発動に | P1 |
| T4 | 24-48h 固定運転後のデータで 708# 変更効果を測定 | 706# P0 | P0 |

### 4.2 Codex に委託するタスク

| # | タスク | 理由 | 詳細 |
|---|--------|------|------|
| CX1 | skip_gate score 品質分析・二峰性検証 | 大規模データ分析 + model architecture 理解が必要 | prompt A |
| CX2 | entry_gate selectivity redesign | CalibrationMap の EV 改善 + dead code 修正 | prompt B |
| CX3 | SAG penalty 比例化設計 | offset pipeline 改修 | prompt C |

---

## 5. 706# / 707# への総合評価

### 706# の評価

**長所**:
- fill_records 実測に基づく定量的主張
- protocol 695 の再利用による trend_5s 否定が説得力あり
- 「safety / gate の多層化が真のボトルネック」という構造的診断は正確
- P0（計測固定化）は運用上正しい原則

**短所**:
- counterfactual 数値の再現性に疑義（final_clamp: 706#=+5.455 vs 本検証=+3.768）
- sell 側を全体として肯定したが trending_down 単体の悪化を見落とし
- AS trailing gate の不活性を見逃し
- P0 と P1-P4 の同時並列が矛盾

### 707# の評価

**長所**:
- 理論的フレームワーク（情報理論、GIGO）の適用は正しい
- 自己の理想論（Continuous EV Pricing）を自ら反証する誠実さ
- API rate limit / quote churn の運用リスク指摘は実践的
- 「ベースオフセット引き上げ = Robust Prior の再設定」という再解釈は秀逸

**短所**:
- **OBI の U 字型関係を見落とし** — 最も重要な反証ポイント
- データに基づく定量検証が完全に欠落（数値なし、protocol 未参照）
- Coincheck 固有の制約（spread 分布、流動性）への理解が浅い
- Rev.1→Rev.2 のピボットが大きく、初版の信頼性に疑問符

### 合意点と不合意点

| 論点 | 706# | 707# | 本検証 |
|------|------|------|--------|
| buy base_offset 引上げ | ○ P3 | ○ P0 | **○ 即実行** |
| trend_5s 無効化 | ○ P1 | ─ | **○ 即実行** |
| entry_gate dead | ○ §3 | ─ | ○ 確認済 |
| skip_gate selector 不全 | ○ §4 | ─ | ○ 確認済 |
| OBI deadband 引上げ | ─ | ○ P2 | **✗ U字型で不支持** |
| SAG penalty 改善 | △ 再設計 | ○ P1 精緻化 | ○ 比例化推奨 |
| Continuous EV Pricing | ─ | ✗ 自己棄却 | ✗ GIGO |

---

## 6. 再現コマンド

```bash
# 本稿の検証スクリプト
.venv/Scripts/python.exe temp/verify_706_707.py

# counterfactual 再計算（order_price + future mid）
# temp/verify_706_707.py 内 v9b セクション

# protocol 695 (trend_5s) — 706# が引用
.venv/Scripts/python.exe -m scripts.v460.analysis.run_protocol \
  --protocol 695_trend5s --start 2026-04-01 --end 2026-04-06

# AS trailing gate 検証
# temp/verify_706_707.py 内 v10 セクション
```
