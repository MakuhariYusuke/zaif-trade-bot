# 084# ph2 impl: 083 レビュー評価 — コード/データ照合と盲点指摘

| key | value |
|---|---|
| 番号 | 084 |
| フェーズ | ph2 |
| 種別 | impl |
| 対象文書 | `docs/v460/083_ph2_rev_082.md` |
| 参照 | `docs/v460/082_ph2_fill_test_deep_dive_for_codex.md`, `scripts/v460/run_fill_test.py`, `scripts/v460/ml/skip_gate.py`, `scripts/v460/lib/param_adapter.py`, `configs/v460/fill_test.yaml`, `results/v460/fill_test/`, `results/v460/verification_077/`, `tools/ab_test_runner.py`, `tools/ab_param_search.py` |
| 作成日 | 2026-02-17 |
| Git HEAD | `206d33137` |
| 結論 | **083# の 6 つの事実修正は全件コード/データで裏付けられ AGREE。ただし 083# 自身が見落としている構造的盲点が少なくとも 8 件ある。最大の盲点は「spread_adaptive が全注文に 2.0× boost しており、実効 offset が yaml 表記の 2 倍で運用されている事実」を議論していない点。** |

---

## §0 エグゼクティブサマリ

1. 083# §1 の 6 修正は **全件 AGREE** (コード / JSONL / ログで直接確認)。
2. 083# §2–§3 の Gate 現況認識 (E1/E4/E5 FAIL) と統計判定 (p=0.109) も正確。
3. しかし 083# は **8 つの構造的盲点** を抱えており、特に以下が収益判断に直結する:
   - **盲点 A**: spread_adaptive がほぼ全件に 2.0× boost → 実効 offset が yaml 値の 2 倍
   - **盲点 B**: param_adapter が AS 優先で offset を下げ続ける負のスパイラル
   - **盲点 C**: Wait 長い方が PnL 良好なのに timeout 短縮を推奨
4. 083# §4 の「48 時間施策」は部分的に正しいが、盲点を踏まずに実装すると **逆効果になるリスク** がある。

---

## §1 083# §1 修正ポイントの検証結果

### #1 SkipGate「機能不全」→「解釈修正」— **AGREE (補足あり)**

**083# の主張**: "最新 run では score は連続記録され `-5` 近傍"

**検証結果**:
- JSONL 上で SkipGate score が記録されているのは **last 2 run_ids のみ (10/501)**
  - `1771227270_eca8d769`: 3 件, score range [-5.08, -4.75]
  - `1771228012_ec0d219b`: 7 件, score range [-5.31, -4.91]
- 残り 491 件は score = 0 (記録なし)
- `fill_test.log` 内の SkipGate ログエントリ: **0 件** (loading 含む)

**補足**: "run 混在の影響が大きい" は事実だが、もう一歩踏み込むと:
- `skip_gate.py:193`: `pred_pnl = -pred_prob * 10` → score = -10 × P(AS)
- score -5.0 = P(AS) 0.50, threshold = 0.65
- score range [-5.31, -4.75] → P(AS) range [0.475, 0.531]
- **結論: モデルは動いているが、P(AS) が閾値 0.65 に全く届かず「判別力がない」**
- 機能不全ではなく「判別不能」— 082# の懸念は方向として正しい

**判定**: ✅ AGREE (083# は「run 混在」で説明を止めているが、根本原因は「モデル判別力不足」)

---

### #2 集計粒度修正 — **AGREE**

- run_id 分布:
  - `None` (quarantine): 149 件
  - `4f513c12`: 136, `383ebf85`: 106, `2dfc8dfb`: 70, `2dfec424`: 30, `ec0d219b`: 7, `eca8d769`: 3
- clean (run_id あり) = 352, quarantine = 149
- SkipGate score は last 2 runs (計 10 件) のみ

**判定**: ✅ AGREE

---

### #3 Early Exit 発火 — **AGREE**

- `fill_test.log`: `2026-02-16 16:39:40 [early_exit] Loss threshold hit at 25s: -5.10 bps < -5.0`
- 082# 時点で 0 回 → その後 1 回発火

**判定**: ✅ AGREE (更新済み事項)

---

### #4 Adaptation 記録 — **AGREE (重要発見あり)**

- `run_fill_test.py:1462`: `effective_offset_used` 記録あり (latest 10 件)
- `spread_offset_ratio`: 226 件が記録済み

**intra-run 適応の証拠** (run `383ebf85`, n=106):
```
records [0-19]:   unique ratios = [0.04, 0.05]
records [20-39]:  unique ratios = [0.04]
records [40-59]:  unique ratios = [0.04]
records [60-79]:  unique ratios = [0.03, 0.04]
records [80-105]: unique ratios = [0.03]
```
→ 0.05 → 0.04 → 0.03 と段階的に低下 (AS 回避方向に適応) — **§2 盲点 B で詳述**

**判定**: ✅ AGREE

---

### #5 side_offset は乗算ではなく上書き — **AGREE**

- `run_fill_test.py:267-271`: `side_offset.sell: 0.12` → `spread_offset_ratio_sell = 0.12`
- `run_fill_test.py:668-669`: sell 側は `effective_offset_ratio = self.config.spread_offset_ratio_sell` (直接代入)
- 082# の「×0.12」は誤り。正しくは `0.12 に置換` (置換後 spread_adaptive 等で further modify)

**判定**: ✅ AGREE

---

### #6 計算誤り `113.6JPY ≈ 0.001bps` — **AGREE**

- 正: `113.6 / 10,569,106 × 10000 = 0.107 bps`
- 082# の `0.001 bps` は 2 桁の桁誤差

**判定**: ✅ AGREE

---

## §2 083# の盲点 — コード/データが示す未指摘事項

### 盲点 A: spread_adaptive が全注文に 2.0× boost (CRITICAL)

**事実**:
- `spread_at_order` が記録された 172 件のうち、**100% が narrow spread (<10 bps)**
- Median spread: 2,236 JPY = **2.08 bps** (narrow_spread_bps 閾値 10.0 を大幅に下回る)
- `run_fill_test.py:687-694`: narrow spread 判定 → `effective_offset_ratio *= narrow_spread_boost (2.0)`

**影響**:
```
yaml 設定          spread_adaptive 適用後     実効 offset
─────────────────  ─────────────────────────  ────────────
buy:  0.05         0.05 × 2.0 = 0.10          0.10
sell: 0.12         0.12 × 2.0 = 0.24          0.24
```

- **全注文が yaml base 値の 2 倍の offset で運用されている**
- 082# は `spread_offset_ratio: 0.05` を「小さすぎる」と問題提起したが、実際は 0.10 で運用
- 083# は side_offset の「乗算 vs 上書き」は修正したが、spread_adaptive の 2.0× boost には言及なし
- **offset 0.24 で sell PnL がまだ -0.95 bps**ということは、offset 問題ではなく AS の構造的要因が大きい

**083# への影響**: §4.1-3「固定時間フィルタの縮退」や §4.2-1「Sell 専用ポリシー化」の設計前提が狂う

---

### 盲点 B: param_adapter が AS 優先で負のスパイラル (HIGH)

**事実** (`scripts/v460/lib/param_adapter.py:107-120`):
- `high_as AND low_fill` → **AS 回避優先で offset 縮小**
- Run `383ebf85`: offset 0.05 → 0.04 → 0.03 (AS > max_as_ratio 0.15 が続いた)

**問題**:
- offset 縮小 = 注文がスプレッド内で less aggressive = fill rate がさらに低下
- fill rate 低下 → low_fill 条件も同時成立 → しかし AS 優先で offset 縮小続行
- **デッドロック**: AS は下がるかもしれないが fill rate も下がる → 結局 PnL 改善せず

**実測証拠**:
```
run_id                   fill_rate   offset 方向
4f513c12 (初期)          86.8%       base 0.05
2dfc8dfb                 85.7%       0.05 固定
383ebf85                 77.4%       0.05 → 0.03 (AS 優先で低下)
ec0d219b (最新)          57.1%       0.10-0.24 (spread_adaptive × 2.0)
```

- fill rate は run ごとに低下しているが、offset 方向と因果関係が不明確
- 083# §3-5「レジーム unknown」は指摘しているが、適応方向の問題は見落とし

---

### 盲点 C: Wait 長い方が PnL 良好なのに timeout 短縮を推奨 (HIGH)

**083# §4.1-4**: "timeout 短縮 + 再見積" を推奨

**データ反証**:
```
Wait Quintile    Range (s)    n     Mean PnL (JPY)    AS rate
Q1               5-6         75    -0.495             41%
Q2               6-12        75    -1.127             47%
Q3               12-25       75    -1.194             41%
Q4               28-63       75    -0.234             39%
Q5               64-304      79    -0.119             28%  ← 最良
```

- **Wait ≥ 64s の Q5 が最良パフォーマンス** (PnL -0.119, AS 28%)
- Wait < 18s は mean PnL = -0.838 JPY, AS = 42.2%
- Wait ≥ 18s は mean PnL = -0.397 JPY, AS = 35.6%

**timeout 短縮のリスク**: 高 PnL の long-wait fill を切り捨てる → 全体平均悪化

**083# §5-Q7** は「Wait と AS の因果は交絡の可能性」と慎重だが、§4.1-4 では断定的に短縮を推奨 → **矛盾**

**推奨**: timeout はむしろ維持し、**短 wait fill の AS 防衛を強化**する方向が合理的

---

### 盲点 D: AS_raw 52.1% vs AS_deadzone 34.1% — 18pp のマスキング (MEDIUM)

**事実**:
- `adverse_selected_raw` が記録された 290 件中:
  - AS_raw: 151/290 = **52.1%**
  - AS_deadzone (2.5bps 適用): 99/290 = **34.1%**
  - Deadzone がマスキング: 52 件 = **17.9pp**

**問題**:
- Monitor の E5 = 34.1% も deadzone 版
- 実際の AS は **52.1%** — 取引の半数以上が逆選択を受けている
- 083# は E5=34.1% を引用するが、raw との差には言及なし
- `as_deadzone_bps: 2.5` の妥当性自体が問われるべき

---

### 盲点 E: Time filter が極端に制限的 — Buy 8h/Sell 6h のみ稼働 (MEDIUM)

**事実** (yaml 設定):
```
Buy  open hours: [0, 3, 4, 5, 6, 7, 20, 22]      — 8/24h (33%)
Sell open hours: [0, 6, 7, 10, 11, 20]             — 6/24h (25%)
```

**問題**:
- 082# と 083# は「n=137/352 まで削減」と記述するが、これは既に time filter 適用後の結果
- 実際は **全 24 時間のうち 1/3 しか取引機会がない**
- 083# §3-3「高頻度目標と逆方向」は正しい指摘だが、**程度を過小評価**
- `cycle_interval=120s` × 33% 稼働時間 ≈ 2.4 cycle/h → 実効 fill は **約 1.8/h**
- "粗い近似で +2 JPY/時" (083# §2.3) すら楽観的

---

### 盲点 F: API エラーが Cancel の 28% (MEDIUM)

**Cancel 理由**:
```
timeout:        51 件 (41.8%)  mean_wait=303.2s
api_error:      34 件 (27.9%)  mean_wait=0.0s
unknown:        26 件 (21.3%)  mean_wait=70.0s
status_unknown: 10 件 (8.2%)   mean_wait=5.8s
None:            1 件 (0.8%)
```

**問題**:
- 083# §4.1-4 は timeout 短縮のみ議論
- しかし **api_error (28%) は wait=0.0s** — 発注自体が失敗している
- これは市場構造ではなくインフラ要因
- api_error を削減すれば fill rate 改善余地がある
- `status_unknown` (8%) も調査不足

---

### 盲点 G: 時系列でパフォーマンスが悪化 (LOW)

**事実**:
```
First  half (189 fills): mean PnL = -0.349 JPY
Second half (190 fills): mean PnL = -0.907 JPY
```

- 後半でパフォーマンスが 2.6× 悪化
- alpha decay、市場環境変化、または param_adapter の不適切な方向 (盲点 B) が原因候補
- 083# は時系列推移を分析していない

---

### 盲点 H: E1 threshold 90% は非現実的 (LOW)

**事実**:
- 最良 run (4f513c12) でも fill rate = 86.8%
- Maker 指値 + time filter + spread offset の戦略で 90% は構造的に困難
- E1_fill_rate_p90 = 66.15% は p90 計測 (rolling window のボトム)
- 083# は E1 FAIL を記載するが、閾値の妥当性を問うていない

---

## §3 083# のアクション提案に対する評価

### §4.1-1 run_id 分離評価 — ✅ 同意 (即実行すべき)

- 混在データの全体集計は misleading
- `--run-id` / `--git-sha` フィルタは低コストで高価値
- ただし run_id=None (149 件) の正体解明が先

### §4.1-2 SkipGate 可観測性 — ✅ 同意 (P(AS) 直接記録)

- score = -10*P(AS) は解釈困難
- `P(AS)`, `threshold_used`, `model_version` を FillRecord に追加
- 再学習の前に「そもそもモデルの AUC は十分か」の評価が必要

### §4.1-3 固定時間フィルタの縮退 — ⚠️ 条件付き同意

- 方向は正しいが、現在 Buy 8h / Sell 6h しか開いていない事実を踏まえると
- 「縮退」ではなく「まず開いている時間帯の中でマイクロ条件を導入」が先
- wide/strict を micro 条件に置き換える前に、**最悪帯 (08 UTC 等) の time-filter 効果を定量化**すべき

### §4.1-4 timeout 短縮 — ❌ 反対 (盲点 C)

- データが wait 長 = PnL 良好を示す
- timeout 短縮は Q5 の best-performing fill を排除する
- 代替案: **短 wait fill の AS 防衛** (fast_fill_defense の閾値調整) + **cancel-replace** (stale 化した注文のみ再見積)

### §4.2-1 Sell 専用ポリシー — ✅ 同意

- sell PnL = -0.95 bps vs buy = -0.32 bps の差は有意
- ただし sell offset は既に **0.24** (spread_adaptive 2.0×) であり、offset 増加余地は限定的
- **AS 構造的要因** (BTC 上昇バイアス) への対策が本丸

### §4.2-2 イベント駆動フィルタ — ✅ 同意 (最重要)

- hour 固定 → spread/flow/volatility イベント駆動は正しい方向
- 高頻度と危険回避の両立に必須

### §4.2-3 G1.1 補助 KPI — ✅ 同意

- round-trip と net inventory の常時併記は必要

---

## §4 083# §5 (Q&A) への補足

### Q3 offset 最適化: 目的関数の前提に注意

- 083# 提案: `E[pnl_per_attempt] = fill_rate * mean_pnl_fill`
- **補足**: spread_adaptive が全件 2.0× boost している現状では、探索すべきは `narrow_spread_boost` と `side_offset.sell` の組み合わせ
- base offset (0.05) の探索よりも spread_adaptive parameters の探索が本命

### Q7 Wait と AS の因果: データは明確

- Wait Q5 (64-304s): AS = 28%, PnL = -0.119 JPY (最良)
- Wait Q1-Q3 (5-25s): AS = 41-47%, PnL = -0.50 to -1.19 JPY
- 交絡はあり得るが (e.g., 穏やかな市場 → wait 長 & AS 低)、**timeout 短縮が逆効果な方向は明確**

### Q8 ロット増加: Gate 閾値再検討が先

- 083# の「Gate 未通過でロット増は不可」は正論
- ただし E1 threshold 90% が非現実的なら **Gate 閾値自体の再設計** が必要
- さもなければ永遠に Gate を通過できず、system が前に進まない

---

## §5 083# §6 (v458 再利用) への評価

### tools/ab_test_runner.py — ✅ 再利用可能

- multi-seed, 並列実行, 集計済み
- ただし fill_test 向けに adapter が必要 (現在は training experiment 前提)
- `run_fill_test.py` の run_id 単位実行を `ab_test_runner` でラップするのは有効

### tools/ab_param_search.py — ✅ 条件付き再利用

- grid search + bandit 的探索基盤
- ただし fill_test の 1 イテレーションが長い (120s × n cycles) ため、grid は小規模に限定
- **spread_adaptive parameters** (narrow_spread_boost, narrow_spread_bps) の 2D grid に最適

### v458/v457 docs の教訓 — ✅ 適用すべき

- 評価汚染・二重計上・設定未配線 (`docs/v458/19`)
- 因果性とリーク回避 (`docs/v457/34`, `docs/v457/36`)
- 「queue_wait に基づく戦略は事後情報依存」は正当な警告

---

## §6 優先順位の再整理 (084# 提案)

### Tier 1: 即実行 (24h)

| # | 施策 | 根拠 | 期待効果 |
|---|---|---|---|
| 1 | run_id 分離評価の実装 | 083# §4.1-1 + 084# 同意 | 評価精度向上、バイアス除去 |
| 2 | SkipGate P(AS) 直接記録 | 083# §4.1-2 + 084# 同意 | 判別力評価の前提条件 |
| 3 | api_error の原因調査 | 084# 盲点 F | fill rate 改善 (28% 削減余地) |

### Tier 2: 検証後実装 (48-72h)

| # | 施策 | 根拠 | 注意点 |
|---|---|---|---|
| 4 | spread_adaptive parameters 探索 | 084# 盲点 A | narrow_spread_boost 2.0 は最適か? 1.5, 2.5 等と比較 |
| 5 | param_adapter の方向ロジック修正 | 084# 盲点 B | AS & low_fill 同時 → hold (縮小停止) に変更 |
| 6 | AS_raw 指標の並行記録 | 084# 盲点 D | deadzone 版と raw 版を常時比較可能に |

### Tier 3: 中期 (1 週間)

| # | 施策 | 根拠 | 注意点 |
|---|---|---|---|
| 7 | イベント駆動フィルタ (hour → micro) | 083# §4.2-2 | hour 固定 buy=8h/sell=6h を段階的に緩和 |
| 8 | fast_fill_defense による短 wait AS 防衛 | 084# 盲点 C 対策 | timeout 短縮ではなく短 wait 側を制御 |
| 9 | E1 threshold / Gate 閾値の再設計 | 084# 盲点 H | 90% は構造的に非現実的 → 85% or rolling 改善率へ |

---

## §7 結論

083# は **082# の事実誤認を正確に指摘** しており、修正内容は全件コード/データで裏付けられた。
Gate 現況認識 (E1/E4/E5 FAIL) と統計判定 (permutation p=0.109, power 65%) も正確。

しかし 083# 自身が **8 つの構造的盲点** を抱えている:

1. (**A**) spread_adaptive 2.0× boost が全注文に適用 → 実効 offset が議論と乖離
2. (**B**) param_adapter の AS 優先縮小が負のスパイラルを形成
3. (**C**) Wait 長 = PnL 良好 ←→ timeout 短縮推奨の矛盾
4. (**D**) AS_raw 52.1% が deadzone で 34.1% にマスクされている
5. (**E**) Time filter が Buy 8h/Sell 6h しか開いておらず高頻度と矛盾
6. (**F**) api_error 28% が未対処のまま
7. (**G**) 時系列で後半パフォーマンスが 2.6× 悪化
8. (**H**) E1 threshold 90% が非現実的

**最短ルートは**: run_id 分離 → SkipGate 可観測化 → api_error 削減 → spread_adaptive 探索。
timeout 短縮は保留、param_adapter のロジック修正は検証後に実施。
