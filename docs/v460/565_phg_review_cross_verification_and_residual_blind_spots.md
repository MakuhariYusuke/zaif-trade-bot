# 565# 560–564 横断検証と残存盲点の拾い上げ

- **日付**: 2026-03-23
- **目的**: 560#(実測)・561#(Gemini理論)・562#(レビュー)・563#(セカンドオピニオン)・564#(統合解剖+金融工学)の各意見をコードベース実態と照合し、全著者が一致して見落としている盲点を特定・提示する
- **手法**: git HEAD + configs/v460/fill_test.yaml + 実装コードレベルの突合検証

---

## §0 総括

5ドキュメントは以下の論点をほぼ網羅している：

| 論点 | 560 | 561 | 562 | 563 | 564 |
|------|-----|-----|-----|-----|-----|
| Clamp飽和 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Sell AS率/テール | ✅ | ✅ | ✅ | 補正 | ✅ |
| Buy側の過小評価 | — | — | — | ✅ | ✅ |
| DRC動的天井 | — | ✅ | 批評 | 批評 | 改良 |
| 乗算チェーン爆発 | — | — | ✅ | — | ✅ |
| inv_skew trending無効 | — | — | — | — | ✅ |
| CV favorable_tighten | ✅ | — | ✅ | — | ✅ |
| Same-SHA純度 | — | — | ⚠️ | ✅ | ✅ |
| AS Risk Score統合 | — | — | ✅ | 批評 | — |
| preflight原因論 | ✅ | — | — | ✅ | ✅ |

**しかし、全員が見落としている重大な盲点が8つある**。以下、実装コードの直接検証に基づいて列挙する。

---

## §1 各ドキュメントの主張検証（コードベース照合）

### 1.1 ファクト正誤表

| # | ドキュメント | 主張 | コード実態 | 判定 |
|---|------------|------|-----------|------|
| F1 | 560# | ceiling = 0.25 | `offset_ceiling_ratio_buy/sell: 0.30` (542#) | ❌ **誤り** |
| F2 | 561# | ceiling = 0.25 | 同上 | ❌ **誤り** |
| F3 | 562# | `sell_dynamic_kill.enabled: false` | `enabled: true` (L790) | ❌ **誤り** |
| F4 | 560# | sell_dynamic_kill 653回発動 | enabled=true で整合 | ✅ 整合 |
| F5 | 562# | ceiling 0.30 | `offset_ceiling_ratio_buy: 0.30, sell: 0.30` | ✅ 正確 |
| F6 | 562# | hour_ceiling_mult 実装済み | UTC 13:×1.5, 14:×2.0, 15:×1.3, 17:×1.5, 18:×2.0 | ✅ 正確 |
| F7 | 564# | inv_skew trending時無効 | `inv_skew_regime_gate_enabled` → `is_trending`時ブロック (249#) | ✅ 正確 |
| F8 | 564# | 9段乗算チェーン | offset_pipeline.py L61-380: 正確に9段、`mult *= factor` | ✅ 正確 |
| F9 | 562# | favorable_tighten buy/sell共通 | 単一bool `cross_venue_favorable_tighten_enabled: true` | ✅ 正確 |
| F10 | 560# | AS判定 = PnL < 0 | `post_fill_pnl < -as_deadzone_bps` (deadzone=2.5bps) | ⚠️ 不正確：deadzone付き |
| F11 | 全員 | PnL計測は30秒後 | **sellは90秒後** (168# `post_fill_wait_sec_sell: 90.0`) | ⚠️ **重大な誤認** |

### 1.2 重要な矛盾の影響分析

#### F3: sell_dynamic_kill の状態認識齟齬
562# は `enabled: false` を前提に「killが無効なのにAS率が高い」と論じた。実際は `enabled: true` で window=30, threshold=-0.5bps, EWMA alpha=0.05 で稼働中。つまり **killは動いているのにAS 33.1%** という、562# の結論とは逆のメッセージになる：「killが効いてもなおAS率が高い」。これは kill の閾値/タイミングの問題であり、kill の有無の問題ではない。

#### F11: Sell PnL計測ウィンドウの誤認
**全5ドキュメントが「30s PnL」として議論しているが、sell側は実際には90秒後のPnLである。** pnl_measurer.py の実装：

```python
# L81-83: sell側は90s wait
wait_sec = cfg.post_fill_wait_sec  # 30.0
if side == "sell" and cfg.post_fill_wait_sec_sell is not None:
    wait_sec = cfg.post_fill_wait_sec_sell  # 90.0

# L127-132: wait_sec後（sell=90s）に価格取得
await asyncio.sleep(wait_sec)
m.mid_30s_after = await get_mid_price()  # ← 変数名は "30s" だが実際は90s
```

従って **560# の「sell avg PnL = -0.17 bps」は30sではなく90s後の値**。これは以下の分析に影響する：

- sell のAS判定: 90s後に-2.5bps以下 → ASと分類。30s時点ではASでなくても90sでASになる（遅延逆選択）ケースがAS率を膨らませている可能性
- buy(-0.43bps@30s) vs sell(-0.17bps@90s) の比較: **異なる時間窓の値を直接比較している**。同一窓での比較がなされていない
- 563# の「buy側が平均で悪い」結論: sell が90s計測で0に近づいているだけで、30s時点ではsell の方が悪い可能性

---

## §2 全ドキュメントが見落としている盲点

### 盲点 1: PnL計測ウィンドウの非対称性（Critical）

上記 F11 の詳細。buy=30s, sell=90s という非対称な計測窓が、**全分析の基盤を歪めている**。

**影響範囲**:
1. Side別PnL比較が無効（異なる窓の値を比較）
2. AS率の意味がside間で異なる（30s AS vs 90s AS）
3. 562# の損益分岐分析（§5.8: $\bar{L} < 6.12$ bps）が buy/sell 混在で計算

**さらに深刻**: E3 60s/120s計測はベース `post_fill_wait_sec`(=30s) を基準に計算する：

```python
e3_target_60s = cfg.post_fill_wait_sec * cfg.e3_60s_multiplier  # 30 × 2.0 = 60s
```

sell側では90s既に待っているため、`e3_wait_60 = max(0, 60-90) = 0` → **sell の "60s PnL" は実質90s PnL（"30s PnL"と同一タイミング）に崩壊**。sell側の30-90s間のPnL推移は計測不能。

| 計測名 | Buy実質 | Sell実質 | Sell問題 |
|--------|---------|---------|---------|
| "30s PnL" | 30s | **90s** | 名称と実態の乖離 |
| "60s PnL" | 60s | **~90s** | "30s"と同一地点に崩壊 |
| "120s PnL" | 120s | ~120s | 正常 |

**提案**: E3計測のベースを実際の `wait_sec` に変更するか、sell PnLフィールド名を `post_fill_90s_pnl` にリネームして混乱を排除。

---

### 盲点 2: Execution Quality分解（spread_capture + AS cost）の完全未活用

pnl_measurer.py (L145-154) にKissell & Glantz (2003) の分解が実装済み：

```python
# PnL = spread_capture + adverse_selection_cost
m.spread_capture_bps = _side_pnl_bps(side, fill_price, mid_at_fill)   # fill価格 vs mid
m.adverse_selection_cost_bps = _side_pnl_bps(side, mid_at_fill, mid_30s_after)  # mid変動分
```

fill_config_results.py (L87-91) で fill_records に保存されている。**しかし、14の分析スクリプト（scripts/v460/analysis/）のいずれも `spread_capture_bps` を参照していない**（grep確認済み）。

**なぜこれが重要か**:
- `spread_capture_bps` = MMの付加価値（発注価格がmidよりどれだけ有利か）
- `adverse_selection_cost_bps` = 約定後の不利な価格移動
- 両者を分離すれば「offset戦略の質（spread capture）」と「市場環境の毒性（AS cost）」を独立に評価できる
- 560# の「Non-AS +2.31 bps」は両者の合算。分解すれば、Non-ASの+2.31がoffsetの質によるものか、偶然の価格回復かが判別可能

**提案**: analyze_fill_logs.py に `spread_capture_bps`・`adverse_selection_cost_bps` のside×regime別集計を追加。これにより「ceiling引上げによるspread captureの改善量」を直接推定できる。

---

### 盲点 3: Regime遷移遅延（最短6分）のAS影響が定量化されていない

macro_regime.py の hysteresis_count=3, cycle=120s → **regime切替に最短360秒（6分）**。BTC市場の急変は数十秒で完了するため：

- trending初動の最初の6分間は、まだ `ranging` パラメータで発注
- ranging用の浅いoffset → trending初動のtoxic flowに無防備
- これが「ranging regimeなのにAS率が高い」事象の主因である可能性

**検証方法**: fill_recordsの `regime` フィールドと、実際の価格変動を突合。rangingと記録されたfillの直前/直後にtrendingに切り替わったケースのPnLを分離せよ。

**全ドキュメントでの扱い**: 562# がregime遅行性を一般論として触れるのみ。具体的な「遷移遅延中のAS発生率」の定量化は誰もやっていない。

---

### 盲点 4: AS burstの形式的自己相関テスト欠如

AS填充が時間的にクラスタリング（連続発生）することは観察的に知られているが：

- formal autocorrelation test（ラグ1-5のAS発生の自己相関係数）が未実装
- 独立性仮定が崩れると、560# の「1,590 fills」の実質サンプルサイズは大幅に縮小
- 信頼区間が過小評価されている可能性

**なぜ重要か**:
- ASがburst的なら「1回ASを受けたら次のN秒は発注を見送る」クールダウン戦略が理論的に最適
- sell_dynamic_kill はrolling mean → kill → resume のサイクルだが、**burst検出 → 即時短期退避**とは異なる
- burst autocorrelation係数が高い(>0.3)なら、kill thresholdを待たずに1-fill退避が合理的

**提案**: fill_records の AS フラグを時系列として取り出し、lag-1~lag-5 の φ₁ を算出する分析スクリプトを追加。

---

### 盲点 5: 曜日効果の未分析

560# は11日間（3/12(木)〜3/22(日)）の日別推移を示しているが、曜日パターン分析が欠落。BTC/JPY市場の特性：

- **週末**: 機関投資家不在、流動性低下、spread拡大、個人トレーダー主体
- **平日**: 機関参加、流動性高、tight spread、アルゴ充実

560# の日別データを曜日でマッピング：

| 曜日 | 該当日 | avg PnL | 特徴 |
|------|--------|---------|------|
| 木 | 3/12 | +0.32 | 好調 |
| 金 | 3/13 | -0.79 | 急転 |
| 土 | 3/14, 3/21 | -0.47, -0.15 | 中程度損失 |
| 日 | 3/15, 3/22 | -1.44, -0.47 | **一貫して損失** |
| 月 | 3/16 | +0.86 | 好調 |
| 火 | 3/17 | -0.84 | 損失 |
| 水 | 3/18 | -0.13 | ほぼ均衡 |
| 木 | 3/19 | -1.26 | 損失 |
| 金 | 3/20 | +0.03 | 均衡 |

**日曜日が一貫して最悪日**（3/15: -1.44, 3/22: -0.47）。低流動性の週末にMMを稼働させることの是非は、時間帯分析（JST 22-23h）と同等以上に重要な制御レバーである可能性がある。

**提案**: 曜日別の fill_rate / AS_rate / avg_PnL を集計し、曜日ベースのlotスケーリングまたはskip ruleを検討。

---

### 盲点 6: Kelly有効 / lot_sizing無効の矛盾

現行config：

```yaml
lot_sizing:
  enabled: false        # ← 動的ロット無効
kelly:
  enabled: true         # ← Kelly計算は有効
  equity_btc: 0.002
  fraction: 0.5         # half-Kelly
  max_fraction: 0.25
order_quantity: 0.001   # ← 固定ロット
```

**矛盾**: Kelly基準の計算は実行されているが、lot_sizing が無効のため**出力が使われていない**。全取引が固定0.001 BTCで実行される。

- 562# (P-E) がKelly基準を用いた時間帯別lot縮小を提案
- 564# がA-S在庫ペナルティに言及
- しかし**現行でKellyが無視されている事実**を誰も指摘していない

**影響**: edge（Non-AS +2.31 bps）が存在するなら、Kelly最適ロットは現行固定値と異なるはず。特にAS率が極端に高い時間帯/regime/side でのlot縮小は、Kelly fraction の自然な帰結である。

---

### 盲点 7: 乗算チェーン結果の「実分布」が不在

564# は理論上の乗算爆発（$1.2^9 \approx 5.16$）を指摘するが、**実際のpre_clamp offset分布**が一度も提示されていない。

- 9段全てが最大乗数になるケースは実際にどの頻度で発生するのか？
- pre_clamp offset の中央値・p90・p99 はいくつか？
- 全体の何%が ceiling に到達しているのは事実（99%）だが、到達前の分布形状が不明

**なぜ重要か**:
- 乗算爆発が「理論的に起こり得る」と「実際に頻繁に起きている」は異なる
- もし pre_clamp の中央値が0.32（ceiling=0.30のわずか上）なら、ceiling を0.35にするだけで飽和率が50%以下に下がる可能性
- 逆に中央値が2.0なら、564# の言うように乗算構造自体の改修が必須

**提案**: offset_pipeline.py のログまたは fill_records の `pre_clamp_offset_ratio`（存在すれば）から分布を抽出。このデータなしにceiling引上げ幅を決めるのは根拠なき推定。

---

### 盲点 8: inv_skew trending無効化の「なぜ」に対する批判的検証の欠如

564# は「inv_skew の trending 時無効化はA-Sモデルの原則と真逆」と断じる。理論的には正しいが、**249# がこれを実装した理由**を誰も検証していない。

考えられる合理的理由：
1. **方向性ベットとの干渉**: trending時にinv_skewが在庫を中立に戻そうとすると、トレンド方向の建玉を早期に手放してしまう（逆方向に指値を有利化 → fill → 在庫中立化 → トレンド利得を逃す）
2. **Gate/Killとの役割分担**: trending時の防御はinv_skewではなく、sell_dynamic_kill/trending_sell_skip/velocity_skip が担当する設計思想
3. **計測期間の問題**: inv_skewは在庫偏りを「悪」と仮定するが、trending期間中の一時的偏りは「正常」であり、強制修正すると保有コスト vs 機会利得のトレードオフが歪む

564# はこの実装理由を無視して「復活すべき」と結論しているが、**理由なき復活は過去に解決した問題の再導入リスクがある**。249# のコミットログ / ドキュメントで当時の設計判断を確認した上で、条件付き復活（例：trending でも在庫偏りが extreme の場合のみ適用）を検討すべき。

---

## §3 各ドキュメントの提案に対する実装順序の再評価

盲点を踏まえて、560-564 の提案を再優先順位付けする。

### 3.1 即時実施すべきもの（計測基盤の修正）

| # | 施策 | 対象 | 理由 |
|---|------|------|------|
| I1 | PnL期間名の修正or E3ベース計算見直し | pnl_measurer.py | sell "60s PnL"が"30s PnL"と同一タイミングに崩壊 → データ品質問題 |
| I2 | spread_capture/AS cost分解を分析スクリプトに追加 | analyze_fill_logs.py | 既に記録済みデータの活用。560-564全ての議論精度が向上 |
| I3 | pre_clamp offset分布の可視化 | analyze_fill_logs.py or 新規 | ceiling引上げ幅の根拠データ |

### 3.2 計測基盤修正後に実施すべきもの（パラメータ変更）

| # | 施策 | 出典 | 修正条件 |
|---|------|------|---------|
| P1 | Ceiling引上げ (0.30→sell:0.40, buy:0.35) | 562# P-B | I3 のpre_clamp分布で妥当性確認後 |
| P2 | CV favorable_tighten sell側無効化 | 562# P-A, 564# | sell widen -1.10bps の内訳（spread_capture vs AS cost）確認後 |
| P3 | Stage max_mult導入 (各段上限2.0) | 562# P-C | P1と同時に実施（相補的） |

### 3.3 データ蓄積を要するもの（中期）

| # | 施策 | 出典 | 前提 |
|---|------|------|------|
| M1 | Regime遷移ASインパクト分析 | 本文盲点3 | fill_recordsのregimeフィールドとprice移動の突合 |
| M2 | AS burst autocorrelation → クールダウン戦略 | 本文盲点4 | φ₁ 係数の計測 |
| M3 | 曜日効果検証 → 週末lotスケーリング | 本文盲点5 | 少なくとも4週分のデータ |
| M4 | Kelly lotの実運用化 | 本文盲点6 | lot_sizing.enabled = true + 段階評価 |

### 3.4 アーキテクチャ変更（長期）

| # | 施策 | 出典 | 注意 |
|---|------|------|------|
| L1 | DRC統合（既存ceiling系の一本化） | 561#, 562# P-G, 563# | 「新レイヤ」ではなく「既存機構の統合」として |
| L2 | 乗算→加算型Pipeline移行 | 564# | pre_clamp分布(I3)結果次第でスコープ決定 |
| L3 | inv_skew 条件付き復活 | 564# | 249#設計理由の確認 + extreme-only guard |
| L4 | AS Risk Score max/RMS/capped-sum比較 | 562#, 563# | 結合則のA/B test |

---

## §4 五者全体の構造的バイアス

### 4.1 「AS防衛偏重」— 攻めの議論が薄い

5ドキュメント全てが「ASをいかに避けるか」に集中し、「Non-ASからいかに多く取るか」の議論が薄い。562# P-H が損益分岐分析で最も近いが、具体的な「Non-AS利得の最大化」施策は提示されていない。

- Non-AS +2.31 bps のうち spread_capture がいくらか不明 → offset を浅くすれば fill rate↑ だがspread capture↓ のトレードオフ
- **「守り」と「攻め」の最適バランス点の特定**が最終的な目標であるべき

### 4.2 「Sell偏重」— 563# の指摘が十分浸透していない

563# がsell-only narrative を補正したが、564# は再びsell+inv_skewに傾斜。buy側の「平均的な出血」（-0.43 bps @ 30s vs sell -0.17 bps @ 90s）は、**同一窓で比較すれば差が縮まるか逆転する**可能性すらある。

### 4.3 「理論先行・計測後回し」— データに聞く前に仮説を決めている

561#のDRC、562#のAS Risk Score、564#の加算型Pipeline — いずれも魅力的だが、**前提となる計測（pre_clamp分布、spread_capture分解、regime遷移AS率、burst autocorrelation）が一切なされていない**。理論構築のペースが計測のペースを上回っている。

---

## §5 結論：次に何をすべきか

**理論はもう十分にある。欠けているのは計測である。**

1. **I1-I3（計測基盤3件）を優先着手** — 理論の検証材料がなければ、DRC/AS Risk Score/Pipeline改修のいずれも「筋の良い推測」の域を出ない
2. pre_clamp offset分布が判明して初めて、ceiling引上げ幅(P1)と乗算構造変更(L2)のスコープが確定する
3. spread_capture分解（I2）が判明して初めて、Non-ASの質とoffset最適化の方向性が確定する
4. buy vs sell のPnL窓非対称性（I1修正後）が解消して初めて、side別処方箋の妥当性が検証可能

即ち：**計測の不備を放置したまま構造変更に進むと、560-564 と同じ「mixed-SHA/mixed-config」問題を次世代に持ち越す。**
