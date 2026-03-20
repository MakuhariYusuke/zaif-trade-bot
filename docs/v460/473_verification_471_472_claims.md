# 473# 検証レポート：471# / 472# の主張に対するコード・データ照合

**種別**: rev  
**日付**: 2026-03-18  
**対象**: 470# (原レポート), 471# (第1レビュー), 472# (第2レビュー)

---

## §0 結論

471#と472#はいずれも**本物のバグを指摘している**。しかし、両者とも
**そのバグが実運用でどの方向に効いているか** を誤認している。

コードトレースとデータ照合の結果:

| 主張 | バグの存在 | 実影響の方向 | レビュアーの結論 | 実際 |
|------|-----------|------------|----------------|------|
| 471# `_apply_offset_multiplier` 方向 | ✅ 正確 | — | — | — |
| 471# `_recalc` half-spread 不整合 | ✅ 本物のバグ | Sell: **保護的** (+278 JPY) | 「危険」 | **逆** |
| 471# micro-timeout 価格比率 | ✅ 本物のバグ | **休眠中** (0回発火) | 「危険」 | **未発動** |
| 472# Phantom Midpoint | ✅ 計算は正確 | Sell: **保護的** | 「壊滅的攻撃位置」 | **逆** |

**最重要発見**: `_recalc` バグは売り側では**偶発的な保護層**として機能しており、「修正」すると
売り性能が**悪化**する。一方、買い側では逆に**攻撃的**（mid寄り）に作用しており、
修正すると買い性能が**改善**する。

---

## §1 検証方法

1. `maker_price.py` の基底価格式をコードレベルで確認
2. `pre_order_adjustments.py` の `_apply_offset_multiplier` と `_recalc_price_with_new_offset` を逐語的に追跡
3. `offset_pipeline.py` の9段パイプラインと `final_clamp` の呼出箇所を確認
4. `fill_cycle_executor.py` の micro-timeout re-quote コードを確認
5. 3/16-3/18の192約定レコードに対し、パイプラインをPythonで再現し、`_recalc`前後の価格差を実測

---

## §2 471# の主張検証

### §2.1 「`_apply_offset_multiplier` は sell を mid から遠ざける」 → ✅ 正確

[pre_order_adjustments.py](scripts/v460/lib/pre_order_adjustments.py#L55)のコード:

```python
# Conservative mode (default):
if side == "sell":
    order_price = round(order_price + delta)  # delta > 0 → price UP → away from mid
```

470#が「全ブーストが逆方向（midに向かう）」と書いたのは、`maker_price` 内部の
`_scale_offset_ratio`（ratio のみ操作、price は最後に一括計算）の動作を
executor段の `_apply_offset_multiplier`（ratio と price を同時に操作）に
誤って外挿した結果。

**470#の誤り**: maker_price 内では ratio↑ = mid寄り = 攻撃的 だが、
executor pipeline 内では multiplier > 1 で sell price を UP (mid から遠ざける)。
両者は**異なる操作**。471#はこの区別を正しく指摘している。

### §2.2 「`_recalc_price_with_new_offset` が half-spread / full-spread 不整合」 → ✅ バグは本物。但し影響方向が逆

**バグの存在**: 確認済み。

[pre_order_adjustments.py](scripts/v460/lib/pre_order_adjustments.py#L46):
```python
mid_est = order_price - spread_at_order * old_ratio / 2   # ← half-spread
```

[maker_price.py](scripts/v460/lib/maker_price.py#L1051):
```python
offset = spread * effective_offset_ratio                   # ← FULL spread
price = best_ask - offset
```

`_recalc`は `spread * ratio / 2` を前提にmidを逆算するが、基底式は `spread * ratio`（full spread）。

**数学的証明** (472#の例を使用):
```
Mid=10000, Ask=10100, Spread=200, base_ratio=0.30

Step 1: maker_price → price = 10100 - 200×0.30 = 10040 (mid+40)
Step 2: _apply_mult ×2.0 (conservative) → price=10100 (mid+100), ratio=0.60
Step 3: _recalc(10100, 200, 0.60, 0.50):
  mid_est = 10100 - 200×0.60/2 = 10040 (真mid=10000, error=+40)
  new_price = 10040 + 200×0.50/2 = 10090 (mid+90)

「正しい」価格 (base式 ratio=0.50): 10100 - 200×0.50 = 10000 (mid+0)
_recalc結果: 10090 (mid+90)
誤差: +90 JPY → sell価格がmidより90円上 → 正しい値(mid)より保護的
```

**472#は「ratio 0.05の超攻撃位置」と主張したが、実際のmidからの距離は
90 JPY（half-spreadの90%）であり、「正しい」結果（mid+0）より保護的。**

**実データでの検証** (34件の sell clamp fills):

| 指標 | 値 |
|------|-----|
| 保護的（err > 0, midから遠い） | **33/34 = 97%** |
| 攻撃的（err < 0, midに近い） | 1/34 = 3% |
| 平均誤差 | **+278 JPY** (midからの距離が278 JPY増加) |
| 中央値 | +307 JPY |

**買い側** (84件の buy clamp fills):

| 指標 | 値 |
|------|-----|
| 攻撃的（err > 0, midに近い） | **79/84 = 94%** |
| 保護的（err < 0, midから遠い） | 5/84 = 6% |
| 平均誤差 | **+360 JPY** (midに360 JPY近づく) |

**結論**: `_recalc`バグは **側ごとに逆方向に効く**:
- Sell: +278 JPY 保護（midから離す）→ 偶発的防御層
- Buy: +360 JPY 攻撃（midに寄せる）→ 買い性能を悪化

**471#も472#もこの非対称性を見落としている。**
固定すると sell は悪化し、buy は改善する。

### §2.3 「micro-timeout が ratio を価格比率として使用」 → ✅ バグは本物。但し休眠中

[fill_cycle_executor.py](scripts/v460/lib/fill_cycle_executor.py#L1100) 付近:
```python
if side == "sell":
    order_price = _rq_mid * (1 + effective_offset_ratio)
```

`effective_offset_ratio` = スプレッドに対する比率 (例: 0.20) を `mid × (1 + 0.20)` として使用。
mid = 11.8M JPY の場合、`11.8M × 1.20 = 14.16M` → 完全に異常な価格。

**しかし、実データで micro_timeout は 0回発火。reprice_count も全レコード 0。**

このバグは「地雷」であるが、3/16-3/18の損失には**一切寄与していない**。
471#の指摘は将来的に重要だが、現在の損失原因としては無関係。

### §2.4 471#の「三重セマンティクス」フレーミングの評価

471#は
> 設計が逆なのではなく、設計が一致していない

と整理し、これ自体は技術的に正確。しかし実用上の問題としては:

1. **micro-timeout セマンティクス**: 正しいが休眠中 → 現在の損失には無関係
2. **_recalc セマンティクス**: 正しいがsell側では保護的 → 「修正」は短期的に逆効果
3. **maker_price vs executor セマンティクス**: 正しい指摘だが、executorは sell を正しくmid から遠ざけている

よって471#の推奨する優先順位「offset契約一貫化 → micro-timeout → _recalc → 売りパラメータ」は、
**現在の損失を止めるためには順序が逆**。売りパラメータ修正が先で、セマンティクス統一は中期課題。

---

## §3 472# の主張検証

### §3.1 「Phantom Midpoint — 架空の中値による価格ワープ」 → ✅ 数学は正確。但し結論は逆

472#の数値例を忠実に再現した結果:

```
472#の例: ratio 0.30 → ×2.0 → 0.60 → clamp to 0.50
_recalc結果: 10090 (mid+90)
「正しい」結果: 10000 (mid+0)
差: +90 JPY → sell が mid より上 → 保護的
```

472#は「true ratio 0.05（超攻撃位置）」と主張するが、これは **ask からの距離で測定した値**。
sell にとって重要なのは **mid からの距離** であり、mid+90 は mid+0 より安全な位置。

**472#の主張「Botは安全な0.50にいると思い込んでいるが実は0.05の最前線」は
フレーミングの誤り。実際の市場位置は mid+90 であり、「正しい」mid+0 より保護的。**

### §3.2 「SAC学習への致命的ノイズ注入」

472#が指摘する「保守的行動を取ったはずなのに大怪我 → 学習発散」のシナリオは、
_recalcバグが攻撃方向に効くケース（buy側）では理論的に妥当。
しかしsell側では保護方向に効いているため、report通りの問題にはならない。

### §3.3 Free Option理論 → ✅ 正確かつ重要

472#の

> sell_offset_floor はオプションをタダで配り回る行為に等しい

は的確。sell_offset_floor (0.30) + パイプラインブースト → ceiling (0.50) → mid直上 という
経路は、flow toxicity が高い環境で「撤退不能な義務」を課す構造であり、
Glosten-Milgrom / Free Option 理論から見て自殺行為。この分析は470#の結論を**強く補強**する。

### §3.4 「絶対値JPYオフセットへのプロトコル統一」

方向性としては正しい。ratio を介在させず、`offset_jpy = (ask からの距離)` で
一貫して管理すれば、three-semantics 問題は構造的に解消する。但し大規模リファクタリングが必要。

---

## §4 両者が見落としている点

### §4.1 _recalc の側別非対称性

**最も重大な見落とし**。_recalcバグは sell では +278 JPY (保護的)、
buy では +360 JPY (攻撃的) であり、「修正」の影響は方向が逆:

| アクション | Sell への影響 | Buy への影響 |
|-----------|-------------|-------------|
| _recalc を修正 | -278 JPY (悪化) | +360 JPY (改善) |
| _recalc を放置 | 現状維持 | 現状維持 |

**安直に「バグ修正」すると、sell が現在より 278 JPY mid に接近し、さらに悪化する。**

### §4.2 Final Clamp の発火率と損失の因果

```
Final clamp fired (filled): 118/192 = 61%
  Buy:  84/97 = 87%
  Sell: 34/95 = 36%
```

- **Sell**: clamp 発火分の PnL = **-1.9 bps** / 非発火分 = **+0.6 bps**
- しかしこれは clamp → 損失 の因果ではなく、**trending regime → clamp 発火 AND PnL 悪化** の交絡
- レンジ相場: PnL +0.2, clamp 発火少 → clamp なしでも OK
- トレンド相場: PnL -2.1, clamp 発火多 → clamp があっても不足

**結論**: final clamp を「ない方がいい」と解釈するのは間違い。
clamp があっても不十分なだけで、clamp なし(= ratio 0.72) はさらに悪い可能性がある。

### §4.3 61/95 の売りフィルは clamp なしで mid+24 に配置

最も重要な事実: **売りの64%（61/95）は final clamp が発火していない**。
これらは maker_price の ceiling (0.50) で直接 ratio=0.50 に抑制され、
executor EV (×0.96) で微小に保護された後、そのまま注文される。

```
maker_price ceil=0.50 → price = ask - spread×0.50 = mid+0
executor EV ×0.96 → price ≈ mid+24
No final clamp (_recalc は呼ばれない)
```

つまり、売り損失の **64% は _recalc とは無関係**。`_recalc` バグが寄与するのは
34/95=36% の fills のみであり、しかもその寄与は保護方向。

**470#が特定した「売りが mid 直上に配置される」問題は、_recalc ではなく、
maker_price の ceiling_sell=0.50 と floor=0.30 が主因。**

### §4.4 472#のアーキテクチャ提案の問題点

472#はmmap、XGBoost micro-prediction、Shadow Orderを提案しているが、
いずれも**現在の中核的損失原因（sell が mid+24 に配置される）とは無関係**。
P0問題を解決する前にインフラ投資するのは優先順位の逆転。

---

## §5 統合的結論：実際の損失の因果パス

```
maker_price:
  sell base=0.18 → floor(0.30)で強制引上 → 各段ブースト → ceiling(0.50)でフタ
  → sell 注文 = ask - spread × 0.50 = mid + 0 (事実上 mid)
  ↓
executor pipeline (EV ×0.96):
  → sell 注文 = mid + 24 (微小保護) ← 61/95 のフィルはここで注文配置
  ↓
市場の逆選択:
  → 待機中に mid が +936 JPY 上昇 (68%の確率で逆行)
  → 約定時: mid - 750 JPY (水没)
  ↓
結果:
  → sell スプレッド獲得 = -0.64 bps (負値)
  → 理論エッジ 1,196 JPY の 100% が消失
```

_recalcバグは34/95のフィルにのみ関与し、その影響は保護方向（+278 JPY）。
micro-timeout は 0回発火。Phantom Midpoint は数学的に存在するが、売りでは保護方向。

**したがって、損失の主因は変わらず「sell ceiling=0.50 + floor=0.30 がsellをmidに固定する」
ことであり、470#の原分析の核心は正しい。**

---

## §6 修正優先順位の再整理

471#と472#の指摘を統合した上での推奨:

### P0: 売りパラメータ修正 [即時] ← 470#の方向性を維持

```yaml
# 現在                     → 修正案
offset_ceiling_ratio_sell: 0.50 → 0.20  # 買いと同等
sell_guard.offset_floor:   0.30 → 0.05  # または無効化
```

これにより sell は mid+718 (= bid + spread×0.20 の対称位置) に配置され、
936 JPY の逆行に対し718 JPYのバッファが確保される。

### P1: _recalc 数式修正 [高・但し sell 悪化に注意]

full-spread に統一: `mid_est = price ∓ spread * old_ratio` (半分ではなく全量)

**注意**: P0 を先に実施すること。P0 なしで P1 だけ直すと sell が -278 JPY 悪化。
P0 で ceil=0.20 にした後なら、_recalc の pre_clamp ratio も小さくなり（0.20台）、
誤差は `spread * (0.5 - 1.5 * 0.25) ≈ +300 JPY` → 誤差の縮小方向に動く。

### P2: micro-timeout 数式修正 [高・地雷除去]

`mid * (1 ± ratio)` → `mid ± spread * ratio` に修正。
現在は休眠中だが、将来的に発動した瞬間に壊滅的な価格を生成する地雷。

### P3: offset セマンティクス統一 [中期]

471#/472# の指摘する三重セマンティクス問題は実在。
中期的に absolute JPY offset ベースに統一するか、
ratio の定義を全モジュールで厳密に統一するリファクタリングが有益。

---

## §7 470# への修正

470#の分析で修正が必要な点:

1. ~~「全ブーストが逆方向」~~ → executor pipeline のブーストは sell を mid から遠ざける（正しい方向）。
   maker_price 内の ratio 操作と executor の price 操作を混同していた。

2. ~~「単一の構造的バグに帰着」~~ → 複数のバグが存在（_recalc, micro-timeout）。
   ただし主損失原因は依然として sell floor + ceiling による mid 固定。

3. 470#の **数値分析**（スプレッド獲得 -0.64bps、逆選択率68%、理論エッジ消失100%）は正確。
   結論の方向性（sell offset 縮小が P0）も正しい。

---

*検証実行日: 2026-03-18*  
*検証データ: 192 fills (3/16-3/18) のパイプライン再現トレース*  
*コード参照: maker_price.py, pre_order_adjustments.py, offset_pipeline.py, fill_cycle_executor.py*
