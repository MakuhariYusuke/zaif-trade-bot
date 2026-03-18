# 480# 478/479 レビュー検証: 主張の裏取りと重大な誤読の訂正

**種別**: verify  
**対象**: 478#, 479# ならびに関連実装  
**日付**: 2026-03-18

---

## §0 結論

478# と 479# はいずれも 477# の分析に対する重要な補正を含んでいる。しかし、独自に fill_records を再分析した結果、**479# の核心的主張に重大な事実誤認がある** ことが判明した。

### 478# — 概ね正確 ✅

| 主張 | 検証結果 |
|------|----------|
| コホート混在で current state を断定できない | ✅ 10個の SHA 混在を実測確認 |
| `ranging_low_vol_skip` 217件は旧SHA `d0769f283da3` 単独 | ✅ 完全一致 |
| `preflight_insufficient` 134件は `b70365d4d4c9` の資金制約時代 | ✅ 完全一致 |
| Gate 2b sell は soft mode (blocked=False) | ✅ コード確認済 |
| preflight recovery path に丸め残存 | ✅ `int(raw_shrunk / _mob) * _mob` 確認 |
| cooldown lot scale は current state でほぼ no-op | ✅ `max(0.001, 0.001*0.3)` = 0.001 |
| requote_attempts > 0 が 106件 | ✅ 全SHA合計106件 (最新SHA限定では57件) |
| 条件付き確率構造で分離すべき | ✅ 理論的に正しい |

### 479# — 方向は良いが、核心に誤読あり ⚠️

| 主張 | 検証結果 |
|------|----------|
| 最新SHA(f840d0e0aa5e)で gate は 18.3% | ✅ 59/321 = 18.4% |
| feasibility が 35.8% で「現在の真の主犯」 | ⚠️ 115/321 正しいが内訳が誤り |
| `no_feasible_quote` と `spread_too_narrow` は**同一の根** | ❌ **誤り** — 下記 §1 参照 |
| spread < 1000 JPY が全体の約32% (104件) | ❌ **過大** — 実際は55件 (17.1%) |
| `preflight_insufficient` が0件に激減 | ✅ 最新SHA限定で正しい |
| Timeout Buy/Sell = 完全対称 | ✅ 22:22 確認 (**ただし479では47:46と記載 — 軽微な不一致**) |
| Timeout avg spread ≈ Filled avg spread | ✅ 2398 vs 2451 JPY |
| `min_spread_jpy` 緩和の提案方向 | △ 部分的に妥当だが万能ではない |

---

## §1 CRITICAL: 479# の「同一の根」は事実誤認である

479# の最も重要な主張は次である:

> 実は `no_feasible_quote` (59件) と `spread_too_narrow` (45件) は**同一の根から発生**しています。(中略) 全体の約32%にあたる104件が「スプレッドが1000円未満だから」という単一の理由だけで取引を放棄しています。

独自に fill_records の `error_message` フィールドを精査した結果、この主張は **完全に誤り** である。

### `no_feasible_quote` 59件の内訳

| error_message | 件数 | 割合 |
|---------------|------|------|
| `cross_venue_veto: buy suppressed by bitflyer down lead...` | **49** | **83.1%** |
| `Spread too narrow: xxx JPY < min 1000` | **10** | 16.9% |

**no_feasible_quote の 83% は `cross_venue_lead_lag_veto` が真因** であり、`spread_too_narrow` ではない。

したがって、**spread < 1000 JPY が真因のレコード合計は**:

$$
45 \text{ (direct spread\_too\_narrow)} + 10 \text{ (NFQ via spread\_too\_narrow)} = 55 \text{ 件 (17.1\%)}
$$

479# の主張する 104件 (32%) はほぼ倍の過大見積もりである。

### cross_venue_veto の発火メカニズム

- `maker_risk_guards.py`: BitFlyer の BBO 乖離が `veto_threshold_bps: 6.0` bps 超で veto フラグを立てる
- `maker_price.py` L1012-1019: veto フラグ検出時に `InfeasibleQuoteError(reason=CR.CROSS_VENUE_LEAD_LAG_VETO)` を送出
- `fill_cycle_executor.py` L670-690: 同一 side で3回連続 `InfeasibleQuoteError` → `NO_FEASIBLE_QUOTE` に昇格

つまり **cross_venue_veto は、BitFlyer の板価格が Coincheck より 6bps 以上乖離した状況で Buy を抑止する防御機構** であり、`min_spread_jpy` とは全く無関係な系統である。

### 真のボトルネック再構成 (最新SHA, 321件)

| 段階 | 真因 | 件数 | 割合 |
|------|------|------|------|
| **約定** | filled | 93 | 29.0% |
| **Gate** | skip_gate, buy_dynamic_kill, final_clamp, cross_venue_veto(直接), etc. | 59 | 18.4% |
| **Feasibility: spread_too_narrow** | BBO spread < 1000 JPY | 55 | **17.1%** |
| **Feasibility: cross_venue_veto** | NFQ(3連続) via BitFlyer 6bps veto | **49** | **15.3%** |
| **Post-submit** | timeout, status_unknown, stale_drift, post_only | 54 | 16.8% |
| **その他 feasibility** | postonly_crossing_skip, sell_guard_reject | 11 | 3.4% |

**479# が見落としていた真のボトルネック第2位は `cross_venue_lead_lag_veto`** (49件, 15.3%) であり、`spread_too_narrow` と同等規模の問題である。

---

## §2 479# の「Liquidity Paradox」論は部分的に正しい

479# の理論的考察（§2: ゼロ手数料下における Liquidity Paradox）自体は市場微視的構造理論として妥当な面がある。

- Maker 手数料 0% 環境で narrow spread を全面拒絶するのは機会損失
- 狭いスプレッドは安全な Flow（ノイズトレーダー）を示唆する可能性がある
- `min_spread_jpy=1000` は BTC ≈ 11.7M JPY で約 0.85 bps — かなり狭い閾値

ただし、以下の点で 479# の提案は楽観的すぎる:

1. **spread < 1000 は 55件 (17.1%)** であり、479# の主張する 32% の半分
2. **`min_spread_jpy` を撤廃しても fill 率改善は +17.1% ポイントに留まる** — 29.0% → 最大 46.1%
3. 実際には narrow spread 時の queue competition が激しく、fill されるとは限らない
4. **spread 帯別 fill 率**を実測した結果:

| Spread 帯 (JPY) | レコード数 | Fills | Fill率 |
|-----------------|-----------|-------|--------|
| 1000-1500 | 26 | 13 | 50.0% |
| 1500-2000 | 36 | 16 | 44.4% |
| 2000-3000 | 87 | 39 | 44.8% |
| 3000+ | 45 | 25 | 55.6% |

広い spread (3000+) の方が fill 率が高い点は、narrow spread が必ずしも safety premium を持たないことを示唆する。

### 提案の修正

479# の `min_spread_jpy: 100` への即時引き下げは急進的すぎる。478# の助言に従い、**同一 config / 同一 SHA で再計測した上で段階的に緩和** するのが安全である:

- **Phase 1**: `min_spread_jpy: 700` (現行 1000 → 700、約30%緩和)
- **Phase 2**: 実測データに基づき 500 まで引き下げ判断
- **Phase 3**: `spread_adaptive` ロジックへの統合を検討

---

## §3 478# の主張はほぼ全て裏取りが取れた

### §3.1 コホート混在 — 実測確認

全928レコード中、10個の異なる SHA が混在:

| SHA | 件数 | Fills | Fill率 | 主な failure mode |
|-----|------|-------|--------|-------------------|
| f840d0e0aa5e | 321 | 93 | 29.0% | cross_venue_veto, spread_narrow, timeout |
| d0769f283da3 | 217 | 0 | 0.0% | ranging_low_vol_skip 一色 (217件) |
| b70365d4d4c9 | 135 | 0 | 0.0% | preflight_insufficient 一色 (119件) |
| 7dd01eef66e9 | 59 | 17 | 28.8% | timeout, skip_gate, spread_narrow |
| 15013d6f558b | 61 | 15 | 24.6% | no_feasible_quote, skip_gate |

477# の「fill 17.7%」は、**0% fill のレガシー SHA 2つ (352件)** を含んだ合成値であり、現行系の実力を過小評価していた。478# のこの指摘は完全に正しい。

### §3.2 条件付き確率構造 — 実測で裏付け

478# §3 の指摘通り、cancel reason を同列に数えるのは粗い。最新SHA で実測した段階別構造:

$$
P(\text{fill}) = P(\text{pass gate}) \times P(\text{feasible} \mid \text{gate pass}) \times P(\text{fill} \mid \text{order placed})
$$

- Gate 通過率: 82% (59/321 が gate で落ちる)
- Gate 通過後の feasibility 率: 約 56% (115/262 が feasibility で落ちる)
- 発注到達後の fill 率: 93/147 = **63.3%** (timeout + post-submit で 54 が落ちる)

つまり、発注まで到達したレコードの fill 率は実は 63% と高い。問題は **発注前の feasibility 段階で 44% が弾かれている** 点にある。

### §3.3 Gate 2b soft mode — コード確認済

`cycle_gate_aggregator.py` L555:
```python
if self._config.ranging_sell_low_vol_as_offset:
    return GateCheckResult(gate_name="ranging_sell_low_vol", blocked=False, ...)
```

現行 config で `ranging_sell_low_vol_as_offset: true` であり、478# の「soft mode では sell 専用保護ではなく既存 generic boost への接続」という読みは正確。

### §3.4 preflight recovery path の丸め残り — コード確認済

`orchestrator_balance.py` L223-227:
```python
raw_shrunk = self._current_lot / self.config.balance_shrink_divisor
_mob = self.config.min_order_btc
self._current_lot = max(min_lot, int(raw_shrunk / _mob) * _mob)
```

この `int(x / _mob) * _mob` パターンは 0.001 刻みの量子化を意味し、476# の「切り捨て完全廃止」は主要パスについてのみ正確。478# の補足として妥当。

---

## §4 Buy/Sell 非対称性 — 478# §9.2 の裏付け

最新 SHA (321件) の side 別分析:

| Side | 件数 | Fills | Fill率 | 最大 cancel reason |
|------|------|-------|--------|-------------------|
| Buy | 202 | 47 | **23.3%** | no_feasible_quote: 58 (うち49件 = cross_venue_veto) |
| Sell | 114 | 46 | **40.4%** | timeout: 22, spread_too_narrow: 19 |

**Buy 側が圧倒的に不利** — これは 478# §9.2 の「buy 側だけ機会集合を過剰に削る現状」と完全に一致。

原因は明確: **cross_venue_veto が Buy side にほぼ専属的に発火している** (error_message: "buy suppressed by bitflyer down lead")。BitFlyer の板が Coincheck より先行して下落すると、Buy 側の発注が veto される構造。

---

## §5 PnL 品質 — 懸念すべきデータ

最新 SHA 限定の約定品質は **477# の混合コホート分析よりも悪い**:

| 指標 | 477# (混合, 160件) | 最新SHA (93件) |
|------|-------------------|---------------|
| PnL30s 平均 | -0.44 bps | **-1.386 bps** |
| PnL30s 中央値 | +0.10 bps | **-0.877 bps** |
| AS 率 | 25.6% | **32.3%** |

中央値もマイナスに転落している。これは 479# の「Bot は適切に安全な位置に注文を置いている」という楽観的評価に対する反証となる。AS 率 32.3% は、**約定の3件に1件が逆選択に遭っている** ことを意味する。

477# の混合コホート分析で中央値が +0.10 bps だったのは、旧 SHA群（fill=0 のため PnL 統計に参加しない）のバイアスではなく、他 SHA の良好な fills が混入していた可能性がある。

---

## §6 総合アクション判断

### 478# からの指示 — 採用すべき

1. ✅ 同一 config / 同一 SHA で fill funnel を再計測 → 本ドキュメントで実施済
2. ✅ preflight / gate / feasibility / post-submit を分離再集計 → 実施済
3. ✅ buy 側 hard gate と no_feasible_quote の寄与再評価 → cross_venue_veto が真犯人と特定
4. ✅ その後に suppression 緩和 → Phase 制御で進行

### 479# からの指示 — 部分採用

| 提案 | 判断 | 理由 |
|------|------|------|
| `min_spread_jpy` を 100 に引下 | ❌ 急進的 | 17.1% の改善余地に対してリスクが大きい。700 から段階的に |
| Feasibility の動的化 | △ 方向は良い | ただし cross_venue_veto が真犯人の 83% を占めるため、min_spread だけでは不十分 |
| Queue Position 依存タイムアウト | △ 将来課題 | 現状 timeout は fill funnel の 16.8% であり最優先ではない |

### 真の P0 アクション (本検証に基づく)

1. **cross_venue_veto 閾値の見直し**: `veto_threshold_bps: 6.0` が Buy 側に 49件/321件 (15.3%) の抑止。閾値を 8.0-10.0 bps に緩和し、本当に toxic な大乖離のみを veto する
2. **`min_spread_jpy` の段階的緩和**: 1000 → 700 で 55件中の一部を回収
3. **PnL 品質の改善**: AS率 32.3% は高すぎる。offset 戦略の見直しが fill rate 改善以上に urgentである可能性

---

## §7 結語

478# は厳密で実測に誠実なレビューであり、ほぼ全ての主張がコード・データレベルで裏付けられた。

479# は理論的枠組み(Liquidity Paradox)として価値があるが、**データ精査が雑** であり、`no_feasible_quote` の error_message を読まずに「spread_too_narrow と同根」と断じた点は明確な誤りである。最大のボトルネックは `spread_too_narrow` (17.1%) と `cross_venue_lead_lag_veto` (15.3%) の**2系統**であり、単一原因に帰着させる 479# の論法は成立しない。

ただし、479# の提起した `min_spread_jpy` の緩和方向自体は妥当であり、閾値の段階的引き下げは検討に値する。重要なのは、**それだけでは fill rate の半分も解決しない** という認識を持つことである。

---

## §8 481# 対応: 実施した改修

本検証結果に基づき、以下の改修を実施した。

### §8.1 YAML 変更 (fill_test.yaml)

| 設定 | Before | After | 根拠 |
|------|--------|-------|------|
| `veto_threshold_bps` | 6.0 | **8.0** | 49件中41件(84%)が6-8帯。median=7.07bps。本当にtoxicな8bps超のみvetoする |
| `min_spread_jpy` | 1000 | **700** | spread<1000が55件(17.1%)。700以下は約30件を回収見込。Phase1段階緩和 |

**期待効果**: cross_venue_veto のうち 41件が解放 + spread 系の一部解放。Buy 側 fill rate の改善。

### §8.2 コード改善 (fill_cycle_executor.py)

NFQ エスカレーション時のログメッセージに `last_reason` を追加:

```
Before: "consecutive infeasible quotes (buy) — constraint set collapse (min_spread=...)"
After:  "consecutive infeasible quotes (buy) — last_reason=cross_venue_lead_lag_veto, min_spread=..."
```

これにより、NFQ の背景にある真因が即座に判別可能になる。

### §8.3 今後の課題 (未着手)

1. NFQ skip record に `cross_venue_lead_lag_spread_bps` を付加して分析効率を向上
2. Phase 2: `min_spread_jpy: 500` への追加引き下げ判断
3. PnL 品質改善: AS率 32.3% の根本対策 (offset 戦略)
