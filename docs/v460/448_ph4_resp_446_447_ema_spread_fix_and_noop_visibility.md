# 448# 446/447# レビュー応答: EMA spread 混線修正 + No-Op 可視化

> **種別**: resp
> **対象**: 446# / 447#
> **日付**: 2026-03-16

---

## §0 レビュー指摘の検証結果サマリ

446# (Codex) と 447# (Gemini) の指摘を個別に検証した結果、全 Finding が正しいことを確認した。

| Finding | 検証結果 | 本セッションでの対応 |
|---------|---------|---------------------|
| F1: mixed-SHA A/B データ | **確認**: 168 rows / 5 SHA 混在。`f34467b5c2f8`（79 rows）は cv 未搭載で最多 | P1 — 別チケット（A/B 基盤改修） |
| F2: ceiling no-op | **確認**: buy ceiling=0.20/0.30, boost×1.25 → cap で `0.30→0.30` | **P0 修正済**: cap_hit 可視化 |
| F3: EMA/point spread 混線 | **確認**: L316 で `spread_bps=point_spread` を返しつつ direction は EMA | **P0 修正済**: spread 統一 |
| F4: buy+ranging 過剰 skip | **確認**: ranging_low_vol_skip 57/168 件。fill rate 31.4%→10.5% | P1 — 別チケット |
| F5: heuristic toxicity 再校正 | **確認**: ML 棄却は妥当だが heuristic veto は live で 367 set / 53 block | P1 — 別チケット |
| F6: deadlock/dynamic_kill 支配的 | **確認**: route_to_kill_deadlock 24 件, sell_dynamic_kill 15 件 | P1 — 別チケット |

---

## §1 P0 修正: F3 EMA spread 混線の解消

### §1.1 問題

`compute_cross_venue_lead_lag_hint()` の confidence（EMA）モードにおいて:

- **direction**: `ema_spread_bps` の符号から決定 ✓
- **spread_bps**: 生の point spread をそのまま返却 ✗
- **veto 判定**: `abs(hint.spread_bps)` = point spread で判定 ✗

例: `ema_spread_bps = -2.26bps`（down）だが `point_spread_bps = +0.15bps` → direction=down なのに spread=+0.15 という矛盾。veto threshold 6.0bps に対して point spread では到達しないため、veto 発火数 = 0。

### §1.2 修正内容

**方針**: EMA モードでは `hint.spread_bps = ema_spread_bps` に統一。生の point spread は新フィールド `point_spread_bps` として診断用に保持。

| ファイル | 変更 |
|---------|------|
| `cross_venue_lead_lag.py` L68-82 | `CrossVenueLeadLagHint` に `point_spread_bps: float \| None` 追加 |
| `cross_venue_lead_lag.py` L316-330 | EMA モード: `spread_bps=ema_spread_bps`, `point_spread_bps=spread_bps` |
| `cross_venue_lead_lag.py` L87-98 | `build_cross_venue_event_details()` に `point_spread_bps` 追加 |
| `cross_venue_lead_lag.py` L103-155 | `build_cross_venue_fill_fields()` に `point_spread_bps` 追加 |
| `maker_risk_guards.py` L240-249 | veto ログに `pt_spread` 情報追加 |
| `maker_risk_guards.py` L293-320 | boost ログに `pt_spread` + `CAP_HIT=NO-OP` 表示 |

### §1.3 効果

修正後、veto 判定は `abs(ema_spread_bps) >= 6.0` で評価される。EMA spread が安定的に大きい乖離を示す場合にのみ veto が発火するため、point spread のノイズに惑わされない。direction / veto / boost が全て同じ情報源（EMA）に基づくようになり、意味の一貫性が担保される。

---

## §2 P0 修正: F2 No-Op (Cap Hit) 可視化

### §2.1 問題

`_scale_offset_ratio()` が `max_ratio` でクランプするため、offset が既に上限に達している場合は boost が掛かっても `0.30 → 0.30` の no-op になる。この事実がログにも FillRecord にも記録されず、「applied=true だが実効果ゼロ」を発見できなかった。

### §2.2 修正内容

| ファイル | 変更 |
|---------|------|
| `maker_risk_guards.py` L51-55 | 型スタブに `_cross_venue_lead_lag_pre_offset`, `_post_offset`, `_cap_hit` 追加 |
| `maker_risk_guards.py` L227-231 | guard 関数でこれらを初期化 |
| `maker_risk_guards.py` L293-299 | boost 後に `cap_hit = (mult≈1.0 && actual_boost>1.0)` で検出 |
| `maker_risk_guards.py` L305-322 | ログに `CAP_HIT=NO-OP` を付加 |
| `fill_record_builder.py` L201-213 | FillRecord に `cross_venue_lead_lag_pre_offset`, `post_offset`, `cap_hit` フィールド追加 |

### §2.3 効果

- `cap_hit=true` で grep すれば no-op 件数が即座にわかる
- A/B 分析時に `applied=true && cap_hit=true` を除外することで、真に効力があったケースのみ評価可能
- Ceiling 値の再検討の判断材料になる

---

## §3 F1 に対する見解: mixed-SHA A/B のクリーンルーム化

446# P0 指摘の「`ab_offset_comparison.py` に `--git-sha` filter を追加」は重要だが、本セッションのスコープ外とする。別チケットで対応予定。

### 検証結果

```
Total rows: 168
SHA count: 5
f34467b5c2f8: 79 rows, cv_fields=0  ← Cross-Venue 未搭載なのに最多
a9714ad9af85: 50 rows, cv_fields=6
e23a063923ee: 23 rows, cv_fields=3
1d64e64db506: 10 rows, cv_fields=0
c38c15ec943c:  6 rows, cv_fields=1
```

`+1.391bps` の改善値は、cv 未搭載の `f34467` の好成績に引っ張られている可能性が高い。

---

## §4 その他の Finding に対する方針

### F4: buy+ranging 過剰 skip
ranging_low_vol_skip が 57/168 件は確かに強い。438# のロジック自体は効いている可能性があるが、参加率低下（31.4%→10.5%）が在庫バランスや trade starvation を生むリスクがある。controlled lane（microprice/imbalance 条件付き通過）の検討は P1。

### F5: heuristic toxicity 再校正
ML toxicity は凍結継続だが、既存の heuristic toxicity（`toxic_veto_set: 367`, `toxic_veto_block: 53`）は active。これと regime-side asymmetry の掛け合わせによる再校正は P1。

### F6: deadlock / dynamic_kill
`route_to_kill_deadlock` 24 件、`sell_dynamic_kill` 15 件は cross-venue 以前の既存問題。alpha 改善の真価評価にはこれらの安定化が前提となる。P1。

---

## §5 447# 新エッジ提案の評価

| 提案 | 評価 | 備考 |
|------|------|------|
| A: Asymmetric Inventory Sponging | ○ 面白い | Offset ceiling 問題を迂回する発想。ただし在庫リスク増大の安全弁が必要 |
| B: Micro-Timeout (TIF) | ◎ 有望 | buy+ranging に最も直接的に効く。15秒放置キャンセルは実装コストも低い。F4 の代替策としても機能 |
| C: Global Spread Shadowing | ○ 面白い | BitFlyer スプレッド急拡大 → Coincheck 退避は合理的。ただし BF spread データの品質依存 |

P2 として提案 B の検討を優先する。

---

## §6 テスト結果

- `test_439_cross_venue_lead_lag.py`: **29 passed**
- `test_336_fill_config_parser.py`: **30 passed**
- 全 cross_venue 関連テスト: **33 passed**

---

## §7 変更ファイル一覧

| ファイル | 変更種別 |
|---------|---------|
| `scripts/v460/lib/cross_venue_lead_lag.py` | F3 修正: spread 統一 + point_spread_bps |
| `scripts/v460/lib/maker_risk_guards.py` | F3/F2 修正: veto/boost ログ + cap_hit |
| `scripts/v460/lib/fill_record_builder.py` | F2 修正: cap_hit FillRecord 出力 |
