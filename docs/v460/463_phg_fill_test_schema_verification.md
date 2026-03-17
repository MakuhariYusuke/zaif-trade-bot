# 463# Fill Test 10日間分析 — Schema-Aware 検証

> **種別**: rpt / fix
> **フェーズ**: phg (フェーズ横断)
> **日付**: 2026-03-18
> **前提**: 461# fill test 深堀り分析 + 462# レビュー指摘

---

## 概要

461# (fill test 10日間深堀り分析) に対する 462# (レビュー) の指摘を実データで全件検証し、461# の修正と分析基盤の改善を実施。

### 成果物

| 成果物 | 内容 |
|---|---|
| `temp/analyze_fill_test_v3.py` | schema-aware 分析スクリプト (7パート) |
| `temp/fill_v3_output.txt` | v3 全出力 |
| 461# §10 追記 | 修正箇所一覧 + 検証結果サマリー |
| 462# §10 追記 | 検証結果 + 新発見 |
| 本ドキュメント (463#) | 検証の完全記録 |

---

## 1. 検証動機

462# は 461# に対し 5 つの drift (schema / run / config / population / market) を指摘。461# の §9 (Reviewer追記) でも同様の懸念を表明。しかしいずれも「可能性」の指摘であり、実データでの検証は未実施だった。

---

## 2. Schema Presence Audit — フィールド不在の確定

### 2.1 結果

| 日付 | `execution_pre_clamp_offset` | `cross_venue_lead_lag_applied` | `resolved_side_reason` | `start_git_sha` |
|---|---|---|---|---|
| 3/8 | ❌ | ❌ | ❌ | ❌ |
| 3/9 | ❌ | ❌ | ❌ | ❌ |
| 3/10 | ❌ | ❌ | ❌ | ❌ |
| 3/11 | ❌ | ❌ | ❌ | ❌ |
| 3/12 | ❌ | ❌ | ❌ | ❌ |
| 3/13 | ❌ | ❌ | ❌ | ❌ |
| 3/14 | ❌ | ❌ | ❌ | ❌ |
| 3/15 | ✅ | ❌ | ✅ | ✅ |
| 3/16 | ✅ | ✅ | ✅ | ✅ |
| 3/17 | ✅ | ✅ | ✅ | ✅ |

### 2.2 含意

- **3/8-3/14 (7日間)**: 4つの late-added フィールドが全て不在
- **3/15**: `pre_clamp`, `resolved`, `start_sha` が出現 (cross_venue はまだ不在)
- **3/16+**: 全フィールド揃う

これにより:
- 461# の「旧 SHA ceiling clamp 0%」は**計測不能 (N/A)** に修正
- 461# の「旧 SHA balance_switch 0%」は `resolved_side_reason` 不在による擬似ゼロ。ただし `balance_forced_switch` フィールドは全 date に存在

### 2.3 常在フィールド

以下のフィールドは全期間で利用可能:
- `run_id` — 全 date に存在
- `balance_forced_switch` — 全 date に存在 (True / None)
- `side`, `effective_offset_used`, `pnl_30s`, `cancel_reason` 等

---

## 3. Balance Switch 二重計測問題

### 3.1 発見

| SHA | `balance_forced_switch=True` | `resolved_side_reason="balance_switch"` | 使用フィールド |
|---|---|---|---|
| eb24cf4 (3/8) | **21/89** (23.6%) | フィールド不在 | `balance_forced_switch` のみ |
| 0d22298 (3/9) | 0/62 (0%) | フィールド不在 | 同上 |
| 5c3238f (3/13) | 0/187 (0%) | フィールド不在 | 同上 |
| f840d0e (3/17) | **0/88** (0%, 全件 None) | **51/88** (58%) | `resolved_side_reason` のみ |
| d0769f2 (3/16-17) | 0/12 (0%) | 7/12 (58%) | `resolved_side_reason` のみ |

### 3.2 解釈

- `balance_forced_switch` と `resolved_side_reason` は**異なる計測系**
- 旧コード: `balance_forced_switch=True` で在庫スイッチを記録
- 新コード: `resolved_side_reason="balance_switch"` で記録するようになり、`balance_forced_switch` は None のまま
- **両者の cross-schema 比較は直接的にはできない**

### 3.3 eb24cf4 の balance switch 効果

| 区分 | n | PnL 30s | AS% |
|---|---|---|---|
| balance_forced_switch=True | 21 | **+0.10** | 23.8% |
| balance_forced_switch=None | 68 | -0.06 | 22.1% |

eb24cf4 では balance switch fills のほうが PnL が良い。「balance switch = 悪」は f840d0e 限定の観察であり、メカニズム自体の否定には使えない。

### 3.4 461# への影響

- §5.4.4: eb24cf4 「Balance Switch: 0%」→「23.6% (21/89)」に修正済
- §6.4: 「Balance Switch の逆機能」は f840d0e 固有 + 計測方法差異の注記を追加済

---

## 4. Ceiling Clamp 再計測

### 4.1 旧 SHA (3/8-3/14)

`execution_pre_clamp_offset` フィールド不在のため **計測不能 (N/A)**。

461# の「0%」は v2 スクリプトが `pre_clamp is not None` フィルタで 0 件を抽出し、分母 (全 side fills) で割った結果。

### 4.2 新 SHA (3/15+)

| SHA | Buy Ceiling | Sell Ceiling | 備考 |
|---|---|---|---|
| d0769f2 | 5/5 (100%) | 3/3 (100%) | 初版 461#: 83.3%/50.0% (分母誤り) |
| f840d0e | 41/41 (100%) | 17/17 (100%) | 初版 461#: 93.2%/38.6% (分母誤り) |

v3 では pre_clamp フィールドが存在する fills のみを分母とした。結果、**pre_clamp が記録された fills は全件 clamped**。

### 4.3 v2 との差異の原因

v2 スクリプトは分母に全 side fills を使用。f840d0e buy 44 fills 中 41 に pre_clamp がある → 41/44 = 93.2%。v3 は 41/41 = 100%。残り 3 fills は pre_clamp=None (理由は不明、同日なのにフィールドが欠落)。

解釈: 「ceiling 93.2%」は「全件の 93.2% が clamped」、「ceiling 100%」は「clamp 観測可能な fills の 100% が clamped」。**後者のほうが ceiling の深刻度を正確に表現する**。

---

## 5. EV Score 0.5-1.0 帯の検証

### 5.1 全母集団分析結果

| SHA | EV 0.5-1.0 (filled+cancel) | 全 EV 既知件数 | 比率 |
|---|---|---|---|
| 5c3238f | 22 | 262 | 8.4% |
| bff652e | 25 | 186 | 13.4% |
| 92c588e | 19 | 187 | 10.2% |
| 819ec73 | 12 | 171 | 7.0% |
| eb24cf4 | 13 | 155 | 8.4% |
| 0d22298 | 9 | 109 | 8.3% |
| **d0769f2** | **0** | **25** | **0.0%** |
| **f840d0e** | **0** | **166** | **0.0%** |

### 5.2 結論

- f840d0e: filled=0, cancel=0, total=0 — 462# が懸念した「filled-only のアーティファクト」ではなく**構造的空白**
- d0769f2 でも 0% — 458# 以降で EV 計算に不連続が発生したことの強い証拠
- 旧 SHA は 7-13% で安定
- 462# §5/§7.3 の「疑い止まり」は全母集団検証で**構造的空白として確定**

---

## 6. Cross-Venue Selection Bias 検証

### 6.1 Side 分離分析 (f840d0e)

| Side | CV 区分 | n | PnL 30s | AS% |
|---|---|---|---|---|
| Buy | cv_on | 29 | **+0.25** | 24.1% |
| Buy | cv_off | 15 | -1.43 | 20.0% |
| Sell | cv_on | 8 | -2.60 | 25.0% |
| Sell | cv_off | 36 | -2.05 | 44.4% |

### 6.2 解釈

- **Buy 側**: CV 効果は明確 (+0.25 vs -1.43)。ただし sample size の差 (29 vs 15) に注意
- **Sell 側**: CV 適用時のほうが PnL 悪化 (-2.60 vs -2.05)。AS は CV 適用時のほうが良い (25% vs 44.4%) が PnL は悪い
- **結論**: 462# §7.2 の selection bias 懸念は sell 側で裏付け。CV 適用率引上げ (P1-3) は **buy 限定で再評価すべき**

---

## 7. sell_dynamic_kill Timeline

### 7.1 SHA 別推移

```
3/08: eb24cf4=27%  fea7911=37%  d4db827=56%  e5d4937=74%  bb59fb1=39%
3/09: 819ec73=25%  0d22298=51%  22a4fc5=6%   06f0ba2=100%
3/10: 27d6acd=43%  22a4fc5=0%
3/11: b2a902c=13%
3/12: 92c588e=6%   66165ee=0%   bff652e=0%
3/13: bff652e=73%  5c3238f=41%  c6ded4a=80%
3/14: 5c3238f=80%  c8a5488=34%  1a84c04=0%   f11e97a=0%   e632954=0%
3/15: 9d8cf7b=6%   f34467b=0%   e934ac3=0%   1a84c04=0%
3/16: c7ebd8c=4%   52627ff=9%   d0769f2=0%   f34467b=0%   a9714ad=14%
3/17: f840d0e=0%   d0769f2=0%
```

### 7.2 観察

- **同一 SHA でも日によって大幅変動**: bff652e は 3/12 に 0% だが 3/13 に 73%。5c3238f は 3/13 に 41% だが 3/14 に 80%
- **同一日でも SHA 間で大きな差**: 3/14 に 5c3238f=80%, 1a84c04=0%
- **漸進的減少**: 3/15 以降は 0-14% に低下し、3/17 で完全消滅
- **461# の「458#/459# が kill を変質させた」は過度に強い因果断定**。sell_dynamic_kill は市場条件にも強く依存しており、コード変更のみに帰すことはできない

---

## 8. Run-Based Analysis

### 8.1 全 28 Runs

v3 PART2 で 28 の run_id を確認。top 5 runs by fill count:

| run_id | n | fills | fill% | SHAs | 期間 |
|---|---|---|---|---|---|
| 1773388585 | 600 | 196 | 32.7% | 5c3238f, c6ded4a | 3/13 07:56 - 3/14 07:54 |
| 1772917391 | 440 | 127 | 28.9% | eb24cf4, fea7911 | 3/8 00:00 - 3/8 15:00 |
| 1773339543 | 379 | 104 | 27.4% | bff652e | 3/12 18:19 - 3/13 07:54 |
| 1773046798 | 252 | 96 | 38.1% | 819ec73 | 3/9 09:00 - 3/9 18:09 |
| 1773732128 | 307 | 88 | 28.7% | f840d0e | 3/17 07:22 - 3/17 18:20 |

### 8.2 Hot-reload の確認

- run `1772917391` (3/8) は eb24cf4 と fea7911 の 2 SHA を含む → hot-reload 発生
- run `1773388585` (3/13-14) は 5c3238f と c6ded4a の 2 SHA → 同上
- **git_sha[:7] だけでは run 境界を正しく切れない** という 462# §3 の指摘を裏付け

---

## 9. 改善提案の再評価

462# 検証結果を踏まえ、461# の P0-P2 提案を再評価する。

| 提案 | 初版評価 | 検証後評価 | 理由 |
|---|---|---|---|
| P0-1: Buy ceiling 0.20→0.35 | 即時 | **⚠ 保留** | 旧SHA ceiling が N/A (計測不能) なので「ceiling なし=好成績」の根拠が不十分。ただし f840d0e ceiling 100% は確かに過剰 |
| P0-2: Deep-night 停止 | 即時 | **✅ 維持** | AS 57-100%, PnL -3〜-13bps は schema 問題と無関係な事実。実行すべき |
| P0-3: status_unknown 調査 | 即時 | **✅ 維持** | 同上 |
| P1-1: ranging_low_vol_skip 緩和 | 短期 | **✅ 維持** | 主収益レジームの遮断は戦略の自己否定 |
| P1-2: balance_switch 厳格化 | 短期 | **⚠ 要再検討** | eb24cf4 では正の効果。f840d0e での悪化は ceiling との複合要因の可能性。計測方法差異も要考慮 |
| P1-3: CV 適用率引上げ | 短期 | **⚠ Buy 限定に修正** | Sell 側は CV 適用で PnL 悪化。Buy 側のみ有効 |
| P1-4: EV 0.5-1.0 空白修正 | 短期 | **✅ 維持 (強化)** | 全母集団で構造的空白が確定。458# 以降の EV 計算変更が原因の可能性大 |
| P2-1: Ranging ceiling-free | 中期 | **⚠ 要再検討** | 旧SHA ceiling N/A のため「ceiling-free で好成績」の証拠が弱い |
| P2-2: sell_dynamic_kill 再設計 | 中期 | **⚠ 要再検討** | 消滅はコード変更のみに帰せない。市場条件依存が大きい |

---

## 10. 残タスク

| 優先 | タスク | 備考 |
|---|---|---|
| P0 | FillRecord に `config_hash` 追加 | 462# §4 config drift 対応。現在の FillRecord では config 変更を追跡できない |
| P0 | `balance_forced_switch` と `resolved_side_reason` の整合性確保 | 新コードで `balance_forced_switch` が None のまま — 旧フィールドも正しく記録すべき |
| P0 | EV 0.5-1.0 帯消失の原因調査 | 458# 以降の EV 計算変更を特定 |
| P1 | hour 固定の matched comparison 追加 | v3 では side/regime 固定まで。時間帯固定が未実施 |
| P1 | pre_clamp 不在 fills (f840d0e 3/44 buy, 27/44 sell) の原因調査 | 同日なのにフィールドが欠落する理由 |
