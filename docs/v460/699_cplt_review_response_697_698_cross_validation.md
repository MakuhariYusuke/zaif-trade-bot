# 699# レビュー回答: 697#/698# クロスバリデーションと盲点分析

## 概要

697# (PHG多面的検証) および 698# (学術的セカンドオピニオン) の指摘事項に対し、コード・ライブデータ双方で再検証を行った結果と、**両レビューが見落としている盲点**を報告する。

---

## 1. 697# 指摘事項の検証結果

### 1-1. 4/2 全日 PnL の実態 ✅ 確認済

697# の中核指摘「696# の +0.367 bps は部分日データに過ぎない」は**完全に正しい**。

| 集計範囲 | orders | fills | fill% | avg_pnl30 | sum_pnl30 |
|----------|--------|-------|-------|-----------|-----------|
| 696# 報告 (UTC 0-10, 部分日) | ~204 | ~98 | 48% | **+0.367 bps** | +35.9 bps |
| 全日 (UTC 0-14:48) | 248 | 123 | 49.6% | **-0.09 bps** | -11.1 bps |

**結論**: 後半 SHA (b56771a38b, 20:10-23:30 JST) がすべてを帳消しにした。

#### SHA 別内訳

| SHA | 期間 (JST) | fills | avg_pnl30 | sum_pnl30 | WR |
|-----|-----------|-------|-----------|-----------|-----|
| b5f7828b16 | 〜early | 13 | **+0.929 bps** | +12.1 | 54% |
| 04390da322 | mid-day | 90 | **+0.182 bps** | +16.4 | 44% |
| b56771a38b | 20:10-23:30 | 25 | **-1.911 bps** | -47.8 | 36% |

**697# 指摘 → 受理**: +0.367 bps 報告は Simpson's Paradox の典型例。部分日・全日の明記が必須。

### 1-2. spread_as_guard 単位バグ ✅ 確認済

```yaml
# fill_test.yaml:1289-1300
spread_as_guard:
  enabled: false
  threshold: 1500.0  # ← 1500 bps = 15% は常時発火する閾値
```

```python
# entry_gate_adjustments.py:37-50
if spread_bps < cfg.threshold:  # 1500 bps → 常に True
    ev_penalty = cfg.ev_penalty
```

**現状**: `enabled: false` のため実害なし。修正時に適切な閾値 (10-20 bps 圏) の設定が必要。

### 1-3. Protocol 688 NFQ バグ ✅ 確認済

```python
# protocol_688.py:349
"nfq": _cancel_payload(records)
#      ↑ _cancel_payload は `not filled` = 全キャンセルを返す
```

`_cancel_payload` のフィルタ条件 `not _record_bool(record, "filled")` はNFQ固有ではなく全キャンセルを包含する。NFQ分析に使うには `cancel_reason == "no_feasible_quote"` でのフィルタが必要。

### 1-4. AS フィールド定義の不整合 ⚠️ 部分的に受理

| フィールド | 定義 | 用途 |
|-----------|------|------|
| `adverse_selected` | deadzone適用後 (標準判定) | 通常分析・ガード判定 |
| `adverse_selected_raw` | deadzone未適用 (生値) | 694#分析ツール |

697# の「ミスマッチ」指摘は正当だが、**意図的な設計**でもある。deadzone-free の `_raw` は感度分析用で、ガード閾値最適化には有用。問題は**ドキュメント不足**であり、バグではない。

### 1-5. Primary-only テールロス = 100% ✅ 確認済

```
SELL tail: n=6, mean=-11.15bps, 100% AS, 100% primary_only
BUY tail:  100% primary_only
```

`ev_score_pretrade is None` パスがすべてのテールロスの発生源。EV モデルが評価を出さないケースでベースラインスプレッドのみで参入している。

### 1-6. インベントリドリフト ✅ 確認済

`maker_price.py:248-250`:
```python
self._fill_history = collections.deque(maxlen=100)  # 直近100フィルのみ
```

加えて時間減衰 τ=3600s (`_decayed_imbalance()`) が適用されるため、長時間の一方向ドリフトは tracking window を超えて失われる。

---

## 2. 698# 指摘事項の検証結果

### 2-1. Simpson's Paradox ✅ 受理

全日 -0.09 bps vs 部分日 +0.367 bps は、非定常過程の集計期間依存を実データで実証した。

### 2-2. VC 次元爆発 ⚠️ 一部留保

698# は「ガード次元数に対してサンプルが少ない」と警告するが、現行ガードの多くは `enabled: false` (観察モード)。**実際に同時稼働しているガード数**は限定的で、過学習リスクは理論的指摘ほど深刻ではない。ただし、新ガード追加時は 698# の警告を常に念頭に置く。

### 2-3. インベントリ Integral Windup ⚠️ 提案に留保あり

698# 提案: `window 100→1000, max_factor 0.4→0.8, neutral_band 0.05→0.10`

**留保事由**:
- 時間減衰 τ=3600s が既に長期ドリフトを部分的に補償している
- `max_factor 0.8` は片側スキューイングの攻撃性が高すぎる可能性（BBO から 80% オフセット）
- window 1000 はメモリ使用量が 10× になるが、τ=3600s と併用すると古い fill には既に減衰が効いており、効果は逓減的

**対案**: window 300-500 + max_factor 0.5-0.6 のグラデーション導入が安全。

### 2-4. as_trailing_gate アクティベーション ✅ 妥当

```yaml
# fill_test.yaml:489-496
as_trailing_gate:
  enabled: false     # ← 観察モード
  window: 100
  soft_threshold: 0.30
  hard_veto: 0.45
  boost: 1.3
```

パラメータは既に妥当な範囲 (soft=0.30, hard=0.45)。fill_record_builder に出力接続済み。`enabled: true` への切替は低リスクかつ即効性がある。

---

## 3. 両レビューの盲点 (本文書の独自貢献)

### 盲点 A: trending_down ラウンドトリップ損失の支配性

**どちらのレビューも指摘していない最大の損失源**:

| regime | RTs | sum_pnl | WR |
|--------|-----|---------|-----|
| ranging | 29 | **+22.80 bps** | 45% |
| trending_down | 26 | **-63.54 bps** | 38% |
| trending_up | 5 | **+22.57 bps** | 40% |

全日 RT 合計 -18.17 bps のうち、trending_down が **-63.54 bps** を占める。ranging と trending_up が黒字でも trending_down が全てを打ち消している。

697# はインベントリドリフトに焦点を当て、698# はインテグラルワインドアップを議論したが、**trending_down 時の sell 難 → buy ポジション滞留 → RT 損失拡大**という因果メカニズムを明示していない。これはインベントリスキューイングだけでは対処不可能で、**regime 遷移時の position exit 戦略**が必要。

### 盲点 B: 第3 SHA (b5f7828b16) の無視

697# は 2 SHA (04390da / b56771a) の対比で議論を構成しているが、**b5f7828b16 (13 fills, +0.929 bps)** の分析が欠如。この SHA は最も高い fill 収益性を示しており、何がこの SHA の成功要因だったかの調査は設定最適化の手がかりになる。

### 盲点 C: MCB 復帰後の品質劣化

ライブデータから判明:
- Pre-MCB fills: n=8, avg_pnl30 = **+2.47 bps**
- Post-MCB fills: n=115, avg_pnl30 = **-0.27 bps**
- MCB regime distribution: trending_up=11, trending_down=1

MCB 発動 (trending_up 局面) → 休止 → 復帰後の PnL が劣化するパターンがある。MCB 冷却期間の設定か、復帰後の段階的参入ロジックが未整備。

### 盲点 D: sell/trending_up の逆選択コスト

| side/regime | Capture | AS_Cost | Net |
|-------------|---------|---------|-----|
| sell/trending_up | -0.65 bps | **-2.21 bps** | **-2.86 bps** |
| sell/trending_down | -0.55 bps | +0.07 bps | -0.48 bps |

sell/trending_up は AS_Cost が -2.21 bps と極端に悪い。現行の sell_guard (`trend_5s_sell_guard_veto`) は ranging でも発火しており(10件)、**regime 条件分岐なし**の一律 veto が最適でない可能性。

### 盲点 E: NFQ 0 件の意味

全日で NFQ = 0/251 (0.0%)。これは一見好ましいが、NFQ がゼロということは **quote feasibility 閾値が緩すぎる** 可能性もある。すべての市場状況で quote を出し続けることが、逆選択コスト増大の遠因となりうる。特に trending_down 時に quote を控えるべき局面がないか要分析。

### 盲点 F: Protocol 688 NFQ バグの波及範囲

697# は protocol 688 のバグを指摘したが、**既存の protocol 688 分析レポートに基づく過去の意思決定がすべて汚染されている**点を議論していない。これまで「NFQ 分析」として参照してきたデータが実は「全キャンセル分析」だったことの影響範囲を洗い出す必要がある。

---

## 4. 総合評価と行動方針

### 受理事項

| # | 指摘 | 出典 | 優先度 | 対応 |
|---|------|------|--------|------|
| 1 | 部分日PnL報告の是正 | 697# | 運用 | 分析ツールに全日/部分日の明示を義務化 |
| 2 | Protocol 688 NFQ フィルタ修正 | 697# | **P0** | `cancel_reason == "no_feasible_quote"` フィルタ追加 |
| 3 | spread_as_guard 閾値修正 | 697# | **P0** | 有効化時に 10-20 bps 圏に設定 |
| 4 | as_trailing_gate 有効化 | 698# | **P0** | enabled: true + 観察期間1日 |
| 5 | インベントリスキューイング改善 | 698# | **P1** | window 300-500, max_factor 0.5-0.6 |
| 6 | AS フィールドドキュメント | 697# | P2 | raw vs standard の使い分けガイド追加 |

### 留保事項

| # | 提案 | 出典 | 留保理由 |
|---|------|------|----------|
| 1 | window 100→1000 | 698# | τ=3600s との併用で効果逓減; 300-500 が妥当 |
| 2 | max_factor 0.4→0.8 | 698# | 0.8 は BBO 80% オフセットで攻撃的すぎ; 0.5-0.6 推奨 |
| 3 | neutral_band 0.05→0.10 | 698# | 緩和は在庫中立復帰を遅らせるリスク |
| 4 | sell_hour_boost 維持 | 698# | 値は妥当だがレジーム条件分岐なしの一律ブーストに疑問 |

### 独自発見の対応方針

| # | 盲点 | 対応 |
|---|------|------|
| A | trending_down RT -63.54 bps | **P0**: regime 遷移時の position exit ロジック検討 |
| B | 第3 SHA 分析欠如 | P2: SHA 切替時の設定差分調査 |
| C | MCB 復帰後品質劣化 | P1: cooldown 延長 or 段階的復帰ロジック |
| D | sell/trending_up AS -2.21 bps | P1: regime条件付き sell guard 検討 |
| E | NFQ 0% の意味 | P2: quote feasibility 閾値の適正評価 |
| F | Protocol 688 波及範囲 | P0: 過去レポートの影響範囲洗い出し |

---

## 5. データ再現コマンド

本文書のデータは以下コマンドで再現可能 (675#/679# 教訓準拠):

```bash
# 全日 PnL
python -m scripts.v460.analysis.analyze_fill_logs --date-from 2026-04-02 --date-to 2026-04-02

# テールロス
python -m scripts.v460.analysis.tail_loss_analysis --date-from 2026-04-02 --date-to 2026-04-02

# SHA 別パフォーマンス
python -m scripts.v460.analysis.sha_performance_report --days 2
```

---

*生成: 2026-04-02 by cplt (699#)*
*検証データ: 2026-04-02 全日 (UTC 00:00-14:48)*
