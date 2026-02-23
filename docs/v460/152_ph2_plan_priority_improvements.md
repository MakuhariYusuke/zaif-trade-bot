# 152# 代替優先施策 — 計画・検証・実装

> 149# §10.3 の代替優先 3 施策について、Phase C データに基づく検証結果と実装計画。

| 項目 | 概要 | 結論 |
|------|------|------|
| 144# CRITICAL | preflight-lot 整合 | §8.1/#1-#3 + §9.1/#1-#4 + §8.1 #6 は 145#/151# で**全件対応済み** |
| P3-03 | confidence_lot 有効化判定 | データ分析の結果 **有効化見送り** (99.7% min_lot で no-op) |
| P3-02 | advanced_regime_detector | unknown 8.8% → PnL 最悪レジーム。**ヒステリシス改善**で削減 |

---

## §1 Phase C データに基づく現状分析

### 1.1 データ概要 (2026-02-13～2026-02-22)

```
Total records: 2,193
Records with order_quantity: 1,560
Filled: 1,272
Regime-tagged records: 1,240
  ranging:  874 (70.5%)
  trending: 257 (20.7%)
  unknown:  109 (8.8%)
```

#### 再現条件 (§8 #2 対応)

| 項目 | 値 |
|------|----|
| データソース | `results/v460/fill_test/fill_records_*.jsonl` |
| 期間 | 2026-02-13 ～ 2026-02-22 (Phase C 開始後の全日付ファイル) |
| run_id フィルタ | なし (全 run_id を含む) |
| clean 判定 | `filled == true` かつ `post_fill_30s_pnl != null` のレコードを PnL 集計対象 |
| 集計スクリプト | `scripts/v460/analysis/analyze_fill_records.py` (regime 分布・PnL) |
| 補助スクリプト | `scripts/v460/analysis/analyze_fill_detail.py` (run_id 別・日次分解) |

> **注意**: run_id 混在時は件数が変動する。特定 run のみ対象にする場合は `analyze_fill_detail.py` の run_id 別出力で該当 ID を確認し、手動フィルタすること。

### 1.2 レジーム別 PnL

| Regime | fills | avg PnL (bps) | sum PnL (bps) |
|--------|-------|---------------|----------------|
| ranging | 685 | -0.222 | -152.31 |
| trending | 227 | -0.086 | -19.56 |
| **unknown** | **93** | **-0.891** | **-82.86** |

**unknown は全レジーム中 PnL 最悪** (ranging の 4 倍、trending の 10 倍悪い)。

### 1.3 ロット分布

```
0.0010 BTC: 1,556 (99.7%)  ← ほぼ全件が min_order_btc
0.0015 BTC:     1 (0.1%)
0.0017 BTC:     2 (0.1%)
0.0020 BTC:     1 (0.1%)
```

### 1.4 AS 確率分布

```
[0.0, 0.1): 0.3%
[0.1, 0.2): 2.8%
[0.2, 0.3): 5.5%
[0.3, 0.4): 6.9%
[0.4, 0.5): 31.8%  ← 最頻
[0.5, 0.6): 51.7%  ← 最頻
[0.6+):     0.9%
mean=0.474, median=0.503
```

---

## §2 144# CRITICAL — 対応済み確認と残課題

### 2.1 145# で対応済みの項目

| §8.1/§9.1 | 重大度 | 問題 | 対応状況 |
|---|---|---|---|
| §8.1 #1 | CRITICAL | preflight 前に regime lot 未反映 | ✅ 145# `run_continuous()` L1580: `_regime_mult` を `_check_balance_for_side()` に渡す |
| §8.1 #2 | HIGH | `_current_lot` 乗算的増加 | ✅ 145#/151# `_regime_lot` を per-cycle 算出、永続化しない |
| §8.1 #3 | HIGH | 縮小レジームで preflight 過大 | ✅ 145# `_check_balance_for_side(regime_mult=)` + BalanceChecker 対応 |
| §9.1 #1 | CRITICAL | reprice 時の数量が `current_lot` | ✅ 145# `_monitor_fill_polling(order_lot=_order_lot)` → monitor に正しいロット転送 |
| §9.1 #2 | HIGH | timeout ラベル不整合 | ✅ 145# `monitor.effective_timeout` 取得 → cancel_reason 判定に使用 |

### 2.2 残課題 (本セッションで対応)

| # | 重大度 | 問題 | 対応方針 |
|---|---|---|---|
| §8.1 #6 | MEDIUM | `regime_timeout_multipliers` / `regime_reprice_adjustments` の値域バリデーションなし | ✅ 145# `__post_init__` で正値チェック済み (L296-310) |
| §9.1 #3 | HIGH | OB 特徴量で `.price/.quantity` 前提 → tuple 形式と不整合 | ✅ 145# で対応済み (`ob_utils.extract_price/depth_volume` で tuple/object 両対応) |
| §9.1 #4 | MEDIUM | SkipGate 判定が lot 適応前 | ✅ 151# §10 #4 で対応済み (`_regime_lot` を先に算出して SkipGate に渡す) |

**結論**: CRITICAL/HIGH/MEDIUM は**全件** 145#/151# で対応完了。§9.1 #3 (OB 特徴量) も 145# で `ob_utils` 導入済み → **残課題ゼロ**。

---

## §3 P3-03 — confidence_lot 有効化判定

### 3.1 判定基準 (151# §11.3)

> `order_quantity == min_order_btc` の比率が 80% 未満なら有効化。

### 3.2 Phase C データ検証結果

```
min_lot (0.001 BTC) 比率: 99.7% (1,556/1,560)
```

**80% 未満の基準を大幅に超過 → 有効化見送り。**

### 3.3 根本原因分析

confidence_lot は `regime_lot × confidence_factor` で算出し、`max(min_order_btc, ...)` でクランプ。

現状: `order_quantity (0.001) = min_order_btc (0.001)` のため:
- confidence_factor < 1.0 → `regime_lot * factor < 0.001` → クランプで 0.001 → **無意味**
- confidence_factor = 1.0 → そのまま 0.001 → **変化なし**

**confidence_lot は「縮小専用」設計のため、ベースロットが min_order_btc では動作しない。**

### 3.4 シミュレーション

| base_lot | 有効ロット比率 | 平均実ロット | 評価 |
|----------|--------------|-------------|------|
| 0.001 (現行) | 0.0% | 0.001 | 完全に no-op |
| 0.003 (3倍) | 99.8% | 0.0016 | AS 高 → 0.001、AS 低 → 0.003 |

### 3.5 将来の有効化条件

confidence_lot を有効活用するには、以下のいずれかが必要:

1. **order_quantity を min_order_btc 超に引き上げ** (推奨: 0.003 BTC)
2. **「拡大併用」モード追加**: confidence が高いとき factor > 1.0 で増量

いずれも Phase C の安定性を損なうため、Phase D 以降で検討。

### 3.6 結論

| 決定 | `enabled: false` 維持 |
|------|----------------------|
| 理由 | 99.7% min_lot → 完全 no-op |
| 次のアクション | Phase D で order_quantity 引き上げ時に再評価 |

---

## §4 P3-02 — レジーム検知改善 (unknown 削減)

### 4.1 unknown の発生メカニズム

`FillTestRegimeDetector` の `_classify()` は常に trending/ranging/high_vol を返す。
UNKNOWN は以下の 2 ルートのみ:

1. **データ不足**: `len(prices) < window (20)` → 起動後 20 サイクル (≈40 分)
2. **ヒステリシス未確定**: `_confirmed_regime = UNKNOWN` (初期値) のまま、3 連続同一分類が成立しない

**min_confidence ゲート (0.3) は実質不発** — `_classify()` の最低 confidence は 0.4 (RANGING)。

### 4.2 改善設計

#### A: 初回遷移の accelerated hysteresis

```python
# UNKNOWN → first regime は 2 連続で確定 (通常は 3)
if self._confirmed_regime == FillTestRegime.UNKNOWN:
    threshold = max(2, self.config.hysteresis_count - 1)
else:
    threshold = self.config.hysteresis_count
```

**期待効果**: 起動後の unknown 期間を 1 サイクル (≈2 分) 短縮。市場が choppy な場合の初回確定も早まる。

#### B: 「最頻分類」フォールバック

window 観測が溜まり、ヒステリシスが未確定 (初回 UNKNOWN) の場合、直近 N 回の raw 分類の最頻値で仮確定:

```python
if self._confirmed_regime == FillTestRegime.UNKNOWN and len(self._raw_history) >= self.config.hysteresis_count:
    from collections import Counter
    recent_raw = self._raw_history[-self.config.hysteresis_count * 2:]
    non_unknown = [r for r in recent_raw if r != FillTestRegime.UNKNOWN]
    if non_unknown:
        majority, _ = Counter(non_unknown).most_common(1)[0]
        self._confirmed_regime = majority
```

**期待効果**: choppy な市場でも「いつまでも unknown」を防止。

#### C: min_confidence 閾値調整 (優先度: A/B の後)

現行 0.3 → **0.0 に下げる**（またはゲート自体を削除）。
既に `_classify()` の最低値が 0.4 で常にパスするため、実質的変更なし。
しかしドキュメント上の意図を明確にし、将来の classification 拡張を考慮して 0.0 ではなく **0.2** に引き下げ。

> **§8 #5 対応**: 本案は A/B の効果検証完了後に着手する。A/B の採否判定が優先。

### 4.3 AdvancedRegimeDetector の採用可否

| 要素 | FillTestRegimeDetector | AdvancedRegimeDetector |
|------|----------------------|----------------------|
| 入力データ | mid_price のみ | price + high + low |
| 状態数 | 4 | 12 |
| 指標 | trend% + vol_ratio | RSI + ADX + momentum + MACD |

**結論**: fill_test は mid_price しか取得できず、AdvancedRegimeDetector の high/low 入力を満たせない。
mid_price のみで RSI/ADX を近似しても精度は低い。**FillTestRegimeDetector の改善が現実的。**

### 4.4 工数見積

| 作業 | 工数 |
|------|------|
| A: accelerated hysteresis | 0.05 日 |
| B: 最頻フォールバック | 0.1 日 |
| C: min_confidence 調整 | 0.02 日 |
| テスト追加 | 0.1 日 |
| **合計** | **0.27 日** |

### 4.5 期待効果

- unknown 比率: 8.8% → 推定 2-3% (起動直後の window 充填期間のみ)
- PnL 改善: unknown 93 fills × avg -0.891 → 正しいレジーム分類で -0.222 に近づく場合、
  **最大 +62 bps の改善余地**

### 4.6 採用/棄却 Gate (§8 #3 対応)

> A/B 比較ハーネスで以下を測定し、3 基準すべてクリアで採用。

| Gate | 基準 | 閾値 | 測定方法 |
|------|------|------|----------|
| G1 | unknown 比率の減少 | **≤ 3%** (現行 8.8%) | `analyze_fill_records.py` の regime 分布 |
| G2 | regime 別 PnL の非悪化 | ranging / trending の avg PnL が **±0.1 bps 以内** | 同スクリプトの Regime × PnL 出力 |
| G3 | 全体 PnL の非悪化 | 全 fill の sum PnL が **改善 or ≤ 5 bps 悪化** | A (現行) vs B (改善後) の差分 |

**棄却時**: A/B 比較で G1 未達 → 設計見直し。G2/G3 未達 → fallback ロジックの副作用調査。

---

## §5 実装計画

### 5.1 実施順序

| 順 | 作業 | ファイル | 工数 |
|----|------|---------|------|
| 1 | §4.2 A: accelerated hysteresis | `regime_detector.py` | 0.05 日 |
| 2 | §4.2 B: 最頻フォールバック | `regime_detector.py` | 0.1 日 |
| 3 | §4.2 C: min_confidence YAML 更新 | `fill_test.yaml` | 0.02 日 |
| 4 | §3.6 P3-03 結論をドキュメント反映 | `fill_test.yaml` コメント | 0.02 日 |
| 5 | テスト追加 (regime_detector) | `test_143_regime_utilization.py` | 0.1 日 |
| **合計** | | | **0.29 日** |

> 144# §8.1 #6 は 145# で対応済みのため、スコープから除外。

### 5.2 変更対象ファイル一覧

| ファイル | 変更種別 | 内容 |
|---------|---------|------|
| `scripts/v460/lib/regime_detector.py` | 機能改善 | A: accelerated hysteresis + B: majority fallback |
| `configs/v460/fill_test.yaml` | 設定変更 | min_confidence 0.3→0.2, P3-03 有効化見送りコメント |
| `tests/unit/v460/test_143_regime_utilization.py` | テスト追加 | accelerated hysteresis + fallback テスト |
| `docs/v460/152_ph2_plan_priority_improvements.md` | 新規 | 本ドキュメント |

---

## §6 Codex レビュー依頼

### 6.1 144# CRITICAL 対応状況の検証

- §2.1 の「145# で対応済み」判定は正しいか
- 見落としている未修正項目はないか
- §9.1 #3 (OB 特徴量) の優先度判断は適切か

### 6.2 P3-03 有効化見送りの妥当性

- §3.2 の「99.7% min_lot → no-op」評価は正しいか
- 有効化見送りの代わりに即座に取るべきアクションはあるか
- order_quantity 引き上げ (§3.5 #1) のリスク評価

### 6.3 P3-02 レジーム検知改善

- §4.2 の 3 改善案 (A/B/C) の実装は妥当か
- B (最頻フォールバック) が既存の hysteresis 設計思想と矛盾しないか
- AdvancedRegimeDetector 不採用の判断は適切か

### 6.4 全体評価

- 149# §10.3 の 3 施策のうち、2 つが「見送り/対応済み」に終わる結論は妥当か
- 工数 0.29 日の投資対効果 (ROI) は十分か
- 次に着手すべき施策の提案

---

## §7 変更履歴

| 日付 | 内容 |
|------|------|
| 2026-02-23 | 初版: 3 施策の計画・検証・実装方針 |
| 2026-02-23 | §8 Codex レビュー全 5 件対応 (§10 追加) |

---

## §8 Codex レビュー結果 (2026-02-23)

### 8.1 総評

- 方針の大枠は妥当です。特に **P3-03 を即時有効化しない判断** は、現状ロット分布と整合しています。  
- 一方で、優先度と事実関係にズレが一部あり、ここを直さないと Phase C の時間を消耗する可能性があります。

### 8.2 指摘事項 (重大度順)

| # | 重大度 | 指摘 | 根拠 | 推奨対応 |
|---|---|---|---|---|
| 1 | MEDIUM | §2.2 の「§9.1 #3 (OB 特徴量) は 153# 以降」は現状と不整合 | `scripts/v460/lib/skip_gate_evaluator.py` では `extract_price/depth_volume` を使用し、tuple/object 両対応に修正済み | §2.2 を更新し、`#3` は「残課題」から外す。優先度を `unknown 改善の定量検証` に振り替える |
| 2 | MEDIUM | §1 データ集計の再現条件が不足 | 同期間でも `run_id` 混在有無で件数が変わるため、`2,193` 等の数値の再現手順が本文にない | §1.1 に「対象 run_id / clean 判定条件 / 集計スクリプト」を追記して再現可能化 |
| 3 | MEDIUM | P3-02 の成功判定が弱い | §4.5 は期待効果のみで、採用/棄却の閾値が未定義 | 「unknown 比率」「regime別PnL」「全体PnL悪化許容」を Gate 化し、A/B 比較で判断する |
| 4 | LOW | 工数記載が不一致 | §5.1 合計は `0.29 日`、§6.4 は `0.34 日` | どちらかに統一し、差分 0.05 日の内訳を明記 |
| 5 | LOW | min_confidence 調整の優先度が相対的に低い | 本文自身で「classify 最低値 0.4 で不発に近い」と記載 | C 案は後順位にし、A/B の効果検証を先行する |

### 8.3 152 計画に対する判定

- **採択**: P3-03 見送り、P3-02 集中は妥当。  
- **修正必須**: §2.2 の残課題整理（OB #3 の扱い）と、§1 数値の再現性明文化。  
- **着手順**: A/B 実装前に「測定設計」を先に固定する。

---

## §9 Phase C 裏作業 (今すぐ並行でできること)

> 入金待ちや live 実行待ちの間でも進められる、高収益寄与の高い順に列挙。

| 優先 | 作業 | 目的 | 成果物 |
|---|---|---|---|
| P0 | **集計再現スクリプト固定** (`run_id`/期間/clean 条件を引数化) | 152 の数値ぶれ防止、意思決定の再現性確保 | `scripts/v460/analysis/reproduce_152_metrics.py` + 実行ログ |
| P0 | **regime A/B 比較ハーネス** (現行 vs A vs A+B) | unknown 削減が実PnL改善に効くかを事前判定 | `results/v460/ab_regime/*.csv` |
| P0 | **寄与分解レポート自動化** (side×regime×hour×cancel_reason) | どこで負けているかを固定フォーマットで可視化 | `docs/v460/xxx_attribution_report.md` |
| P1 | **SkipGate 閾値のカウンターファクト掃引** (ログ再評価) | live再実行なしで閾値候補を絞る | `scripts/v460/analysis/sweep_skip_gate_thresholds.py` |
| P1 | **unknown 遷移イベント監査** (transition 直前後のPnL/AS) | P3-02 改修の副作用検知 | `results/v460/regime_transition_audit.csv` |
| P1 | **confidence_lot 有効化前ガード** (no-op率チェック) | 誤って `enabled=true` にしても無意味投入を防止 | 起動時チェック + warning/abort 条件 |
| P2 | **テストの安定化** (`unit` mark warning の解消) | CI ノイズ削減、レビュー効率化 | warning 0 化 |
| P2 | **run_fill_test の追加分割設計** (lot/regime/skip 判定の責務分離) | 今後の機能追加コストを圧縮 | 小規模リファクタ設計メモ |

### 9.1 推奨する次の 3 手

1. `P0-1` 集計再現スクリプトを先に作り、152 の数値を「再生成可能」にする。  
2. `P0-2` regime A/B 比較ハーネスで、A/B 案の採否基準を数値で固定する。  
3. `P1-4` SkipGate 閾値のオフライン掃引で、live 実験前に候補を 2-3 個へ圧縮する。  

---

## §10 Codex レビュー対応結果 (§8.2 全 5 件)

| # | 重大度 | 指摘 | 対応内容 | 対応箇所 |
|---|---|---|---|---|
| 1 | MEDIUM | §2.2 の §9.1 #3 が残課題のまま | `skip_gate_evaluator.py` L537-551 で `ob_utils.extract_price/depth_volume` 使用を確認。§2.2 を「✅ 145# で対応済み」に更新。結論も「残課題ゼロ」に修正 | §2.2 |
| 2 | MEDIUM | §1 データ集計の再現条件不足 | §1.1 に再現条件テーブル追加: データソース、期間、run_id フィルタ、clean 判定、集計スクリプト、注意事項 | §1.1 |
| 3 | MEDIUM | P3-02 成功判定が弱い | §4.6「採用/棄却 Gate」新設: G1 (unknown ≤ 3%)、G2 (regime PnL ±0.1 bps)、G3 (全体 PnL ≤ 5 bps 悪化)。棄却時の次ステップも明記 | §4.6 |
| 4 | LOW | 工数不一致 (§5.1: 0.29 日 vs §6.4: 0.34 日) | §6.4 を 0.29 日に統一 (§5.1 と整合)。差分 0.05 日は §8.1 #6 のスコープ除外による | §6.4 |
| 5 | LOW | C 案 (min_confidence) の優先度が低い | §4.2 C に「優先度: A/B の後」を明記。A/B の採否判定完了後に着手する旨を追記 | §4.2 C |
