# 152# 代替優先施策 — 計画・検証・実装

> 149# §10.3 の代替優先 3 施策について、Phase C データに基づく検証結果と実装計画。

| 項目 | 概要 | 結論 |
|------|------|------|
| 144# CRITICAL | preflight-lot 整合 | §8.1/#1-#3 + §9.1/#1-#2 は 145# で**対応済み**。残 §9.1 #3/#4 + §8.1 #6 を対応 |
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
| §9.1 #3 | HIGH | OB 特徴量で `.price/.quantity` 前提 → tuple 形式と不整合 | 153# 以降で検討 (Phase C での SkipGate 精度影響は限定的) |
| §9.1 #4 | MEDIUM | SkipGate 判定が lot 適応前 | ✅ 151# §10 #4 で対応済み (`_regime_lot` を先に算出して SkipGate に渡す) |

**結論**: CRITICAL/HIGH/MEDIUM は全件 145#/151# で対応完了。残りの §9.1 #3 (OB 特徴量) は 153# 以降。

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

#### C: min_confidence 閾値調整

現行 0.3 → **0.0 に下げる**（またはゲート自体を削除）。
既に `_classify()` の最低値が 0.4 で常にパスするため、実質的変更なし。
しかしドキュメント上の意図を明確にし、将来の classification 拡張を考慮して 0.0 ではなく **0.2** に引き下げ。

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
- 工数 0.34 日の投資対効果 (ROI) は十分か
- 次に着手すべき施策の提案

---

## §7 変更履歴

| 日付 | 内容 |
|------|------|
| 2026-02-23 | 初版: 3 施策の計画・検証・実装方針 |
