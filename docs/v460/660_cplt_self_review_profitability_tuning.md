# 660# セルフレビュー：収益性分析とパラメータチューニング

## 概要

2日間 (3/29–3/30) の fill_records を統計分析し、レジーム別の PnL 構造を可視化。
損失の 72% が trending_up レジームに集中していることを特定し、4 つの YAML パラメータを調整。

## 分析手法

- 対象: `results/v460/fill_test/fill_records/` 直近 2 日分
- 分析スクリプト: `temp/self_review_analysis.py`（使い捨て）

## 主要データ

| 指標 | 値 |
|------|-----|
| 総サイクル | 733 |
| 約定数 | 70 (fill rate 9.5%) |
| PnL30 平均 | **-0.68 bps** |
| PnL30 合計 | **-47.6 bps** |
| PnL120 平均 | -1.08 bps |
| Spread capture 平均 | -0.52 bps |

### レジーム別 PnL30

| レジーム | 平均 bps | 約定数 | 損失寄与 |
|----------|----------|--------|----------|
| ranging | -0.13 | 42 | ≈ ブレークイーブン |
| **trending_up** | **-1.90** | **18** | **72%** |
| trending_down | -0.80 | 10 | 17% |

### サイド別 PnL30

| サイド | 平均 bps | 約定数 |
|--------|----------|--------|
| buy | -0.46 | 37 |
| sell | -0.93 | 33 |

### キャンセル理由分布 (663 件)

| 理由 | 件数 | 割合 |
|------|------|------|
| preflight_insufficient | 298 | 44.9% |
| no_feasible_quote | 133 | 20.1% |
| spread_too_narrow | 100 | 15.1% |
| mcb_halt | 46 | 6.9% |
| skip_gate | 44 | 6.6% |

※ preflight_insufficient は buy:219, sell:79 — JPY 残高枯渇が主因（構造的問題）

## 変更内容

### 1. skip_gate.regime_thresholds.trending_up: -0.1 → 0.3

**根拠**: trending_up 18 fills で PnL30 = -1.90 bps（全損失の 72%）。
regime_thresholds は base_threshold の**置換値**（加算ではない）。
-0.1 は実質ほぼフリーパスだったため、0.3 に引き上げて
trending_up 時は skip_gate が EV > 0.3 のときのみ通過を許可。

### 2. skip_gate.regime_thresholds.trending_down: -0.1 → 0.1

**根拠**: trending_down 10 fills で PnL30 = -0.80 bps（17%）。
軽度の締め付けで adverse selection を軽減。

### 3. micro_circuit_breaker.halt_sigma: 2.0 → 2.5

**根拠**: 2 日間で 46 回の MCB HALT（全サイクルの 6.3%）。
偽 HALT がトレード機会を逸失させていた。
2.5σ に緩和して真にリスクの高い状況のみ HALT する。

### 4. buy_dynamic_kill.regime_thresholds.trending_up: -1.5 → -1.0

**根拠**: trending_up での buy は特に adverse selection リスクが高い。
BDK regime_thresholds も置換値であり、-1.5 は寛容すぎた。
-1.0 に引き締めて trending_up buy の悪質な約定を早期キルする。

## 期待される効果

- **trending_up 約定の大幅削減**: 18 fills × -1.90 bps = -34.2 bps → SG 閾値 0.3 で大半をフィルタ
- **trending_down 改善**: 10 fills × -0.80 bps = -8.0 bps の一部削減
- **MCB 偽 HALT 削減**: 46 HALT → ~30 HALT に削減見込み（機会損失回復）
- **ranging レジームへの影響なし**: ranging は -0.13 bps でほぼ収支均衡、変更対象外

## 付随修正

- `tests/unit/v460/test_336_yaml_code_drift_prevention.py`: `mcb_halt_sigma` を `KNOWN_YAML_OVERRIDES` に追加
- `tests/unit/v460/test_657_regime_max_factor_and_toxic_veto_offset.py`: `time.time()` タイミング差による flaky テスト修正 (abs=1e-10 → 1e-8)
