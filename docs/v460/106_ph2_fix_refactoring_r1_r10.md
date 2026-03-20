# 106# リファクタリング調査 + 即時修正

- **parent**: `105_ph2_fix_sell_offset_balance.md`
- **status**: committed (継続更新中)
- **commit**: (後記)

## 背景

前ターンのコードレビューで R1～R10 のリファクタリング余地を特定。
本ドキュメントは全項目の記録と、再起動前に安全に実行可能な修正の実施報告。

---

## R1～R10 全項目一覧

### 高優先度 (収益直結)

| # | 項目 | 判定 | 理由 |
|---|------|------|------|
| **R1** | `run_fill_test.py` God Object 解体 | **✅ 119#–121# で段階実施** | 3411→2701 (119#) →1912 (120#) →~1000 (121#)。詳細は §4 参照 |
| **R2** | `BPS_FACTOR = 10_000` 定数化 | **✅ 106#で実施済** | `run_fill_test.py` 14箇所 + `lot_sizer.py` 1箇所 |
| **R3** | SkipGate `evaluate()`/`warm_start` 単体テスト不足 | **✅ 後続で大幅補強** | 当時は「後日」判断だったが、session037 で canonical 化と並行して `test_enricher_skip_gate.py` / `test_skip_gate_v3.py` / migration test を拡充し、result metadata / FillRecord payload / runtime helper 境界まで回帰を追加 |

### 中優先度 (保守性)

| # | 項目 | 判定 | 理由 |
|---|------|------|------|
| **R4** | ドキュメント命名違反 28件 | **後日** | 060-098番台に `phX`/`type` 欠落多数。大量リネームは運用に影響なし |
| **R5** | lib → ztb 移動検討 | **✅ session037 で大幅前倒し** | `KillSwitch` / `OrderState` に加え、`cancel_reasons` / `param_adapter` / `lot_sizer` / `fast_fill_defense` / `sac_common` / `regime_detector` / `bayesian_regime_filter` まで canonical 化を実施。残りは `maker_price` / `skip_gate_evaluator` / `order_monitor` の split-first 終盤 |
| **R6** | utils 70+ファイルの分割 | **後日** | God package。safety.py, rate_limiter.py が埋もれている。大規模リファクタ |
| **R7** | `config/` vs `configs/`, `reporting/` vs `reports/` の重複ディレクトリ整理 | **後日** | 影響範囲が広い |

### 低優先度

| # | 項目 | 判定 | 理由 |
|---|------|------|------|
| **R8** | `# type: ignore` 3箇所の解消 | **✅ 部分実施** | 1/3 解消 (`regime_detector.update` → assert ガード)。残り2箇所は正当 (psutil=untyped, SIGBREAK=Windows固有attr) |
| **R9** | インライン import 移動 | **✅ 実施済** | `import random as _rng` をトップレベルに移動。psutil は lazy import 維持 (3rd party、未インストール環境対応) |
| **R10** | 100番の番号重複解消 | **✅ 105#で解消済** | 100→101→102→103→104 cascade rename |

---

## §4 R1 詳細: run_fill_test.py God Object 解体 (119#–121#)

### 行数推移

| ターン | 行数 | 削減率 | 主な抽出 |
|--------|------|--------|----------|
| 106# | 3411 | — | R1 問題特定 |
| 113# | 2701 | 21% | `run_single_cycle` 分割: 3メソッド+3 dataclass |
| 119# | 2701 | — | FillTestConfig, BatchPersistence, ResultsAnalyzer 抽出 |
| 120# | 1912 | 44% | MakerPriceCalculator, OrderMonitor, PnlMeasurer, AdaptationEngine 抽出 |
| 121# | ~1000 | **71%** | SkipGateEvaluator, BalanceChecker, TimeFilter, SideSelector, RecordBuilder, CLI 抽出 |

### 抽出済みモジュール一覧 (121# 時点)

| モジュール | ターン | 行数 | 責務 |
|-----------|--------|------|------|
| `lib/fill_config.py` | 119# | 528 | 設定 dataclass + YAML マッピング |
| `lib/batch_persistence.py` | 119# | 140 | バッチ保存 + 緊急ダンプ + TTL flush |
| `lib/results_analyzer.py` | 119# | 160 | メトリクス集計 + 判定出力 |
| `lib/maker_price.py` | 120# | 250 | 板価格算出 + imbalance + volatility guard |
| `lib/order_monitor.py` | 120# | 310 | 約定ポーリング + stale reprice |
| `lib/pnl_measurer.py` | 120# | 130 | post-fill PnL 計測 (30/60/120s) |
| `lib/adaptation_engine.py` | 120# | 290 | 方策A/B 適応 + 動的 loss_cap |
| `lib/skip_gate_evaluator.py` | 121# | ~180 | SkipGate ML 判定 + 特徴量構築 |
| `lib/balance_checker.py` | 121# | ~170 | 残高 pre-flight + ロット自動縮小 |
| `lib/time_filter.py` | 121# | ~80 | 時間帯フィルター判定 |
| `lib/side_selector.py` | 121# | ~60 | buy/sell 交互 + Smart Side |
| (既存) `lib/fast_fill_defense.py` | 100# | 180 | 即約定防御 (side-aware) |
| (既存) `lib/resilience.py` | 113# | 380 | CircuitBreaker + Health + State |
| (既存) `lib/regime_detector.py` | 037# | 200 | レジーム検知 |

---

## 実施済み変更の詳細

### §1 R2: `_BPS_FACTOR` 定数化

**Before**: `* 10000` / `* 1e-4` がファイル内に散在 (14+1箇所)

**After**: クラス定数 `_BPS_FACTOR: int = 10_000` を定義し、全箇所を統一

```python
# run_fill_test.py — クラス定数
_BPS_FACTOR: int = 10_000

# 使用例 (ratio → bps)
mid_trend_bps = (mid_price - prev) / prev * self._BPS_FACTOR

# 使用例 (bps → ratio)
cumulative_pnl_jpy += pnl_bps / self._BPS_FACTOR * price * qty
```

**対象ファイル**:
- `scripts/v460/run_fill_test.py`: 14箇所 (全 `* 10000` → `* self._BPS_FACTOR`, 全 `* 1e-4` → `/ self._BPS_FACTOR`)
- `scripts/v460/lib/lot_sizer.py`: 1箇所 (`* 1e-4` → `/ 10_000`)

### §2 R8: `# type: ignore` 解消

| 箇所 | Before | After | 判定 |
|------|--------|-------|------|
| L2081 `regime_detector.update()` | `# type: ignore[arg-type]` | `assert r.mid_at_fill is not None` ガード | **解消** |
| L1147 `import psutil` | `# type: ignore[import-untyped]` | 維持 | 正当: psutil にスタブなし |
| L3033 `signal.SIGBREAK` | `# type: ignore[attr-defined]` | 維持 | 正当: Windows 固有属性 |

### §3 R9: インライン import 整理

| import | Before | After |
|--------|--------|-------|
| `random` | L1867 インライン `import random as _rng` (毎サイクル実行) | トップレベル stdlib import に移動 |
| `psutil` | L1147, L2123 インライン lazy import | 維持: 3rd party、未インストール環境対応 |

---

## 未実施項目の優先順位付け (次回以降)

| 優先 | # | 推奨タイミング |
|------|---|---|
| ~~1~~ | ~~R1~~ | **121# で ~1000行まで到達** (目標達成) |
| 2 | R3 | 次回 SkipGate 再訓練時 |
| ~~3~~ | ~~R5~~ | **session037 で主要移行を前倒し**。残りは `maker_price` / `skip_gate_evaluator` / `order_monitor` の終盤整理 |
| 4 | R4 | ドキュメント整理一括作業時 |
| 5 | R6/R7 | リポジトリ構造整理フェーズ |

## テスト結果

- 811 passed, 0 failed (v460 unit tests, 106# 時点)
- 857 passed, 0 failed (113# R1 分割後, 2026-02-19)
- 878 passed, 0 failed (120# God Object 分割後, 2026-02-21)
- 878 passed, 0 failed (121# 最終抽出後, 1912→1568行)

## 2026-03-21 補遺

106# 時点では `v461` 以降送りとしていた `lib -> ztb` 移行と SkipGate 周辺のテスト補強は、
session037 でかなり前倒しされた。

- canonical 化済み:
  - `cancel_reasons`
  - `param_adapter`
  - `lot_sizer`
  - `fast_fill_defense`
  - `sac_common`
  - `regime_detector`
  - `bayesian_regime_filter`
- `skip_gate_evaluator` は
  - result metadata
  - FillRecord extra payload
  - final FillRecord context/builder
  まで canonical 側へ整理済み
- `maker_price` も
  - inventory math
  - offset math
  - loss boost decay
  - spread adaptive
  - spread guard finalization
  - final ceiling clamp
  の pure/stage helper を `ztb.trading.pricing` 側へ抽出済み

このため、106# の deferred 表記は「当時の判断」としては妥当だったが、現状認識としては更新が必要な状態になっている。
