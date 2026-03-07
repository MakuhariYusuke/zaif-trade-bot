# 324# 未達事項消化 + Regime 既存実装活用

## 概要

321# 以前の未達事項を消化し、ztb 既存 regime 実装の活用を図る。

- **M-2**: per-side unknown counter（buy/sell 混合カウント問題の修正）
- **L-3**: dynamic_cycle_interval の重複懸念を YAML 文書化で解消
- **L-4**: Feature Graveyard (imbalance/smart_side/early_exit) の YAML 文書化
- **Regime RSI**: ztb advanced_regime_detector の RSI アルゴリズムを live detector に統合

## 1. M-2: per-side unknown counter

### 問題

`cycle_gate_aggregator.py` の `_consecutive_unknown_blocks` が単一 `int` で管理されていた。
buy/sell が交互に評価されると、片方の連続ブロック数が他方を加算してカウントされ、
バイパス閾値 (`UNKNOWN_REGIME_MAX_CONSECUTIVE = 10`) への到達が不正確になる。

### 修正

| 箇所 | Before | After |
|---|---|---|
| `__init__` L158 | `int = 0` | `dict[str, int] = {"buy": 0, "sell": 0}` |
| `evaluate()` L205 | `self._consecutive_unknown_blocks` | `self._consecutive_unknown_blocks.get(side, 0)` → `_side_count` |
| Gate 1 block L222 | `+= 1` | `[side] = _side_count + 1` |
| Gate 1 pass L228 | `= 0` | `[side] = 0` |
| Gate 7 block L345 | `+= 1` | `[side] = _side_count + 1` |

### テスト更新 (3 ファイル)

- `test_220_deadlock_fixes.py`: 5 テスト — `["buy"]`/`["sell"]` access に変更。`test_mixed_buy_sell_unknown_counts_together` → `test_per_side_independence` にリネーム・再実装
- `test_229_cleanup_counter_rename.py`: 6 箇所 — per-side dict アクセスに変更
- `test_277_magic_number_grounding.py`: 1 箇所 — `["buy"]` access に変更

## 2. L-3: dynamic_cycle_interval 文書化

### 懸念

`regime_policy.dynamic_cycle` と `dynamic_cycle_interval` が競合する可能性。

### 解消

コード調査の結果、両者は **composable（合成可能）** であることを確認:

1. `regime_policy.dynamic_cycle`: regime → base interval のマップ（`CycleStrategy.effective_interval()`）
2. `dynamic_cycle_interval`: σ ベースのスケーリング（`_compute_dynamic_interval()`）

fill_test.yaml に 4 行のコメントブロックで関係を文書化。

## 3. L-4: Feature Graveyard 文書化

`fill_test.yaml` の `imbalance` / `smart_side` / `early_exit` セクションに
Feature Graveyard コメントを追加。071# / 120# で無効化済みだが YAML 参照が
残存するため温存。削除条件も明記。

## 4. Regime RSI 統合

### 設計

ztb/analysis/regime/advanced_regime_detector.py の `TechnicalIndicators.calculate_rsi()`
（Wilder RSI アルゴリズム）を live `FillTestRegimeDetector._classify()` に inline 統合。

import 依存を回避し、3 行の核心計算で実装。

### 変更点

| ファイル | 変更 |
|---|---|
| `regime_detector.py` RegimeConfig | `rsi_modulation: bool = True`, `rsi_period: int = 14` 追加 |
| `regime_detector.py` _classify() | trending 判定後に RSI 変調呼び出し追加 |
| `regime_detector.py` | `_apply_rsi_modulation()` メソッド新設 (55 行) |

### RSI 変調ロジック

| 条件 | 効果 |
|---|---|
| RSI がトレンド方向を確認 (up + RSI≥55 / down + RSI≤45) | confidence +0.10 |
| RSI がトレンドと不一致 (divergence) | confidence -0.15 |
| RSI 45-55 (中立) | 変更なし |

### 理論的根拠

J. Welles Wilder Jr. (1978) "New Concepts in Technical Trading Systems".
RSI-trend divergence は反転兆候として広く認識されており、
trending regime の confidence を適切にモデレートする。

## 5. 未着手事項の棚卸し

| ID | 内容 | 判定 |
|---|---|---|
| M-3b | offset_mult < 1.0 無視 | **意図的設計** — conservative モードで aggressive 動作を防止 |
| M-1/L-2 | velocity_ema_alpha = 1.0 | **保留** — 321# 記載通り「データ蓄積後に 0.6 有効化」 |
| S-2 | Sell Hour Boost | **保留** — post-310# データ分析が前提 |
| S-6 | buy ev_offset | **保留** — 分析先行 |

## 6. テスト結果

```
4096 passed, 0 failed (30.26s)
```

323# の 3971 passed から 125 テスト増加（外部追加分含む）。

## 7. 変更ファイル一覧

| ファイル | 変更種別 |
|---|---|
| `scripts/v460/lib/cycle_gate_aggregator.py` | M-2 per-side counter |
| `scripts/v460/lib/regime_detector.py` | RSI modulation |
| `configs/v460/fill_test.yaml` | L-3/L-4 YAML comments |
| `tests/unit/v460/test_220_deadlock_fixes.py` | M-2 test update |
| `tests/unit/v460/test_229_cleanup_counter_rename.py` | M-2 test update |
| `tests/unit/v460/test_277_magic_number_grounding.py` | M-2 test update |
| `docs/v460/322_phg_refactor_maker_price_god_object_split.md` | 新規作成 |
| `docs/v460/324_phg_fix_residual_tasks_and_regime_reuse.md` | 新規作成 |
| `docs/v460/index.md` | 322#/323#/324# 追加 |
