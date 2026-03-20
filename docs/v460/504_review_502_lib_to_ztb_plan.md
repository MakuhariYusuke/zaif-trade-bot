# 504# レビュー: 502# `scripts/v460/lib` → `ztb` 移行計画

**種別**: review  
**日付**: 2026-03-20  
**対象**: 502# PHG: `scripts/v460/lib` → `ztb` 移行 / オブジェクト分割 実行計画  
**ステータス**: レビュー完了

---

## 1. 総合評価

502# は `scripts/v460/lib` の domain logic を `ztb` へ段階的に移行する計画書として、**方向性・構造ともに妥当**。
3 分類（lib 残留 / 低リスク移行 / 分割先行）は正しい判断基準に基づいている。

ただし実コードベースを検証した結果、**行数の不正確さ、依存グラフの過小評価、既存 ztb 構造との配置衝突**が複数確認された。
以下にデータに基づく是正事項を記す。

---

## 2. ファクトチェック: 行数

502# に記載された行数と実測値を比較:

| ファイル | 502# 記載 | 実測 | 差異 |
|---------|----------|------|------|
| `fast_fill_defense.py` | 315 | 314 | ≈一致 |
| `param_adapter.py` | 312 | 311 | ≈一致 |
| `lot_sizer.py` | 446 | 445 | ≈一致 |
| `regime_detector.py` | 684 | 683 | ≈一致 |
| `sac_common.py` | 495 | 494 | ≈一致 |
| `maker_price.py` | 1092 | 1091 | ≈一致 |
| `skip_gate_evaluator.py` | 867 | 866 | ≈一致 |
| `order_monitor.py` | 646 | 645 | ≈一致 |
| `adaptation_engine.py` | 635 | 634 | ≈一致 |
| `fill_config.py` | 874 | 873 | ≈一致 |

**評価**: 行数は概ね正確（±1 の誤差は末尾改行の数え方の差）。ファクトとして問題なし。

---

## 3. 重要な見落とし・リスク

### 3.1 依存グラフの過小評価

502# は低リスク移行候補を「依存が薄い」としているが、**実際の被参照数は大きく異なる**:

| ファイル | 502# 評価 | 実被参照数 | 主要参照元 |
|---------|----------|-----------|-----------|
| `fast_fill_defense.py` | 「依存薄い」 | **23 ファイル** | maker_price, conftest, テスト 20 件 |
| `param_adapter.py` | 「汎用」 | 3 ファイル | adaptation_engine, テスト 2 件 |
| `lot_sizer.py` | 「fill test 以外でも使える」 | 4 ファイル | adaptation_engine, テスト 3 件 |
| `regime_detector.py` | 「判定ロジック自体は reusable」 | **30 ファイル** | maker_price, order_monitor, adaptation_engine, テスト 20+ 件 |
| `sac_common.py` | 「一部 shared 済み」 | 8 ファイル | retrain_scheduler, テスト 4 件 |

**問題点**:
- `fast_fill_defense.py` は 23 ファイルから参照されており、「低リスク移行」とは言い難い。特に `tests/unit/v460/conftest.py` から import されており、テストフィクスチャの根本に存在する
- `regime_detector.py` は **30 ファイルから参照** — `lib` 内だけでも `maker_price`, `order_monitor`, `maker_microstructure`, `maker_regime_boost`, `adaptation_engine`, `fill_record_helpers` の 6 モジュールが依存。移行時の import パスの一斉変更は大きなリファクタリングになる

**是正案**: 
- façade (re-export) 戦略は 502# に記載済みだが、**特に `fast_fill_defense` と `regime_detector` は façade 必須**と明記すべき
- `param_adapter` と `lot_sizer` は被参照が少なく、本当に低リスクと言える

### 3.2 既存 `ztb` import 違反

ztb → scripts.v460.lib の逆方向 import が **1 件確認**:

```
ztb/metrics/fill_quality.py:1597: from scripts.v460.lib.cancel_reasons import AUDIT_CANCEL_REASONS
```

**影響**:
- この違反は `cancel_reasons.py` (212 行) を **最初に移行しなければならない**ことを意味する
- 502# の Phase 1 リストに `cancel_reasons.py` が含まれていない — **欠落**

**是正案**: Phase 0 か Phase 1 の冒頭に `cancel_reasons.py` → `ztb/trading/constants/` への移行を追加

### 3.3 既存 `ztb` namespace との配置衝突

502# の推奨配置と既存の `ztb` 構造の整合性:

| 502# 推奨配置 | 既存 ztb 構造 | 衝突/問題 |
|--------------|--------------|----------|
| `ztb/trading/execution/fast_fill_defense.py` | `ztb/trading/execution/` に既存 3 ファイル (model.py, pseudo_hft.py, realistic.py) | **意味的衝突**: 既存は simulation/backtest 用 execution。fill_test の FFD とは文脈が異なる |
| `ztb/trading/sizing/lot_sizer.py` | `ztb/trading/` に `sizing/` なし | 新規作成が必要だが問題なし |
| `ztb/trading/sizing/param_adapter.py` | 同上 | 同上 |
| `ztb/trading/regime/regime_detector.py` | `ztb/trading/` に `regime/` なし | 新規作成が必要。ただし `ztb/trading/live/` 配下にすべきか要検討 |
| `ztb/ml/sidecar_signal_io.py` | `ztb/ml/` に既存の `skip_gate.py`, `skip_gate_features.py` | 問題なし |

**是正案**:
- `fast_fill_defense.py` は `ztb/trading/execution/` ではなく `ztb/trading/risk/fast_fill_defense.py` が適切（リスク管理ロジック）
- あるいは `ztb/trading/defense/` として独立 namespace

### 3.4 `fill_config.py` 分割の設計判断

502# は `fill_config.py` を「分割先行候補」としているが、**329# で既に 4 分割が実施済み**:

| ファイル | 行数 | 役割 |
|---------|------|------|
| `fill_config.py` | 873 | schema (dataclass 定義) + from_yaml façade |
| `fill_config_parser.py` | 1117 | YAML → kwargs パース |
| `fill_config_validation.py` | 431 | バリデーションルール |
| `fill_config_results.py` | 124 | 結果 dataclass (FillMonitorResult 等) |

502# が「schema / defaults / yaml mapping / result view が混在」と評価しているのは **329# 以前の状態を見ている可能性がある**。現状 `fill_config.py` の本体は schema 定義に集中しており、これ以上の分割は over-engineering の恐れがある。

**是正案**: `fill_config.py` は Phase 3 分割リストから除外するか、「329# で分割済み。残るは schema 定義 873 行だが、フィールド数が多いだけで責務は単一。追加分割は不要」と記載

### 3.5 `sac_common.py` の import 依存

502# は `sac_common.py` を低リスク移行候補としているが:
- `ztb.utils.memory_utils` に依存あり → `ztb` 内依存なので移行にプラス
- `numpy`, `pandas` 依存あり → 問題なし
- `scripts/v460/ml/sac_retrain_scheduler.py` から import → 移行後は `ztb` import に変更で問題なし

**評価**: sac_common は本当に低リスク。ただし `ztb/training/sac.py` (既存) と `sac_common` 移行先の namespace 整理が必要

### 3.6 lib ファイル総数と分類の網羅性

`scripts/v460/lib/` には **77 .py ファイル** が存在するが、502# で明示的に分類されているのは約 25 ファイル。残り 52 ファイルについて:

| 未分類の重要ファイル | 行数 | 推奨分類 |
|-------------------|------|---------|
| `ab_judgment.py` | 1178 | 分割先行 (God Object 級) |
| `cycle_gate_aggregator.py` | 935 | lib 残留 (orchestration 寄り) |
| `stopgap_health.py` | 849 | lib 残留 (v460 運用固有) |
| `config_hot_reload.py` | 825 | `ztb` 移行候補 (reusable) |
| `daily_drawdown_guard.py` | 667 | `ztb/trading/risk/` 移行候補 |
| `maker_risk_guards.py` | 484 | → `maker_price.py` 分割の一部として整理 |
| `phantom_position_guard.py` | 483 | `ztb/trading/risk/` 移行候補 |
| `bayesian_regime_filter.py` | 576 | `ztb/trading/regime/` 移行候補 (regime_detector と同時) |
| `offset_pipeline.py` | 352 | lib 残留 (v460 固有 pipeline) |
| `cross_venue_lead_lag.py` | 345 | `ztb` 移行候補 (reusable) |
| `cancel_reasons.py` | 212 | **Phase 0 必須** (ztb 違反解消) |

**是正案**: 502# に全 77 ファイルの分類表を追記すべき。少なくとも 500 行超のファイルは全て明示的に分類する

---

## 4. アーキテクチャ判断への異論

### 4.1 `ztb.trading.fill_test` は作らない — 同意

502# のこの判断は正しい。`fill_test` は v460 オーケストレーションであり、canonical domain ではない。

### 4.2 Phase 順序の問題

502# の Phase 順序:
```
Phase 0: 事前固定 → Phase 1: 低リスク移行 → Phase 2: SAC shared → Phase 3: God Object 分割 → Phase 4: import 収束
```

**問題**: `cancel_reasons.py` → `ztb` 移行が Phase 0 に含まれていない。これは既存の ztb import 違反であり、Phase 1 より先に解消すべき。

**是正案**:
```
Phase 0: 事前固定 + cancel_reasons.py 移行 (ztb 違反解消)
Phase 1: 低リスク移行 (param_adapter, lot_sizer, sac_common) ← fast_fill_defense と regime_detector は中リスクに格上げ
Phase 1.5: fast_fill_defense, regime_detector 移行 (façade 必須、テスト影響大)
Phase 2: SAC shared runtime
Phase 3: God Object 分割
Phase 4: import 収束
```

### 4.3 façade 戦略の具体性不足

502# は「thin wrapper に縮める」と記載しているが、具体的な façade パターンが示されていない。
推奨パターン:

```python
# scripts/v460/lib/regime_detector.py (façade after migration)
"""Compatibility shim — canonical は ztb.trading.regime.regime_detector."""
from ztb.trading.regime.regime_detector import (  # noqa: F401
    FillTestRegime,
    RegimeDetector,
    RegimeResult,
)
```

この 1 ファイル化により、既存の 30 ファイルの import を即座に壊さず移行できる。

### 4.4 テスト戦略の不足

502# は「守るべきテスト」を 5 件列挙しているが、影響の大きい `conftest.py` への言及がない。
`tests/unit/v460/conftest.py` は `fast_fill_defense` を import しており、これが壊れると **全 v460 テストが失敗する**。

---

## 5. Phase 3 分割設計の補足意見

### 5.1 `maker_price.py` (1091 行)

502# は「pricing / guards / cross-venue / microstructure が混在」と評価。
ただし **329# パターンの再適用** で考えると:
- `maker_risk_guards.py` (484 行) は **既に分割済み** — 502# が見落としている可能性
- `maker_microstructure.py` (361 行) も **既に分離済み**
- 残る本体 1091 行は pricing core + inventory skewing + OB 操作

**是正案**: 実態を再確認の上、「maker_price.py は 3 分割済み (maker_risk_guards, maker_microstructure, 本体)。さらなる分割は pricing core vs inventory skewing の 2 分割が妥当」

### 5.2 `ab_judgment.py` (1178 行) — 502# 未記載

502# に含まれていないが、lib 内で **`fill_cycle_executor.py` (1374) に次ぐ 2 番目の大ファイル**。
AB テスト判定ロジックが集約されており、分割候補に含めるべき。

### 5.3 `order_monitor.py` (645 行) の分割案

502# は「polling / stale judgement / retry / logging が混在」と評価。適切。
推奨分割:
- `ztb/trading/execution/order_polling.py` — ポーリングループの core logic
- `scripts/v460/lib/order_monitor.py` — v460 固有の stale reprice / chase 配線

---

## 6. 実行優先度の再提案

### 即時実行可能 (低リスク、高価値)

1. **`cancel_reasons.py` → `ztb/trading/constants/cancel_reasons.py`** (ztb 違反解消、212 行)
2. **`param_adapter.py` → `ztb/trading/sizing/param_adapter.py`** (被参照 3 件のみ)
3. **`lot_sizer.py` → `ztb/trading/sizing/lot_sizer.py`** (被参照 4 件のみ)

### 次フェーズ (façade 必須)

4. **`sac_common.py` → `ztb/training/sac/runtime.py`** (被参照 8 件)
5. **`fast_fill_defense.py` → `ztb/trading/risk/fast_fill_defense.py`** (被参照 23 件、conftest 注意)
6. **`regime_detector.py` + `bayesian_regime_filter.py` → `ztb/trading/regime/`** (被参照 30 件、最大リスク)

### 設計先行 (分割後に移行)

7. `maker_price.py` 追加分割設計
8. `skip_gate_evaluator.py` 分割設計
9. `order_monitor.py` 分割設計

---

## 7. 結論

| 項目 | 評価 |
|------|------|
| **方向性** | ✅ 正しい。3 分類の基準も妥当 |
| **行数データ** | ✅ 正確 (±1) |
| **dependency 分析** | ⚠️ 不足。被参照数を実測すべき |
| **Phase 順序** | ⚠️ `cancel_reasons.py` Phase 0 追加必須 |
| **配置先** | ⚠️ `execution/` vs `risk/` の意味的衝突あり |
| **fill_config 分割** | ❌ 329# で分割済み。Phase 3 から除外推奨 |
| **未分類ファイル** | ⚠️ 77 中 52 が未分類。大ファイル (`ab_judgment` 等) の明示分類が必要 |
| **テスト戦略** | ⚠️ `conftest.py` 影響の明記が必要 |
| **façade パターン** | ⚠️ 具体コード例が必要 |

**総合**: 計画書として 7 割完成。上記の是正を反映すれば実行可能な水準に達する。
