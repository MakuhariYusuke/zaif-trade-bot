# 176# Trending方向×サイド別Offset Asymmetry + 横展開

> **日付**: 2026-02-27  
> **ベース**: `5df74e733` (175# Code Review Sweep #2)  
> **テスト**: 2197 passed (36 new), 0 failed

---

## 1. 概要

10日間のfill_test運用データ分析から、**大きな値動き（trending市場）で収益を逃している**構造的問題を特定。
根本原因3つを解明し、**施策A+B**を実装。加えて横展開で10ファイルの関連不整合を修正。

### 収益インパクト推定

| 問題 | 2/25実績 | 機会損失 |
|------|---------|---------|
| TRENDING sell skip バグ | 118件の sell ブロック | 推定 +178 bps |
| balance_forced_skip カスケード | 246件の deadlock | 推定 +82 bps |
| 方向非対称 offset 未実装 | trending_up buy +4.02 bps を 1.0x boost | 推定 +56 bps |

---

## 2. 根本原因分析

### 2.1 TRENDING (方向不明) sell skip バグ (施策A)

**発見**: `skip_sell_trending_up_only=true` 設定で `trending_up` のみブロックするはずが、
`TRENDING` (方向不明/undirected) もブロックしていた。

```python
# 旧コード (156# D-4): trending_down のみ通過
if _current_regime.value == "trending_down":
    _should_skip = False
# → TRENDING (undirected) は通過条件に該当せず、skip される

# 新コード (176# A): trending_up 以外は全通過
if _current_regime.value != "trending_up":
    _should_skip = False
# → TRENDING, trending_down は全て通過
```

**2/23 実績**: `TRENDING` (undirected) regime で 220 件の sell が不当にブロック。

### 2.2 balance_forced_skip カスケード

sell skip → `_last_side = "sell"` → 次サイクル buy 試行 → BTC 在庫未変化 (sell 未実行) → balance_forced_skip →
deadlock。2/23: 246 件の balance_forced_skip (全て regime=None) がこのカスケードにより発生。

**対策**: `skip_sell_trending: false` により sell skip 自体を廃止 → カスケード根絶。

### 2.3 方向非対称 offset 未実装 (施策B)

**2/25 反実仮想分析**: trending_up 中の fill PnL:
- **BUY**: 19 fills, 平均 **+4.02 bps** (順張り = 有利)
- **SELL**: 14 fills, 平均 **+1.51 bps** (逆張りでも利益!)

→ sell skip は完全に誤判断。offset 非対称化で防御しつつ両方向の fill を確保。

| regime | side | offset boost | 効果 |
|--------|------|-------------|------|
| trending_up | buy | ×0.7 (縮小) | 順張り fill を積極獲得 |
| trending_up | sell | ×1.8 (拡大) | 逆選択防御しつつ利確 |
| trending_down | buy | ×1.8 (拡大) | 逆選択防御 |
| trending_down | sell | ×0.7 (縮小) | 順張り fill を積極獲得 |

---

## 3. 実装内容

### 施策 A: TRENDING sell skip 除外 (HIGH)

| ファイル | 変更 |
|---------|------|
| [fill_loop_orchestrator.py](../../scripts/v460/lib/fill_loop_orchestrator.py) L733-742 | `== "trending_down"` → `!= "trending_up"` |

### 施策 B: 方向×サイド別 offset boost (HIGH)

| ファイル | 変更 |
|---------|------|
| [fill_config.py](../../scripts/v460/lib/fill_config.py) L116-125 | 4 新フィールド追加 (`trending_up_buy/sell`, `trending_down_buy/sell`) |
| [fill_config.py](../../scripts/v460/lib/fill_config.py) L965-968 | YAML パーサーマッピング追加 |
| [maker_price.py](../../scripts/v460/lib/maker_price.py) L305-340 | `_resolve_trending_boost()` 静的メソッド新設 (3段優先順位) |
| [maker_price.py](../../scripts/v460/lib/maker_price.py) L350-365 | `_apply_regime_boosts()` 方向別分岐 |
| [fill_test.yaml](../../configs/v460/fill_test.yaml) L112-124 | 方向別 boost 値設定 |
| [fill_test.yaml](../../configs/v460/fill_test.yaml) L327-332 | `skip_sell_trending: false` |

### Offset Boost 優先順位 (3段フォールバック)

```
1. trending_up_buy_offset_boost (方向×サイド別、最優先)
2. regime_trending_offset_boost_buy (サイド別)
3. regime_trending_offset_boost (共通値)
```

---

## 4. 横展開 — レビュー先回り修正

コードベース全体を走査し、trending 方向別処理が欠落していた箇所を一括修正。

### 4.1 config_hot_reload.py — 新パラメータ未登録 (HIGH)

4 つの方向別 offset boost パラメータが `_HOT_RELOADABLE_FIELDS` に未登録 →
本番 YAML 変更が即時反映されないバグ。

| ファイル | 変更 |
|---------|------|
| [config_hot_reload.py](../../scripts/v460/lib/config_hot_reload.py) L68-71 | 4 フィールド追加 |

### 4.2 ML 特徴量 regime_trending 情報損失 (MED)

`regime == "trending"` 完全一致により、156# D-4 以降のデータ (`trending_up`/`trending_down`)
で `regime_trending` 特徴量が常に 0 になる問題。5 ファイル × 同一パターン。

| ファイル | 変更 |
|---------|------|
| [skip_gate.py](../../scripts/v460/ml/skip_gate.py) L640-642 | `regime.startswith("trending")` |
| [feature_enricher.py](../../scripts/v460/ml/feature_enricher.py) L600-602 | 同上 (1/2) |
| [feature_enricher.py](../../scripts/v460/ml/feature_enricher.py) L764-766 | 同上 (2/2) |
| [data_loader.py](../../scripts/v460/ml/data_loader.py) L146-148 | 同上 (1/2) |
| [data_loader.py](../../scripts/v460/ml/data_loader.py) L207-209 | 同上 (2/2) |

**既存モデルとの互換性**: カラム名 `regime_trending` は変更なし (値域のみ拡大)。
学習済みモデルは trending=0 のデータで訓練されているため、次回 retrain で自然に改善。

### 4.3 YAML regime マップ方向別キー欠落 (LOW)

| ファイル | セクション | 変更 |
|---------|---------|------|
| [fill_test.yaml](../../configs/v460/fill_test.yaml) L281-286 | skip_gate `regime_thresholds` | `trending_up: -0.1`, `trending_down: -0.1` 追加 |
| [fill_test.yaml](../../configs/v460/fill_test.yaml) L454-460 | retrain `regime_sample_weights` | `trending_up: 0.8`, `trending_down: 0.8` 追加 |

### 4.4 retrain_scheduler.py デフォルト dict (LOW)

| ファイル | 変更 |
|---------|------|
| [retrain_scheduler.py](../../scripts/v460/ml/retrain_scheduler.py) L219-227 | `regime_sample_weights` に `trending_up/down` 追加 |
| [retrain_scheduler.py](../../scripts/v460/ml/retrain_scheduler.py) L1870-1878 | `regime_interval_multipliers` に `trending_up/down` 追加 |

### 4.5 compare_regime_ab.py — 分析漏れ (LOW)

| ファイル | 変更 |
|---------|------|
| [compare_regime_ab.py](../../scripts/v460/analysis/compare_regime_ab.py) L244 | `["ranging", "trending"]` → `["ranging", "trending", "trending_up", "trending_down"]` |

### 4.6 CHANGELOG 日付修正 (COSMETIC)

| ファイル | 変更 |
|---------|------|
| [CHANGELOG.md](../../CHANGELOG.md) L9 | 174# 日付 `2026-03-01` → `2026-02-27` (未来日付修正) |

---

## 5. 影響を受ける / 受けないコンポーネント

### 既に方向別対応済み (変更不要)
- `sell_dynamic_kill regime_thresholds` — `trending_up: -0.3`, `trending_down: -1.0`
- `buy_dynamic_kill regime_thresholds` — `trending_down: -0.5`, `trending_up: -1.5`
- `stopgap_health.py` — `.startswith("trending")` で正しく対応
- `skip_gate_evaluator._valid_regimes` — 6 種全て含む
- `side_regime_dashboard.py target_regimes` — 4 種全て含む

### 将来の最適化候補 (今回は見送り)
- `trending_up_buy_offset_boost` / `trending_down_sell_offset_boost` の値を A/B テストで微調整
- ML 特徴量に `regime_trending_up`, `regime_trending_down` の個別カラム追加 (再学習とセット)

---

## 6. テスト

### 新規テスト (36 件)

| クラス | テスト数 | カバー範囲 |
|--------|---------|-----------|
| `TestTrendingSkipExclusion` | 3 | 施策A: AST 解析で `!= "trending_up"` 確認 |
| `TestDirectionalBoostConfig` | 6 | 施策B: config フィールド存在・型・YAML パース |
| `TestResolveTrendingBoost` | 8 | 施策B: 3段優先順位ロジック全パス |
| `TestApplyRegimeBoostsDirectional` | 6 | 施策B: 統合テスト (offset 値検証) |
| `TestHotReloadDirectionalFields` | 2 | 横展開: hot-reload 登録確認 |
| `TestMLFeatureTrendingDirection` | 5 | 横展開: ML 特徴量 startswith 検証 |
| `TestYAMLRegimeDirectionKeys` | 4 | 横展開: YAML マップキー存在確認 |
| `TestChangelogDateConsistency` | 1 | 横展開: 日付整合性検証 |
| **合計** | **36** | |

### 回帰テスト結果

```
2197 passed, 0 failed, 14 warnings
```

---

## 7. 変更ファイル一覧

| # | ファイル | 施策 | 優先度 |
|---|---------|------|--------|
| 1 | `scripts/v460/lib/fill_loop_orchestrator.py` | A | HIGH |
| 2 | `scripts/v460/lib/fill_config.py` | B | HIGH |
| 3 | `scripts/v460/lib/maker_price.py` | B | HIGH |
| 4 | `scripts/v460/lib/config_hot_reload.py` | 横展開 | HIGH |
| 5 | `scripts/v460/ml/skip_gate.py` | 横展開 | MED |
| 6 | `scripts/v460/ml/feature_enricher.py` | 横展開 | MED |
| 7 | `scripts/v460/ml/data_loader.py` | 横展開 | MED |
| 8 | `scripts/v460/ml/retrain_scheduler.py` | 横展開 | LOW |
| 9 | `scripts/v460/analysis/compare_regime_ab.py` | 横展開 | LOW |
| 10 | `configs/v460/fill_test.yaml` | A+B+横展開 | HIGH |
| 11 | `CHANGELOG.md` | 日付修正 | COSMETIC |
| 12 | `tests/unit/v460/test_176_trending_offset_asymmetry.py` | テスト | — |
