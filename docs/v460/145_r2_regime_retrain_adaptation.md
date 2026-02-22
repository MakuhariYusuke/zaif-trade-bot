# 145# R-2 レジーム適応型再学習

## §1 概要

142# §2 Phase R-2 (中期施策) の実装。
retrain パイプラインにレジーム情報を活用し、学習データ重み付けと再学習頻度を適応化。

| 施策 | ステータス | 変更ファイル |
|---|---|---|
| **R-2a** レジーム重み付き再学習 | ✅ 実装完了 | `retrain_scheduler.py` |
| **R-2b** レジーム別 retrain 頻度 | ✅ 実装完了 | `retrain_trigger.py`, `retrain_scheduler.py` |
| **R-2c** RegimeAdaptiveTrainer 設計再利用 | ✅ 分析完了 → R-3a に統合 | doc-only |

**前提**: R-1 完了 (R-1a offset / R-1b lot / R-1c reprice / R-1d timeout — 143#-144#)

---

## §2 R-2a: レジーム重み付き再学習 (sample_weight)

### 設計

LightGBM の `fit(sample_weight=...)` パラメータを活用し、現在のレジームに近いサンプルを upweight。

```
enriched DataFrame
    ↓ regime 列 (high_vol / trending / ranging / unknown)
    ↓
_compute_regime_sample_weights()
    ↓ config weights × current_regime boost × mean=1 正規化
    ↓
sample_weight: np.ndarray (len = n_valid_samples)
    ↓
    ├── _evaluate_wf_multi():  fit(sample_weight=weight[train_start:train_end])
    ├── _evaluate_wf_single(): fit(sample_weight=weight[:split_idx])
    └── retrain_model() final: fit(sample_weight=weight)
```

### 設定パラメータ

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `regime_weighting_enabled` | `False` | 安全デフォルト: 無効 |
| `regime_sample_weights` | `{"high_vol": 1.0, "trending": 1.0, "ranging": 1.0, "unknown": 1.0}` | レジーム → weight マッピング |
| `regime_current_boost` | `1.5` | 直近レジーム一致サンプルへの追加ブースト |
| `regime_current_lookback` | `10` | 直近 N 件から現在レジームを多数決推定 |
| `regime_weight_floor` | `0.1` | 重み最低値 (0 近接回避) |

### 正規化ロジック

1. `regime_sample_weights` から基本重みを付与
2. 直近 N 件多数決で current_regime を推定、一致サンプルに `regime_current_boost` 倍を適用
3. mean=1.0 に正規化 (学習率スケールへの影響を抑制)
4. `weight_floor` を再適用

### Observability

`retrain_model()` の結果 dict に `regime_weighting` キーで以下を記録:
```json
{
  "regime_weighting": "applied",
  "current_regime": "high_vol",
  "regime_distribution": {"high_vol": 15, "trending": 30, "ranging": 50, "unknown": 5},
  "weight_mean": 1.0123,
  "weight_std": 0.3456,
  "weight_min": 0.1000,
  "weight_max": 2.1500
}
```

SkipGate metadata にも `regime_weighting` を記録。

---

## §3 R-2b: レジーム別 retrain 頻度

### 設計

`RetrainTrigger.get_effective_interval()` にレジーム倍率を追加。
high_vol では市場変動が激しいため短い間隔で retrain し、ranging では安定なので長い間隔。

```
effective_interval = base_interval × backoff_factor × regime_multiplier
                     (cap: backoff_max_interval_sec)
```

### フロー

```
retrain_model()
    ↓ result["current_regime"] = 直近多数決 (R-2a と共有ロジック)
    ↓
run_scheduler()
    ↓ trigger.record_result(status, current_regime=...)
    ↓
trigger._current_regime ← 更新
    ↓
trigger.get_effective_interval()
    ↓ base × backoff × regime_multiplier
    ↓
time.sleep(effective_interval)
```

### 設定パラメータ

`RetrainTriggerConfig.regime_interval_multipliers`:

| レジーム | デフォルト倍率 | 効果 |
|---|---|---|
| `high_vol` | `0.5` | 半分の間隔 (高頻度 retrain) |
| `trending` | `0.75` | やや短め |
| `ranging` | `1.5` | 1.5 倍の間隔 (低頻度で十分) |
| `unknown` | `1.0` | デフォルト |

**例**: `base_interval_sec=3600` の場合
- high_vol → 30 分間隔
- ranging → 1.5 時間間隔
- backoff 2 回 + high_vol → 3600 × 4 × 0.5 = 7200 秒 (2h)

### YAML 外部化

```yaml
retrain:
  trigger_regime_interval_multipliers:
    high_vol: 0.5
    trending: 0.75
    ranging: 1.5
    unknown: 1.0
```

---

## §4 R-2c: RegimeAdaptiveTrainer 設計資産分析

### 分析結果

`ztb/training/components/regime_adaptive_trainer.py` の `RegimeAdaptiveTrainerMixin` は:
- **SAC 向け Mixin** (ent_coef, learning_rate, reward_scale 等)
- **MarketRegime** (comprehensive: 20+ alias) ベース

fill_test の retrain は:
- **LightGBM** (sample_weight, n_estimators 等)
- **FillTestRegime** (4 値: high_vol, trending, ranging, unknown)

### 再利用可能な設計概念

| RegimeAdaptiveTrainer 要素 | R-2 での対応 |
|---|---|
| regime_specific_params (per-regime ハイパラ) | ✅ R-2a regime_sample_weights |
| performance_tracking_window | ✅ R-2b 既存 OnlineMonitor (P1-12) |
| adaptation_frequency | ✅ R-2b regime_interval_multipliers |
| regime_classifier_config | ⬜ R-3a (MarketRegime → FillTestRegime マッピング) |

### 結論

R-2a/R-2b でコア概念は吸収済み。残る adapter 層 (MarketRegime → FillTestRegime 変換) は R-3a と一体であり、R-3a に統合して 148#+ で対応する。

---

## §5 テスト

| テストクラス | テスト数 | ファイル |
|---|---|---|
| `TestRegimeSampleWeights` | 7 | `test_retrain_hot_reload.py` |
| `TestRetrainTriggerRegimeInterval` | 7 | `test_136_p1_retrain_kill.py` |

テスト合計: **1270 passed** (144# 1263 + R-2a 7 新規)
- R-2b 7 テストは `test_136_p1_retrain_kill.py` の既存テストと合わせて 25 passed

### テスト項目 (R-2a)

1. `test_uniform_when_no_regime_col` — regime 列なし → 均一重み
2. `test_config_weights_applied` — config weights の正しい適用
3. `test_current_regime_boost` — 直近レジームブースト
4. `test_weights_normalized_mean_1` — 正規化 (mean ≈ 1.0)
5. `test_weight_floor_respected` — 最低重み尊重
6. `test_nan_regime_treated_as_unknown` — NaN → unknown
7. `test_default_config_disabled` — デフォルト config 検証

### テスト項目 (R-2b)

1. `test_high_vol_shortens_interval` — high_vol → 50% interval
2. `test_ranging_lengthens_interval` — ranging → 150% interval
3. `test_regime_with_backoff_combined` — バックオフ × レジーム倍率
4. `test_record_result_updates_regime` — regime 更新
5. `test_unknown_regime_default_multiplier` — 未知レジーム → 1.0x
6. `test_regime_capped_at_max_interval` — max_interval 上限
7. `test_default_config_has_regime_multipliers` — デフォルト config

---

## §6 リスクと緩和策

| リスク | 緩和策 |
|---|---|
| R-2a: 過学習 (特定レジームに偏る) | `regime_weighting_enabled=False` デフォルト、正規化、weight_floor |
| R-2a: レジーム分布の偏り | 正規化 (mean=1.0) で全体学習率を維持 |
| R-2b: high_vol で retrain 頻繁すぎ | `backoff_max_interval_sec` で上限あり |
| R-2b: 不正確なレジーム検出 | 直近 N 件多数決、unknown フォールバック |

---

## §7 YAML 設定例

```yaml
retrain:
  # R-2a: レジーム重み付き再学習
  regime_weighting_enabled: true
  regime_sample_weights:
    high_vol: 1.5
    trending: 1.2
    ranging: 0.8
    unknown: 0.5
  regime_current_boost: 1.5
  regime_current_lookback: 10
  regime_weight_floor: 0.1

  # R-2b: レジーム別 retrain 頻度
  trigger_regime_interval_multipliers:
    high_vol: 0.5
    trending: 0.75
    ranging: 1.5
    unknown: 1.0
```

---

## §8 次ステップ

- **Phase C: 24h 連続 run** — R-1 + R-2 有効化での効果測定
- **R-3a**: MarketRegime → FillTestRegime マッピング層 (R-2c 含む)
- **R-3b/c**: サブレジーム・OB 連携 (データ蓄積待ち)

---

## §9 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/ml/retrain_scheduler.py` | R-2a: `_compute_regime_sample_weights()`, config, fit 注入, R-2b: current_regime 検出, trigger 伝搬 |
| `ztb/ml/retrain_trigger.py` | R-2b: `regime_interval_multipliers`, `update_regime()`, `get_effective_interval()` 倍率 |
| `tests/unit/v460/test_retrain_hot_reload.py` | R-2a: `TestRegimeSampleWeights` (7 tests) |
| `tests/unit/v460/test_136_p1_retrain_kill.py` | R-2b: `TestRetrainTriggerRegimeInterval` (7 tests) |
| `docs/v460/145_r2_regime_retrain_adaptation.md` | 本ドキュメント |

---

## §10 144# レビュー修正 (lot 管理 + timeout label)

### §10.1 修正対象 (144# §8/§9 レビュー指摘)

| ID | 重要度 | 問題 | 修正内容 |
|---|---|---|---|
| §8-#2 | CRITICAL | `_regime_adjusted_lot()` → `_current_lot` 永続化 → 乗法的複利 (0.001→0.0015→0.00225→…) | `_current_lot` への書き戻し削除 |
| §8-#3 | HIGH | `if _order_lot > self._current_lot` — 片側のみ、shrink レジームで `_current_lot` が膨張したまま | 同上 (永続化自体を削除) |
| §9-#1 | CRITICAL | OrderMonitor の reprice が `self._current_lot` (膨張値) を使用 | `_monitor_fill_polling` に `order_lot` パラメータ追加、regime 調整済み lot を渡す |
| §9-#2 | HIGH | cancel_reason が `config.order_timeout_sec` (base) と比較 — regime 短縮時に "unknown" 誤判定 | `FillMonitorResult.effective_timeout` 追加、regime 実効値で比較 |

### §10.2 修正詳細

#### §8-#2/#3: lot 永続化除去

```python
# Before (144#): 乗法的複利 + 片側更新
_order_lot = self._regime_adjusted_lot()
if _order_lot > self._current_lot:
    self._current_lot = _order_lot  # → 次サイクルで base_lot = 膨張値

# After (145# fix): per-cycle のみ、_current_lot 不変
_order_lot = self._regime_adjusted_lot()
# _current_lot は balance_checker が管理する base lot のまま
```

**設計原則**: `_current_lot` は BalanceChecker が管理する「残高ベースのロット」。
regime 調整は per-cycle の `_order_lot` として一時的に適用し、永続化しない。

#### §9-#1: OrderMonitor へ正しい lot を渡す

```python
# Before: 膨張した _current_lot を渡す → reprice が過大ロットで発注
current_lot=self._current_lot

# After: per-cycle の regime 調整済み lot を渡す
_lot = order_lot if order_lot is not None else self._current_lot
current_lot=_lot
```

#### §9-#2: effective_timeout で cancel_reason 判定

```python
# Before: base timeout と比較 → regime 短縮時に "unknown" 誤判定
queue_wait >= self.config.order_timeout_sec

# After: regime 調整済み実効値と比較
queue_wait >= (_effective_timeout or self.config.order_timeout_sec)
```

### §10.3 テスト

| テストクラス | テスト数 | 検証内容 |
|---|---|---|
| `TestLotNoCompounding` | 3 | 乗法的複利なし、双方向更新、balance shrink 追従 |
| `TestMonitorReceivesOrderLot` | 1 | monitor に order_lot が渡される |
| `TestEffectiveTimeout` | 4 | effective_timeout フィールド、cancel_reason logic |
| `TestPreflightLotAlignment` (更新) | 1 | 永続化コード除去の source 検証 |

合計: **8 tests added** (+ 1 updated)

### §10.4 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/run_fill_test.py` | L805-811: 永続化除去, L592-630: `order_lot` param, L923-927: callsite, L987-993: effective_timeout |
| `scripts/v460/lib/fill_config.py` | `FillMonitorResult.effective_timeout` 追加 |
| `scripts/v460/lib/order_monitor.py` | `effective_timeout` を返却値に追加 |
| `tests/unit/v460/test_143_regime_utilization.py` | 8 tests 追加 + 1 test 更新 |
