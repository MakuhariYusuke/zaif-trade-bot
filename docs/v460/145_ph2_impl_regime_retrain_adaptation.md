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

---

## §11 追補レビュー (145# 潰し込み)

145# 実装について、コード実体・テスト実行結果・境界条件を再点検した。
軽微事項も含め、現時点での追加指摘を以下に整理する。

### §11.1 重大度付き指摘

| # | 重大度 | 対象 | 指摘 | 推奨対応 |
|---|---|---|---|---|
| 1 | HIGH | `scripts/v460/ml/retrain_scheduler.py` | `regime_weighting_enabled=True` かつ `len(X_valid)==0` のとき、`_compute_regime_sample_weights()` が空配列に対して `np.min/np.max` を実行し `ValueError` で落ちる (`_compute_regime_sample_weights` の統計計算部)。`retrain_model()` 側では min samples チェックより前に当該関数を呼んでいるため、`skipped` ではなく `error` になり得る。 | `len(valid_index)==0` で早期 return するガードを追加し、`weight_stats` は `{"regime_weighting":"uniform","reason":"empty_valid_index"}` などを返す。併せて `test_empty_valid_index_no_crash` を追加。 |
| 2 | MEDIUM | `scripts/v460/ml/retrain_scheduler.py` | `regime_current_lookback=0` の設定時、`if len(X_valid) >= _regime_lookback` が空データでも真になり、`value_counts().index[0]` で `IndexError` を起こす余地がある (`current_regime` 推定部)。 | `lookback = max(1, safe_to_int(...))` に正規化、または `len(X_valid) > 0` を先に判定してから `index[0]` を参照する。 |
| 3 | MEDIUM | `ztb/ml/retrain_trigger.py` | `get_effective_interval()` が `int(base * regime_mul)` をそのまま返すため、設定値次第で `0` 秒 interval が発生しうる (特に小さい `base_interval_sec` や `regime_mul<=0`)。結果として busy-loop 化のリスク。 | `interval = max(1, int(base * regime_mul))` を適用し、`regime_interval_multipliers` のロード時に `>0` バリデーションを追加。 |
| 4 | MEDIUM | `tests/unit/v460/test_113_resilience.py`, `tests/unit/v460/test_139_review_fixes.py` | `tests/unit/v460` 全体実行で 3 件失敗。失敗原因は実装バグというより、`cancel_reason` 文字列をソース文字列として固定比較しているテストが、`cancel_reasons` 定数参照化 (`CR.*`) 後のコードと齟齬を起こしている点。 | ソース文字列の直比較を廃止し、`CR` 定数の利用確認または実行挙動 (生成 FillRecord の `cancel_reason`) を検証する形へ更新。 |
| 5 | LOW | `docs/v460/145_r2_regime_retrain_adaptation.md` | §5 の「1270 passed」記述が現状と乖離。現時点の全体実行は `1282 passed, 3 failed`、145関連の対象 3 ファイルは `147 passed`。 | 総数の固定値を記述する場合は実行日時を併記。推奨は「対象テスト群のみ件数明記 + 全体は latest CI 参照」に変更。 |
| 6 | LOW | `ztb/trading/live/exchanges/coincheck/adapter.py`, `ztb/trading/live/exchanges/bitflyer/adapter.py` | 継承構造が非対称。`BitFlyerAdapter` は `BaseExchangeAdapter` 継承だが、`CoincheckAdapter` は `IBroker` 直実装で重複責務が残る。今後の保守で差分ドリフトを生みやすい。 | `CoincheckAdapter` を段階的に `BaseExchangeAdapter` へ寄せる移行タスクを作成し、`_check_rate_limit`/dry-run 共通部を統合。 |
| 7 | LOW | `ztb/trading/live/exchanges/bitflyer/adapter.py` | `_make_request()` に重複 docstring が残存し、レビュー/保守のノイズになっている。 | docstring を 1 つに整理して差分可読性を改善。 |

### §11.2 テスト実行メモ (今回)

- 145 関連の主要テスト:
  - `tests/unit/v460/test_retrain_hot_reload.py`
  - `tests/unit/v460/test_136_p1_retrain_kill.py`
  - `tests/unit/v460/test_143_regime_utilization.py`
  - 結果: **147 passed**
- `tests/unit/v460` 全体:
  - 結果: **1282 passed, 3 failed**
  - 失敗は §11.1 #4 のテスト実装側の齟齬が主因。

### §11.3 優先順 (次の1手)

1. #1 の empty sample crash を先に潰す (再学習停止リスクの除去)。
2. #4 の brittle test を修正し、`tests/unit/v460` 全緑を回復する。
3. #2/#3 の境界値ガードを追加して設定変更耐性を上げる。

---

## §12 144# レビュー残項目修正 (第2バッチ)

§10 で §8-#2/#3, §9-#1/#2 を修正、本セクションで §8-#1/#6, §9-#3/#4/#5/#6/#7 を修正。
§11 #4 (brittle test) も本バッチで解消。

### §12.1 修正一覧

| 144#項番 | 修正内容 | 変更ファイル | 影響度 |
|---|---|---|---|
| §8-#1 | **preflight-lot alignment**: `BalanceChecker.check()` に `regime_mult` パラメータ追加。preflight が実効ロット (base × regime倍率) で残高判定し、自動縮小もレジーム倍率を考慮 | `balance_checker.py`, `run_fill_test.py` | HIGH |
| §8-#6 | **regime config 値域バリデーション**: `FillTestConfig.__post_init__` で `regime_timeout_multipliers > 0`, `regime_lot_multipliers > 0`, `\|regime_reprice_adjustments\| ≤ 10` を検証 | `fill_config.py` | MEDIUM |
| §9-#3 | **OB 形式正規化**: `ob_utils.py` 新規作成 (tuple/object 両対応)。SkipGate の `.price/.quantity` アクセスを `extract_price()` / `depth_volume()` に修正 — OB features の silent failure を解消 | `ob_utils.py` (NEW), `skip_gate_evaluator.py` | HIGH |
| §9-#4 | **SkipGate lot 整合**: `_evaluate_skip_gate()` に `order_lot` パラメータ追加。regime-adjusted lot を SkipGate に渡して FillRecord の `order_quantity` を正確化 | `run_fill_test.py` | MEDIUM |
| §9-#5/7 | **DRY ヘルパ**: `_make_skip_record()` + `_new_cycle_id()` を FillTestRunner に追加。11箇所の散在 FillRecord 構築を一元化 | `run_fill_test.py` | MEDIUM |
| §9-#6 | **cancel_reason 定数**: `cancel_reasons.py` 新規作成 (AUDIT/EXEC/GUARD/ORDERBOOK)。`fill_quality.py` 等から参照。文字列ドリフトリスクを排除 | `cancel_reasons.py` (NEW), `fill_quality.py`, `run_fill_test.py` | MEDIUM |
| §11-#4 | **brittle test 修正**: ソース文字列直比較テスト (088/113/139) を CR.定数/`_make_skip_record` ベースに更新 | `test_088_features.py`, `test_113_resilience.py`, `test_139_review_fixes.py`, `test_143_regime_utilization.py` | LOW |

### §12.2 新ファイル

| ファイル | 目的 |
|---|---|
| `scripts/v460/lib/ob_utils.py` | OB 正規化: `extract_price()`, `extract_size()`, `best_bid_ask()`, `depth_volume()` |
| `scripts/v460/lib/cancel_reasons.py` | cancel_reason 定数集約 (AUDIT/EXEC/GUARD/ORDERBOOK カテゴリ) |
| `tests/unit/v460/test_145_structural_fixes.py` | 本バッチの全テスト (53 tests) |

### §12.3 `_regime_lot_multiplier()` 抽出

`_regime_adjusted_lot()` からレジーム倍率取得ロジックを分離。
`run_continuous()` の preflight で `_regime_lot_multiplier()` を呼び、
`_check_balance_for_side(regime_mult=...)` 経由で BalanceChecker に渡す。

```
run_continuous:
  _regime_mult = self._regime_lot_multiplier()     # 1.0 / 1.5 / 0.8 etc.
  _check_balance_for_side(next_side, regime_mult=_regime_mult)
    → BalanceChecker.check(regime_mult=_regime_mult)
      → _check_sell: effective_lot = base × mult, shrink = btc_free / mult
      → _check_buy:  jpy_needed = base × mult × price × margin
```

### §12.4 テスト

| テストクラス | テスト数 | 検証内容 |
|---|---|---|
| `TestObUtilsExtractPrice` | 3 | tuple / list / object 形式 |
| `TestObUtilsExtractSize` | 4 | tuple / list / object(.quantity) / object(.size) |
| `TestObUtilsBestBidAsk` | 3 | 正常OB / 空OB / None |
| `TestObUtilsDepthVolume` | 3 | full/partial/default depth |
| `TestCancelReasons` | 5 | frozenset 整合 / EXEC/GUARD/ORDERBOOK存在 / fill_quality連携 |
| `TestRegimeConfigValidation` | 7 | timeout/lot/reprice 各フィールドの NG 値 + 有効値 |
| `TestPreflightRegimeMult` | 7 | sell/buy regime_mult 反映, 自動縮小, 復元, デフォルト=1.0 |
| `TestRegimeLotMultiplier` | 4 | 倍率辞書なし / detector なし / trending / unknown fallback |
| `TestNewCycleId` | 3 | prefix なし/あり形式 / uniqueness |
| `TestMakeSkipRecord` | 6 | 基本フィールド / auto cycle_id / custom / default lot / extra kwargs |
| `TestSkipGateLotConsistency` | 2 | signature + callsite ソース検証 |
| `TestCheckBalanceAcceptsRegimeMult` | 2 | signature + run_continuous ソース検証 |
| `TestSkipGateObFormat` | 2 | extract_price 使用確認 / .price 直アクセス不在確認 |
| `TestDataQualityFillRecord` (更新) | 2 | _make_skip_record 内の run_id/git_sha + 使用回数 |
| `TestCircuitBreaker` (更新) | 1 | CR.CIRCUIT_BREAKER_OPEN 定数確認 |
| `TestRunContinuous` (更新) | 4 | CR.定数ベースのソース検証 |
| `TestPreflightLotAlignment` (更新) | 1 | lot_floor → _order_lot 順序検証 |
| `TestMinLotUnification` (更新) | 1 | _regime_lot_multiplier バインド修正 |

新規 53 tests + 既存更新 9 tests → **合計 1339 passed** (全緑)

### §12.5 残課題 (§13 スコープ)

§11.1 の #1 (empty sample crash), #2 (lookback=0), #3 (busy-loop interval) は
本バッチのスコープ外。次回 §13 で対応予定。

---

## §13 144# §10 / 145# §11 残項目修正

§12.5 で保留した境界値ガード + 144# §10 の追加指摘を修正。

### §13.1 修正一覧

| 指摘元 | 修正内容 | 変更ファイル | 影響度 |
|---|---|---|---|
| §11-#1 HIGH | **empty valid_index crash guard**: `_compute_regime_sample_weights()` 先頭で `len(valid_index)==0` 判定 → 早期 return (空 ndarray + uniform メタ)。`np.min/np.max` の `ValueError` を回避 | `retrain_scheduler.py` | HIGH |
| §11-#2 MEDIUM | **lookback=0 IndexError guard**: `_compute_regime_sample_weights()` + `retrain_model()` 内で `max(1, safe_to_int(...))` を適用。`value_counts().index[0]` の `IndexError` を防止。空 Series の場合も `"unknown"` フォールバック追加 | `retrain_scheduler.py` | MEDIUM |
| §11-#3 MEDIUM | **busy-loop interval guard**: `get_effective_interval()` で `max(1, int(base * max(regime_mul, 0.0)))` を適用。`RetrainTriggerConfig.__post_init__` で `regime_interval_multipliers > 0` + `base_interval_sec >= 1` をバリデーション | `retrain_trigger.py` | MEDIUM |
| §10.2-#2 / §11-#7 | **bitflyer duplicate docstring**: `_make_request()` の 3 重 docstring を 1 つに整理 | `bitflyer/adapter.py` | LOW |
| §10.1-#1 / §11-#6 | **CoincheckAdapter 継承**: 現状 `IBroker` 直実装。段階的移行は §14 以降。テストで継承構造の差異を記録 | (doc-only + test) | LOW (planned) |

### §13.2 見送り事項 (§14+ スコープ)

| 指摘元 | 内容 | 理由 |
|---|---|---|
| §10.1-#1 | CoincheckAdapter → BaseExchangeAdapter 継承移行 | 853 行クラスの大規模リファクタ。段階的移行タスクとして計画。本番影響リスク大 |
| §10.1-#2 | FillTestRunner 分割 (AbstractCycleRunner) | 2k 行超の設計変更。R-2 完了後の構造整備フェーズで対応 |
| §10.1-#3 | MarketDataAccessorBase 導入 | ob_utils.py で部分的に対応済み。追加抽象化は次期アーキテクチャ整備 |
| §11-#5 | doc テスト件数更新 | 本セクションで記載 |

### §13.3 新規テストファイル

`tests/unit/v460/test_145_s13_boundary_guards.py` — 18 tests

| テストクラス | テスト数 | 検証内容 |
|---|---|---|
| `TestEmptyValidIndexNocrash` | 3 | 空 index / numpy エラーなし / 1 件 index |
| `TestLookbackZeroGuard` | 4 | lookback=0 / IndexError 回避 / 負値クランプ / ソース検証 |
| `TestBusyLoopIntervalGuard` | 7 | small base / tiny mul / 正常値 / config reject 0/負値/base=0 / ソース検証 |
| `TestBitflyerDocstringCleanup` | 1 | 重複 docstring 不在確認 |
| `TestCoincheckAdapterInheritance` | 3 | IBroker 実装 / BaseExchangeAdapter 継承 / 共通メソッド |

### §13.4 テスト結果

- 新規: 18 passed
- 全体: **1357 passed** (全緑)
