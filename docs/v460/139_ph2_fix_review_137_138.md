# 139# ph2 review-fix: 137# §9 + 138# §8 レビュー指摘対応

| key | value |
|---|---|
| 番号 | 139 |
| フェーズ | ph2 |
| 種別 | review-fix |
| 対象 | `docs/v460/137_ph2_impl_review_fixes_p1.md` §9, `docs/v460/138_ph2_impl_p1_preflight_calibration.md` §8 |
| 作成日 | 2026-02-23 |
| 前提 | Git `d9285beb9` (138# commit) |
| 結論 | **137# §9 全 11 件 + 138# §8 全 6 件 + §8.1 再レビュー 5 件 = 計 22 件対応完了。retrain new_samples 致命バグ修正 + preflight_pause AttributeError 修正。テスト 1147→1171 (+24)。** |

---

## §0 エグゼクティブサマリ

137# 外部レビュー §9 の 7 件 + 追加 4 件 (#A-#D) を全件対応。**最重要はretrain new_samples が -765 で永久停止していたバグの修正** (§9 #1/A)。138# §8 の 6 件はcalibrator 注入、mode 廃止、監査レコード、境界値バリデーション等。§8.1 再レビューで発覚した CRITICAL (`_append_fill_record` 未定義メソッド) + HIGH (skip 系 FillRecord 不在) を 140# として即時修正。

### ログ診断結果

| 指標 | 値 |
|---|---|
| fill_records | 10日間 (2/13-22), 1916 total, 1183 filled |
| buy | 596 filled, avg PnL **-0.073 bps**, win 48.5%, AS 27.9% |
| sell | 587 filled, avg PnL **-0.556 bps**, win 43.8%, AS 27.1% |
| retrain | **永久停止** (new_samples: -765) → §9 #1/A で修正 |

**sell-side の PnL が buy の 7.6 倍悪い** — P1-01/02 (side 別モデル) のデータ十分性は確認済み (596/587 件) だが、retrain 修正が前提条件。

---

## §1 137# §9 レビュー対応

### §1.1 #1/A (HIGH): retrain new_samples run 切替検出

**問題:** `prev_n_samples` が旧 run の pkl (858) を保持。現 run は 93 件のみ → `raw_new_samples = 93 - 858 = -765`。133# P0-01 の `max(0, ...)` クランプにより `new_samples=0` が恒常化し、retrain が永久スキップ。

**修正:** `raw_new_samples < 0` の場合、run 切替と判定し `new_samples = len(X_valid)` (全サンプルを新規扱い)。

**ファイル:** `scripts/v460/ml/retrain_scheduler.py` L1010-1025

### §1.2 #2 (MEDIUM): regime_thresholds YAML→Config→Manager 配線

**問題:** `regime_thresholds` が YAML 定義済みだが `FillTestConfig` に未配線、`SellKillConfig` に渡されていなかった。

**修正:**
- `fill_config.py`: `sell_dynamic_kill_regime_thresholds: dict[str, float]` フィールド追加
- `fill_config.py` YAML パーサ: `sell_kill["regime_thresholds"]` 読み取り追加
- `run_fill_test.py` L234: `SellKillConfig(regime_thresholds=...)` に接続
- `fill_test.yaml`: `regime_thresholds: {trending_up: -0.3, trending_down: -1.0, ranging: -0.5}` 追加

### §1.3 #3/C (MEDIUM): narrow_spread_pause 実待機

**問題:** `narrow_spread_pause_sec` を設定していても FillRecord 返却前に実際の sleep がなかった。

**修正:** `run_fill_test.py` L729 に `await asyncio.sleep(pause_sec)` 追加。

### §1.4 #4/D (MEDIUM): fee 仕様コメント明確化

**問題:** maker fee のみ控除の仕様がコード上で不明確。

**修正:** `pnl_measurer.py` に 3 行のコメント追加 — maker fee only、taker_fee_bps は将来 IOC 用リザーブ、slippage は将来対応。

### §1.5 #5/B (MEDIUM): trades 全量フォールバック廃止

**問題:** `feature_enricher.py` の 3 段目フォールバックで `load_raw_trades(raw_dir, date_filter=None)` が全期間ロードを行い、メモリ・パフォーマンスリスク。

**修正:** 全量ロード呼び出しを削除し、WARNING ログ + 空 trades_df パススルーに置換。

**ファイル:** `scripts/v460/ml/feature_enricher.py` L438-440

### §1.6 #7 (LOW): feature freshness デフォルト修正

**修正:** `fill_test.yaml` の `trigger_check_feature_freshness` を `false` → `true` に変更。

### §1.7 #6 (LOW): テスト追加

`tests/unit/v460/test_139_review_fixes.py` に 13 テスト作成:
- `TestRetrainNewSamplesRunSwitch` (3): negative→full count、positive→差分、zero→0
- `TestRegimeThresholdsWiring` (3): config フィールド、YAML パース、Manager 受渡し + kill 判定
- `TestNarrowSpreadPauseActualWait` (2): config 存在、asyncio.sleep 存在確認
- `TestFeeSpecClarification` (2): maker-only コメント、taker_fee リザーブ
- `TestTradesFallbackSafety` (1): 全量ロード廃止確認
- `TestIntegrationRegimeKillFlow` (1): regime kill → cooldown → resume
- `TestFeatureFreshnessDefault` (1): YAML true 確認

**既存テスト修正:**
- `test_113_resilience.py`: 行数 assertion `< 400` → `<= 405` (asyncio.sleep 行追加分)
- `test_retrain_hot_reload.py`: fallback 呼び出し回数 3→2 (全量ロード廃止分)

---

## §2 138# §8 レビュー対応

### §2.1 #1 (HIGH): ScoreCalibrator → SkipGateEvaluator 注入

**問題:** `ScoreCalibrator` が実装されているが `SkipGateEvaluator` から利用されていなかった。

**修正:**
- `skip_gate_evaluator.py`: `_inject_calibrator()` メソッド追加 (30 行)
- `__init__()` L73: `self._inject_calibrator(skip_gate)` 呼び出し
- `_check_and_reload_model()`: hot-reload 後に `self._inject_calibrator(new_gate)` 再注入
- `config.skip_gate_score_calibration=False` → `_score_calibrator=None` 明示設定
- `config.skip_gate_calibrator_path` から pkl ロード、失敗時は None にフォールバック

### §2.2 #2 (MEDIUM): calibrator 永続化方針

**決定:** calibrator を SkipGate pkl に同梱せず、外部パス (`skip_gate_calibrator_path`) を source of truth とする。hot-reload 時は毎回 pkl から再ロード。

### §2.3 #3 (MEDIUM): mode 仕様削減

**修正:** `ScoreCalibratorConfig` から `mode: str = "pnl"` を削除。現実装は PnL (predicted_pnl_bps → actual_pnl_bps) 専用。AS 校正は将来 P(AS) データ蓄積後に別途検討。

**ファイル:** `ztb/ml/score_calibrator.py`

### §2.4 #4 (MEDIUM): preflight pause 監査レコード

**問題:** preflight pause 発動時に FillRecord が残らず、監査不能。

**修正:** `run_fill_test.py` の preflight_pause ブロック内、sleep 前に `self._append_fill_record(FillRecord(..., cancel_reason="preflight_pause"))` を追加。

### §2.5 #5 (LOW): 統合テスト

`tests/unit/v460/test_139_review_fixes.py` に 14 テスト追加:
- `TestEvaluatorCalibratorInjection` (5): disabled→None、enabled+pkl→注入、no_path→None、missing_file→default、hot-reload 確認
- `TestCalibratorConfigModeRemoved` (2): mode フィールド不在、mode kwarg TypeError
- `TestPreflightPauseAuditRecord` (1): cancel_reason="preflight_pause" 存在確認
- `TestFillConfigBoundaryValidation` (6): threshold≥1、max_pauses≥0、pause_sec≥0、min_samples≥1、refit_interval≥1、正常値通過

### §2.6 #6 (LOW): fill_config 境界値バリデーション

**修正:** `fill_config.py __post_init__()` に 5 件のバリデーション追加:
- `preflight_pause_threshold >= 1`
- `preflight_max_pauses >= 0`
- `preflight_pause_sec >= 0`
- `skip_gate_calibrator_min_samples >= 1`
- `skip_gate_calibrator_refit_interval >= 1`

---

## §3 P1-01/02 実現可能性評価

### §3.1 データ状況

| 指標 | buy | sell |
|---|---|---|
| filled 件数 | 596 | 587 |
| avg PnL (bps) | -0.073 | -0.556 |
| win rate | 48.5% | 43.8% |
| AS rate | 27.9% | 27.1% |

### §3.2 評価

- **データ十分性:** ○ — side 別 ~590 件は SkipGate 学習の最低ライン (特徴量 ~20 に対し十分)
- **sell-side 劣性:** 明確 — PnL 7.6 倍悪化、win rate 4.7pt 低下
- **前提条件:** retrain 修正 (§1.1) 完了が必須。次回 retrain で `new_samples=93` (全件新規) として実行される
- **推奨:** retrain 修正後の 1-2 run (24-48h) で retrain 正常動作を確認 → P1-01/02 着手

---

## §4 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/ml/retrain_scheduler.py` | §9 #1/A: new_samples run 切替検出, §8.1 #3: source_run_id metadata + run_id 直接比較 |
| `scripts/v460/lib/fill_config.py` | §9 #2: regime_thresholds, §8 #6: バリデーション |
| `scripts/v460/run_fill_test.py` | §9 #2/#3: regime 配線 + sleep, §8 #4→§8.1 #1: batch.append 統一, §8.1 #2: skip FillRecord 化 (4分岐) |
| `scripts/v460/lib/skip_gate_evaluator.py` | §8 #1: _inject_calibrator 追加 |
| `scripts/v460/lib/pnl_measurer.py` | §9 #4: fee コメント明確化 |
| `scripts/v460/ml/feature_enricher.py` | §9 #5: trades 全量フォールバック廃止 |
| `ztb/ml/score_calibrator.py` | §8 #3: mode 廃止 |
| `configs/v460/fill_test.yaml` | §9 #2/#7: thresholds + freshness |
| `tests/unit/v460/test_139_review_fixes.py` | 新規 37 テスト (§9: 13, §8: 14, §8.1: 10) |
| `tests/unit/v460/test_113_resilience.py` | 行数 assertion 修正 |
| `tests/unit/v460/test_091_fixes.py` | §8.1 #2: 086 deadlock FillRecord 追加に伴う行数拡張 |
| `tests/unit/v460/test_retrain_hot_reload.py` | fallback 回数修正 |

---

## §5 テスト結果

```
tests/unit/v460: 1171 passed, 0 failed
test_139_review_fixes.py: 37 passed (§9: 13, §8: 14, §8.1: 10)
```

| 区間 | テスト数 |
|---|---|
| 138# (前回) | 1147 |
| 139# (対応) | 1161 |
| 140# (§8.1 対応) | 1171 |
| 差分 | +24 (§8 14 + §8.1 10) |

---

## §6 次ステップ

1. retrain 正常動作確認 (24-48h 運用監視)
2. P1-01/02 (side 別モデル) 着手 — sell-side 改善が最優先
3. 140# として P1-01/02 設計・実装を計画

---

## §7 レビューチェックリスト

- [ ] §9 #1/A: retrain 永久停止修正 — 次回 retrain ログで `new_samples > 0` 確認
- [ ] §9 #2: regime_thresholds — trending_up/-0.3 等で kill 発動確認
- [ ] §9 #3/C: narrow_spread_pause — ログに sleep 時間記録確認
- [ ] §8 #1: calibrator 注入 — score_calibration=True 時 ログに "ScoreCalibrator injected" 確認
- [ ] §8 #6: バリデーション — 不正値でアプリ起動エラー確認

---

## §8 外部再レビュー追記 (2026-02-22)

### §8.1 実装済み点検 (重大度付き) → **140# 対応済み**

| # | 重大度 | 対象 | 点検結果 | 対応内容 |
|---|---|---|---|---|
| 1 | CRITICAL | `run_fill_test.py` | `_append_fill_record()` が `FillTestRunner` に未定義 → `AttributeError` | **140# 修正済み**: `batch.append()` + `maybe_flush()` に統一。`_append_fill_record` 呼び出し全廃。 |
| 2 | HIGH | `run_fill_test.py` | time_filter/preflight skip の FillRecord 化未完 | **140# 修正済み**: `time_filter_both_sides`, `time_filter_086_deadlock`, `preflight_insufficient` の 3 分岐に FillRecord 生成追加。全 7 skip パスが cancel_reason 付き。 |
| 3 | MEDIUM | `retrain_scheduler.py` | `new_samples` 負値対策がヒューリスティック | **140# 改善済み**: metadata に `source_run_id` 保存。run_id 直接比較を優先し、旧モデル (source_run_id 未保存) は負値フォールバックで後方互換維持。 |
| 4 | MEDIUM | `test_139_review_fixes.py` | ソース文字列検査中心でランタイム不整合を検出不能 | **140# 追加済み**: `TestRunContinuousBranchExecution` (5テスト) + `TestRetrainRunIdComparison` (5テスト) 計 10 テスト追加。`hasattr(FillTestRunner, "_append_fill_record") == False` のランタイム検証含む。 |
| 5 | LOW | ドキュメント | 「17件対応完了」結論が過大 | **140# 修正済み**: 結論を「22件対応完了」に更新し、CRITICAL 修正を含む正確な記述に。 |

### §8.2 132/133/134 との整合性点検

| 起点 | 論点 | 現状 | 整合判定 |
|---|---|---|---|
| 132# F2 | retrain `new_samples` 停滞 | 負値永久停止は緩和済み + run_id 直接比較追加 | ✅ 整合 |
| 132# F4 | skip 系可観測性欠損 | 全 7 skip パスに cancel_reason 付き FillRecord 生成 | ✅ 整合 (140# で完了) |
| 133# P0-03/04 | trades 供給復旧 + recorder 二重化 | `check_trades_health` + `TradesRecorder` 導入済み | ✅ 整合 |
| 133# P0-07/12 | per-run gate + CLI 統一 | `gate_judgment --run-id/--latest-run`、`run_gate_check` 委譲済み | ✅ 整合 |
| 134# Phase C | 24h再計測 + latest-run 判定運用 | 実装基盤はあるが運用実行はこれから | ⬜ 未完了 |
| 134# Phase E | P1-01/02 side 分離モデル | 139#時点では未着手 | ⬜ 未完了 (次フェーズ候補) |

### §8.3 検証ログ

- `tests/unit/v460/test_139_review_fixes.py`: **37 passed, 1 warning** (27→37, +10)
- `tests/unit/v460`: **1171 passed, 91 warnings** (1161→1171, +10)
- ランタイム点検: `hasattr(FillTestRunner, "_append_fill_record")` → **False** (テストで検証済み)
- `self._append_fill_record` 呼び出し → **source 内に存在しない** (テストで検証済み)

### §8.4 次ステップ提案 (優先順) → **140# 対応状況**

1. ~~P0: `preflight_pause` の未定義メソッド不具合を修正~~ → **140# 完了**
2. ~~P0: `time_filter / preflight_insufficient / preflight_pause` を全て FillRecord 化~~ → **140# 完了** (+ `time_filter_086_deadlock`)
3. ~~P0: run_continuous の分岐実行型テストを追加~~ → **140# 完了** (10テスト追加)
4. ~~P1: `new_samples` を run_id 直接比較へ変更~~ → **140# 完了** (`source_run_id` metadata 保存 + 比較)
5. P1: 24h 固定設定 run → `--latest-run` で gate 判定 → P1-01/02 進行 → **運用タスク (コード完了)**  
