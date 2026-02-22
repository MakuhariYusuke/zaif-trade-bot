# 138# ph2 impl: P1-10 / P1-03 + 134# ロードマップ状況確認

| key | value |
|---|---|
| 番号 | 138 |
| フェーズ | ph2 |
| 種別 | impl (実装) |
| 対象 | `docs/v460/134_ph2_rev_133_validity_evaluation.md` §7 Phase A-E |
| 作成日 | 2026-02-22 |
| 前提 | Git `a598f72db` (137# doc commit) |
| 結論 | **P1-10 (preflight pause) + P1-03 (score calibration) 実装完了。134# ロードマップ Phase A/B は 135# で実装済みを確認。テスト 1118→1134。** |

---

## §0 エグゼクティブサマリ

134# ロードマップの全 Phase (A〜E) の現状を体系的に精査し、未実装タスクに着手。Phase A (P0-03/04/P2-09→P1) と Phase B (P0-07/P0-12) は 135# で既に実装済みであることを確認。Phase E 残タスクとして P1-10 (preflight pause) と P1-03 (score calibration) を新規実装し、テスト 16 件を追加した。

---

## §1 134# ロードマップ状況精査

### §1.1 Phase A: データインフラ復旧 → **135# 実装済み**

| ID | 施策 | 状態 | 確認結果 |
|---|---|---|---|
| P0-03 | trades 欠損原因特定 + run_observation 再起動 | ✅ 運用対応済み | 135# で対応 |
| P0-04 | TradesRecorder fill_test 内蔵化 | ✅ **135# 実装済み** | `ztb/data/trades_recorder.py` 226 行。重複排除 + watermark + メモリ保護。`run_fill_test.py:L230` で有効化。L665-668 で毎サイクル記録 |
| P2-09→P1 | run 開始時 trades 当日ファイル必須チェック | ✅ **135# 実装済み** | `run_fill_test.py:L1024-1037` で `check_trades_health()` 呼び出し |

### §1.2 Phase B: 観測性強化 → **135# 実装済み**

| ID | 施策 | 状態 | 確認結果 |
|---|---|---|---|
| P0-07 | gate_judgment.py に per-run 評価追加 | ✅ **135# 実装済み** | `_filter_by_run_id()`, `--run-id`, `--latest-run` CLI 引数。ALL vs LATEST 対比表示 (Simpson 型リスク検出) |
| P0-12 | run_gate_check 統一 | ✅ **135# 実装済み** | `run_gate_check.py:L213` deprecated 注記 + `gate_judgment.run_gate_judgment()` に委譲 |
| P2-10 | latest-run hard floor | ✅ **P0-07 統合** | `--latest-run` で最新 run のみ G1.2 評価 |

### §1.3 Phase C: 再計測 → **運用タスク (コード不要)**

24h 連続 run + per-run Gate 判定。Phase B 実装済みにより実行可能。

### §1.4 Phase D: retrain 再始動 → **136# で基盤実装済み**

136# P1-01 (RetrainTrigger) + P1-02 (FeatureFreshness) により、retrain 定期ループ再開の技術基盤が整備済み。

### §1.5 Phase E: P1 群着手 → **本 138# + 137# で対応**

| ID | 施策 | 状態 | 実装ドキュメント |
|---|---|---|---|
| P1-01/02 | buy/sell 分離モデル + target 二層化 | ⬜ データ蓄積後 | — |
| P1-03 | score 校正 (isotonic regression) | ✅ **138# 実装** | 本ドキュメント §3 |
| P1-06 | reprice 売側上限縮小 AB | ✅ **137# 実装** | 137# §2 |
| P1-08 | narrow spread pause | ✅ **137# 実装** | 137# §3 |
| P1-10 | preflight 失敗連続→run pause | ✅ **138# 実装** | 本ドキュメント §2 |
| P1-11 | PnL fee 控除 | ✅ **137# 実装** | 137# §4 |

---

## §2 P1-10: preflight pause (dead-cycle 防止)

### §2.1 問題

現行 044# SAFE_STOP は `max_preflight_skip` (=10) 回の連続 preflight 失敗で即座に `_kill_switch.kill()` → プロセス終了。一時的な残高不足 (決済待ち、取引所メンテナンス等) でもプロセスが死亡し、手動再起動が必要。

### §2.2 設計

SAFE_STOP に至る前に「一時停止 → 自動再開」ステップを挿入:

1. 連続 preflight 失敗が `preflight_pause_threshold` (=5) に到達
2. `preflight_pause_sec` (=300s) の間スリープ
3. `_preflight_skip_count` をリセットして再開
4. run 内の累積 pause 回数が `preflight_max_pauses` (=3) を超過した場合のみ SAFE_STOP

### §2.3 実装詳細

**`fill_config.py`** — 4 フィールド追加:
```python
preflight_pause_enabled: bool = True       # SAFE_STOP 前 pause 有効
preflight_pause_threshold: int = 5         # pause 発動閾値
preflight_pause_sec: float = 300.0         # pause 秒数
preflight_max_pauses: int = 3              # run 内最大 pause 回数
```

**`run_fill_test.py`** — `__init__` に `_preflight_pause_count: int = 0` 追加。`run_continuous()` の balance_shrink → SAFE_STOP 間に pause 判定を挿入:
```
balance_shrink → [NEW: preflight_pause 判定] → SAFE_STOP
```

**`fill_test.yaml`** — `loss_control.preflight_pause` セクション追加:
```yaml
preflight_pause:
  enabled: true
  threshold: 5
  pause_sec: 300.0
  max_pauses: 3
```

### §2.4 シーケンス例

```
cycle 1-5: preflight fail (both sides insufficient)
→ pause #1 (300s sleep)
cycle 6-10: preflight fail
→ pause #2 (300s sleep) 
cycle 11-15: preflight fail
→ pause #3 (300s sleep)
cycle 16-20: preflight fail
→ max_pauses (3) 超過 → SAFE_STOP
```

従来: 10 回失敗で即死 (最短 10 * cycle_interval = 100s)
改善: 5 * 3 = 15 回失敗 + 3 * 300s = 900s = **合計 ~15分の自動回復猶予**

---

## §3 P1-03: score calibration (isotonic regression)

### §3.1 背景

134# §3 より:「FillRecord に `skip_gate_score` + `post_fill_30s_pnl_bps` が蓄積済み。事後分析から着手可。」

既存の 088# `_calibrate_threshold()` は目標 skip 率に閾値を合わせる仕組みだが、raw score 自体の精度改善には寄与しない。P1-03 は raw score → 実績 PnL の写像を isotonic regression で学習し、予測精度自体を改善する。

### §3.2 ScoreCalibrator 設計

**新規ファイル:** `ztb/ml/score_calibrator.py`

```
ScoreCalibratorConfig:
  enabled: bool           # 校正有効/無効
  min_samples: int        # 最小サンプル数 (30)
  mode: str               # "pnl" or "as"
  incremental: bool       # 増分更新 (True)
  refit_interval: int     # 自動 refit 間隔 (100)
  persist_path: str|None  # pkl 永続化パス

ScoreCalibrator:
  add_observation(raw, actual) → bool  # 蓄積 + 自動refit
  fit(scores, actuals) → CalibrationStats
  calibrate(raw) → float              # 校正 (未学習時パススルー)
  save(path) / load(path)             # 永続化
  from_fill_records(records) → cls    # FillRecord から学習
```

**sklearn.isotonic.IsotonicRegression** を使用:
- `increasing=True` (高スコア → 高 PnL)
- `out_of_bounds="clip"` (外挿時はクランプ)
- `CalibrationStats` で MAE 改善率を定量化

### §3.3 SkipGate 統合

`SkipGate.__init__()` に `score_calibrator` 引数を追加。`evaluate()` の PnL 予測部分で:

```python
# 138# P1-03: score calibration
if self._score_calibrator is not None:
    cal = self._score_calibrator
    if hasattr(cal, "is_fitted") and cal.is_fitted:
        pred_pnl = cal.calibrate(pred_pnl)
```

既存の 088# `_calibrate_threshold()` (閾値較正) と直交する設計。両方を同時に有効化可能。

### §3.4 設定

**`fill_config.py`** — 4 フィールド追加:
```python
skip_gate_score_calibration: bool = False      # 校正有効/無効
skip_gate_calibrator_path: str | None = None   # pkl パス
skip_gate_calibrator_min_samples: int = 30     # 最小サンプル数
skip_gate_calibrator_refit_interval: int = 100 # refit 間隔
```

**`fill_test.yaml`** — `skip_gate` セクションに追加:
```yaml
score_calibration: false
calibrator_path: null
calibrator_min_samples: 30
calibrator_refit_interval: 100
```

### §3.5 安全設計

- **デフォルト無効** (`enabled: false`) — AB テスト対応
- **未学習時パススルー** — calibrator 未 fit なら raw score をそのまま使用
- **NaN/inf 安全** — 非有限値は無視・パススルー
- **自動 refit** — `add_observation()` で蓄積、interval 到達で自動 fit
- **pkl 永続化** — プロセス再起動後も学習状態を引き継ぎ
- **メモリ安全** — 増分蓄積のみ、外部 cap なし (将来要検討)

---

## §4 変更ファイル一覧

### 新規

| ファイル | 行数 | 概要 |
|---|---|---|
| `ztb/ml/score_calibrator.py` | 290 | P1-03: ScoreCalibrator (isotonic regression) |
| `tests/unit/v460/test_138_p1_preflight_calibration.py` | 230 | P1-10 + P1-03 テスト (16件) |

### 修正

| ファイル | 変更概要 |
|---|---|
| `scripts/v460/lib/fill_config.py` | P1-10: preflight_pause 4 フィールド + P1-03: calibrator 4 フィールド + YAML パース |
| `scripts/v460/run_fill_test.py` | P1-10: `_preflight_pause_count` + pause ロジック (SAFE_STOP 前) |
| `scripts/v460/ml/skip_gate.py` | P1-03: `score_calibrator` 引数 + evaluate() 内 calibration 呼び出し |
| `configs/v460/fill_test.yaml` | P1-10: preflight_pause セクション + P1-03: calibrator 設定 |

---

## §5 テスト結果

### 対象テスト (16件)

**`test_138_p1_preflight_calibration.py`:**

| # | テスト | 検証内容 |
|---|---|---|
| 1 | TestPreflightPause::test_config_defaults | デフォルト値 |
| 2 | TestPreflightPause::test_yaml_parsing | YAML パース |
| 3 | TestPreflightPause::test_pause_fires_before_safe_stop | pause 条件 |
| 4 | TestPreflightPause::test_pause_disabled_goes_straight_to_safe_stop | 無効時 SAFE_STOP |
| 5 | TestScoreCalibrator::test_uncalibrated_passthrough | 未学習パススルー |
| 6 | TestScoreCalibrator::test_disabled_passthrough | 無効パススルー |
| 7 | TestScoreCalibrator::test_fit_and_calibrate | fit→校正→単調増加 |
| 8 | TestScoreCalibrator::test_min_samples_guard | 最小サンプル数ガード |
| 9 | TestScoreCalibrator::test_add_observation_auto_refit | 自動 refit |
| 10 | TestScoreCalibrator::test_save_load_roundtrip | pkl 往復 |
| 11 | TestScoreCalibrator::test_from_fill_records | FillRecord 学習 |
| 12 | TestScoreCalibrator::test_nan_handling | NaN/inf 安全 |
| 13 | TestScoreCalibrationConfig::test_config_defaults | fill_config デフォルト |
| 14 | TestScoreCalibrationConfig::test_yaml_parsing | YAML パース |
| 15 | TestSkipGateCalibrationIntegration::test_skip_gate_with_calibrator | SkipGate 統合 |
| 16 | TestSkipGateCalibrationIntegration::test_skip_gate_without_calibrator | 無 calibrator |

### フルスイート

v460 unit tests: **1134 passed** (1118→1134, +16)

> **注記:** フルスイートのテスト数は実装環境の依存パッケージ構成に依存する。上記数値は実装時環境 (Python 3.11.9, sklearn installed) で確認。

---

## §6 134# ロードマップ完全状況

| Phase | 施策 | 状態 | 実装 # |
|---|---|---|---|
| **A** | P0-03 trades 復旧 | ✅ | 135# |
| **A** | P0-04 TradesRecorder | ✅ | 135# |
| **A** | P2-09→P1 trades チェック | ✅ | 135# |
| **B** | P0-07 per-run Gate | ✅ | 135# |
| **B** | P0-12 gate_check 統一 | ✅ | 135# |
| **C** | 24h 再計測 | ⬜ 運用 | — |
| **D** | retrain 再始動 | ✅ 基盤 | 136# |
| **E** | P1-01/02 side 分離 | ⬜ Data | — |
| **E** | P1-03 score 校正 | ✅ | **138#** |
| **E** | P1-06 reprice sell | ✅ | 137# |
| **E** | P1-08 narrow pause | ✅ | 137# |
| **E** | P1-10 preflight pause | ✅ | **138#** |
| **E** | P1-11 fee 控除 | ✅ | 137# |

**未実装残:**
- P1-01/02 (buy/sell 分離モデル + target 二層化) — データ蓄積待ち
- P2 群 (logging, parallelism, oracle KPI)

---

## §7 コミット履歴

| SHA | 内容 |
|---|---|
| `a10520fb4` | 137# §9 review fixes + P1-06/08/11 |
| `a598f72db` | 137# ドキュメント + 136# §9 #C 注記 |
| *(pending)* | **138# P1-10 + P1-03 実装 + テスト 16 件** |

---

## §8 外部レビュー追記欄

(レビュー結果はここに追記)
