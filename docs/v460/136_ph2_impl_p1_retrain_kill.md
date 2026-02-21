# 136# ph2 impl: P1 残課題 (retrain trigger / feature staleness / sell dynamic kill)

| key | value |
|---|---|
| 番号 | 136 |
| フェーズ | ph2 |
| 種別 | impl (実装) |
| 対象 | 135# §7 残課題: P1-01, P1-02, P1-03 |
| 作成日 | 2026-02-22 |
| 前提 | Git `2d3a99ccd` (135# §10 レビュー対応完了) |
| 結論 | **retrain trigger・feature staleness・sell dynamic kill の 3 施策を実装。テスト 1104 passed (1089→1104, +15)。** |

---

## §0 エグゼクティブサマリ

135# §7 で残課題として挙がっていた P1-01/02/03 を実装。retrain の無駄な起動を抑止する事前トリガー、特徴量データの鮮度監視、sell 動的 kill のテスト可能なクラス抽出+レジーム別閾値を完了。

**成果物**:
- 新規 ztb モジュール 2 件 (`ztb/ml/retrain_trigger.py`, `ztb/risk/sell_dynamic_kill.py`)
- 既存モジュール拡張 1 件 (`ztb/data/trades_health.py` に `check_feature_freshness` 追加)
- 既存モジュール修正 2 件 (`retrain_scheduler.py`, `run_fill_test.py`)
- テスト 15 件追加
- 回帰テスト: 1104 passed, 0 failed

---

## §1 実装項目一覧

| 135# §7 ID | 内容 | 成果物 | 状態 |
|---|---|---|---|
| P1-01 | retrain trigger ロジック | `ztb/ml/retrain_trigger.py` + `retrain_scheduler.py` 統合 | ✅ 完了 |
| P1-02 | feature staleness monitoring | `ztb/data/trades_health.py` 拡張 | ✅ 完了 |
| P1-03 | sell dynamic kill チューニング | `ztb/risk/sell_dynamic_kill.py` + `run_fill_test.py` 委譲 | ✅ 完了 |

---

## §2 P1-01: RetainTrigger (データ駆動トリガー)

### §2.1 問題

retrain_scheduler の固定 1h interval loop は:
1. fill_records 未更新でも毎回データロード → CPU/I/O の浪費
2. trades データ欠損 (GIGO) でも retrain 試行 → 品質劣化リスク
3. 連続 skip でも常に 1h 間隔 → 不要なログノイズ

### §2.2 設計

```
ztb/ml/retrain_trigger.py
├── RetainTriggerConfig   # 設定 dataclass
│   ├── check_fill_records_mtime: bool   # fill_records mtime 事前チェック
│   ├── check_trades_health: bool        # trades 健全性ガード
│   ├── backoff_multiplier: float        # 連続skip倍率 (default: 2.0)
│   └── backoff_max_interval_sec: int    # バックオフ上限 (default: 14400s = 4h)
└── RetainTrigger         # トリガーマネージャ
    ├── should_retrain() → (bool, reason)
    ├── record_result(status)
    └── get_effective_interval() → int
```

### §2.3 retrain_scheduler 統合

`run_scheduler()` の while loop 内で:
1. `trigger.should_retrain()` → False なら sleep (adaptive interval)
2. True なら `retrain_model()` を実行
3. `trigger.record_result(status)` でバックオフ状態更新
4. `trigger.get_effective_interval()` で次回 sleep 時間を決定

バックオフ: `base * 2^consecutive_skips`、最大 4h。deploy/error でリセット。

---

## §3 P1-02: Feature Staleness Monitor

### §3.1 問題

`trades_health.py` は日単位のファイル存在チェックのみ。retrain に必要な時間粒度（数時間以内の鮮度）を検証する機能がない。

### §3.2 設計

```
ztb/data/trades_health.py (追加)
├── FeatureFreshnessResult    # dataclass: fresh, trades_stale_hours, ob_stale_hours
├── _latest_mtime_hours()     # ディレクトリ内最新 mtime → 経過時間
└── check_feature_freshness() # trades + OB の鮮度を包括判定
```

- trades/OB のそれぞれの最新ファイル mtime を取得
- 経過時間が閾値 (default: 6h) 以上なら STALE
- RetainTrigger.should_retrain() 内で trades health ガードとして使用

---

## §4 P1-03: SellDynamicKillManager

### §4.1 問題

133# P0-10 の sell dynamic kill は `run_fill_test.py` に直接埋め込み:
- 単体テスト不可（async クラスの内部メソッド）
- レジーム別の閾値調整ができない
- テレメトリ返却がない（ログのみ）

### §4.2 設計

```
ztb/risk/sell_dynamic_kill.py
├── SellKillConfig          # dataclass
│   ├── window, threshold_bps, resume_window
│   └── regime_thresholds: dict[str, float]   # レジーム別閾値
├── SellKillTelemetry       # dataclass (判定結果の詳細)
│   ├── killed, cooldown_remaining, rolling_mean
│   ├── threshold_used, regime, total_kills
│   └── total_cooldown_cycles
└── SellDynamicKillManager
    ├── track(pnl_bps)            # PnL 追跡
    ├── check_kill(regime?) → (bool, Telemetry)
    └── reset()
```

### §4.3 run_fill_test.py 統合

- `__init__`: `SellDynamicKillManager(SellKillConfig(...))` インスタンス化
- `_is_sell_killed()`: `self._sell_kill_mgr.check_kill()` に委譲
- `_track_sell_pnl()`: `self._sell_kill_mgr.track()` に委譲
- 旧 `_sell_pnl_history` / `_sell_kill_cooldown` インスタンス変数を除去

---

## §5 テスト結果

```
tests/unit/v460/test_136_p1_retrain_kill.py: 15 passed
  TestRetainTrigger: 5 tests (mtime check, update pass, trades unhealthy, backoff, reset)
  TestFeatureFreshness: 3 tests (fresh, stale, partial)
  TestSellDynamicKillManager: 7 tests (insufficient, killed, cooldown, above threshold,
                                       regime override, disabled, memory limit)

v460 全体: 1104 passed, 0 failed, 91 warnings
テスト増分: +15 (1089 → 1104)
```

---

## §6 ファイル変更一覧

### 新規作成

| ファイル | 行数 | 目的 |
|---|---|---|
| `ztb/ml/__init__.py` | 1 | ml パッケージ初期化 |
| `ztb/ml/retrain_trigger.py` | 148 | P1-01: データ駆動 retrain トリガー |
| `ztb/risk/sell_dynamic_kill.py` | 166 | P1-03: sell 動的 kill マネージャ |
| `tests/unit/v460/test_136_p1_retrain_kill.py` | 287 | 136# テスト全体 |

### 修正

| ファイル | 変更概要 |
|---|---|
| `ztb/data/trades_health.py` | P1-02: `check_feature_freshness()` + `FeatureFreshnessResult` 追加 |
| `scripts/v460/ml/retrain_scheduler.py` | P1-01: `run_scheduler()` に RetainTrigger 統合 |
| `scripts/v460/run_fill_test.py` | P1-03: `_is_sell_killed/_track_sell_pnl` を SellDynamicKillManager に委譲 |

---

## §7 残課題 (134# ロードマップ Phase E 以降)

| 134# ID | 内容 | 優先度 | 備考 |
|---|---|---|---|
| P1-01/02 | buy/sell 分離モデル + target 二層化 | P1 | モデル構造変更。データ蓄積後に着手 |
| P1-03 (134#) | score 校正 (isotonic/quantile) | P1 | FillRecord データで事後分析→リアルタイム化 |
| P1-06 | reprice 売側上限縮小 AB | P1 | AB テスト容易 |
| P1-08 | spread 狭小時の「休む」判定 | P1 | too_narrow 拡張 |
| P1-10 | preflight 失敗連続→run pause | P1 | dead-cycle 抑止 |
| P1-11 | PnL 評価 fee/slippage 控除 | P1 | 実収益一致 |
| P2 群 | logging 改善, parallelism, oracle 日次 KPI 等 | P2-P3 | 工数対効果で優先 |

---

## §8 コミット履歴

| SHA | 内容 |
|---|---|
| `b96ac2ef3` | 135# §9 review fixes (watermark dedup, flush retry, etc.) |
| `2d3a99ccd` | 135# §10 レビュー対応結果ドキュメント追記 |
| `af30e12b1` | 136# P1-01/02/03 実装 + テスト 15 件 |
