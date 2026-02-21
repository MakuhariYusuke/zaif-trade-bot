# 135# ph2 impl: Data Infrastructure & Gate Per-Run

| key | value |
|---|---|
| 番号 | 135 |
| フェーズ | ph2 |
| 種別 | impl (実装) |
| 対象 | 134# ロードマップ Phase A/B (P0-03/04, P2-09→P1, P0-07, P0-12) |
| 作成日 | 2026-02-22 |
| 前提 | Git `77fbe8ef0` (134# 妥当性評価完了) |
| 結論 | **trades データ基盤完成 + per-run Gate 評価 + CLI 統合。テスト 1082 passed (1061→1082, +21)。** |

---

## §0 エグゼクティブサマリ

134# ロードマップの Phase A/B を実装。trades データ欠損問題の根本解決（fill_test 内蔵 TradesRecorder）、run 開始時の健全性チェック、per-run Gate 評価による Simpson 型リスク検出、CLI 統合を完了。

**成果物**:
- 新規 ztb モジュール 3 件 (共通 JSONL.gz, TradesRecorder, trades_health)
- 既存モジュール修正 5 件 (OBRecorder, fill_test, gate_judgment, run_gate_check, ztb/io)
- テスト 22 件追加 + 既存テスト修正 2 件 (G1.1 delegation 対応)
- 回帰テスト: 1082 passed, 0 failed

---

## §1 実装項目一覧

| 134# ID | 内容 | 成果物 | 状態 |
|---|---|---|---|
| P0-03 | trades データ欠損の原因特定 | `run_observation.py` 停止を確認。OB は fill_test 内蔵だが trades は別プロセス依存 → 単一障害点 | ✅ 完了 |
| P0-04 | TradesRecorder fill_test 内蔵化 | `ztb/data/trades_recorder.py` + `run_fill_test.py` 統合 | ✅ 完了 |
| P2-09→P1 | trades 健全性チェック | `ztb/data/trades_health.py` + run_continuous 起動時統合 | ✅ 完了 |
| P0-07 | per-run Gate 評価 | `gate_judgment.py` に `--run-id` / `--latest-run` 追加 | ✅ 完了 |
| P0-12 | CLI 統合 | `run_gate_check.py:run_g1_1()` を `gate_judgment` に委譲 + DeprecationWarning | ✅ 完了 |
| (副次) | OBRecorder JSONL.gz 共通化 | `ztb/io/jsonl_gz.py` 抽出 + OBRecorder リファクタ | ✅ 完了 |

---

## §2 アーキテクチャ詳細

### §2.1 JSONL.gz 共通ユーティリティ (`ztb/io/jsonl_gz.py`)

**動機**: OBRecorder, TradesRecorder, MarketDataCollector で同一の gzip JSONL 書き込みロジックが重複。DRY 原則に基づき共通化。

```
ztb/io/jsonl_gz.py
├── append_jsonl_gz(path, records) → int    # gzip append 書き込み
└── read_jsonl_gz(path) → list[dict]        # gzip 読み込み (malformed line handling)
```

- `gzip.open("at")` で追記モード
- `Sequence[dict]` 型 (list/tuple 両対応)
- malformed 行はスキップ + WARNING ログ
- 空 records → 早期 return (ファイル作成なし)

**影響**: `ztb/io/__init__.py` に `append_jsonl_gz`, `read_jsonl_gz` を export 追加。

### §2.2 TradesRecorder (`ztb/data/trades_recorder.py`)

**設計思想**: OBRecorder (129#) と対称的な buffer→flush パターン。`__slots__` でメモリ効率化。

```
TradesRecorder
├── record_trades(trades: list[dict]) → int        # dict 入力 (dedup 付き)
├── record_from_adapter(trade_records) → int       # TradeRecord duck-typing
├── flush() → int                                  # data/v460/raw/trades/YYYYMMDD.jsonl.gz
├── buffer_size: int                               # 現在バッファサイズ
└── total_written: int                             # 累計書き込み数
```

**重複排除**:
- `TradeEntry` NamedTuple: `(ts, price, amount, side)` の composite key
- `_last_trade_key`: flush 後も保持。ts ≤ 前回最大の trade をスキップ
- `_seen_keys`: 同一 flush バッチ内の重複防止 (flush 時にリセット)

**メモリ保護**:
- `_BUFFER_CAP = 10,000`: 上限到達で強制 flush
- `__slots__` 使用: dict ベースより ~50% メモリ削減
- `_seen_keys` は flush でクリア (無限成長防止)

**fill_test 統合箇所**:
1. `__init__`: `self._trades_recorder = TradesRecorder(enabled=True)` (OBRecorder 直後)
2. `run_single_cycle` (~L658): OB 記録後に `adapter.get_recent_trades(symbol, limit=100)` → `record_from_adapter()`
3. `_cleanup_sync`: trades recorder final flush (OB recorder flush 直後)

### §2.3 trades 健全性チェック (`ztb/data/trades_health.py`)

**目的**: retrain が全量 fallback して特徴量時間整合性が崩れることを run 開始時に検出。

```
TradesHealthResult (frozen dataclass)
├── healthy: bool
├── available_days: list[str]
├── missing_days: list[str]
├── stale_hours: float
└── message: str

check_trades_health(raw_dir, expected_days, lookback_days=3, stale_threshold_hours=36.0)
```

- `lookback_days` 日分のファイル存在チェック
- 最新ファイルの mtime から `stale_hours` 計算
- `missing_days > 0` OR `stale_hours >= threshold` → unhealthy
- CLI 対応: `python -m ztb.data.trades_health --days 3`

**fill_test 統合箇所**: `run_continuous` の lock 取得直後、loss_cap チェック前で実行。unhealthy の場合は WARNING ログ + `run_observation.py` 再起動を示唆。

### §2.4 per-run Gate 評価 (`gate_judgment.py` P0-07)

**動機**: 133# C1 Simpson 型リスク — 全体 WATCH でも最新 run が FAIL の場合を検出できない。

**追加関数**:
- `_filter_by_run_id(records, run_id=None, *, latest=False)`: run_id / latest でフィルタ
- `_get_unique_run_ids(records)`: ユニーク run_id を timestamp 昇順で返す

**CLI 拡張**:
```bash
# 全レコード (従来互換)
python -m scripts.v460.gate_judgment

# 特定 run_id の結果のみ
python -m scripts.v460.gate_judgment --run-id "dry_20260220_1423"

# 最新 run vs ALL 比較 (Simpson 型リスク検出)
python -m scripts.v460.gate_judgment --latest-run
```

**最新 run 比較出力**:
```
┌─────────── ALL vs LATEST ───────────┐
│ [ALL]    gate_result = WATCH        │
│ [LATEST] gate_result = FAIL         │
│ ⚠️ ALL ≠ FAIL だが LATEST = FAIL:   │
│   Simpson 型リスク — 最新 run が     │
│   全体評価を希釈している可能性       │
└─────────────────────────────────────┘
```

### §2.5 CLI 統合 (`run_gate_check.py` P0-12)

`run_g1_1()` の内部実装を `gate_judgment.run_gate_judgment()` に委譲:
- `DeprecationWarning` 発行 (stacklevel=2)
- 返り値は `result["g1_1_quick"]` で後方互換維持
- gate 名は `G1.1-exec` → `G1.1-quick` に変更 (g1_1_quick_judgment 由来)

---

## §3 OBRecorder リファクタ

`scripts/v460/lib/ob_recorder.py` の `flush()` を `ztb/io/jsonl_gz.py` 共通化:

**Before**:
```python
import gzip, json
with gzip.open(path, "at", encoding="utf-8") as f:
    for rec in self._buffer:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
```

**After**:
```python
from ztb.io.jsonl_gz import append_jsonl_gz
append_jsonl_gz(path, self._buffer)
```

行数削減: flush() 内 5行 → 1行。機能的に等価。

---

## §4 型安全・メモリ保護まとめ

| 施策 | 対象 | 効果 |
|---|---|---|
| `__slots__` | TradesRecorder | dict ベースより ~50% メモリ削減 |
| `NamedTuple` (TradeEntry) | dedup composite key | 型安全な比較 + ハッシュ |
| `frozen=True` dataclass | TradesHealthResult | immutable 保証 |
| `_BUFFER_CAP = 10,000` | TradesRecorder | メモリ爆発防止 |
| `_seen_keys` flush クリア | TradesRecorder | set 無限成長防止 |
| `Sequence[dict]` | append_jsonl_gz | list/tuple 両対応 |
| duck-typing (`getattr`) | record_from_adapter | TradeRecord 型依存排除 |

---

## §5 テスト結果

### §5.1 新規テスト (22 件)

`tests/unit/v460/test_135_trades_and_gate.py`:

| セクション | テスト数 | カバレッジ |
|---|---|---|
| §1 JSONL.gz ユーティリティ | 4 | append, empty, multiple, read_nonexistent |
| §2 TradesRecorder 基本 | 4 | record, disabled, flush_empty, total_written |
| §3 TradesRecorder 重複排除 | 2 | same_trade, old_trades_skipped |
| §4 TradesRecorder flush | 1 | creates_jsonl_gz + 中身検証 |
| §5 TradesRecorder adapter | 1 | record_from_adapter duck-typing |
| §6 trades_health | 4 | healthy, missing, stale, no_dir |
| §7 per-run Gate | 4 | filter_specific, filter_latest, no_args, unique_ids |
| §8 OBRecorder 回帰 | 2 | flush_creates, append_existing |

### §5.2 既存テスト修正 (2 件)

| ファイル | 変更内容 |
|---|---|
| `test_gate_check.py` TestRunG1_1 | delegation 後の gate 名 `G1.1-quick` + check key `K1-K6` に対応。`_make_fill_records` に `run_id`/`git_sha` 追加 (quarantine filter 対応)。deprecation_warning テスト追加 |
| `test_fill_quality.py` TestGateCheckG11 | `G1.1-exec` → `G1.1-quick` |

### §5.3 回帰結果

```
1082 passed, 0 failed, 91 warnings
テスト増分: +21 (1061 → 1082)
```

---

## §6 ファイル変更一覧

### 新規作成

| ファイル | 行数 | 目的 |
|---|---|---|
| `ztb/io/jsonl_gz.py` | 69 | 共通 JSONL.gz 読み書き |
| `ztb/data/trades_recorder.py` | 192 | fill_test 内蔵 trades 記録 |
| `ztb/data/trades_health.py` | 124 | trades 健全性チェック |
| `tests/unit/v460/test_135_trades_and_gate.py` | 258 | 135# テスト全体 |

### 修正

| ファイル | 変更概要 |
|---|---|
| `scripts/v460/lib/ob_recorder.py` | flush() を `append_jsonl_gz` 共通化 |
| `scripts/v460/run_fill_test.py` | TradesRecorder 統合 (init, cycle, cleanup) |
| `scripts/v460/gate_judgment.py` | per-run フィルタ + `--run-id`/`--latest-run` CLI |
| `scripts/v460/run_gate_check.py` | `run_g1_1()` → `gate_judgment` 委譲 + 非推奨警告 |
| `ztb/io/__init__.py` | jsonl_gz exports 追加 |
| `tests/unit/v460/test_gate_check.py` | G1.1 delegation 対応 |
| `tests/unit/v460/test_fill_quality.py` | G1.1 gate 名修正 |

---

## §7 残課題 (134# ロードマップ Phase C 以降)

| 134# ID | 内容 | 優先度 | 備考 |
|---|---|---|---|
| P1-01 | retrain_scheduler trigger ロジック | P1 | スケジュール見直し |
| P1-02 | feature staleness monitoring | P1 | trades_health の拡張版 |
| P1-03 | sell dynamic kill チューニング | P1 | rolling window / threshold 最適化 |
| P2 群 | logging 改善, parallelism 等 | P2-P3 | 工数対効果で優先 |

---

## §8 外部レビュー向けチェックリスト

- [ ] `ztb/io/jsonl_gz.py`: append mode での gzip 正常性 (大量レコード)
- [ ] TradesRecorder dedup: 時系列逆転ケースの扱い (API 応答順保証なし)
- [ ] trades_health: UTC 日付境界のタイムゾーン一貫性
- [ ] gate_judgment per-run: run_id が None/blank の旧レコード混在時の挙動
- [ ] run_g1_1() deprecation: 既存スクリプト/CI からの呼び出し箇所の棚卸し
