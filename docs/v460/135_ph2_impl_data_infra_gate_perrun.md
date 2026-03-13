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

---

## §9 外部レビュー追記 (2026-02-22)

### §9.1 重大度付きレビュー結果

| # | 重大度 | 対象 | 問題 | 推奨対応 |
|---|---|---|---|---|
| 1 | HIGH | `ztb/data/trades_recorder.py` | flush 跨ぎ dedup が API 降順レスポンスで破綻。`_last_trade_key` を buffer 最終要素で更新しているため、旧 trade が再流入する。 | watermark を `last` ではなく `max(ts,price,amount,side)` で更新。`record_from_adapter()` 側で timestamp 昇順正規化。 |
| 2 | MEDIUM | `ztb/data/trades_recorder.py` | flush 失敗時に buffer を破棄するため、I/O 一時障害で trade データ欠損が確定する。 | flush 失敗時は buffer 保持 + 次回再試行。必要なら emergency dump 経路を追加。 |
| 3 | MEDIUM | `scripts/v460/run_fill_test.py` | trades 収集が orderbook prefetch の外側 try に従属し、OB 失敗時に trades 収集までスキップされる。 | OB 収集と trades 収集を独立 try に分離し、片系障害でも記録継続。 |
| 4 | LOW | `scripts/v460/gate_judgment.py` | `--latest-run` 指定時、run_id 有効データが無いと silently ALL にフォールバックし誤読を招く。 | fallback 時に WARNING + `run_scope=ALL_FALLBACK` を明示、またはエラー終了。 |
| 5 | LOW | `scripts/v460/run_gate_check.py` | `run_g1_1()` は成功時 `G1.1-quick`、NO_DATA 時 `G1.1-exec` を返し、互換 API の戻り値契約が不安定。 | `gate` 名を常に統一。未使用 `thresholds` 引数は削除または活用。 |
| 6 | LOW | `scripts/v460/run_fill_test.py` | trades 健全性 warning が `run_observation.py` 再起動を推奨しており、135# 方針（fill_test 内蔵 recorder）と運用メッセージが不整合。 | 運用メッセージを「fill_test 内 recorder 状態確認」に更新。 |

### §9.2 追加見落とし点 (再点検)

| # | 重大度 | 対象 | 問題 | 推奨対応 |
|---|---|---|---|---|
| A | MEDIUM | `ztb/data/trades_recorder.py` | flush 跨ぎ判定が `key[:3]` 比較で `side` を無視しており、同一 ts/price/amount で side が異なる正当 trade を取りこぼす可能性。 | dedup 方針を「完全 key」または `trade_id` 基準に統一し、仕様として明文化。 |
| B | LOW | `ztb/data/trades_health.py` | `lookback_days` 自動生成が「当日(UTC)含む」ため、日跨ぎ直後の起動で false warning が出やすい。 | 運用上は「昨日まで N 日」モードを追加し、起動直後の誤警報を抑制。 |
| C | LOW | `tests/unit/v460/test_135_trades_and_gate.py` | 降順レスポンス + flush 跨ぎ重複、flush 失敗時再試行のテストが未カバー。 | 回帰テストを 2-3 ケース追加し、今回指摘の再発を防止。 |

### §9.3 再検証ログ (このレビュー時点)

- `tests/unit/v460/test_135_trades_and_gate.py`: **22 passed**
- `tests/unit/v460/test_gate_check.py -k "run_g1_1 or g1_1"`: **6 passed**
- 全体 `pytest -q`: 本環境では依存不足 (`stable_baselines3`, `torch`, 他) により収集エラー。  
  したがって「1082 passed」は当該実装時の環境前提として扱うのが妥当。

### §9.4 優先修正順 (提案)

1. P0: TradesRecorder watermark 修正 + 降順/flush 跨ぎ回帰テスト追加  
2. P1: flush 失敗時のデータ保全 (再試行 or dump)  
3. P1: run_fill_test の OB/trades 収集を障害分離  
4. P2: Gate CLI fallback 表示整備 + `run_g1_1` 戻り値契約統一

---

## §10 レビュー対応結果 (2026-02-22)

**コミット**: `b96ac2ef3` — テスト 1089 passed (1082→+7)

### §10.1 対応一覧

| §9 # | 重大度 | 対応内容 | 状態 |
|---|---|---|---|
| 1 | HIGH | `_last_trade_key` → `_watermark` に改名。`max()` で更新。`sorted(trades, key=ts)` で API 降順レスポンスを昇順正規化。 | ✅ |
| A | MEDIUM | `key[:3]` 比較 → 完全 key (`key <= self._watermark`) に変更。side 含む NamedTuple 比較で正当 trade の取りこぼし解消。 | ✅ |
| 2 | MEDIUM | flush 失敗時 buffer 保持 + 次回再試行。`_flush_fail_count` で 3 連続失敗 → 緊急 drop (OOM 防止)。 | ✅ |
| 3 | MEDIUM | `run_fill_test.py`: OB/trades 取得を独立 try block に分離。OB 失敗でも trades 記録継続。 | ✅ |
| 4 | LOW | `gate_judgment.py`: `_filter_by_run_id(latest=True)` で有効 run_id なし時、WARNING ログ + `run_scope=ALL_FALLBACK` 明示。 | ✅ |
| 5 | LOW | `run_gate_check.py`: NO_DATA 時 gate 名を `G1.1-exec` → `G1.1-quick` に統一。 | ✅ |
| 6 | LOW | `run_fill_test.py`: 健全性 warning を「fill_test 内蔵 TradesRecorder の動作状態を確認」に更新。 | ✅ |
| B | LOW | `trades_health.py`: lookback を `days=i` → `days=i+1` (昨日起算) に変更。UTC 日付境界での偽警告を防止。 | ✅ |
| C | LOW | テスト 8 件追加: 降順 sort・flush 跨ぎ dedup・side 非 dedup・flush retry/emergency drop・lookback yesterday・fallback warning。 | ✅ |

### §10.2 補足修正

- `gate_judgment.py`: `import logging` + `logger = logging.getLogger(__name__)` 追加 (§9.1 #4 の WARNING 出力に必要)
- 既存 health テスト 2 件 (`test_healthy_when_all_present`, `test_unhealthy_when_missing`) を lookback 昨日起算に合わせて更新

### §10.3 §8 チェックリスト解消状況

- [x] TradesRecorder dedup: 降順ケース対応完了 (sorted + watermark max)
- [x] trades_health: UTC 日付境界 → 昨日起算で解消
- [x] gate_judgment per-run: None/blank run_id → WARNING + ALL_FALLBACK
- [ ] `ztb/io/jsonl_gz.py`: 大量レコード append mode gzip 正常性 (未追加テスト、既存運用で問題なし)
- [ ] `run_g1_1()` deprecation: CI 棚卸し (既存呼び出し箇所なし、当面維持)
