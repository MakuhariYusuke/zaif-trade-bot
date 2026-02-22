# 022# — Fill Test JSONL データロス調査依頼（外部AIレビュー）

| key | value |
|---|---|
| 番号 | 022 |
| フェーズ | ph2 (G1.1-exec) |
| 種別 | ext (外部AIレビュー依頼) |
| 作成日 | 2026-02-14 |
| 状態 | **OPEN** |
| 緊急度 | **P0-CRITICAL** — データ毎分喪失中 |
| 前提文書 | 009# (G1.1 計画), 019# (fill test 分析), 020# (O1-O5 修正) |

---

## 0. 概要

**fill test プロセス (PID 48100) は稼働中で Coincheck 上の取引も成約しているが、JSONL ファイルへの書き込みが UTC 日付変更 (JST 03:00 = UTC 18:00) 以降停止している。**

データはメモリ上にのみ存在し、プロセスクラッシュでデータが完全喪失するリスクがある。

---

## 1. 症状

| 項目 | 詳細 |
|---|---|
| プロセス状態 | PID 48100: CPU=572s, WS=18MB, Threads=18, Responding=True |
| 取引実行 | Coincheck 取引履歴に 03:14:33 JST の 0.001 BTC 取引確認 |
| 最終JSONL書込 | `fill_records_20260213.jsonl`: 105レコード, mtime 03:00 JST |
| 新規JSONL | `fill_records_20260214.jsonl` **存在しない** |
| ロス開始時刻 | 2026-02-14 03:00 JST (= 2026-02-13 18:00 UTC) |
| 推定ロスサイクル | 120秒/サイクル × 数時間 = **60+サイクル** のデータ喪失推定 |

### タイムライン

```
2026-02-13 21:41 JST  PID 48100 起動 (--hours 72 --cycle-interval 120 --start-side sell)
                       ↓ 正常動作: fill_records_20260213.jsonl に順次追記
2026-02-14 02:59 JST  最終レコード書込 (n=105, timestamp in JSONL)
2026-02-14 03:00 JST  UTC日付変更 (UTC 18:00 → 新ファイル名 "20260214" が必要に)
                       ↓ ここで JSONL 書込が停止
2026-02-14 03:14 JST  Coincheck 取引履歴に約定確認 (=プロセスは稼働中)
2026-02-14 06:00+ JST 現在も PID 48100 稼働中、JSONL 書込なし
```

---

## 2. プロセスアーキテクチャ

```
PID 3440 (.venv\Scripts\python.exe)       ← 親プロセス (zombie/waiting)
  └─ PID 48100 (system Python 3.11)       ← 実ワーカー (actively running)
       コマンド: scripts/v460/run_fill_test.py --hours 72 --cycle-interval 120 --start-side sell
       CPU: 572s, WS: 18MB, Threads: 18

PID 24148 (.venv\Scripts\python.exe)       ← 親プロセス (observation)
  └─ PID 30024 (system Python 3.11)       ← observation ワーカー
       CPU: 3845s, WS: 198MB, Threads: 18
```

**重要**: PID 48100 は `5f7e48805` コミット **前** に起動されたため、020# O1-O5 修正（`run_id`, `git_sha`, `adverse_selected_raw`）を含まない**旧コード**で動作中。

---

## 3. 関連コード

### 3.1 `run_continuous()` — メインループ (scripts/v460/run_fill_test.py L415-476)

```python
async def run_continuous(self, hours: float) -> list[FillRecord]:
    end_time = time.time() + hours * 3600

    existing_records = self.resume_from_existing()
    records: list[FillRecord] = list(existing_records)
    batch: list[FillRecord] = []
    batch_size = 10  # 10 サイクルごとに保存

    while time.time() < end_time and not self._shutdown_requested:
        try:
            record = await self.run_single_cycle()
            records.append(record)
            batch.append(record)

            # バッチ保存
            if len(batch) >= batch_size:
                self._save_batch(batch)     # ← ★ ここで例外 → batch = [] に到達しない
                batch = []                  # ← ★ _save_batch 成功時のみ実行

            # 進捗ログ (50サイクルごと)
            if self._cycle_count % 50 == 0:
                filled_count = sum(1 for r in records if r.filled)
                logger.info(...)

        except KeyboardInterrupt:
            self._shutdown_requested = True
            break
        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)   # ← ★ 全例外を飲み込む
            await asyncio.sleep(self.config.cycle_interval_sec)
            continue                                            # ← ★ batch は未クリアのまま

        await asyncio.sleep(self.config.cycle_interval_sec)

    # 残りバッチを保存
    if batch:
        self._save_batch(batch)   # ← ★ ここも例外なら未保存のまま終了

    return records
```

### 3.2 `_save_batch()` — JSONL 保存 (L477-481)

```python
def _save_batch(self, batch: list[FillRecord]) -> None:
    """日別 JSONL ファイルにバッチ保存."""
    from datetime import datetime, timezone

    day_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = self._results_dir / f"fill_records_{day_str}.jsonl"
    save_fill_records(batch, path)
```

### 3.3 `save_fill_records()` — ファイル書込 (ztb/metrics/fill_quality.py L296-303)

```python
def save_fill_records(records: list[FillRecord], path: str | Path) -> None:
    """JSONL 形式で FillRecord を保存."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} fill records to {p}")
```

### 3.4 `FillRecord.to_dict()` — dataclass.asdict

```python
@dataclass
class FillRecord:
    cycle_id: str
    timestamp: float
    side: str
    order_price: float
    order_quantity: float
    fill_price: Optional[float] = None
    filled: bool = False
    cancelled: bool = False
    queue_wait_sec: float = 0.0
    mid_at_fill: Optional[float] = None
    mid_30s_after: Optional[float] = None
    post_fill_30s_pnl: Optional[float] = None
    adverse_selected: Optional[bool] = None
    adverse_selected_raw: Optional[bool] = None   # 020# で追加
    cancel_reason: Optional[str] = None
    run_id: Optional[str] = None    # 020# で追加
    git_sha: Optional[str] = None   # 020# で追加

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FillRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
```

---

## 4. 仮説

### H1: `_save_batch()` 内で例外発生（最有力）

**根拠**:
- `_save_batch` は `try` ブロック内で呼ばれる
- 例外は `except Exception: continue` で飲み込まれる
- `batch = []` (バッチクリア) は `_save_batch` の **直後** にあるため、例外時は未クリア
- 次の10サイクル目で再度 `_save_batch` が呼ばれるが、同じ例外で再び失敗 → 永久ループ
- batch は grow し続ける → **メモリリーク** かつ **データロス**

**可能性のある例外原因**:
1. **ファイルパス問題**: `self._results_dir` の Path オブジェクトと `f"fill_records_{day_str}.jsonl"` の結合で何らかの問題
2. **権限問題**: 新日付ファイル作成時のファイルシステム権限
3. **`to_dict()` シリアライズエラー**: `asdict()` が何らかのフィールドで失敗
4. **ディスク容量**: 書込不可

### H2: `run_single_cycle()` 内で例外発生

- `run_single_cycle()` で例外 → `except Exception: continue`
- record が作られない → batch は成長しない
- ただし Coincheck で約定確認されている → 実際にはサイクルは完了

### H3: batch_size=10 に到達しない

- 10サイクル = 120s × 10 = 20分
- 03:00〜06:00 = 3時間 = **約9バッチ** 分
- しかし1バッチ目(最初の10サイクル)すら保存されていない → H3 単独では説明不可

### H4: 旧コードとの不整合

- PID 48100 は 020# コミット **前** のコードで動作
- 旧コードの `FillRecord` には `adverse_selected_raw`, `run_id`, `git_sha` フィールドが**存在しない**
- しかし `to_dict()` は `asdict()` を使用 → フィールドがなければ出力しないだけ → エラーにならないはず
- ただし `from_dict()` 側（`resume_from_existing()` 内）が影響する可能性は？

---

## 5. 質問事項

外部AIレビューアーに以下の点を確認・分析依頼:

### Q1: データロスの根本原因特定
上記コードと症状から、JSONL が書き込まれない原因として最も可能性が高いものは何か？
特に `_save_batch()` → `save_fill_records()` → `open()` の連鎖で、UTC日付変更後のみ発生する条件を考慮せよ。

### Q2: `except Exception: continue` パターンの危険性
このパターンが本件にどう寄与しているか、また修正案を提示せよ。
ログには `Cycle error:` が出ているはずだが、確認手段は？

### Q3: メモリ上データの救出可能性
PID 48100 のプロセスメモリ上に蓄積された `records` リストと `batch` リストを、プロセスを殺さずに抽出する方法はあるか？
（例: `py-spy`, `pyrasite`, `/proc/<pid>/mem`, `gdb python` 等）

### Q4: 再発防止のコード改善案
以下を満たす改善コードを提示せよ:
1. `_save_batch` 失敗時に batch を消失させずリトライ
2. 例外の詳細をログに残す（現状 `continue` で握り潰し）
3. バッチ外のフォールバック保存（メモリ上 records を定期的にフルダンプ）
4. atexit / signal handler でのメモリ上データ保存

### Q5: プロセスアーキテクチャ
`.venv/Scripts/python.exe` が system Python を子プロセスとして spawn する挙動は正常か？
`.venv` 内で system Python が使われている場合、パッケージ解決に問題は生じないか？

---

## 6. 補足情報

### ディレクトリ構造

```
results/v460/fill_test/
├── fill_records_20260213.jsonl   ← 105レコード (最終更新 03:00 JST)
└── (fill_records_20260214.jsonl は存在しない)
```

### 起動コマンド

```powershell
# .venv 内から実行 (02/13 21:41 JST)
.venv\Scripts\python.exe scripts/v460/run_fill_test.py --hours 72 --cycle-interval 120 --start-side sell
```

### ログファイル確認方法

```powershell
# ログ出力先（stdoutに出力、ファイルリダイレクト不明）
Get-CimInstance Win32_Process -Filter "ProcessId = 48100" | Select-Object CommandLine
```

### Git 状態

```
HEAD: 5f7e48805 (020# P0 commit)
Branch: main
```

---

## 7. 期待するアウトプット

1. **根本原因の特定**（または最も可能性の高い仮説の検証手順）
2. **即座に実行可能な対応策**（データ救出 or 安全な再起動手順）
3. **再発防止のコード修正PR案**（差分形式）
4. **`except Exception: continue` の改善パターン**
