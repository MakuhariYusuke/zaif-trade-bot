# 649# retrain_scheduler データ鮮度チェック分離

## 概要
`sac_retrain_scheduler.py` の `ensure_data_fresh()` 呼び出しが `retrain_once()` 内に
しか存在せず、`should_retrain()` が `data_unchanged` を返す限り到達不可能だった
**chicken-and-egg デッドロック** を解消。

## 問題の構造

```
run_scheduler() main loop:
  ┌─ should_retrain() ← OHLCV mtime 変化を監視
  │   └─ data_unchanged → skip retrain ──┐
  │                                        │
  └─ retrain_once() ← should_retrain=true のみ到達
      └─ ensure_data_fresh() ← OHLCV を更新   ← 到達不可
                                               │
  ∴ OHLCV 更新されない → mtime 変化なし ← ───┘ (無限ループ)
```

- `should_retrain()` は OHLCV ファイルの mtime 変化を retrain trigger の一つとして使用
- `ensure_data_fresh()` が OHLCV を API から取得・更新する唯一の手段
- `ensure_data_fresh()` は `retrain_once()` 内にのみ存在
- `should_retrain()` が `data_unchanged` を返す → `retrain_once()` 呼ばれない → データ永久に古いまま

## 修正内容

### 1. `run_scheduler()` に独立した周期的データ鮮度チェックを追加

```python
# run_scheduler() main loop 内 (while not _shutdown_event.is_set():)
now = time.time()
if now - _last_data_freshness_check >= cfg.data_freshness_check_interval_sec:
    _last_data_freshness_check = now
    try:
        updated = ensure_data_fresh(
            cfg.ohlcv_path,
            max_stale_hours=cfg.max_data_stale_hours,
        )
        if updated:
            logger.info(
                "[649#] Data refreshed by periodic check — "
                "next retrain trigger should detect mtime change"
            )
    except Exception as e:
        logger.warning(f"[649#] Periodic data freshness check failed: {e}")
```

- `should_retrain()` の **前** に実行 → retrain trigger に依存しない
- `retrain_once()` 内の既存呼び出しはバックアップとして保持
- 失敗しても `logger.warning` のみ → メインループ死亡を防止

### 2. `SACRetrainConfig` に新フィールド追加

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `data_freshness_check_interval_sec` | `int` | `3600` (1h) | データ鮮度チェック間隔 |
| `max_data_stale_hours` | `float` | `48.0` (48h) | この時間を超えると自動更新 |

### 3. バリデーション追加
- `data_freshness_check_interval_sec >= 60` (60s 未満はエラー)
- `max_data_stale_hours > 0` (ゼロ以下はエラー)

### 4. YAML 設定追加

```yaml
# configs/v460/experiments/g2_sac_train.yaml
sac_retrain:
  # 649# データ鮮度チェック (retrain trigger 非依存)
  data_freshness_check_interval_sec: 3600  # 鮮度チェック間隔 (1h)
  max_data_stale_hours: 48.0               # この時間超過で自動更新
```

### 5. `retrain_once()` 内の既存呼び出しを設定化

```python
# before (ハードコード)
ensure_data_fresh(cfg.ohlcv_path, max_stale_hours=48.0)

# after (設定から参照)
ensure_data_fresh(cfg.ohlcv_path, max_stale_hours=cfg.max_data_stale_hours)
```

## 修正後のフロー

```
run_scheduler() main loop:
  ┌─ [649#] periodic ensure_data_fresh()  ← retrain trigger に非依存
  │   └─ OHLCV 更新 → mtime 変化
  │
  ├─ should_retrain() ← mtime 変化を検出 → true
  │
  └─ retrain_once()
      └─ ensure_data_fresh() (バックアップ)
```

## 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/ml/sac_retrain_scheduler.py` | SACRetrainConfig 2フィールド追加、バリデーション追加、run_scheduler() に周期チェック追加、retrain_once() 内 max_stale_hours 設定化 |
| `configs/v460/experiments/g2_sac_train.yaml` | `data_freshness_check_interval_sec`, `max_data_stale_hours` 追加 |
| `AGENTS.md` | 現行アーキテクチャ要点を 649# 時点に更新、retrain_scheduler 記載追加 |
| `CHANGELOG.md` | 649# エントリ追加 |
| `tests/unit/v460/test_sac_retrain_scheduler.py` | `TestDataFreshnessDecoupling649` (7 cases) 追加 |

## テスト

`TestDataFreshnessDecoupling649` — 7テストケース:

| テスト | 内容 |
|---|---|
| `test_config_defaults` | 新フィールドのデフォルト値 (3600s, 48.0h) |
| `test_config_from_yaml` | YAML パース (カスタム値 1800s, 24.0h) |
| `test_config_validation_interval_too_small` | interval < 60 でバリデーションエラー |
| `test_config_validation_stale_hours_zero` | stale_hours <= 0 でバリデーションエラー |
| `test_periodic_data_check_called_in_scheduler` | run_scheduler が retrain trigger 前にデータチェックを呼ぶ |
| `test_periodic_check_failure_resilience` | ensure_data_fresh 失敗時もループ継続 |
| `test_interval_respected` | チェック間隔が設定値を尊重 |

## 運用上の注意
- デフォルト設定 (1h チェック, 48h stale 閾値) は保守的。OHLCV 更新頻度に応じて調整可能。
- `data_freshness_check_interval_sec` を短くしすぎると API レートリミットに注意。
- 650# 分析で判明: sidecar が 93% stale だったのは retrain 自体が 1回も成功していない可能性がある。本修正により OHLCV は更新されるが、retrain 成功は別問題 (モデル品質、OOS 評価等)。
