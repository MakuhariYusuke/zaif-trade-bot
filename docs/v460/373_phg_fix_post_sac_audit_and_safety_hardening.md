# 373# Post-SAC 安全性監査 & ハードニング

**Date**: 2026-03-10
**Scope**: ztb/, scripts/v460/lib/, scripts/v460/ml/ — 372# SAC/skip_gate 以外の未対処問題
**Commits**: `43159ff31` (F1-F4), `TBD` (F6/F8/F9)

---

## Finding 1: `order_quantity` / `min_order_btc` 未検証 — ゼロ or 負値で API 400 & ゼロ除算

**Severity**: CRITICAL
**File**: [fill_config_validation.py](scripts/v460/lib/fill_config_validation.py)
**Issue**: `validate_fill_config()` に `order_quantity > 0` と `min_order_btc > 0` の検証が欠落。
`balance_checker.py` L148 で `int(max_base / self._min_order_btc)` としてゼロ除算が発生する。
YAML 設定ミスで `order_quantity: 0` や `min_order_btc: 0` と書くだけで本番クラッシュ。

**Affected code**: `balance_checker.py` L148, `lot_manager._clamp_lot()`, `fill_cycle_executor` place_order

**Fix**:
```python
# fill_config_validation.py に追加
if config.order_quantity <= 0:
    raise ValueError(f"order_quantity must be > 0, got {config.order_quantity}")
if config.min_order_btc <= 0:
    raise ValueError(f"min_order_btc must be > 0, got {config.min_order_btc}")
if config.order_quantity < config.min_order_btc:
    raise ValueError(
        f"order_quantity ({config.order_quantity}) must be >= "
        f"min_order_btc ({config.min_order_btc})"
    )
```

---

## Finding 2: `balance_checker.check()` が全例外を飲み込んで `False` (= 注文続行) を返す

**Severity**: CRITICAL
**File**: [balance_checker.py](scripts/v460/lib/balance_checker.py#L131)
**Issue**: `check()` の `except Exception` で `False` を返すと「残高チェックをスキップして発注続行」になる。
API 接続断、認証エラー、レート制限エラーなど致命的な障害時にも注文が発行される。
残高 0 なのに buy 注文を出す、BTC 0 なのに sell 注文を出す、等の財務リスク。

```python
except Exception as e:
    logger.warning(f"[balance] Pre-flight check failed — proceeding: {e}")
return False  # ← 注文続行
```

**Fix**: ネットワーク/認証系エラーを分離し、接続障害時は `True` (= スキップ) を返す:
```python
except (ConnectionError, TimeoutError, aiohttp.ClientError) as e:
    logger.error(f"[balance] Pre-flight check network error — SKIPPING cycle: {e}")
    return True  # 安全側: スキップ
except Exception as e:
    logger.warning(f"[balance] Pre-flight check failed — proceeding: {e}")
    return False
```

---

## Finding 3: `SACRetrainConfig` にバリデーション不在 — 無効な学習設定がサイレントに適用

**Severity**: IMPORTANT
**File**: [sac_retrain_scheduler.py](scripts/v460/ml/sac_retrain_scheduler.py#L62)
**Issue**: `SACRetrainConfig` に `__post_init__` バリデーションがない。
`learning_rate: -1`, `batch_size: 0`, `gamma: 2.0`, `val_ratio: 1.5` etc. がすべて受け入れられる。
無意味なモデルが deploy → 投資判断に使用される。

**Fix**: `__post_init__` で基本的な値域チェックを追加:
```python
def __post_init__(self) -> None:
    if self.learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
    if self.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
    if not (0.0 < self.gamma <= 1.0):
        raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
    if not (0.0 < self.val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {self.val_ratio}")
    if self.total_timesteps < 1:
        raise ValueError(f"total_timesteps must be >= 1, got {self.total_timesteps}")
    if self.retrain_interval_sec <= 0:
        raise ValueError(f"retrain_interval_sec must be > 0, got {self.retrain_interval_sec}")
    if self.check_interval_sec <= 0:
        raise ValueError(f"check_interval_sec must be > 0, got {self.check_interval_sec}")
```

---

## Finding 4: `sidecar_signal_io.read_sidecar_signal()` TOCTOU 競合

**Severity**: IMPORTANT
**File**: [sidecar_signal_io.py](scripts/v460/lib/sidecar_signal_io.py#L103)
**Issue**: `path.exists()` チェック後、`path.read_text()` までの間に SAC scheduler が atomic write (rename) を実行すると `FileNotFoundError` 発生の可能性がある。
現在は `OSError` が catch されるので crash にはならないが、`exists()` チェックと `read_text()` の間の一瞬、Windows の `os.replace()` がブロックする場合に読込失敗となり `None` 返却 → sidecar バイアスが一時的にゼロにフォールバック。

```python
if not path.exists():   # ← check
    return None
try:
    raw = path.read_text(encoding="utf-8")  # ← use (race window)
```

**Fix**: `exists()` チェックを削除し read_text の例外で統一:
```python
try:
    raw = path.read_text(encoding="utf-8")
except FileNotFoundError:
    return None  # ファイル未存在は正常 (初回起動時)
except OSError as e:
    logger.warning(f"Sidecar signal read error: {e}")
    return None
```

---

## Finding 5: `config_hot_reload._do_reload()` が mid-cycle で config を in-place 変更

**Severity**: IMPORTANT
**File**: [config_hot_reload.py](scripts/v460/lib/config_hot_reload.py#L613)
**Issue**: `maybe_reload()` は `_post_cycle_sleep()` 内で呼ばれるため、通常はサイクル間の安全ポイント。
しかし `current_values[field_name] = new_val` は Python dict への直接代入であり、コンポーネント再構築中に次サイクルのプリフライトが始まると、中途半端な状態のコンフィグでサイクルが実行される可能性がある。
Python GIL により dict 代入は atomic だが、複数フィールドの更新は非 atomic。
_例_: `spread_offset_ratio` だけ更新され、`min_offset_jpy` が旧値のまま → offset 計算の不整合。

**Impact**: 単一 asyncio イベントループ (シングルスレッド) なので実際に同時実行は起きないが、
`_post_cycle_sleep` 中に `maybe_reload → _do_reload` 内で `runner._rebuild_*()` が `await` を含む場合、
イベントループの yield ポイントで他タスクが config を読む可能性がある。
現時点では `_do_reload` は同期関数のため **低リスク** だが、将来の非同期化で顕在化する。

**Fix (防御的)**: フィールド更新をバッチ化してから1回 commit するか、docstring に「sync-only」制約を明記。

---

## Finding 6: `maker_microstructure.py` の `type: ignore[union-attr]` が None 参照を隠蔽

**Severity**: IMPORTANT
**File**: [maker_microstructure.py](scripts/v460/lib/maker_microstructure.py#L222)
**Issue**: `getattr(self, "_fill_prob_model", None)` で None チェック後に `self._fill_prob_model.k` を参照しているが、`type: ignore[union-attr]` で mypy を黙らせている。`getattr` の結果が `None` でないことを保証しているのはランタイム `getattr` だが、mypy が検出できる None 安全性を失っている。
また `_fill_prob_model` の型宣言がクラスレベルで明示されていない (duck-typing 依存)。

```python
if (
    getattr(self, "_fill_prob_model", None) is not None
    and cfg.glft_dynamic_k_enabled
    and self._fill_prob_model.k > 0  # type: ignore[union-attr]
):
    k = self._fill_prob_model.k  # type: ignore[union-attr]
```

**Fix**: クラスレベルで型宣言し、narrowing で type: ignore を排除:
```python
_fill_prob_model: FillProbModelLike | None = None

# usage:
_fpm = self._fill_prob_model
if _fpm is not None and cfg.glft_dynamic_k_enabled and _fpm.k > 0:
    k = _fpm.k
```

---

## Finding 7: `sac_retrain_scheduler.from_yaml_dict()` — 大量の `# type: ignore[assignment]`

**Severity**: IMPORTANT
**File**: [sac_retrain_scheduler.py](scripts/v460/ml/sac_retrain_scheduler.py#L119)
**Issue**: `cfg.get("data", {})` の戻り値は `dict | object` 型だが、`dict` に直接キャストしている。
7箇所の `# type: ignore[assignment]` が連続しており、YAML に不正な値が入った場合 (例: `data: null`)
後続の `.get()` で `AttributeError` がサイレントに発生する。

```python
data_cfg: dict = cfg.get("data", {})  # type: ignore[assignment]
sac_cfg: dict = cfg.get("sac_hyperparameters", {})  # type: ignore[assignment]
# ... 5 more
```

**Fix**: 明示的な型チェック or `cast` with validation:
```python
data_cfg = cfg.get("data") or {}
if not isinstance(data_cfg, dict):
    raise TypeError(f"Expected dict for 'data', got {type(data_cfg).__name__}")
```

---

## Finding 8: `fill_cycle_executor` — `_order_lot` の最終 `max_lot` 再検証なし

**Severity**: IMPORTANT
**File**: [fill_cycle_executor.py](scripts/v460/lib/fill_cycle_executor.py#L750-L800)
**Issue**: `_effective_order_lot()` で `max_lot` クランプ済みだが、その後の乗数チェーン:
- `_alert_lot_mult` (clamped to [0.01, 1.0] → OK)
- `_recovery_lm` (< 1.0 条件付き → OK)
- `_dd_side_scale` (< 1.0 条件付き → OK)
- `_cd_lm` (< 1.0 条件付き → OK)

各乗数は `max(self.config.order_quantity, lot * mult)` でフロア処理している。
**現状は全乗数が ≤ 1.0 のため max_lot 超えは発生しない**。
ただし今後 > 1.0 の乗数が追加された場合、max_lot を超えるロットが exchange に送られる。

**Fix (防御的)**: 最終ロットにも `min(max_lot)` クランプを追加:
```python
if self.config.max_lot > 0:
    _order_lot = min(_order_lot, self.config.max_lot)
```

---

## Finding 9: `order_monitor.monitor()` — poll loop 内の `except Exception` が全エラーを飲み込む

**Severity**: IMPORTANT
**File**: [order_monitor.py](scripts/v460/lib/order_monitor.py#L411)
**Issue**: ポーリングループ内の `except Exception as e: logger.warning(f"Poll error: {e}")` が
`KeyboardInterrupt` 以外の全例外を飲み込む。
`asyncio.CancelledError` (Python 3.9+ は `BaseException`) は通常 `except Exception` では捕捉されないが、
`SystemExit` は捕捉される。より重要なのは、連続的に `get_order_status` が失敗する場合、
タイムアウトまでループが空回りし、注文のキャンセルタイミングを逸する可能性がある。

```python
except Exception as e:
    logger.warning(f"Poll error: {e}")
    # → continue (次の poll_interval_sec 後にリトライ)
```

**Fix**: 連続エラーカウンタを追加し、閾値超過でループを抜ける:
```python
_consecutive_poll_errors = 0
# ... in loop:
except Exception as e:
    _consecutive_poll_errors += 1
    logger.warning(f"Poll error ({_consecutive_poll_errors}): {e}")
    if _consecutive_poll_errors >= 5:
        logger.error("Too many consecutive poll errors — cancelling order")
        cancel_reason_poll = CR.POLL_ERROR_LIMIT
        break
```

---

## Finding 10: 単体テスト欠落 — `balance_checker` / `order_monitor` の行動テスト不在

**Severity**: INFORMATIONAL
**File**: tests/unit/v460/
**Issue**: `balance_checker.py` と `order_monitor.py` の現在のテストは**ソースコード検査テスト**
(「文字列 X がソースに含まれるか」) が主体であり、実際の行動テスト (mock 使用のシナリオ実行) は
`test_237_phantom_position_guard.py` 等に部分的にあるのみ。

とくに以下のシナリオが未テスト:
- `balance_checker.check()` 例外時のフォールバック挙動
- `balance_checker._check_buy()` で `price=None` が返る場合
- `order_monitor.monitor()` の連続リトライ失敗パス
- `dust_buy_pending` と `dust_sweep_active` の並行状態遷移
- `config_hot_reload` 中のコンポーネント再構築失敗時の復旧

**Fix**: 上記シナリオのモックベーステストを追加。

---

## Summary

| # | Severity | Area | Issue | Status |
|---|----------|------|-------|--------|
| 1 | CRITICAL | Config Validation | `order_quantity` / `min_order_btc` ゼロ除算 | ✅ `43159ff31` |
| 2 | CRITICAL | Exception Handling | `balance_checker.check()` 全例外飲み込み→注文続行 | ✅ `43159ff31` |
| 3 | IMPORTANT | Config Validation | `SACRetrainConfig` バリデーション不在 | ✅ `43159ff31` |
| 4 | IMPORTANT | Concurrency | `read_sidecar_signal()` TOCTOU 競合 | ✅ `43159ff31` |
| 5 | IMPORTANT | Concurrency | Config hot-reload 非 atomic 更新 | ⚪ WONTFIX (asyncio single-thread) |
| 6 | IMPORTANT | Type Safety | `maker_microstructure` type:ignore 隠蔽 | ✅ 本コミット |
| 7 | IMPORTANT | Type Safety | `sac_retrain_scheduler` type:ignore 連続 | ⚪ ACCEPTED (YAML dict 特性) |
| 8 | IMPORTANT | Financial Risk | `_order_lot` 最終 max_lot 未検証 | ✅ 本コミット |
| 9 | IMPORTANT | Exception Handling | `order_monitor` poll 全例外飲み込み | ✅ 本コミット |
| 10 | INFO | Test Coverage | `balance_checker` / `order_monitor` 行動テスト不在 | ✅ 本コミット (F2/F9 テスト) |
