# Codex Prompt: 408#/409# 統合修正 — CRITICAL/HIGH 課題修正 + デッドコード清掃

## ミッション

以下 2 つの調査レポートで発見された **CRITICAL/HIGH** の課題を修正し、
各修正に対するユニットテストを `tests/unit/v460/test_codex_408_409_fixes.py` に追加してください。

- `docs/v460/408_phg_rpt_dead_code_analysis.md` — デッドコード調査・重複解消
- `docs/v460/409_phg_rpt_broad_discovery_scan.md` — 広域課題スキャン

> **重要**: 既存テスト (2208 tests) を壊さないこと。修正後に `python -m pytest tests/ -x --tb=short -q --no-cov --ignore=tests/integration/test_comprehensive_fixes.py` で全テスト通過を確認すること。

---

## カスケード発見プロトコル

修正中に **タスク一覧にない関連課題** を発見した場合、以下のプロトコルに従ってください。

1. **即座に修正可能** (S/M 工数、影響範囲が限定的) → その場で修正し、テストを追加する。
2. **大規模リファクタ** (L 工数、影響範囲が広い) → 修正せず、`CASCADE_DISCOVERIES.md` にエントリを追記する。
3. すべてのカスケード発見 (修正済み・未修正問わず) を `CASCADE_DISCOVERIES.md` に記録する。

### `CASCADE_DISCOVERIES.md` フォーマット

```markdown
# カスケード発見ログ

| # | 発見元タスク | ファイル:行 | 概要 | 重要度 | 対処 |
|---|---|---|---|---|---|
| D1 | T3 | replay_market.py:92 | 型アノテーション不足 | LOW | 修正済み |
| D2 | T14 | forced_balance.py:200 | penalty 上限なし | HIGH | 未修正 (要別タスク) |
```

> **判断基準**: 「修正しないと他のタスクの修正品質が落ちる」場合は即修正。「独立していて後回しにできる」場合は記録のみ。

---

## プロジェクト概要

- **Python 3.11**, SB3 (stable-baselines3), gymnasium, PyTorch, LightGBM
- **SAC 訓練**: `scripts/v460/lib/tasks/sac_train.py` → `HeavyTradingEnv` (1835 行)
- **報酬**: `RewardCalculator` (2252 行, 50 メソッド — God Object)
- **ライブ基盤**: `ztb/trading/live/` — IdempotencyStore, ServiceRunner, HealthMonitor 等
- **テスト**: 8,662 collected, 2208 passed, ~21% coverage
- **HEAD**: `5c3238f25` (latest)

---

## 修正タスク一覧

### Part A: 409# 広域スキャン由来 (T1–T10)

### タスク 1: IdempotencyStore の非原子的ロック修正 — CRITICAL

**ファイル**: `ztb/trading/live/core/idempotency_store.py` L43-70
**現状**:
```python
while not lock_acquired:
    try:
        with open(self._lock_file, "w") as f:
            f.write(str(os.getpid()))
        lock_acquired = True
    except (OSError, IOError):
        time.sleep(0.01)
```
`open(..., "w")` は既存ファイルを無条件に上書きするため、複数プロセスが同時にロックを取得できてしまう。

**修正方針**:
1. Windows 環境なので `msvcrt.locking()` または `os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)` を使い、原子的排他ロックに置き換える。
2. `O_CREAT | O_EXCL` がファイル存在時に `FileExistsError` を発生させる性質を利用する。
3. タイムアウト付きリトライ (最大 5 秒、0.01 秒間隔) を実装。
4. `finally` ブロックで確実にロックファイルを削除。
5. stale lock 検出: ロックファイルの PID が生存していなければ削除して再取得。

**テスト要件**:
```python
def test_process_lock_is_exclusive(tmp_path):
    """二重ロック取得が失敗すること"""

def test_process_lock_releases_on_exit(tmp_path):
    """contextmanager 終了で lock file が削除されること"""

def test_stale_lock_recovery(tmp_path):
    """存在しない PID の lock file は回収されること"""
```

---

### タスク 2: reward_components が最終 reward と乖離する — HIGH

**ファイル**: `ztb/trading/environment/heavy_env/core.py` L1426 付近
**現状**:
```python
# L1426
info["reward_components"] = reward_components.copy()
info.update(debug_info)
# ... 30行後 ...
# L1460: bankruptcy penalty が reward に適用されるが reward_components には反映されない
reward -= bankruptcy_penalty * self.config.reward_scaling
# L1481: drawdown penalty も同様
reward -= drawdown_penalty
```

**修正方針**:
1. bankruptcy penalty / drawdown penalty 適用後に `reward_components` へ追記する。
2. `info["reward_components"]` を最終 reward と整合させる。具体的には:
```python
# bankruptcy/drawdown penalty 適用後
if "bankruptcy" in info:
    reward_components["bankruptcy_penalty"] = -(bankruptcy_penalty * self.config.reward_scaling)
if "drawdown_penalty" in info:
    reward_components["drawdown_penalty"] = -info["drawdown_penalty"]
reward_components["final_reward"] = reward
info["reward_components"] = reward_components.copy()
```
3. `info["reward_components"]` の代入タイミングを後段に移動する。

**テスト要件**:
```python
def test_reward_components_include_bankruptcy_penalty():
    """bankrupt 時に reward_components["bankruptcy_penalty"] が存在し、final_reward と reward が一致すること"""

def test_reward_components_include_drawdown_penalty():
    """drawdown 超過時に reward_components["drawdown_penalty"] が存在すること"""

def test_reward_components_match_final_reward():
    """通常ケースで reward_components["final_reward"] == returned reward であること"""
```

---

### タスク 3: ReplayMarket.get_progress() ゼロ除算ガード — HIGH

**ファイル**: `ztb/trading/live/simulation/replay_market.py` L84-88
**現状**:
```python
def get_progress(self) -> float:
    if self._data is None:
        return 0.0
    return min(1.0, self._current_index / len(self._data))
```
`self._data` が空 DataFrame の場合、`len(self._data) == 0` でゼロ除算。

**修正方針**:
```python
if self._data is None or len(self._data) == 0:
    return 0.0
```

**テスト要件**:
```python
def test_replay_progress_none_data():
    assert market.get_progress() == 0.0

def test_replay_progress_empty_dataframe():
    # _data = pd.DataFrame() (空)
    assert market.get_progress() == 0.0

def test_replay_progress_normal():
    assert 0.0 < market.get_progress() <= 1.0
```

---

### タスク 4: service_runner の成功時無条件 restart 修正 — HIGH

**ファイル**: `ztb/trading/live/core/service_runner.py` L129-133
**現状**:
```python
def _should_restart(self, cycle_success: bool) -> bool:
    if cycle_success:
        return True  # 成功しても restart → perpetual restart loop
```

**修正方針**:
1. 成功時は `return False` に変更 (連続稼働は呼び出し側の `while` ループが担う)。
2. `_should_restart` は失敗時のリトライ判定のみに限定する。

**テスト要件**:
```python
def test_should_restart_false_after_success():
    assert runner._should_restart(True) is False

def test_should_restart_true_after_failure_within_limit():
    assert runner._should_restart(False) is True

def test_should_restart_false_when_max_exceeded():
    # max_restarts 回失敗後
    assert runner._should_restart(False) is False
```

---

### タスク 5: HealthMonitor の 1 秒ブロック解消 — HIGH

**ファイル**: `ztb/trading/live/core/health_monitor.py` L20-55
**現状**:
```python
cpu_percent = psutil.cpu_percent(interval=1)  # 1秒ブロック
process = psutil.Process()  # 毎回再生成
```

**修正方針**:
1. `psutil.cpu_percent(interval=None)` に変更 (non-blocking, 前回呼び出しからの差分)。
2. `psutil.Process()` を `__init__` で 1 回生成し、インスタンス変数でキャッシュする。
3. `process.cpu_percent()` も `interval=None` に明示。

**テスト要件**:
```python
def test_health_status_is_non_blocking(monkeypatch):
    """get_health_status() が 1 秒未満で完了すること"""
    start = time.monotonic()
    monitor.get_health_status()
    elapsed = time.monotonic() - start
    assert elapsed < 0.5  # 1秒ブロックがないこと

def test_process_handle_reused():
    """Process ハンドルが __init__ で生成されていること"""
    assert hasattr(monitor, '_process')
```

---

### タスク 6: 環境 import 失敗の握り潰し修正 — HIGH

**ファイル**: `ztb/trading/environment/__init__.py` L10-16
**現状**:
```python
try:
    from .environment import FlipHeavyTradingEnv, HeavyTradingEnv
except Exception:
    FlipHeavyTradingEnv = None
    HeavyTradingEnv = None
```
全例外を `None` に変換。依存エラーが後段で `NoneType` として爆発する。

**修正方針**:
1. `ImportError` のみを捕捉し、`warnings.warn()` で通知する。
2. それ以外の例外 (`RuntimeError`, `TypeError` 等) は re-raise する。
```python
try:
    from .environment import FlipHeavyTradingEnv, HeavyTradingEnv
except ImportError as e:
    import warnings
    warnings.warn(
        f"HeavyTradingEnv could not be imported (likely missing torch): {e}",
        ImportWarning,
        stacklevel=2,
    )
    FlipHeavyTradingEnv = None  # type: ignore[assignment,misc]
    HeavyTradingEnv = None  # type: ignore[assignment,misc]
```

**テスト要件**:
```python
def test_import_error_emits_warning(monkeypatch):
    """ImportError 時に warning が出て None がセットされること"""

def test_runtime_error_is_not_silenced(monkeypatch):
    """RuntimeError は re-raise されること"""
```

---

### タスク 7: sac_train.py の assert を ValueError に置換 — MEDIUM

**ファイル**: `scripts/v460/lib/tasks/sac_train.py` L445-447
**現状**:
```python
if oos_enabled:
    assert oos_eval_env is not None
    assert best_model_path is not None
```

**修正方針**:
```python
if oos_enabled:
    if oos_eval_env is None:
        raise ValueError("OOS evaluation enabled but oos_eval_env is None")
    if best_model_path is None:
        raise ValueError("OOS evaluation enabled but best_model_path is None")
```

**テスト要件**:
```python
def test_oos_missing_env_raises_value_error():
    """oos_enabled=True, env=None で ValueError"""

def test_oos_missing_path_raises_value_error():
    """oos_enabled=True, path=None で ValueError"""
```

---

### タスク 8: fill_test_cli の 10 秒固定 sleep 除去 — HIGH

**ファイル**: `scripts/v460/lib/fill_test_cli.py` L328 付近
**現状**:
```python
retrain_proc = subprocess.Popen(...)
time.sleep(10)
if retrain_proc.poll() is not None:
```

**修正方針**:
1. 固定 10 秒 sleep を短い poll ループに置換:
```python
# 最大10秒、0.5秒おきにプロセス生存確認
for _ in range(20):
    time.sleep(0.5)
    if retrain_proc.poll() is not None:
        break
```
2. これにより成功時は即座に次処理に進み、失敗時のみ最大 10 秒待つ。

**テスト要件**:
```python
def test_start_retrain_scheduler_exits_early_on_success(monkeypatch):
    """プロセスがすぐ起動成功した場合、10秒待たないこと"""
```

---

### タスク 9: tests/conftest.py の import 例外縮小 — HIGH

**ファイル**: `tests/conftest.py` L15-53
**現状**:
```python
except Exception:
    torch = None
# ...
except Exception:
    pass
```

**修正方針**:
1. `except Exception` → `except (ImportError, ModuleNotFoundError)` に縮小する。
2. `pass` → `logging.getLogger(__name__).debug("...", exc_info=True)` に変更する。
3. numpy 互換チェックは維持するが、numpy 自体の import 失敗は `ImportError` のみに限定。

**注意**: テスト collection が壊れないよう慎重に。torch/sb3 の optional 性は維持しつつ、unexpected error の握り潰しだけ排除する。

**テスト要件**:
```python
def test_conftest_limits_catch_to_import_error():
    """conftest.py 内の except が ImportError/ModuleNotFoundError に限定されていることを静的にチェック"""
    import ast
    # conftest.py を AST 解析し、bare except / except Exception がないことを確認
```

---

### タスク 10: behavior_optimization whitelist mapping の堅牢化 — HIGH

**ファイル**: `ztb/trading/environment/utils/config.py` L410 付近
**現状**:
```python
if "action_balance_target" in behavior_opt:
    instance.reward_settings.action_balance_target = float(...)
if "balance_penalty" in behavior_opt:
    instance.reward_settings.balance_penalty = float(...)
# 手動ホワイトリスト — 新キー追加時に漏れる
```

**修正方針**:
1. `RewardSettings` の `__dataclass_fields__` (または `__init__` のパラメータ) から許可キー一覧を自動生成する。
2. behavior_opt 内のキーが `RewardSettings` のフィールドに存在すれば自動で `setattr` する。
3. 未知キーは `logger.warning(f"Unknown behavior_optimization key: {key}")` で通知。
4. 型変換は `float()` をベースに、`bool` / `int` フィールド型に応じて分岐。

```python
# 方針案
_rs_fields = {f.name: f.type for f in dataclasses.fields(RewardSettings)}
for key, value in behavior_opt.items():
    if key in _rs_fields:
        setattr(instance.reward_settings, key, float(value))
    else:
        logger.warning(f"Unknown behavior_optimization key ignored: {key}")
```

**テスト要件**:
```python
def test_behavior_optimization_auto_maps_to_reward_settings():
    """behavior_optimization の全キーが RewardSettings に反映されること"""

def test_unknown_behavior_optimization_key_warns():
    """未知キーで warning が出ること"""
```

---

### Part B: 408# デッドコード調査由来 (T11–T16)

### タスク 11: デッドコード 3 ファイルを archived/ へ移動 — HIGH

**対象ファイル**:
1. `ztb/trading/environment/components/calculators/simplified_reward_calculator.py` (37 LOC) — tests-only、現行 v460 未使用
2. `ztb/trading/environment/components/reward/metrics.py` (301 LOC) — `LongTermMetrics` の非テスト caller なし
3. `ztb/trading/environment/bridge.py` (867 LOC) — 外部 import を確認できない

**修正方針**:
1. 上記 3 ファイルを以下に `git mv` する:
   - `ztb/trading/environment/archived/reward/simplified_reward_calculator.py`
   - `ztb/trading/environment/archived/reward/metrics.py`
   - `ztb/trading/environment/archived/bridge/bridge.py`
2. 移動先ディレクトリが存在しなければ作成する。
3. 移動後、既存テストに影響がないことを確認する。もし tests 側で import しているなら import path を更新する。
4. `ztb/trading/environment/components/reward_calculator.py` (17 LOC proxy shim) で `SimplifiedRewardCalculator` の re-export を削除する。

**テスト要件**:
```python
def test_dead_code_files_are_archived():
    """3 ファイルが archived/ 配下に存在し、元パスに存在しないこと"""
    import pathlib
    base = pathlib.Path("ztb/trading/environment")
    assert not (base / "components/calculators/simplified_reward_calculator.py").exists()
    assert not (base / "components/reward/metrics.py").exists()
    assert not (base / "bridge.py").exists()
    assert (base / "archived/reward/simplified_reward_calculator.py").exists()
    assert (base / "archived/reward/metrics.py").exists()
    assert (base / "archived/bridge/bridge.py").exists()
```

---

### タスク 12: `test_reward_calculation()` を本番コードから tests/ へ抽出 — HIGH

**ファイル**: `ztb/trading/environment/components/calculators/reward_calculator.py` L1947 付近
**現状**:
```python
def test_reward_calculation(self) -> dict:
    """Test reward calculation with dummy data."""
    ...
```
本番クラス `RewardCalculator` 内に self-test メソッドが存在する。外部 caller なし。

**修正方針**:
1. `test_reward_calculation()` メソッドを `RewardCalculator` から **削除** する。
2. 同等のテストロジックを `tests/unit/v460/test_codex_408_409_fixes.py` 内の `TestT12RewardCalculatorSelfTest` に移植する。
3. テスト内では `RewardCalculator` のインスタンスを生成して外部から同等の検証を行う。

**テスト要件**:
```python
def test_reward_calculator_no_self_test_method():
    """RewardCalculator に test_reward_calculation メソッドが存在しないこと"""
    assert not hasattr(RewardCalculator, 'test_reward_calculation')

def test_reward_calculation_via_external_test():
    """旧 self-test と同等の検証を外部テストで行う"""
```

---

### タスク 13: RewardCalculator の dead メソッドに deprecation warning を付与 — MEDIUM

**ファイル**: `ztb/trading/environment/components/calculators/reward_calculator.py`
**対象メソッド**:
- `get_current_regime()` (L196) — 外部 caller なし
- `reset_episode_state()` (L707) — 外部 caller なし

**修正方針**:
1. 各メソッドの先頭に `warnings.warn("...", DeprecationWarning, stacklevel=2)` を追加する。
2. docstring に `.. deprecated::` ディレクティブを追加する。
3. メソッド削除は行わない (将来のリリースで削除する前提)。

**テスト要件**:
```python
def test_get_current_regime_emits_deprecation():
    with pytest.warns(DeprecationWarning, match="get_current_regime"):
        calc.get_current_regime()

def test_reset_episode_state_emits_deprecation():
    with pytest.warns(DeprecationWarning, match="reset_episode_state"):
        calc.reset_episode_state()
```

---

### タスク 14: forced-balance mapping helper の二重定義解消 — HIGH

**ファイル**:
- `ztb/trading/environment/components/calculators/reward_calculator.py` L600-676 (`_map_forced_balance_penalty`, `_map_forced_balance_bonus`)
- `ztb/trading/environment/components/rewards/forced_balance.py` L85-152 (同名メソッド)

**現状**: 同等のロジックが `RewardCalculator` と `ForcedBalanceReward` の両方に存在する。

**修正方針**:
1. `ForcedBalanceReward` 側の実装を **正** (canonical) とする。
2. `RewardCalculator` 側の `_map_forced_balance_penalty()` / `_map_forced_balance_bonus()` を、`ForcedBalanceReward` のメソッドへ委譲するように書き換える:
```python
def _map_forced_balance_penalty(self, deviation: float) -> float:
    return ForcedBalanceReward._map_forced_balance_penalty_static(deviation)
```
3. `ForcedBalanceReward` に `@staticmethod` の正規実装を持たせる。
4. 既存の振る舞いが変わらないことをテストで確認する。

**テスト要件**:
```python
def test_forced_balance_penalty_canonical_matches():
    """RewardCalculator 経由と ForcedBalanceReward 経由で同一結果"""
    assert rc_result == fb_result

def test_forced_balance_bonus_canonical_matches():
    """同上 (bonus)"""
```

---

### タスク 15: hot path の `gc.collect()` を条件付き化 — HIGH

**ファイル**: `ztb/trading/environment/heavy_env/mixins/initialization.py` L192, L323
**関連**: `ztb/trading/environment/components/data_processor.py` L118
**現状**:
```python
self.df = self.data_processor.preprocess_data(base_df)
if df is None:
    del base_df
gc.collect()
```
env 初期化の hot path で無条件に `gc.collect()` を呼び、stop-the-world GC が発生する。

**修正方針**:
1. 無条件 `gc.collect()` を削除し、メモリ水準超過時のみ実行する条件付き GC に置き換える:
```python
import psutil
if psutil.virtual_memory().percent > 85:
    gc.collect()
```
2. `data_processor.py` 内の `gc.collect()` も同様に条件付き化する。
3. psutil が利用不可の場合は GC をスキップする (try/except ImportError)。

**テスト要件**:
```python
def test_env_init_does_not_force_gc_on_normal_path(monkeypatch):
    """メモリ使用率 50% 以下で gc.collect() が呼ばれないこと"""

def test_env_init_forces_gc_on_high_memory(monkeypatch):
    """メモリ使用率 90% 以上で gc.collect() が呼ばれること"""
```

---

### タスク 16: `assert True` の integration テストを実質的アサーションに置換 — MEDIUM

**ファイル**: `tests/integration/test_v459_phase0_integration.py` L270 付近
**現状**:
```python
# 全コンポーネントが正常動作
assert True
```

**修正方針**:
1. `assert True` を削除し、前段で生成されたデータに対する実質的な検証に置換する。
2. 少なくとも以下のいずれかを assert する:
   - `len(trade_history) > 0` (取引が発生していること)
   - portfolio value が初期値から変動していること
   - state transition が発生していること
3. 前段のコンテキストを読んで最も適切な invariant を選ぶ。

**テスト要件**: テスト自体の修正なので、追加テストは不要。修正後の integration テストが通ること。

---

## 制約事項

1. **テストファイル**: 全テストを `tests/unit/v460/test_codex_408_409_fixes.py` に集約。
2. **既存テスト不変**: `python -m pytest tests/ -x --tb=short -q --no-cov --ignore=tests/integration/test_comprehensive_fixes.py` で 2208+ passed を維持すること。
3. **型安全**: `Any` 型は使わない。`mypy --config-file mypy.ini` でエラーゼロを維持。
4. **命名規則**: テストクラス名は `TestT{N}{Description}` (例: `TestT1IdempotencyLock`, `TestT12SelfTestExtraction`)
5. **コミット不要**: 変更はステージングまで。コミットはレビュー後に行う。
6. **カスケード発見**: 修正中に関連課題を見つけたら、即修正できるものは修正し、できないものは `CASCADE_DISCOVERIES.md` に記録する (詳細はカスケード発見プロトコル参照)。

## docs 更新

修正完了後、以下を更新してください:

1. `docs/v460/409_phg_rpt_broad_discovery_scan.md` の「409# セッションで対処済みの項目」テーブルに Codex 修正分を追記。
2. `docs/v460/408_phg_rpt_dead_code_analysis.md` の末尾に「Codex 修正済み項目」セクションを追加し、T11–T14 の対処状況を記載。
3. `CHANGELOG.md` に 1 行追記: `- 2026-03-XX: Codex 408#/409# fixes — CRITICAL/HIGH 修正 + デッドコード清掃 (T1-T16)`

---

## 優先順位

| 順位 | タスク | 由来 | 重要度 | 理由 |
|---|---|---|---|---|
| 1 | T1 IdempotencyStore ロック | 409# | CRITICAL | 実口座の注文重複リスク |
| 2 | T2 reward_components 乖離 | 409# | HIGH | 学習品質・checkpoint 選定への影響 |
| 3 | T3 ReplayMarket ゼロ除算 | 409# | HIGH | 即座にクラッシュ |
| 4 | T4 service_runner restart | 409# | HIGH | perpetual restart loop |
| 5 | T5 HealthMonitor ブロック | 409# | HIGH | レイテンシ悪化 |
| 6 | T6 __init__.py 例外握り潰し | 409# | HIGH | デバッグ困難化 |
| 7 | T12 test_reward_calculation 抽出 | 408# | HIGH | 本番コードに self-test が混入 |
| 8 | T11 デッドコード archive | 408# | HIGH | 保守コスト・認知負荷 |
| 9 | T14 forced-balance 重複解消 | 408# | HIGH | DRY 違反・ドリフト源 |
| 10 | T10 behavior_opt mapping | 409# | HIGH | サイレント設定無効化 |
| 11 | T15 gc.collect() 条件付き化 | 409# | HIGH | 訓練 throughput 悪化 |
| 12 | T9 conftest.py 例外縮小 | 409# | HIGH | テスト品質 |
| 13 | T7 assert → ValueError | 409# | MEDIUM | 本番安全性 |
| 14 | T8 10秒 sleep 除去 | 409# | HIGH | 起動レイテンシ |
| 15 | T13 dead メソッド deprecation | 408# | MEDIUM | 将来の削除準備 |
| 16 | T16 assert True 置換 | 408# | MEDIUM | テスト品質向上 |
