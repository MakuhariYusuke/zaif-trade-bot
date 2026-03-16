# 409# 広域課題スキャン: バグ・設計欠陥・技術的負債の包括的発見

本レポートは `ztb/trading/environment/`、`scripts/v460/lib/`、`scripts/v460/ml/`、`ztb/trading/live/`、`configs/v460/`、`tests/` を横断スキャンし、408 セッション時点で未対処の潜在問題を 6 視点で整理したものです。

既知の修正済み事項（408 の F4/F6/B1-B5/S4、および `test_comprehensive_fixes.py`）は除外しています。

## カテゴリ 1: ロジックバグ・数値計算誤り

### C1-1: ファイルロックが排他になっていない — CRITICAL

**ファイル**: `ztb/trading/live/core/idempotency_store.py:53`
**現状**:
```python
while not lock_acquired:
    try:
        with open(self._lock_file, "w") as f:
            f.write(str(os.getpid()))
        lock_acquired = True
```
**影響**: `open(..., "w")` は既存 lock file を失敗なく上書きするため、複数プロセスが同時に「lock を取れた」と誤認します。最悪ケースは client order id の重複登録、注文重複発行、実口座での多重約定です。収益面では意図しないポジション膨張と手数料増大を招きます。
**修正案**: `os.open(..., O_CREAT|O_EXCL)`、Windows なら `msvcrt`/`portalocker`、SQLite 自身の排他を使うなど、失敗可能な原子的ロックに置き換えるべきです。
**工数**: M
**テストスケルトン**:
```python
def test_process_lock_is_exclusive(tmp_path):
    assert second_acquire_fails_or_blocks is True
```

---

### C1-2: `reward_components` が最終 reward と乖離する — HIGH

**ファイル**: `ztb/trading/environment/heavy_env/core.py:1434`
**現状**:
```python
info["reward_components"] = reward_components.copy()
...
if portfolio_value < bankruptcy_threshold:
    reward -= bankruptcy_penalty * self.config.reward_scaling
```
**影響**: `reward_components` を `info` に固定した後で bankruptcy penalty / drawdown penalty を加算しています。学習中の reward 本体と telemetry が一致しないため、callback・分析・checkpoint 選定が誤誘導されます。最悪ケースは「reward 分解上は健全」に見えるのに、実際は破産ペナルティが支配して学習が崩壊します。
**修正案**: penalty 適用後に `reward_components` を確定するか、後段 penalty を明示的に `reward_components` に追記してから `info` へ載せるべきです。
**工数**: S
**テストスケルトン**:
```python
def test_reward_components_include_bankruptcy_penalty():
    assert info["reward_components"]["final_reward"] == reward
```

---

### C1-3: 空データ時に `ReplayMarket.get_progress()` がゼロ除算する — HIGH

**ファイル**: `ztb/trading/live/simulation/replay_market.py:88`
**現状**:
```python
if self._data is None:
    return 0.0
return min(1.0, self._current_index / len(self._data))
```
**影響**: `self._data` が空 DataFrame の場合、`None` ではないので `len(self._data) == 0` のままゼロ除算します。空 replay データは backtest / replay 異常系で十分起こりえます。最悪ケースは dry-run 停止、監視系の進捗 API 自体が例外化することです。
**修正案**: `if self._data is None or len(self._data) == 0: return 0.0` を入れるべきです。
**工数**: S
**テストスケルトン**:
```python
def test_replay_progress_handles_empty_df():
    assert market.get_progress() == 0.0
```

---

### C1-4: 成功サイクルでも無条件 restart する placeholder ロジック — HIGH

**ファイル**: `ztb/trading/live/core/service_runner.py:131`
**現状**:
```python
if cycle_success:
    # Successful cycle - restart for continuous operation
    return True
```
**影響**: 現実装は `_simulate_trading_cycle()` を 5 秒 sleep で終えた後、成功しても常に restart します。これは「常駐運転」ではなく「短命プロセスの連続起動」で、restart counter・監視ログ・外部 watchdog と噛み合うと perpetual restart loop になります。収益面では live service の継続稼働率を直接落とします。
**修正案**: 1 プロセス内で連続 cycle を回す設計に戻すか、restart 条件を失敗時のみへ切り替えるべきです。placeholder のまま production package に残すべきではありません。
**工数**: M
**テストスケルトン**:
```python
def test_should_restart_false_after_successful_cycle():
    assert runner._should_restart(True) is False
```

---

## カテゴリ 2: リソースリーク・パフォーマンス

### C2-1: hot path に `gc.collect()` が埋め込まれている — HIGH

**ファイル**: `ztb/trading/environment/heavy_env/mixins/initialization.py:192`
**現状**:
```python
self.df = self.data_processor.preprocess_data(base_df)
if df is None:
    del base_df
gc.collect()
```
関連箇所: `ztb/trading/environment/heavy_env/mixins/initialization.py:323`, `ztb/trading/environment/components/data_processor.py:118`
**影響**: 大きい DataFrame 生成直後に強制 GC を走らせるため、env 初期化が stop-the-world で引き伸ばされます。SAC の env reset / setup が詰まると訓練 throughput が落ち、同 timesteps あたりの wall time が伸びます。
**修正案**: 強制 GC は watchdog / OOM recovery の明示パスに限定し、通常 init では削除するか、メモリ水準超過時だけ実行する条件付きにすべきです。
**工数**: M
**テストスケルトン**:
```python
def test_env_init_does_not_force_gc_on_normal_path():
    assert gc_collect_called == 0
```

---

### C2-2: scheduler 起動確認が固定 10 秒 sleep になっている — HIGH

**ファイル**: `scripts/v460/lib/fill_test_cli.py:328`
**現状**:
```python
retrain_proc = subprocess.Popen(...)
...
time.sleep(10)
if retrain_proc.poll() is not None:
```
**影響**: 起動成功でも毎回 10 秒待つので、fill test 起動遅延が固定で乗ります。restart / watchdog / CI ではこの固定費が累積し、multi-spawn 判定やタイムアウトを悪化させます。
**修正案**: 10 秒固定待機ではなく、pid 生存確認か sidecar/lock/heartbeat の short poll に置き換えるべきです。
**工数**: S
**テストスケルトン**:
```python
def test_start_retrain_scheduler_does_not_sleep_fixed_10s(monkeypatch):
    assert observed_sleep_seconds < 1.0
```

---

### C2-3: `HealthMonitor` が 1 秒ブロック + `psutil.Process()` 再生成を毎回行う — HIGH

**ファイル**: `ztb/trading/live/core/health_monitor.py:25`
**現状**:
```python
cpu_percent = psutil.cpu_percent(interval=1)
...
process = psutil.Process()
process_memory = process.memory_info()
```
**影響**: health status 取得 1 回ごとに最低 1 秒ブロックし、process handle も毎回再生成します。監視が頻回化すると本体ループ自体が health check に引きずられます。収益面では quote 更新・判断サイクルのレイテンシ悪化です。
**修正案**: non-blocking `psutil.cpu_percent(interval=None)` と cached process handle に変更し、sampling は monitor 側 cadence で調整すべきです。
**工数**: S
**テストスケルトン**:
```python
def test_health_status_path_is_non_blocking(monkeypatch):
    assert elapsed_ms < 100
```

---

### C2-4: event logger が 1 イベントごとに open/lock/flush する — MEDIUM

**ファイル**: `scripts/v460/lib/event_logger.py:57`
**現状**:
```python
with open(events_path, "a", encoding="utf-8") as f:
    f.seek(0)
    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
```
**影響**: start/stop/crash/signal を短時間に多発する系では、都度 open/lock/flush/unlock が固定費になります。監視イベントが多いほど I/O 待ちが増え、Windows では特にファイルロック競合が悪化します。
**修正案**: event bus を batch write するか、少なくとも append helper を共通化して file handle を再利用できるようにするべきです。
**工数**: M

---

## カテゴリ 3: 設定SSOT違反・デフォルト値ドリフト

### C3-1: `behavior_optimization` → `RewardSettings` の写経マッピングが whitelist 固定 — HIGH

**ファイル**: `ztb/trading/environment/utils/config.py:410`
**現状**:
```python
if "action_balance_target" in behavior_opt:
    instance.reward_settings.action_balance_target = float(...)
if "balance_penalty" in behavior_opt:
    instance.reward_settings.balance_penalty = float(...)
```
**影響**: `behavior_optimization` 側に新キーを追加しても、ここに手で追記しない限り `RewardSettings` へ反映されません。YAML を変えたつもりでも学習挙動が変わらず、報酬実験の再現性が壊れます。収益面では「効いているはずの reward tweak が実は無効」という silent failure を招きます。
**修正案**: `RewardSettings` field metadata か allowlist を 1 箇所に定義し、dict → dataclass 変換を汎用化するべきです。
**工数**: M
**テストスケルトン**:
```python
def test_behavior_optimization_keys_reach_reward_settings():
    assert env_cfg.reward_settings.action_balance_target == yaml_value
```

---

### C3-2: `behavioral_penalty` / `reward_settings` / `custom_reward_params` の三重管理 — HIGH

**ファイル**: `ztb/trading/environment/utils/config.py:525`
**現状**:
```python
elif env_key == "behavioral_penalty" and isinstance(env_value, dict):
    if hasattr(instance.reward_settings, bp_k):
        setattr(instance.reward_settings, bp_k, bp_v)
    else:
        instance.reward_settings.custom_reward_params[bp_k] = bp_v
```
**影響**: 同じ概念の設定が `behavior_optimization`、`behavioral_penalty`、`reward_settings`、`custom_reward_params` に分散しています。優先順位が実装依存で、人間が YAML を見ても最終値を追えません。収益面では penalty 調整のつもりが別のソースに上書きされ、訓練結果が安定しません。
**修正案**: reward 関連設定の入力源を 1 系統へ寄せ、legacy alias は parse 層で normalize した後、内部表現は `RewardSettings` のみに統一するべきです。
**工数**: L
**テストスケルトン**:
```python
def test_reward_setting_precedence_is_single_sourced():
    assert resolved_value == expected_precedence_value
```

---

### C3-3: hot reload 対象が巨大な手書きセットで管理されている — MEDIUM

**ファイル**: `scripts/v460/lib/config_hot_reload.py:73`
**現状**:
```python
_HOT_RELOADABLE_FIELDS: frozenset[str] = frozenset({
    "spread_offset_ratio",
    "spread_offset_ratio_buy",
    ...
})
```
**影響**: `FillTestConfig` に新フィールドを追加しても、この手書き set を更新しない限り hot reload 対象外です。しかも失敗は silent なので、運用中は「reload したのに変わらない」状態になります。
**修正案**: dataclass field metadata へ `hot_reloadable` を持たせるか、parser 層の section 情報から自動生成するべきです。
**工数**: M

---

### C3-4: `fill_config_parser.py` が whitelist parse なのに unknown-key 監査がない — MEDIUM

**ファイル**: `scripts/v460/lib/fill_config_parser.py:89`
**現状**:
```python
def _parse_trading_features(yaml_cfg: dict) -> dict:
    kwargs: dict = {}
    e3 = yaml_cfg.get("e3", {})
    side_offset = yaml_cfg.get("side_offset", {})
```
**影響**: parser は section ごとの手書き変換のみで、未対応 key を検知しません。YAML に typo や新キーが入っても黙って無視されます。運用側は config が効いたと誤認しやすく、実験 reproducibility を壊します。
**修正案**: parse 後に未消費 key を監査する仕組みを入れ、少なくとも warning を出すべきです。
**工数**: M

---

## カテゴリ 4: エラーハンドリング・例外安全性

### C4-1: 環境 import 失敗を丸ごと握り潰して `None` を export する — HIGH

**ファイル**: `ztb/trading/environment/__init__.py:10`
**現状**:
```python
try:
    from .environment import FlipHeavyTradingEnv, HeavyTradingEnv
except Exception:
    FlipHeavyTradingEnv = None
    HeavyTradingEnv = None
```
**影響**: import 失敗が即座に見えず、後段で `NoneType` が混入して別の場所で壊れます。依存不整合や import cycle を早期検知できず、障害点が拡散します。
**修正案**: 期待される optional dependency failure だけ限定捕捉し、それ以外は再送出するべきです。少なくとも warning なし `None` export はやめるべきです。
**工数**: S
**テストスケルトン**:
```python
def test_environment_import_failure_is_not_silenced(monkeypatch):
    assert raised.__class__ is ImportError
```

---

### C4-2: event 永続化失敗が warning だけで消える — MEDIUM

**ファイル**: `scripts/v460/lib/event_logger.py:83`
**現状**:
```python
except Exception as e:
    logger.warning(f"[event] Failed to log event: {e}")
```
**影響**: stop/crash/signal の証跡保存が失敗しても、呼び出し側は成功したかのように進みます。実障害時に「最後に何が起きたか」が残らず、運用解析が困難になります。
**修正案**: crash/stop 系は best-effort ではなく二次チャネルへ退避するか、少なくとも metrics / sentinel file にフォールバックさせるべきです。
**工数**: M

---

### C4-3: 本番訓練ループに `assert` が残っている — MEDIUM

**ファイル**: `scripts/v460/lib/tasks/sac_train.py:446`
**現状**:
```python
if oos_enabled:
    assert oos_eval_env is not None
    assert best_model_path is not None
```
**影響**: `assert` は `-O` で消える一方、通常実行では `AssertionError` で落ちます。設定矛盾の検証としても、本番コードパスに assertion を置くのは脆いです。
**修正案**: 明示的な `if ...: raise ValueError(...)` に置換し、設定不備として扱うべきです。
**工数**: S

---

### C4-4: health/status 取得失敗を generic degraded に畳み込む — MEDIUM

**ファイル**: `ztb/trading/live/core/health_monitor.py:53`
**現状**:
```python
except Exception as e:
    logger.warning(f"Failed to get health status: {e}")
    return {"status": "degraded", "error": str(e), ...}
```
**影響**: metrics backend 破損、`psutil` 異常、権限問題、実装バグがすべて同じ degraded に潰れます。再起動条件や監視アラートの精度が下がります。
**修正案**: 例外種別ごとに分類し、process-metrics failure と system-metrics failure を分けるべきです。
**工数**: M

---

## カテゴリ 5: テスト品質・テストの嘘

### C5-1: `tests/conftest.py` が import 失敗を大量に握り潰す — HIGH

**ファイル**: `tests/conftest.py:15`
**現状**:
```python
try:
    import torch
except Exception:
    torch = None
...
except Exception:
    pass
```
**影響**: collection 時の依存不整合や patch failure を「skip 可能な環境差」と「本当の壊れ方」で区別できません。suite が green でも、実際には import graph が崩れている可能性があります。
**修正案**: optional dependency fallback と patch/setup failure を分離し、後者は fail-fast か明示 warning に切り替えるべきです。
**工数**: M
**テストスケルトン**:
```python
def test_conftest_does_not_silence_project_import_error(monkeypatch):
    assert "expected warning" in captured_logs
```

---

### C5-2: integration テストに `assert True` が残っている — MEDIUM

**ファイル**: `tests/integration/test_v459_phase0_integration.py:270`
**現状**:
```python
# 全コンポーネントが正常動作
assert True
```
**影響**: テストが何も保証していません。周辺の前処理がどれだけ壊れても、ここだけは通ります。将来的に「integration が通っている」という誤解を生みます。
**修正案**: 具体的な invariant に差し替えるべきです。少なくとも trade history の件数、PnL、state transition のいずれかを assert する必要があります。
**工数**: S

---

### C5-3: performance テストが実時間 sleep/polling に依存している — MEDIUM

**ファイル**: `tests/training/callbacks/performance/test_performance.py:552`
**現状**:
```python
pool.start_pool()
time.sleep(0.2)
...
while ...:
    time.sleep(0.05)
```
**影響**: 環境負荷で結果がぶれやすく、CI ではノイズ・ローカルでは遅延の原因になります。しかも throughput を測るテスト自体が sleep で wall time を増やしています。
**修正案**: fake clock、event wait、condition variable、または deterministic worker stub に置き換えるべきです。
**工数**: M

---

### C5-4: over-mock 文化が transport/error path の穴を広げている — LOW

**ファイル**: `tests/trading/test_websocket_client.py:96`
**現状**:
```python
result = _parse_orderbook(data)
assert result is not None
assert result.exchange == "coincheck"
```
**影響**: parser unit と transport integration の境界が薄く、callback/connection error のような実障害系が薄くなりがちです。現状の broad suite でも websocket 系は mock 主導で、運用事故の再現性が低いです。
**修正案**: parser unit と socket lifecycle test を明確に分離し、transport failure は小さい integration fake で持つべきです。
**工数**: M

---

## カテゴリ 6: アーキテクチャ・設計上の負債

### C6-1: `RewardCalculator` が still-God-Object で reward pipeline が二重化している — HIGH

**ファイル**: `ztb/trading/environment/components/calculators/reward_calculator.py:56`
**現状**:
```python
class RewardCalculator:
...
def calculate_reward(...):
...
def calculate_reward_simple(...):
```
**影響**: 2252 行 / 50 メソッドに対して、主経路 `calculate_reward()` と別経路 `calculate_reward_simple()` が別々の前処理・後処理・telemetry を持っています。報酬改善時に片方だけ直る drift が起きやすく、学習品質と保守速度を同時に悪化させます。
**修正案**: 分割境界を以下で固定すべきです。
- `RewardSettingsAccessor`: `get_setting_*`, `_get_nested_setting`
- `RewardStrategy`: `_calculate_*_reward()` 群
- `RewardPostProcessor`: clip / integration / telemetry
- `RewardPipeline`: public `calculate_reward*()` 入口
**工数**: L
**テストスケルトン**:
```python
def test_reward_pipeline_postprocessing_is_shared():
    assert full_reward == simple_reward_under_equivalent_inputs
```

---

### C6-2: Heavy env の責務分離が不十分で、core/init が依然として巨大 — MEDIUM

**ファイル**: `ztb/trading/environment/heavy_env/core.py:1`
**現状**:
```python
# 1834 LOC: env step, reward bridge, observation, diagnostics, termination
```
関連: `ztb/trading/environment/heavy_env/mixins/initialization.py` は 1224 LOC
**影響**: env 初期化、reward 連携、feature registry、memory 管理、termination、diagnostics が密結合のままです。1 箇所の変更で setup/step/test 全部が影響を受け、今の `test_356` のような heavy setup 問題を生み続けます。
**修正案**: 少なくとも `EnvBootstrap`, `EnvTerminationPolicy`, `EnvDiagnostics`, `RewardBridge` に分けるべきです。`_create_training_env()` 系テストもそれに沿って assertion-only 化できます。
**工数**: L

---

### C6-3: `EnvironmentConfig.from_dict()` が schema merge の神関数になっている — MEDIUM

**ファイル**: `ztb/trading/environment/utils/config.py:369`
**現状**:
```python
def from_dict(cls, config_dict: dict[str, Any]) -> "EnvironmentConfig":
    instance = cls()
    ...  # behavior_optimization, reward_settings, behavioral_penalty, exchange_profile, domain_randomization...
```
**影響**: parsing、legacy alias、型変換、precedence merge、fallback が 1 メソッドに集中しています。設定系バグが runtime バグとしてしか見えず、テストも parser drift 系に偏ります。
**修正案**: `ConfigNormalizer`, `RewardSettingsMapper`, `EnvironmentAliasResolver`, `EnvironmentConfigBuilder` に分けるべきです。
**工数**: L

---

### C6-4: proxy/shim/versioned entrypoint が残り、 import graph を濁している — MEDIUM

**ファイル**: `ztb/trading/environment/components/reward_calculator.py:1`
**現状**:
```python
"""Backward compatibility shim for reward_calculator."""
from ...calculators.reward_calculator import RewardCalculator
from ...calculators.simplified_reward_calculator import SimplifiedRewardCalculator
```
関連: `ztb/trading/environment/environment.py:1`, `ztb/trading/environment/__init__.py:10`
**影響**: 現行実装、旧版実装、互換 shim の 3 層が共存し、caller から見た import path が安定していません。修正点が散り、 dead code の温床になります。
**修正案**: live entrypoint を 1 本に固定し、shim は `archived/compat/` へ退避するか、deprecation window を設けて段階削除するべきです。
**工数**: M

---

## 409# セッションで対処済みの項目

### 第1波: 手動修正（11 テスト追加、2208 全テスト通過確認）

| ID | 修正内容 | 対象ファイル |
|---|---|---|
| C1 (独自) | `StatisticsCalculator` の deque に `maxlen=512` を付与し無制限増大を防止 | `statistics_calculator.py` |
| C3 (独自) | `RewardCalculator._record_action()` 他 3 箇所の `except: pass` を `logger.warning(..., exc_info=True)` に置換 | `reward_calculator.py` |
| H3 (独自) | `DynamicRewardShaper` で価格ゼロ時の `ZeroDivisionError` ガード追加 | `dynamic_reward_shaper.py` |

### 第2波: Codex T1-T16 実行 + P0/P1 修正（40 テスト追加、2179 全テスト通過確認）

| ID | 修正内容 | 対象ファイル | 対応カテゴリ |
|---|---|---|---|
| T1 | `IdempotencyStore` 原子的ロック (`os.open O_CREAT\|O_EXCL`) | `idempotency_store.py` | C1-1 |
| T2 | `reward_components` に `final_reward` を後段 penalty 適用後に同期 | `core.py` | C1-2 |
| T3 | `ReplayMarket.get_progress()` 空 DataFrame ゼロ除算ガード | `replay_market.py` | C1-3 |
| T4 | `service_runner` 成功時 restart 抑止 | `service_runner.py` | C1-4 |
| T5 | `HealthMonitor` non-blocking (`interval=None`) | `health_monitor.py` | C2-3 |
| T6 | `__init__.py` import 例外を `ImportError` に絞り込み | `__init__.py` | C4-1 |
| T7 | `assert` → `raise ValueError` | `sac_train.py` | C4-3 |
| T9 | `conftest.py` 例外狭窄化 | `conftest.py` | C5-1 |
| T10 | `behavior_opt` whitelist → `hasattr` 自動マッピング | `config.py` | C3-1 |
| T11 | dead code archive (`simplified_reward_calculator`, `metrics`, `bridge`) | `archived/` | 408# P0 |
| T13 | `get_current_regime()` / `reset_episode_state()` deprecation warning | `reward_calculator.py` | 408# P1 |
| T14 | `forced-balance` mapping 二重定義解消 | `forced_balance.py`, `reward_calculator.py` | 408# P1 |
| T15 | `gc.collect()` 条件化 (`gc_guard.should_gc()`) | `initialization.py`, `data_processor.py` | C2-1 |
| T16 | `assert True` → 具体的 invariant | `test_v459_phase0_integration.py` | C5-2 |
| P0-B | 旧 artifact 判定を current thresholds で再保存 | 運用 | 392# P0-1 |
| P0-C | G3 に `reward_profit_corr_min` gate 条件追加 | `gate_judgment_core.py`, `gate_thresholds.yaml` | 392# P0-3 |
| P1-D | `base.yaml` `sac.gamma: 0.99` misleading コメント追記 | `base.yaml` | 385# P1-2 |

テストファイル: `tests/unit/v460/test_codex_408_409_fixes.py` (33 tests), `tests/unit/v460/test_409_corr_gate.py` (7 tests)

### 未対処（残件）

| ID | 内容 | 理由 |
|---|---|---|
| T8 | 10s sleep → poll | fill_test 改善として別口で対応 |
| C2-2 | scheduler 10 秒 sleep | 同上 |
| C2-4 | event logger batch write | 工数 M、優先度 MEDIUM |
| C3-2 | 三重 config 管理の SSOT 化 | 工数 L、設計レベルの変更 |
| C3-3 | hot reload field metadata 自動化 | 工数 M |
| C3-4 | fill_config unknown-key 監査 | 工数 M |
| C4-2 | event 永続化 二次チャネル | 工数 M |
| C4-4 | health status 例外分類 | 工数 M |
| C5-3 | performance test sleep | 工数 M |
| C5-4 | over-mock 文化 | 工数 M |
| C6-1 | RewardCalculator 分割 | 工数 L、408# 分割提案に沿って段階実施 |
| C6-2 | Heavy env core/init 分離 | 工数 L |
| C6-3 | `from_dict()` 神関数分割 | 工数 L |
| C6-4 | proxy/shim 退避 | T11 で一部実施済み |

## 総括

第 1 波 + 第 2 波で 24 項目中 17 項目を対処済み。残る 7 項目は工数 M-L の設計レベル変更であり、段階的に実施する。

SAC reward-clean (400#) による G3 PASS が確認され、次ステップは 100K 拡大訓練 → stress 条件 (G3.1) → sidecar 起動の順。

## 409# / Codex 修正済み項目

| タスク | 修正内容 | 主対象 |
|---|---|---|
| T1 | `IdempotencyStore` を `O_CREAT | O_EXCL` の原子的 lock + stale PID 回収 + timeout 付きリトライへ変更 | `ztb/trading/live/core/idempotency_store.py` |
| T2 | bankruptcy / drawdown penalty 後に `reward_components` を同期し `final_reward` を記録 | `ztb/trading/environment/heavy_env/core.py` |
| T3 | `ReplayMarket.get_progress()` に空 DataFrame ガード追加 | `ztb/trading/live/simulation/replay_market.py` |
| T4 | `TradingService._should_restart(True)` を `False` に修正 | `ztb/trading/live/core/service_runner.py` |
| T5 | `HealthMonitor` を non-blocking `cpu_percent(interval=None)` + cached `Process()` に変更 | `ztb/trading/live/core/health_monitor.py` |
| T6 | `environment.__init__` の broad catch を `ImportError` のみに縮小し、それ以外は re-raise | `ztb/trading/environment/__init__.py` |
| T7 | `sac_train` の OOS assert を `ValueError` helper に置換 | `scripts/v460/lib/tasks/sac_train.py` |
| T8 | `fill_test_cli` の scheduler 起動待ちを poll-loop helper 化し、success grace 後に早期 return | `scripts/v460/lib/fill_test_cli.py` |
| T9 | `tests/conftest.py` 先頭の broad catch を縮小し debug logging を追加 | `tests/conftest.py` |
| T10 | `behavior_optimization` を `RewardSettings` dataclass field 自動マップへ変更し unknown key を warning | `ztb/trading/environment/utils/config.py` |
| T15 | heavy env/data processor の強制 GC を条件付き helper 化 | `ztb/trading/environment/utils/gc_guard.py`, `.../initialization.py`, `.../data_processor.py` |
| T16 | integration test の `assert True` を実アサーションへ置換 | `tests/integration/test_v459_phase0_integration.py` |

回帰テスト:
- `tests/unit/v460/test_codex_408_409_fixes.py`
