# 265# run_continuous Extract Methods + SkipGateAdapter Protocol

| 項目 | 値 |
|---|---|
| Issue | 265# |
| 種別 | refactor + impl |
| フェーズ | phg (横断品質改善) |
| Commit | (pending) |
| テスト | 3639 passed, 32 skipped (変動なし) |
| 元チケット | 257# P1-1 (run_continuous 1694行), 257# P2-5 (adapter: object) |

---

## 背景

`FillLoopOrchestratorMixin.run_continuous()` は 1694 行の god method であり、
初期化・サイクル後処理・進捗/適応・終了処理が一体化して可読性・保守性を著しく損なっていた。

また `SkipGateEvaluator.evaluate()` では `adapter: object` + `getattr` パターンが
型安全性を阻害していた。

## 変更概要

### 1. run_continuous Extract Methods (1694 → 1221 行, 28% 削減)

#### RunSessionState dataclass

ループ内ローカル変数を構造体に集約。メソッド間の状態共有を明示的にした。

```python
@dataclass
class RunSessionState:
    total_count: int = 0
    filled_count: int = 0
    cumulative_pnl_jpy: float = 0.0
    cumulative_btc_delta: float = 0.0
    cumulative_adverse_count: int = 0
    cumulative_adverse_bps: float = 0.0
    batch: list[dict[str, object]] = field(default_factory=list)
    batch_size: int = 50
```

#### 抽出メソッド 4 件

| メソッド | 行数 | 責務 |
|---|---|---|
| `_init_run_session()` | ~175 | ロック取得, trades ヘルスチェック, dynamic loss_cap, stale order cancel, resume, PnL 蓄積, soft_loss_cap 復元, regime/DD/kill warmup → `RunSessionState` 返却 |
| `_process_post_cycle(record, next_side, st)` | ~150 | PnL 追跡, loss cooldown, toxic veto, 累積カウンタ更新, DD 更新, soft/hard loss cap, FastFillDefense, batch 永続化 |
| `_log_progress_and_adapt(next_side, st)` | ~130 | 進捗ログ, MTM equity, AS summary, guard categories, health monitor, state persistence, loss_cap refresh, パラメータ適応, lot sizing, 停止条件判定 |
| `_finalize_run(st, heartbeat_task)` | ~35 | batch 最終保存, state 保存, heartbeat cancel, records reload |

#### 再利用性検討

| メソッド | 他での適用 | 判定 |
|---|---|---|
| `_init_run_session` | run_continuous 専用 (ロック・resume など orchestrator 固有) | 単独使用 |
| `_process_post_cycle` | 将来 backtest/dry-run ループで再利用可能 | 拡張候補 |
| `_log_progress_and_adapt` | adaptation_engine と協調するため orchestrator に留める | 単独使用 |
| `_finalize_run` | 小規模だが finalization パターンとして汎用 | 拡張候補 |

#### 既存実装との重複チェック

- 初期化処理: `_init_run_session` 内の各処理は既存ヘルパー (`_cancel_stale_orders`, `_restore_soft_loss_cap` 等) への委譲が主体。新規ロジック追加なし。
- バッチ永続化: 既存 `_save_batch()` をそのまま使用。
- 状態永続化: 既存 `_save_state()` をそのまま使用。

### 2. SkipGateAdapter Protocol

`skip_gate_evaluator.py` の `adapter: object` を Protocol 化。

```python
class SkipGateAdapter(Protocol):
    async def get_recent_trades(
        self, symbol: str, *, limit: int = 50
    ) -> list[object]: ...

    async def get_orderbook(
        self, symbol: str, *, depth: int = 5
    ) -> object | None: ...
```

| 変更点 | Before | After |
|---|---|---|
| 型注釈 | `adapter: object` | `adapter: SkipGateAdapter` |
| メソッド呼び出し | `getattr(adapter, "get_recent_trades", None)` + `if callable(...)` | `adapter.get_recent_trades(...)` 直接呼び出し |
| 同上 (orderbook) | `getattr(adapter, "get_orderbook", None)` + `if callable(...)` | `adapter.get_orderbook(...)` 直接呼び出し |

### 3. P3-3 config_access docstring

`config_access.py` の `value: object` に設計意図 docstring を追加。
`object` は意図的な使用 (JSON 任意型を表現) であり、Protocol 化対象外であることを明記。

## 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/fill_loop_orchestrator.py` | RunSessionState + 4 extract methods, run_continuous 簡素化 |
| `scripts/v460/lib/skip_gate_evaluator.py` | SkipGateAdapter Protocol, getattr 排除 |
| `scripts/v460/lib/config_access.py` | value:object docstring |
| `tests/unit/v460/test_093_side_params.py` | ソース検査先を `_process_post_cycle` に変更 |
| `tests/unit/v460/test_113_resilience.py` | ソース検査先を `_log_progress_and_adapt` / `_finalize_run` に変更 |
| `tests/unit/v460/test_139_review_fixes.py` | `batch` → `st.batch` 検査 |
| `tests/unit/v460/test_145_structural_fixes.py` | ソース検査先を `_finalize_run` に変更 |
| `tests/unit/v460/test_fill_quality.py` | ソース検査先を `_process_post_cycle` に変更 (3 テスト) |

## リスク評価

| リスク | 対策 |
|---|---|
| Extract method で動作変更 | ロジック変更なし、テスト全パス (3639 passed) |
| RunSessionState のフィールド漏れ | 全ループ変数を網羅的に移行、grep で裸の変数名が残っていないことを確認 |
| SkipGateAdapter 破壊的変更 | CoincheckAdapter は既に両メソッドを実装済み |
| 非同期変更 (`_log_progress_and_adapt`) | `async def` 化、`await` に統一 |

## 今後の拡張候補

- `_process_post_cycle` の backtest ループへの再利用
- `_finalize_run` パターンの他 orchestrator への展開
- `run_continuous` の更なる分割 (1221 行 → 目標 800 行以下)
