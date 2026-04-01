# Codex Task: 688# _execute_skip() 監査 + Decision Path RT ID (684# M2 / 72bd8e713)

## 目的
1. 72bd8e713 で追加された `_execute_skip(update_last_side=...)` の全 call site を監査し、パラメータの正確性を保証する
2. 各サイクルに一意の decision trace ID を付与し、SkipGate → offset pipeline → timeout → fill/cancel の全ステージをトレースできるようにする

## 背景

### 72bd8e713 state separation の監査必要性
- `_execute_skip()` に `update_last_side: bool` パラメータが追加された
- `True`: "この skip は side 試行の結果" → `last_attempted_side` を更新
- `False`: "環境的 halt (MCB/SAD/DD)" → side 選択に影響させない
- 現在 13+ call site があり、各パラメータが意図通りか確認が必要

### 684# M2: Decision Path RT ID
- 現状: `decision_path` (primary_only, ev_offset 等) はあるが、1サイクル内の判断フローを追跡する ID がない
- 問題: ログ分析時に「どの SkipGate 判定が → どの offset → どの timeout → fill/cancel に繋がったか」が不明
- 提案: `cycle_id` とは別のより軽量な decision trace id

### 既存実装の確認ポイント
- `scripts/v460/lib/fill_loop_orchestrator.py`: `_execute_skip()` 定義
- `scripts/v460/lib/orchestrator_pre_cycle.py`: L294, L313, L332, L438, L462, L486, L639 (全て False)
- `scripts/v460/lib/orchestrator_mid_cycle.py`: L67, L91, L300, L389, L424 (全て True)
- `scripts/v460/lib/orchestrator_balance.py`: L109 (True)
- `scripts/v460/lib/fill_cycle_executor.py`: `_new_cycle_id()` 既存
- `scripts/v460/lib/fill_record_builder.py`: L544 `_derive_decision_path()`

## タスク

### Task 1: _execute_skip() 監査 + ドキュメント

**対象**: `scripts/v460/lib/fill_loop_orchestrator.py`

1. `_execute_skip()` メソッドに docstring を追加:
   ```python
   async def _execute_skip(
       self, st, ctx, *, reason: str,
       update_last_side: bool = False,
       sleep: bool = True,
   ) -> None:
       """サイクルスキップを記録する.
       
       Args:
           update_last_side: True なら last_attempted_side を更新する。
               side 試行がブロックされた場合 (freeze, cooldown, 
               preflight_insufficient) に True を設定。
               環境的 halt (MCB, SAD, DD) では False とすること。
           ...
       """
   ```

2. 全 call site の `update_last_side` 値を確認し、必要に応じて修正:
   - `orchestrator_pre_cycle.py`: MCB_HALT, SAD_FROZEN, MCB_SAD_ESCALATION, PER_SIDE_DD_HALT, TOXIC_FILL_SIDE_VETO, OPERATOR_HALT → 全て `False` (環境的 halt)
   - `orchestrator_mid_cycle.py`: ONE_SIDED_FREEZE_SKIP, ONE_SIDED_COOLDOWN_SKIP → `True` (side 試行の結果)
   - `orchestrator_mid_cycle.py`: cycle gate skips → `True` (積極的判断)
   - `orchestrator_balance.py`: PREFLIGHT_INSUFFICIENT → `True` (side 試行の結果)
   
3. 各 call site にインラインコメントで理由を付記:
   ```python
   await self._execute_skip(st, ctx, reason=CR.MCB_HALT, update_last_side=False)  # env halt: don't bias side
   ```

### Task 2: Decision Trace ID

**対象**: 
- `scripts/v460/lib/fill_cycle_executor.py`
- `scripts/v460/lib/fill_record_builder.py`
- `ztb/metrics/fill_quality.py`

1. サイクル開始時に `decision_trace_id` を生成:
   ```python
   import uuid
   decision_trace_id = f"dt_{int(time.time())}_{uuid.uuid4().hex[:6]}"
   ```

2. 主要ステージにログ付加:
   - SkipGate 評価: `"[dt={id}] skip_gate: result={PASS/SKIP}, reason=..."`
   - Offset pipeline: `"[dt={id}] offset: path={decision_path}, final_bps=..."`
   - Timeout 決定: `"[dt={id}] timeout: value={sec}s, reason=..."`
   - Fill/Cancel: `"[dt={id}] outcome: filled={bool}, reason=..."`

3. FillRecord に記録:
   - `decision_trace_id: str | None` フィールド追加

### Task 3: テスト

**対象**: `tests/unit/v460/`

1. _execute_skip の監査テスト:
   - MCB/SAD/DD halt で `last_attempted_side` が変わらないことを確認
   - freeze/cooldown/preflight で `last_attempted_side` が更新されることを確認
2. Decision trace ID テスト:
   - サイクル実行後に FillRecord に `decision_trace_id` が存在する
   - 同一サイクル内のログに同一 ID が含まれる
3. `python -m pytest tests/ -x --tb=short` で全テスト pass

## 受け入れ基準

- [ ] 全 13+ call site の `update_last_side` が意図通りの値で、インラインコメント付き
- [ ] `_execute_skip()` に包括的 docstring
- [ ] FillRecord に `decision_trace_id` が記録される
- [ ] 主要ステージのログに trace ID が含まれる
- [ ] 新規テスト 6 件以上、全テスト pass

## リスク評価

- **低リスク**: 監査は既存動作の確認。trace ID は additive field のみ
- **ロールバック**: trace ID 機能は FillRecord フィールド削除で除去可能
- **依存**: 72bd8e713 (state separation) が merge 済みであること (現在 HEAD)
