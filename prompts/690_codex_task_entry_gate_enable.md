# Codex Task: 690# Entry Gate 有効化 + 安全監視 (555# / 606# 完結)

## 目的
CalibrationMap ベースの entry_gate を observe モード (`enabled: false`) から
段階的有効化モードへ移行する。CalibrationMap は 1,816+ レコードを蓄積済み。
安全なフォールバックと監視メカニズムを追加し、EV≤0 時の block を段階的に有効化する。

## 背景

### 現行アーキテクチャ (606# / 621#)
- `entry_gate_enabled: false` → EV≤0 でも block せず log のみ (observe)
- `CalibrationMap` は EWMA (τ=100) で p_win, avg_win, avg_loss を追跡
- 3 段階 fallback: `{regime}_{action_bin}` → `{regime}` → `"global"`
- `n_min=30.0` のガード: n_eff < n_min なら p_win=0.5 (neutral) にフォールバック
- hot-reload: `entry_gate_enabled` は bool フィールドで config 更新時に反映済み (607#)

### 有効化に必要な安全装置
1. **consecutive block 上限**: 連続 EV≤0 block が N 回続いたら自動 disable (暴走防止)
2. **session-level block rate 上限**: block 率が過大 (e.g. >60%) なら一時 disable
3. **CalibrationMap staleness guard**: 最終更新が M 分以上前なら block 判定を保留
4. **FillRecord 拡張**: entry_gate 判定結果を fill record に記録

### 既存コード位置
- `scripts/v460/lib/orchestrator_mid_cycle.py` L209-260: `_gate_entry_gate_check` 相当ロジック (inline)
- `scripts/v460/run_fill_test.py` L408-440: CalibrationMap load
- `scripts/v460/lib/orchestrator_post_cycle.py` L112-140: CalibrationMap online update
- `ztb/trading/signal/calibration_map.py`: CalibrationMap class
- `scripts/v460/lib/fill_config.py` L994-1005: entry_gate config fields
- `configs/v460/fill_test.yaml` L1259-1270: entry_gate YAML section

## タスク

### Task 1: Entry Gate Safety Guard

**新規作成**: `scripts/v460/lib/entry_gate_guard.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class EntryGateGuardConfig:
    """entry_gate safety guard の設定."""
    max_consecutive_blocks: int = 15       # 連続 block 上限
    max_block_rate: float = 0.6             # session block 率上限
    min_eval_count_for_rate: int = 20       # block rate 計算に必要な最小 eval 数
    staleness_threshold_sec: float = 600.0  # CalibrationMap staleness 閾値 (10min)


@dataclass
class EntryGateGuardState:
    """entry_gate safety guard の状態."""
    consecutive_blocks: int = 0
    total_evals: int = 0
    total_blocks: int = 0
    auto_disabled: bool = False
    auto_disable_reason: str = ""
    last_calibration_update_ts: float = 0.0


class EntryGateGuard:
    """entry_gate 判定の安全装置.
    
    連続 block 上限、session block rate 上限、staleness guard を提供。
    auto_disabled になった場合は entry_gate を一時的に無効化する。
    """
    
    def __init__(self, config: EntryGateGuardConfig) -> None: ...
    
    def should_suppress_block(self, *, ev: float, regime: str, side: str) -> bool:
        """EV≤0 による block を安全装置で抑制すべきか判定.
        
        Returns:
            True: 安全装置発動 → block を抑制して PASS させる
            False: block を許可
        """
    
    def record_eval(self, *, blocked: bool) -> None:
        """entry_gate eval を記録. blocked=True なら consecutive_blocks++."""
    
    def notify_calibration_update(self) -> None:
        """CalibrationMap が更新された際に呼ぶ (staleness 計測用)."""
    
    @property
    def state(self) -> EntryGateGuardState:
        """現在の guard 状態 (observability)."""
    
    def reset_auto_disable(self) -> None:
        """hot-reload で entry_gate_enabled が true に戻された場合の manual reset."""
```

### Task 2: YAML 設定拡張

**対象**: `configs/v460/fill_test.yaml`

```yaml
# Entry Gate (555# / 606#)
entry_gate_enabled: true                   # ← false → true に変更
entry_gate_calibration_map_path: "models/v460/entry_gate_calibration.json"
entry_gate_probability_mode: "lcb"
entry_gate_ewma_tau: 100.0
entry_gate_n_min: 30.0
entry_gate_fee_rate: 0.0
entry_gate_c_spread: 0.3
entry_gate_c_vol: 0.2
entry_gate_c_imp: 0.5
entry_gate_online_update: true
# Safety Guard (690#)
entry_gate_max_consecutive_blocks: 15
entry_gate_max_block_rate: 0.6
entry_gate_min_eval_for_rate: 20
entry_gate_staleness_threshold_sec: 600.0
```

### Task 3: FillConfig 拡張

**対象**: `scripts/v460/lib/fill_config.py`, `fill_config_parser.py`, `fill_config_validation.py`

新規フィールド:
1. `entry_gate_max_consecutive_blocks: int` (default=15)
2. `entry_gate_max_block_rate: float` (default=0.6)
3. `entry_gate_min_eval_for_rate: int` (default=20)
4. `entry_gate_staleness_threshold_sec: float` (default=600.0)

validation:
- `max_consecutive_blocks >= 1`
- `0.0 < max_block_rate <= 1.0`
- `min_eval_for_rate >= 5`
- `staleness_threshold_sec >= 60.0`

### Task 4: orchestrator_mid_cycle 統合

**対象**: `scripts/v460/lib/orchestrator_mid_cycle.py` L209-260

1. `EntryGateGuard` を `FillLoopOrchestrator.__init__` で生成し `orchestrator_mid_cycle` に渡す
2. entry_gate の EV≤0 判定後、`should_suppress_block()` を呼んで安全装置で抑制するかチェック:
   ```python
   if _cal_ev <= 0:
       st.entry_gate_block_count += 1
       if self.config.entry_gate_enabled:
           if self._entry_gate_guard.should_suppress_block(
               ev=_cal_ev, regime=_cal_regime, side=next_side,
           ):
               logger.warning(
                   "[690#] Entry gate block SUPPRESSED by safety guard: %s",
                   self._entry_gate_guard.state.auto_disable_reason,
               )
               # guard が発動 → block せず continue
           else:
               logger.info("[555#] Entry gate BLOCK: ...")
               # 従来の block 処理
       self._entry_gate_guard.record_eval(blocked=self.config.entry_gate_enabled)
   else:
       self._entry_gate_guard.record_eval(blocked=False)
   ```
3. `orchestrator_post_cycle` の CalibrationMap update 後に `notify_calibration_update()` を呼ぶ

### Task 5: FillRecord 拡張

**対象**: `scripts/v460/lib/fill_record_builder.py`, `ztb/metrics/fill_quality.py`

新規フィールド:
1. `entry_gate_ev: float | None` — EV 値
2. `entry_gate_blocked: bool | None` — block されたか (guard 抑制含む)
3. `entry_gate_guard_suppressed: bool | None` — safety guard で抑制されたか
4. `entry_gate_regime: str | None` — 判定時の CalibrationMap regime key

### Task 6: テスト

**対象**: `tests/unit/v460/test_690_entry_gate_guard.py`

1. consecutive block 上限到達 → auto_disabled
2. block rate 上限到達 → auto_disabled
3. staleness guard → block 抑制
4. normal EV>0 → consecutive_blocks リセット
5. auto_disabled 後の reset 動作
6. n_eff < n_min → p_win=0.5 フォールバック時は block されない
7. hot-reload で entry_gate_enabled toggle 時の guard 状態
8. `python -m pytest tests/ -x --tb=short` で全テスト pass

## 動作仕様

1. `entry_gate_enabled: false` → 完全に従来動作 (guard も動かない)
2. `entry_gate_enabled: true` + EV > 0 → PASS (block なし)
3. `entry_gate_enabled: true` + EV ≤ 0 → block (guard 抑制がなければ)
4. 連続 block が `max_consecutive_blocks` に達したら auto_disable → PASS
5. session block rate が `max_block_rate` を超えたら auto_disable → PASS
6. CalibrationMap 最終更新が `staleness_threshold_sec` 超なら block 抑制
7. auto_disable は WARNING ログ + entry_gate_guard_suppressed=True を記録

## 受け入れ基準

- [ ] `entry_gate_enabled: true` で EV≤0 時に block が機能する
- [ ] safety guard 3 種 (consecutive / rate / staleness) が発動する
- [ ] guard 抑制時に FillRecord に記録される
- [ ] `entry_gate_enabled: false` で完全に従来動作
- [ ] hot-reload 対応: `entry_gate_enabled` toggle で guard リセット
- [ ] 新規テスト 8 件以上、全テスト pass
- [ ] CalibrationMap online update 後に staleness タイマーリセット

## リスク評価

- **中リスク**: entry_gate 有効化は収益に直接影響。但し safety guard 3 重で暴走防止
- **ロールバック**: `entry_gate_enabled: false` で即時旧動作復帰
- **観測**: 最初は `max_consecutive_blocks: 15` で寛容に設定、データ蓄積後に引き締め
