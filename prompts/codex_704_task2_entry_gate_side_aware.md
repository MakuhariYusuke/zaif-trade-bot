# 704# Task 2: Entry Gate Guard 段階的ブロック機構の改善

## 背景
3日間のライブデータ分析で、entry_gate が完全に無効化されている問題が判明:
- 429 fills 中 100% (429/429) で EV ≤ 0
- 100% (429/429) が entry_gate_guard_suppressed = True
- EV stats: avg=-0.899, med=-1.164, max=-0.287 (全て負)
- auto_disable 理由: max_consecutive_blocks (15) に即到達 → 全通過

### 根本原因
calibration_map の fallback 統計が構造的にマイナス EV を返しており、
現行の guard パラメータ (max_consecutive_blocks=15, max_block_rate=0.6) では
guard が即 auto-disable → entry_gate が実質 observe モードになる。

704# で `max_consecutive_blocks=50`, `max_block_rate=0.95` に調整済みだが、
EV が **常時** 負のため根本解決にならない。

## 改善方針: 段階的 EV 防御

entry_gate_guard に **side-aware blocking** を追加する。
全てをブロックするのではなく、**損失が大きい sell 側を優先的にブロック** する。

## 修正箇所

### 1. `scripts/v460/lib/entry_gate_guard.py`
`should_suppress_block` に **side-aware suppression logic** を追加:

```python
def should_suppress_block(self, *, ev: float, regime: str, side: str) -> bool:
    """Return True when the guard should suppress an EV<=0 block.
    
    704#: side-aware suppression. sell 側は block を維持しやすく、
    buy 側は suppression しやすくする（buy は構造的にプラス PnL のため）。
    """
    # 既存の auto_disable / staleness / consecutive / rate チェック...
    
    # 704# NEW: side-aware threshold
    # buy 側は EV が軽度マイナスなら suppress (通過許可)
    if side == "buy" and ev > self._config.buy_suppress_ev_threshold:
        return True  # buy で EV が -X bps 以上なら通過
    
    # sell 側は block を維持（suppress しない）
    return False
```

### 2. `scripts/v460/lib/entry_gate_guard.py` — EntryGateGuardConfig
```python
@dataclass(frozen=True)
class EntryGateGuardConfig:
    max_consecutive_blocks: int = 50
    max_block_rate: float = 0.95
    min_eval_count_for_rate: int = 20
    staleness_threshold_sec: float = 600.0
    # 704#: side-aware suppression
    buy_suppress_ev_threshold: float = -0.5  # buy で EV > -0.5 bps なら通過許可
```

### 3. `configs/v460/fill_test.yaml`
```yaml
entry_gate_buy_suppress_ev_threshold: -0.5  # 704#: buy 側の suppress 閾値
```

### 4. `scripts/v460/lib/fill_config.py`
```python
entry_gate_buy_suppress_ev_threshold: float = -0.5
```

### 5. `scripts/v460/lib/fill_config_validation.py`
```python
if not (-5.0 <= config.entry_gate_buy_suppress_ev_threshold <= 0.0):
    raise ValueError(...)
```

### 6. `scripts/v460/lib/fill_config_parser.py`
パーサーで YAML → config マッピング追加

### 7. `scripts/v460/lib/config_hot_reload.py`
hot-reload allowlist に追加

## テスト
`tests/unit/v460/test_704_entry_gate_side_aware.py`

1. **test_buy_suppress_mild_negative_ev**: buy + EV=-0.3 → suppress=True (通過)
2. **test_buy_block_severe_negative_ev**: buy + EV=-1.0 → suppress=False (ブロック)
3. **test_sell_not_suppressed**: sell + EV=-0.3 → suppress=False (ブロック維持)
4. **test_buy_suppress_threshold_boundary**: buy + EV=-0.5 (ちょうど閾値) → suppress=True
5. **test_auto_disable_still_works**: auto_disabled=True → side に関わらず suppress=True
6. **test_staleness_overrides_side_aware**: staleness が検出されたら side-aware より優先して auto-disable

## 設計根拠
- 3日間データ: buy total=+59.42bps, sell total=-167.90bps
- buy は構造的にプラスなので軽度の負 EV でも通過させる価値がある
- sell は全レジームで損失のため、entry_gate のブロックを維持する方がリスク低減
- 閾値 -0.5 bps: 現行 EV 分布 (avg=-0.899) から buy で+EV 寄りのケースを概ね通過させる

## 制約
- 既存テスト (`test_690_entry_gate*`) が引き続きパスすること
- `should_suppress_block` の既存引数 (ev, regime, side) は変更しない（追加パラメータはコンストラクタ経由）
- auto_disable / staleness / rate チェックは既存ロジックを維持（side-aware は最後に追加）
