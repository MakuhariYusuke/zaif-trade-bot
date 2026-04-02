# Codex Prompt: trending_down regime exit 戦略 (700# Task 4)

## Goal
trending_down レジームでのラウンドトリップ損失 (-63.54 bps) を抑制するため、buy ポジション累積上限と aggressive skewing を組み合わせた regime exit 戦略を実装する。

## Context
- **699# 盲点A**: trending_down RT が全日 RT 損失の 349% (-63.54/-18.17 bps) を占める。ranging/trending_up の黒字をすべて打ち消す。
- **メカニズム**: trending_down 時に buy fill が滞留 →売れない → regime 遷移後に損失拡大。
- **699# 盲点C**: MCB 復帰後も品質劣化。trending_up → MCB halt → 復帰後の段階的参入が未整備。
- **699# 盲点D**: sell/trending_up は AS_Cost -2.21 bps。regime 別 sell guard が未実装。

## 既存メカニズムとの関係

| 既存機構 | 内容 | trending_down 時 |
|----------|------|-----------------|
| inventory_skewing | max_factor=0.4 (通常), 0.15 (trending) | 0.15 では弱い |
| inv_skew_regime_gate | 完全停止 or 低減 | legacy モードは完全停止=逆効果 |
| trend_5s_sell_guard | sell side veto | trending_down とは直交 |
| MCB | circuit breaker | trending_up で発動。trending_down は対象外 |
| SAD | sudden adverse detection | 急激な変動のみ。漸進的ドリフトは非検出 |

→ **trending_down 特化のポジション管理が欠如**

## Implementation

### 1. 新モジュール: `scripts/v460/lib/regime_exit_strategy.py`

```python
"""700# Task 4: Regime-aware position exit strategy.

trending_down 時の buy ポジション累積を制限し、
skewing を escalate してポジション解消を促進する。
"""
from __future__ import annotations
import dataclasses
import time

@dataclasses.dataclass(frozen=True)
class RegimeExitConfig:
    enabled: bool = False
    # trending_down 時の buy 累積 fill 上限 (window 内)
    max_trending_down_buy_fills: int = 10
    # window (秒): この期間内の buy fill をカウント
    tracking_window_sec: float = 3600.0
    # 上限超過時の max_factor 引き上げ値
    escalated_max_factor: float = 0.7
    # trending_down で NFQ を有効にする imbalance 閾値
    nfq_trigger_imbalance: float = 0.3

class RegimeExitTracker:
    """trending_down 時の buy exposure を追跡."""
    
    def __init__(self, config: RegimeExitConfig) -> None:
        self._config = config
        self._buy_fills: list[float] = []  # timestamps
    
    def record_fill(self, side: str, timestamp: float) -> None:
        """fill 発生時に呼ばれる."""
        if side == "buy":
            self._buy_fills.append(timestamp)
    
    def evaluate(
        self,
        regime: str,
        imbalance: float,
        now: float,
    ) -> RegimeExitResult:
        """現在のレジームとインバランスから exit 戦略を判定."""
        ...
```

**RegimeExitResult**:
```python
@dataclasses.dataclass(frozen=True)
class RegimeExitResult:
    should_escalate_skewing: bool
    effective_max_factor: float | None  # None = 変更なし
    should_trigger_nfq: bool
    buy_count_in_window: int
    reason: str
```

### 2. MakerPriceCalculator への統合

`maker_price.py` の `_apply_inventory_skew` 内:
```python
# 700# Task 4: regime exit strategy
if self._regime_exit_tracker is not None:
    exit_result = self._regime_exit_tracker.evaluate(
        regime=current_regime_name,
        imbalance=_decayed_imb,
        now=now,
    )
    if exit_result.should_escalate_skewing:
        _effective_max_factor = max(
            _effective_max_factor,
            exit_result.effective_max_factor or _effective_max_factor,
        )
```

### 3. fill_record_builder 追加フィールド

- `regime_exit_escalated: bool`
- `regime_exit_buy_count: int`
- `regime_exit_reason: str`

### 4. Config (fill_test.yaml)

```yaml
regime_exit_strategy:
  enabled: false  # 有効化は手動
  max_trending_down_buy_fills: 10
  tracking_window_sec: 3600
  escalated_max_factor: 0.7
  nfq_trigger_imbalance: 0.3
```

### Test file: `tests/unit/v460/test_700_regime_exit.py`

1. `test_enabled_false_no_op` — disabled なら何もしない
2. `test_buy_fill_tracking` — buy fill が正しくカウントされる
3. `test_window_expiry` — tracking_window_sec 超過した古い fill が除外される
4. `test_escalation_trigger` — buy_fills > max_trending_down_buy_fills で escalation
5. `test_no_escalation_other_regimes` — ranging/trending_up では発動しない
6. `test_nfq_trigger_on_high_imbalance` — imbalance > 0.3 で NFQ 推奨
7. `test_escalated_max_factor_value` — escalate 時の max_factor が config 値
8. `test_sell_fills_not_counted` — sell fill はカウントに含まない
9. `test_result_fields` — RegimeExitResult の全フィールドが正しい型
10. `test_fill_record_fields` — fill_record に regime_exit 関連フィールドが出力される

## 分析タスク (実装と並行)

以下の分析を Codex に実施させ、閾値の妥当性を確認:
- 4/2 trending_down 期間の buy fill 間隔分布
- trending_down → ranging 遷移のタイミングと最後の buy fill の間隔
- hold 時間 (buy fill → 対応 sell fill) の regime 別分布

## Constraints
- `enabled: false` デフォルト。観察モードでの fill_record 出力から始める
- MCB / SAD / inventory_skewing との interaction を壊さない
- RegimeExitTracker は MakerPriceCalculator に注入 (コンストラクタ引数 or setter)
- regime_detector から current_regime を取得する既存パターンに合わせる
- Run: `python -m pytest tests/unit/v460/test_700_regime_exit.py -x --tb=short -q`
- Regression: `python -m pytest tests/ -x --tb=short -q`
