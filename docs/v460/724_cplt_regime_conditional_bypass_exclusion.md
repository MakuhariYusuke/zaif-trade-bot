# 724# regime-conditional bypass exclusion + trending_down_sell_offset_boost 1.3

## 背景

723# の bypass パラドックス深堀り分析により、bypass が全体では +66.0bps 貢献する一方、
**sell/trending_down** では逆効果であることが判明:

- sell/trending_down bypass: n=18, avg=-0.91bps（5日間）
- 5 日間 sell/trending_down 合計: n=66, avg=-0.75, total=-49.3bps, AS=29%
- 04/09 の悪化: AS=53%, avg=-1.80bps

### 722# 後の状況（124 fills）

- 全体: avg=+0.12bps, total=+14.6bps — 正
- sell/trending_down: n=13, AS=62%, avg=-1.81bps — 最大の損失源
- trending_up: n=25, avg=+1.55, AS=8% — 優秀

## 変更内容

### 1. コード変更: regime-conditional bypass exclusion

`_is_bypass_mode_active()` に `regime` パラメータを追加し、`skip_gate_bypass_regime_exclude`
設定で指定された `side/regime` ペアに対して bypass を無効化する汎用機構を実装。

```python
# skip_gate_evaluator.py
def _is_bypass_mode_active(self, side: str, regime: str | None = None) -> bool:
    if regime and self._config.skip_gate_bypass_regime_exclude:
        key = f"{side}/{regime}"
        if key in self._config.skip_gate_bypass_regime_exclude:
            return False
    # ... existing side-specific logic
```

**変更ファイル:**
- `scripts/v460/lib/fill_config.py` — `skip_gate_bypass_regime_exclude: list[str]` フィールド追加
- `scripts/v460/lib/fill_config_parser.py` — YAML マッピング追加
- `scripts/v460/lib/config_hot_reload.py` — hot-reload ホワイトリスト追加
- `scripts/v460/lib/skip_gate_evaluator.py` — `_is_bypass_mode_active` にレジーム除外ロジック追加

### 2. YAML 変更

```yaml
# configs/v460/fill_test.yaml
skip_gate:
  bypass_regime_exclude:
    - "sell/trending_down"    # 724# sell/trending_down bypass 無効化

regime:
  trending_down_sell_offset_boost: 1.3  # 724# 1.0→1.3（sell/trending_down 防御強化）
```

## 期待効果

| 施策 | 想定インパクト |
|------|---------------|
| bypass 除外 | ~18 fills/5d × (-0.91bps avg) → +16.4bps/5d 節約 |
| offset 1.3x | AS 率低減（特に 2-3bps spread 帯: 33%→推定 20%台） |

## テスト

- `test_710_skip_gate_bypass_dryrun.py` — `test_bypass_regime_exclude()` 追加（6 pass）
- `test_336_yaml_code_drift_prevention.py` — フィールド追加（4 pass）
- `test_169_config_hot_reload.py` — hot-reload フィールド追加（23 pass）
- `test_176_trending_offset_asymmetry.py` — アサーション 1.0→1.3 更新
- フルユニットテスト全パス
