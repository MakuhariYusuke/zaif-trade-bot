# Codex Task: 690# regime_timeout_overrides Integration Test (689# フォローアップ)

## 目的
689# で追加された `regime_timeout_overrides` と `get_timeout_with_reason()` の
統合テストを作成し、レジーム × side timeout の priority chain と
fill_cycle_executor との統合を検証する。

## 背景

### 689# で実装済み
- `FillTestConfig.regime_timeout_overrides: dict[str, dict[str, float]]`
- `FillTestConfig.get_timeout_with_reason(side, macro_trend) -> tuple[float, str]`
- `_resolve_cycle_timeout_policy()` in fill_cycle_executor.py
- YAML config section: `regime_timeout_overrides:`
- FillRecord fields: `timeout_applied_sec`, `timeout_reason`

### Priority Chain (get_timeout_with_reason)
1. `regime_timeout_overrides[regime][side]` — 最高優先度 (regime×side直指定)
2. Legacy: `macro_sell_timeout_strong_up` / `macro_sell_timeout_weak_up` (sell only)
3. `order_timeout_sec_sell` (side-specific, sell only)
4. `order_timeout_sec` (global fallback)

### テストが不足している領域
- priority chain の正確性 (全 4 段の fallback 動作)
- regex/全レジーム名との組み合わせ
- 不正な YAML 値のバリデーション
- hot-reload 時の timeout 変更反映
- fill_cycle_executor._resolve_cycle_timeout_policy() のモック統合テスト
- FillRecord への timeout_applied_sec / timeout_reason 記録

### 既存コード位置
- `scripts/v460/lib/fill_config.py` L168-169: フィールド定義
- `scripts/v460/lib/fill_config.py` L1060-1090: `get_timeout_with_reason()`
- `scripts/v460/lib/fill_config_parser.py` L1013-1029: YAML parser
- `scripts/v460/lib/fill_config_validation.py` L72-85: validation
- `scripts/v460/lib/fill_cycle_executor.py` L465-476: `_resolve_cycle_timeout_policy()`

## タスク

### Task 1: get_timeout_with_reason() 単体テスト

**新規作成**: `tests/unit/v460/test_690_timeout_priority.py`

```python
"""get_timeout_with_reason() priority chain の網羅テスト.

689# で追加された regime_timeout_overrides を中心に、
4段 priority chain の正確性を検証する。
"""

import pytest
from scripts.v460.lib.fill_config import FillTestConfig


class TestTimeoutPriorityChain:
    """4段 priority chain のテスト."""

    def test_regime_override_highest_priority(self) -> None:
        """regime_timeout_overrides が最優先."""
        config = FillTestConfig(
            order_timeout_sec=30.0,
            order_timeout_sec_sell=20.0,
            macro_sell_timeout_strong_up=10.0,
            regime_timeout_overrides={"strong_up": {"sell": 5.0}},
        )
        timeout, reason = config.get_timeout_with_reason("sell", "strong_up")
        assert timeout == 5.0
        assert "regime_override" in reason

    def test_legacy_macro_sell_second_priority(self) -> None:
        """Legacy macro sell timeout が 2番目."""
        config = FillTestConfig(
            order_timeout_sec=30.0,
            macro_sell_timeout_strong_up=10.0,
            regime_timeout_overrides={},
        )
        timeout, reason = config.get_timeout_with_reason("sell", "strong_up")
        assert timeout == 10.0

    def test_side_specific_third_priority(self) -> None:
        """order_timeout_sec_sell が 3番目."""
        config = FillTestConfig(
            order_timeout_sec=30.0,
            order_timeout_sec_sell=20.0,
        )
        timeout, reason = config.get_timeout_with_reason("sell", "ranging")
        assert timeout == 20.0

    def test_global_fallback(self) -> None:
        """order_timeout_sec が最終 fallback."""
        config = FillTestConfig(order_timeout_sec=30.0)
        timeout, reason = config.get_timeout_with_reason("buy", "ranging")
        assert timeout == 30.0

    def test_buy_side_ignores_macro_sell_timeout(self) -> None:
        """buy side は macro_sell_timeout を使わない."""
        config = FillTestConfig(
            order_timeout_sec=30.0,
            macro_sell_timeout_strong_up=10.0,
        )
        timeout, reason = config.get_timeout_with_reason("buy", "strong_up")
        assert timeout == 30.0

    def test_override_buy_in_strong_down(self) -> None:
        """strong_down regime で buy の override."""
        config = FillTestConfig(
            order_timeout_sec=30.0,
            regime_timeout_overrides={"strong_down": {"buy": 3.0}},
        )
        timeout, reason = config.get_timeout_with_reason("buy", "strong_down")
        assert timeout == 3.0

    def test_override_missing_side_falls_through(self) -> None:
        """override にレジームはあるが side がない → 次の priority."""
        config = FillTestConfig(
            order_timeout_sec=30.0,
            regime_timeout_overrides={"strong_up": {"buy": 5.0}},  # sell なし
        )
        timeout, reason = config.get_timeout_with_reason("sell", "strong_up")
        assert timeout == 30.0  # global fallback

    def test_none_macro_trend_uses_fallback(self) -> None:
        """macro_trend=None → override 不適用、fallback."""
        config = FillTestConfig(
            order_timeout_sec=30.0,
            regime_timeout_overrides={"strong_up": {"sell": 5.0}},
        )
        timeout, reason = config.get_timeout_with_reason("sell", None)
        assert timeout == 30.0


class TestTimeoutRegimeNames:
    """全レジーム名との組み合わせテスト."""

    REGIMES = ["strong_up", "weak_up", "ranging", "weak_down", "strong_down"]
    SIDES = ["buy", "sell"]

    @pytest.mark.parametrize("regime", REGIMES)
    @pytest.mark.parametrize("side", SIDES)
    def test_all_regime_side_combinations(self, regime: str, side: str) -> None:
        """全 regime × side で例外が出ないこと."""
        config = FillTestConfig(
            order_timeout_sec=30.0,
            regime_timeout_overrides={regime: {side: 7.0}},
        )
        timeout, reason = config.get_timeout_with_reason(side, regime)
        assert timeout == 7.0
```

### Task 2: YAML parser / validation テスト

**対象**: `tests/unit/v460/test_690_timeout_priority.py` (追加)

```python
class TestTimeoutYAMLParsing:
    """regime_timeout_overrides YAML parsing テスト."""

    def test_parse_valid_overrides(self) -> None:
        """正常な YAML が parse される."""

    def test_parse_empty_overrides(self) -> None:
        """空の overrides → 空 dict."""

    def test_parse_nested_structure(self) -> None:
        """regime -> side -> float の 2 段ネスト."""

    def test_invalid_timeout_value_raises(self) -> None:
        """負の timeout 値はバリデーションエラー."""

    def test_case_insensitive_regime_names(self) -> None:
        """YAML のレジーム名は小文字に正規化される."""

    def test_case_insensitive_side_names(self) -> None:
        """YAML の side 名は小文字に正規化される."""
```

### Task 3: _resolve_cycle_timeout_policy() 統合テスト

**対象**: `tests/unit/v460/test_690_timeout_priority.py` (追加)

```python
class TestResolveCycleTimeoutPolicy:
    """fill_cycle_executor._resolve_cycle_timeout_policy() の統合テスト.
    
    FillCycleExecutor のメソッドをモックベースでテストし、
    get_timeout_with_reason() → timeout_applied_sec / timeout_reason の
    end-to-end フローを検証する。
    """

    def test_timeout_propagated_to_fill_record(self) -> None:
        """timeout_applied_sec が FillRecord に記録される."""

    def test_timeout_reason_propagated_to_fill_record(self) -> None:
        """timeout_reason が FillRecord に記録される."""

    def test_macro_trend_from_detector_used(self) -> None:
        """macro_trend は MacroTrendDetector.current_trend から取得."""
```

### Task 4: hot-reload テスト

**対象**: `tests/unit/v460/test_690_timeout_priority.py` (追加)

```python
class TestTimeoutHotReload:
    """regime_timeout_overrides の hot-reload テスト."""

    def test_override_change_reflected_immediately(self) -> None:
        """YAML 変更 → config reload → 次サイクルから新 timeout 適用."""

    def test_override_removal_falls_through(self) -> None:
        """override 削除 → fallback に戻る."""

    def test_global_timeout_change_reflected(self) -> None:
        """order_timeout_sec 変更 → 即反映."""
```

## 動作仕様

1. get_timeout_with_reason() の 4 段 priority chain が正確に動作する
2. 全 regime × side の組み合わせでエラーなし
3. YAML parser が大文字/小文字を正規化
4. 不正な値 (負数, 非数値) はバリデーションで弾かれる
5. hot-reload で即時反映
6. `python -m pytest tests/ -x --tb=short` で全テスト pass

## 受け入れ基準

- [ ] priority chain の全 4 段テスト通過
- [ ] 全 regime × side の parametrize テスト通過
- [ ] YAML parsing / validation テスト通過
- [ ] FillRecord 記録テスト通過
- [ ] hot-reload テスト通過
- [ ] 新規テスト 15 件以上、全テスト pass

## リスク評価

- **低リスク**: テスト追加のみ、ランタイムコード変更なし
- **ロールバック**: テストファイル削除で即時復帰
- **価値**: 689# の regime_timeout_overrides の信頼性保証、regression 防止
