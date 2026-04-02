# Codex Task: 690# _execute_skip 22 コールサイト audit 形式化 (689# フォローアップ)

## 目的
689# で追加された `_execute_skip(update_last_side=...)` の全 22 コールサイトを
形式的に監査し、各パラメータの正当性をテストで保証する。
加えて、cancel_reason の分類体系を整理し、skip 分析の再現性を向上させる。

## 背景

### 689# で実装済みの基盤
- `_execute_skip()` に `update_last_side: bool` パラメータ (687# / 72bd8e713)
- 各 call site にインラインコメント (`# env halt: don't bias side` 等)
- `decision_trace_id` (dt=...) がサイクル全体に伝播 (689#)
- 22 箇所の blocking decision point が確認済み

### 現状の call site 一覧 (13+ 箇所)
| ファイル | 行 | cancel_reason | update_last_side |
|---|---|---|---|
| `orchestrator_balance.py` | L102 | PREFLIGHT_INSUFFICIENT | True |
| `orchestrator_pre_cycle.py` | L294 | MCB_HALT | False |
| `orchestrator_pre_cycle.py` | L314 | SAD_FROZEN | False |
| `orchestrator_pre_cycle.py` | L334 | MCB_SAD_ESCALATION | False |
| `orchestrator_pre_cycle.py` | L441 | PER_SIDE_DD_HALT | False |
| `orchestrator_pre_cycle.py` | L466 | TOXIC_FILL_SIDE_VETO | False |
| `orchestrator_pre_cycle.py` | L491 | (operator halt) | False |
| `orchestrator_pre_cycle.py` | L645 | (daily reset 等) | False |
| `orchestrator_mid_cycle.py` | L67 | ONE_SIDED_FREEZE_SKIP | True |
| `orchestrator_mid_cycle.py` | L88 | ONE_SIDED_COOLDOWN_SKIP | True |
| `orchestrator_mid_cycle.py` | L298 | (entry_gate) | True |
| `orchestrator_mid_cycle.py` | L388 | (skip_gate) | True |
| `orchestrator_mid_cycle.py` | L424 | (cycle gate) | True |

### 必要な audit 作業
1. **全 call site のパラメータ正当性検証テスト**
2. **cancel_reason の enum 化 / 分類** (環境 halt vs side 判断 vs gate block)
3. **skip ceremony 統一性チェック**: heartbeat / state_save / sleep パラメータの一貫性

## タスク

### Task 1: cancel_reason 分類マッピング

**新規作成**: `scripts/v460/lib/cancel_reason_taxonomy.py`

```python
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass

class SkipCategory(str, Enum):
    """skip の分類カテゴリ."""
    ENV_HALT = "env_halt"           # 環境的 halt (MCB/SAD/DD) → update_last_side=False
    SIDE_BLOCK = "side_block"       # side 試行の結果 block → update_last_side=True
    GATE_BLOCK = "gate_block"       # gate 判定による block → update_last_side=True
    RESOURCE = "resource"           # リソース不足 → update_last_side=True
    MAINTENANCE = "maintenance"     # メンテナンス/リセット → update_last_side=False

@dataclass(frozen=True)
class CancelReasonMeta:
    """cancel_reason のメタデータ."""
    reason: str
    category: SkipCategory
    expected_update_last_side: bool
    description: str

# 全 cancel_reason の正規マッピング (689# audit 完了版)
CANCEL_REASON_REGISTRY: dict[str, CancelReasonMeta] = {
    "MCB_HALT": CancelReasonMeta(
        reason="MCB_HALT",
        category=SkipCategory.ENV_HALT,
        expected_update_last_side=False,
        description="Market Circuit Breaker halt",
    ),
    "SAD_FROZEN": CancelReasonMeta(
        reason="SAD_FROZEN",
        category=SkipCategory.ENV_HALT,
        expected_update_last_side=False,
        description="Sudden Anomaly Detector frozen",
    ),
    # ... 全 cancel_reason を登録
}

def validate_skip_consistency(
    cancel_reason: str,
    update_last_side: bool,
) -> bool:
    """cancel_reason と update_last_side の整合性を検証.
    
    テストから呼ばれる。本番では使わない (パフォーマンス理由)。
    """
    meta = CANCEL_REASON_REGISTRY.get(cancel_reason)
    if meta is None:
        return False  # 未登録の cancel_reason
    return meta.expected_update_last_side == update_last_side
```

### Task 2: 静的 audit テスト

**新規作成**: `tests/unit/v460/test_690_skip_audit.py`

```python
"""_execute_skip() 全 call site の audit テスト.

689# で追加された update_last_side パラメータの正当性を
AST 解析 + cancel_reason registry で検証する。
"""

import ast
import inspect
from pathlib import Path
import pytest

from scripts.v460.lib.cancel_reason_taxonomy import (
    CANCEL_REASON_REGISTRY,
    SkipCategory,
    validate_skip_consistency,
)


class TestCancelReasonRegistry:
    """cancel_reason registry の網羅性テスト."""

    def test_all_reasons_have_category(self) -> None:
        """全 cancel_reason がカテゴリ分類されている."""
        for reason, meta in CANCEL_REASON_REGISTRY.items():
            assert meta.category is not None
            assert meta.description != ""

    def test_env_halt_reasons_are_false(self) -> None:
        """ENV_HALT カテゴリは全て update_last_side=False."""
        for reason, meta in CANCEL_REASON_REGISTRY.items():
            if meta.category == SkipCategory.ENV_HALT:
                assert meta.expected_update_last_side is False, f"{reason}"

    def test_side_block_reasons_are_true(self) -> None:
        """SIDE_BLOCK カテゴリは全て update_last_side=True."""
        for reason, meta in CANCEL_REASON_REGISTRY.items():
            if meta.category == SkipCategory.SIDE_BLOCK:
                assert meta.expected_update_last_side is True, f"{reason}"


class TestCallSiteAudit:
    """_execute_skip() call site の AST 解析テスト."""

    ORCHESTRATOR_FILES = [
        "scripts/v460/lib/orchestrator_pre_cycle.py",
        "scripts/v460/lib/orchestrator_mid_cycle.py",
        "scripts/v460/lib/orchestrator_balance.py",
        "scripts/v460/lib/orchestrator_post_cycle.py",
        "scripts/v460/lib/fill_cycle_executor.py",
    ]

    def test_all_call_sites_have_explicit_update_last_side(self) -> None:
        """全 _execute_skip() 呼出しで update_last_side が明示的に指定されている."""
        # AST 解析で _execute_skip の呼出しを検出し、
        # keyword 引数に update_last_side が含まれることを確認

    def test_cancel_reason_matches_registry(self) -> None:
        """全 call site の cancel_reason が registry に登録されている."""

    def test_update_last_side_matches_category(self) -> None:
        """各 call site の update_last_side が cancel_reason のカテゴリと一致."""
```

### Task 3: skip ceremony 一貫性テスト

**対象**: `tests/unit/v460/test_690_skip_audit.py` (追加)

```python
class TestSkipCeremonyConsistency:
    """skip ceremony パラメータの一貫性テスト."""

    def test_env_halt_always_has_heartbeat(self) -> None:
        """環境 halt は heartbeat=True で lock 更新すること."""

    def test_side_block_preserves_sleep(self) -> None:
        """side block は sleep=True (デフォルト) であること."""

    def test_no_duplicate_cancel_reasons(self) -> None:
        """同一 cancel_reason が異なるカテゴリで使われていないこと."""

    def test_inline_comments_present(self) -> None:
        """全 _execute_skip() call site に audit コメントが付いていること.
        
        689# で追加されたインラインコメント (# env halt: ..., # side attempt: ...) が
        全 call site に存在することを確認。
        """
```

### Task 4: cancel_reason 文字列定数化

**対象**: 既存の cancel_reason 文字列を定数化

orchestrator_pre_cycle.py, orchestrator_mid_cycle.py, orchestrator_balance.py で
使用されている cancel_reason 文字列リテラルを、
`cancel_reason_taxonomy.py` の registry と紐付ける。

**注意**: 文字列リテラル自体は変更しない (FillRecord の後方互換性)。
定数定義を追加し、call site で定数参照に置き換える。

## 動作仕様

1. `cancel_reason_taxonomy.py` は本番ランタイムでは import されない (テスト専用 registry)
2. テストが AST 解析で全 call site を自動検出し、registry と照合
3. 新規 `_execute_skip()` call site が追加された場合、テストが自動的に検出して registry 未登録をエラー報告
4. `python -m pytest tests/ -x --tb=short` で全テスト pass

## 受け入れ基準

- [ ] 全 cancel_reason が CANCEL_REASON_REGISTRY に登録されている
- [ ] 全 call site で `update_last_side` が明示的に指定されている (AST テスト)
- [ ] ENV_HALT カテゴリは全て `update_last_side=False`
- [ ] SIDE_BLOCK / GATE_BLOCK カテゴリは全て `update_last_side=True`
- [ ] 新規 call site 追加時にテストが自動検出する
- [ ] 新規テスト 10 件以上、全テスト pass
- [ ] cancel_reason 文字列の後方互換性を維持

## リスク評価

- **低リスク**: テスト専用の静的解析。ランタイムコードの変更は定数化のみ
- **ロールバック**: テストファイル削除で即時復帰
- **価値**: 将来の _execute_skip call site 追加時の regression 防止
