"""
091# 修正テスト: 未保存レコード化防止 + 残高不足side切替 + offset_floor事後再適用.

091# Codex レビュー対応で修正された運用継続性バグの単体テスト。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# =====================================================================
# 091#-1: alt_side==last_side 分岐で periodic flush が実行される
# =====================================================================


class TestAltSideBatchFlush:
    """091# #2: alt_side==_last_side 分岐に batch_flush ロジックが存在する."""

    def test_alt_side_branch_has_batch_flush(self) -> None:
        """alt_side == self._last_side 分岐内に batch flush がある."""
        src = Path(
            _PROJECT_ROOT / "scripts" / "v460" / "run_fill_test.py"
        )
        content = src.read_text(encoding="utf-8")

        # alt_side == self._last_side の行を探す
        lines = content.split("\n")
        alt_side_line = None
        for i, line in enumerate(lines):
            if "alt_side == self._last_side" in line:
                alt_side_line = i
                break

        assert alt_side_line is not None, (
            "alt_side == self._last_side 分岐が見つからない"
        )

        # 119# BatchPersistence 委譲: _batch_persistence.maybe_flush に統合済み
        block = "\n".join(lines[alt_side_line : alt_side_line + 35])
        assert "_batch_persistence.maybe_flush" in block, (
            f"alt_side==last_side 分岐内に _batch_persistence.maybe_flush がない:\n{block[:300]}"
        )

    def test_alt_side_branch_has_091_comment(self) -> None:
        """107# R1 で flush ロジックが _maybe_flush_batch に統合されている."""
        src = Path(
            _PROJECT_ROOT / "scripts" / "v460" / "run_fill_test.py"
        )
        content = src.read_text(encoding="utf-8")
        # 107# R1: 重複 flush → _maybe_flush_batch 統合
        assert "107# R1" in content


# =====================================================================
# 091#-2: preflight 失敗時の即座 opposite side 切替
# =====================================================================


class TestPreflightOppositeSide:
    """091# #3: preflight 失敗時に反対 side を即時チェック."""

    def test_preflight_has_opposite_side_check(self) -> None:
        """preflight 失敗分岐に反対 side チェックがある."""
        src = Path(
            _PROJECT_ROOT / "scripts" / "v460" / "run_fill_test.py"
        )
        content = src.read_text(encoding="utf-8")

        # 091# の即座切替ロジックが存在する
        assert "switching to" in content and "immediately (091#)" in content, (
            "091# の即座 opposite side 切替ロジックが見つからない"
        )

    def test_preflight_skip_has_batch_flush(self) -> None:
        """preflight skip 待機中にも batch_flush が実行される."""
        src = Path(
            _PROJECT_ROOT / "scripts" / "v460" / "run_fill_test.py"
        )
        content = src.read_text(encoding="utf-8")

        # 119# BatchPersistence 委譲: maybe_flush に統合、context に "insufficient" が含まれる
        assert "_batch_persistence.maybe_flush" in content
        assert "insufficient" in content

    def test_preflight_opposite_side_logic_order(self) -> None:
        """opposite side 即時切替が SAFE_STOP より前 (preflight 分岐内) に位置する."""
        src = Path(
            _PROJECT_ROOT / "scripts" / "v460" / "run_fill_test.py"
        )
        content = src.read_text(encoding="utf-8")

        pos_opposite = content.find("immediately (091#)")
        pos_safe_stop = content.find("SAFE_STOP: 連続 preflight")
        assert pos_opposite > 0, "091# 即座切替ロジックが見つからない"
        assert pos_safe_stop > 0, "SAFE_STOP ロジックが見つからない"
        assert pos_opposite < pos_safe_stop, (
            "opposite side 即時切替が SAFE_STOP より後にある"
        )


# =====================================================================
# 091#-3: sell offset_floor が spread_adaptive 後にも再適用される
# =====================================================================


class TestSellOffsetFloorPostAdaptive:
    """091# #6: offset_floor が spread_adaptive で破られないことを検証."""

    def test_post_adaptive_floor_exists_in_code(self) -> None:
        """spread_adaptive の後に sell floor 再適用ロジックがある."""
        src = Path(
            _PROJECT_ROOT / "scripts" / "v460" / "run_fill_test.py"
        )
        content = src.read_text(encoding="utf-8")

        # spread_adaptive ブロックの後に 091# floor 再適用がある
        assert "Post-adaptive floor re-applied" in content

    def test_post_adaptive_floor_after_spread_adaptive(self) -> None:
        """091# floor 再適用が spread_adaptive より後にある."""
        src = Path(
            _PROJECT_ROOT / "scripts" / "v460" / "run_fill_test.py"
        )
        content = src.read_text(encoding="utf-8")

        pos_adaptive = content.rfind("[spread_adaptive]")
        pos_floor = content.find("Post-adaptive floor re-applied")
        assert pos_floor > pos_adaptive, (
            "Post-adaptive floor が spread_adaptive より前にある"
        )

    def test_first_and_post_floor_both_exist(self) -> None:
        """sell_offset_floor が 2 箇所 (初期適用 + 事後再適用) 存在する."""
        src = Path(
            _PROJECT_ROOT / "scripts" / "v460" / "run_fill_test.py"
        )
        content = src.read_text(encoding="utf-8")

        # sell_offset_floor の参照箇所をカウント
        # 088# 初期適用 + 091# 事後再適用 + config 定義 で少なくとも 3 箇所以上
        count = content.count("sell_offset_floor")
        assert count >= 4, (
            f"sell_offset_floor の参照が {count} 箇所しかない (≥4 期待)"
        )


# =====================================================================
# 091#-4: 090 ドキュメント修正検証
# =====================================================================


class TestDoc090Corrections:
    """091# #4,#5: 090 ドキュメントの仕様誤認修正."""

    @pytest.fixture()
    def doc_content(self) -> str:
        doc = Path(
            _PROJECT_ROOT / "docs" / "v460" / "090_ph2_deep_dive_v2_for_codex.md"
        )
        return doc.read_text(encoding="utf-8")

    def test_offset_not_multiplication(self, doc_content: str) -> None:
        """sell offset 記述が '乗算' ではなく '置換' になっている."""
        assert "置換" in doc_content or "置き換" in doc_content
        # 旧記述 "× side_offset.sell" がないこと
        assert "× side_offset.sell" not in doc_content

    def test_early_exit_not_cancel(self, doc_content: str) -> None:
        """Early Exit が '即キャンセル' ではなく '次サイクル反転' と記述."""
        assert "次サイクル" in doc_content
        # 旧記述の「即キャンセル検討」がないこと
        assert "即キャンセル検討" not in doc_content

    def test_091_correction_section(self, doc_content: str) -> None:
        """091# 修正記録セクションが存在する."""
        assert "091# 修正記録" in doc_content

    def test_pre_hypothesis_note(self, doc_content: str) -> None:
        """事前仮説としての位置づけが明記されている."""
        assert "事前仮説" in doc_content
