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
        # 330# extract: time filter ロジックは orchestrator_pre_cycle.py に移動
        src = Path(
            _PROJECT_ROOT / "scripts" / "v460" / "lib" / "orchestrator_pre_cycle.py"
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
        # 140# §8.1-#2 で FillRecord 追加により行数増加→検索範囲拡張
        block = "\n".join(lines[alt_side_line : alt_side_line + 50])
        assert "_batch_persistence.maybe_flush" in block, (
            f"alt_side==last_side 分岐内に _batch_persistence.maybe_flush がない:\n{block[:300]}"
        )

    def test_alt_side_branch_has_091_comment(self) -> None:
        """107# R1 で flush ロジックが _maybe_flush_batch に統合されている."""
        # 330# extract: time filter ロジックは orchestrator_pre_cycle.py に移動
        src = Path(
            _PROJECT_ROOT / "scripts" / "v460" / "lib" / "orchestrator_pre_cycle.py"
        )
        content = src.read_text(encoding="utf-8")
        # 107# R1: 重複 flush → _maybe_flush_batch 統合
        assert "107# R1" in content


# =====================================================================
# 091#-2: preflight 失敗時の即座 opposite side 切替
# =====================================================================


class TestPreflightOppositeSide:
    """091# #3: preflight 失敗時に反対 side を即時チェック.

    522# balance-forcing 完全撤廃: 反対 side への強制切替は廃止。
    残高不足時はスキップし、次サイクルで自然に反対 side が選択される。
    テストは新アーキテクチャに合わせて更新。
    """

    def test_preflight_has_opposite_side_check(self) -> None:
        """522# 残高不足時に side freeze して skip するロジックがある."""
        src = Path(
            _PROJECT_ROOT / "scripts" / "v460" / "lib" / "orchestrator_balance.py"
        )
        content = src.read_text(encoding="utf-8")

        # 522# の freeze + skip ロジックが存在する
        assert "no forced switching" in content, (
            "522# balance-forcing 撤廃ロジックが見つからない"
        )
        assert "freeze_side" in content, (
            "balance 不足時の side freeze がない"
        )

    def test_preflight_skip_has_batch_flush(self) -> None:
        """preflight skip 待機中にも batch_flush が実行される."""
        src = Path(
            _PROJECT_ROOT / "scripts" / "v460" / "lib" / "orchestrator_balance.py"  # 332# Phase 4: balance Mixin に移管
        )
        content = src.read_text(encoding="utf-8")

        # 119# BatchPersistence 委譲: maybe_flush に統合、context に "insufficient" が含まれる
        assert "_batch_persistence.maybe_flush" in content
        assert "insufficient" in content

    def test_preflight_opposite_side_logic_order(self) -> None:
        """522# balance 不足チェックが SAFE_STOP より前に位置する."""
        src = Path(
            _PROJECT_ROOT / "scripts" / "v460" / "lib" / "orchestrator_balance.py"
        )
        content = src.read_text(encoding="utf-8")

        pos_balance_skip = content.find("no forced switching")
        pos_safe_stop = content.find("SAFE_STOP: 連続 preflight")
        assert pos_balance_skip > 0, "522# balance skip ロジックが見つからない"
        assert pos_safe_stop > 0, "SAFE_STOP ロジックが見つからない"
        assert pos_balance_skip < pos_safe_stop, (
            "balance 不足スキップが SAFE_STOP より後にある"
        )


# =====================================================================
# 091#-3: sell offset_floor が spread_adaptive 後にも再適用される
# =====================================================================


class TestSellOffsetFloorPostAdaptive:
    """091# #6: offset_floor が spread_adaptive で破られないことを検証."""

    def test_post_adaptive_floor_exists_in_code(self) -> None:
        """spread_adaptive の後に sell floor 再適用ロジックがある."""
        # 120# _compute_maker_price は maker_price.py に抽出済み
        src = Path(
            _PROJECT_ROOT / "scripts" / "v460" / "lib" / "maker_price.py"
        )
        content = src.read_text(encoding="utf-8")

        # spread_adaptive ブロックの後に 091# floor 再適用がある
        assert "Post-adaptive floor re-applied" in content

    def test_post_adaptive_floor_after_spread_adaptive(self) -> None:
        """091# floor 再適用が spread_adaptive より後にある."""
        # 120# maker_price.py に抽出済み
        src = Path(
            _PROJECT_ROOT / "scripts" / "v460" / "lib" / "maker_price.py"
        )
        content = src.read_text(encoding="utf-8")

        pos_adaptive = content.rfind("[spread_adaptive]")
        pos_floor = content.find("Post-adaptive floor re-applied")
        assert pos_floor > pos_adaptive, (
            "Post-adaptive floor が spread_adaptive より前にある"
        )

    def test_first_and_post_floor_both_exist(self) -> None:
        """sell_offset_floor が 2 箇所 (初期適用 + 事後再適用) 存在する."""
        # 120# maker_price.py + run_fill_test.py 合算で検証
        mp = Path(
            _PROJECT_ROOT / "scripts" / "v460" / "lib" / "maker_price.py"
        )
        rft = Path(
            _PROJECT_ROOT / "scripts" / "v460" / "lib" / "fill_loop_orchestrator.py"  # 163# mixin 分割
        )
        content = mp.read_text(encoding="utf-8") + rft.read_text(encoding="utf-8")

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
