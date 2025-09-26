"""
Tests for Feature Dependencies functionality.
"""

import pytest
from ztb.evaluation.dependency import DependencyGraph, FeatureDependencyManager


class TestDependencyGraph:
    """Test DependencyGraph class"""

    def test_add_dependency(self):
        graph = DependencyGraph()
        graph.add_dependency("child", "parent")

        assert "parent" in graph.graph["child"]
        assert "child" in graph.reverse_graph["parent"]

    def test_has_cycles_no_cycles(self):
        graph = DependencyGraph()
        graph.add_dependency("child", "parent")
        graph.add_dependency("parent", "grandparent")

        assert not graph.has_cycles()

    def test_has_cycles_with_cycles(self):
        graph = DependencyGraph()
        graph.add_dependency("a", "b")
        graph.add_dependency("b", "c")
        graph.add_dependency("c", "a")

        assert graph.has_cycles()

    def test_get_evaluation_order(self):
        graph = DependencyGraph()
        graph.add_dependency("child1", "parent")
        graph.add_dependency("child2", "parent")
        graph.add_dependency("grandchild", "child1")

        order = graph.get_evaluation_order()

        # Parent should come before children
        parent_idx = order.index("parent")
        child1_idx = order.index("child1")
        child2_idx = order.index("child2")
        grandchild_idx = order.index("grandchild")

        assert parent_idx < child1_idx
        assert parent_idx < child2_idx
        assert child1_idx < grandchild_idx

    def test_mark_feature_failed(self):
        graph = DependencyGraph()
        graph.add_dependency("child1", "parent")
        graph.add_dependency("child2", "parent")
        graph.add_dependency("grandchild", "child1")

        graph.mark_feature_failed("parent")

        assert graph.is_feature_blocked("child1")
        assert graph.is_feature_blocked("child2")
        assert graph.is_feature_blocked("grandchild")
        assert "parent" in graph.failed_features

    def test_get_dependency_chain(self):
        graph = DependencyGraph()
        graph.add_dependency("child", "parent")
        graph.add_dependency("parent", "grandparent")

        chain = graph.get_dependency_chain("child")
        assert chain == ["child", "parent", "grandparent"]

    def test_get_blocked_children(self):
        graph = DependencyGraph()
        graph.add_dependency("child1", "parent")
        graph.add_dependency("child2", "parent")
        graph.add_dependency("grandchild", "child1")

        graph.mark_feature_failed("parent")

        blocked = graph.get_blocked_children("parent")
        assert set(blocked) == {"child1", "child2", "grandchild"}

    def test_get_evaluation_candidates(self):
        graph = DependencyGraph()
        graph.add_dependency("child", "parent")

        # No features available
        candidates = graph.get_evaluation_candidates([])
        assert candidates == []

        # Parent available, child not blocked
        candidates = graph.get_evaluation_candidates(["parent"])
        assert "parent" in candidates

        # Both available
        candidates = graph.get_evaluation_candidates(["parent", "child"])
        assert set(candidates) == {"parent", "child"}

        # Parent failed, child blocked
        graph.mark_feature_failed("parent")
        candidates = graph.get_evaluation_candidates(["parent", "child"])
        assert candidates == []


class TestFeatureDependencyManager:
    """Test FeatureDependencyManager class"""

    def test_register_feature(self):
        manager = FeatureDependencyManager()
        manager.register_feature("rsi_14", ["close"], "oscillator")

        assert "rsi_14" in manager.feature_categories
        assert manager.feature_categories["rsi_14"] == "oscillator"
        assert "close" in manager.graph.graph["rsi_14"]

    def test_unregister_feature(self):
        manager = FeatureDependencyManager()
        manager.register_feature("rsi_14", ["close"], "oscillator")

        manager.unregister_feature("rsi_14")

        assert "rsi_14" not in manager.feature_categories
        assert "rsi_14" not in manager.graph.graph

    def test_mark_evaluation_result_success(self):
        manager = FeatureDependencyManager()
        manager.register_feature("rsi_14", ["close"], "oscillator")

        result = manager.mark_evaluation_result("rsi_14", True)
        assert result is None
        assert not manager.graph.is_feature_blocked("rsi_14")

    def test_mark_evaluation_result_failure(self):
        manager = FeatureDependencyManager()
        manager.register_feature("rsi_14", ["close"], "oscillator")
        manager.register_feature("macd", ["rsi_14"], "oscillator")

        error_details = {"error": "computation_error"}
        result = manager.mark_evaluation_result("rsi_14", False, error_details)

        assert result is not None
        assert "dependency_chain" in result
        assert "blocked_children" in result
        assert result["blocked_children"] == ["macd"]
        assert manager.graph.is_feature_blocked("macd")

    def test_get_evaluation_plan(self):
        manager = FeatureDependencyManager()
        manager.register_feature("rsi_14", ["close"], "oscillator")
        manager.register_feature("macd", ["close"], "oscillator")
        manager.register_feature("combined", ["rsi_14", "macd"], "oscillator")

        plan = manager.get_evaluation_plan(["rsi_14", "macd", "combined"])

        assert "evaluation_order" in plan
        assert "blocked_features" in plan
        assert "available_features" in plan

        # Should be able to evaluate rsi_14 and macd first
        assert set(plan["available_features"]) == {"rsi_14", "macd"}

    def test_get_evaluation_plan_with_failures(self):
        manager = FeatureDependencyManager()
        manager.register_feature("rsi_14", ["close"], "oscillator")
        manager.register_feature("macd", ["close"], "oscillator")
        manager.register_feature("combined", ["rsi_14", "macd"], "oscillator")

        # Mark rsi_14 as failed
        manager.mark_evaluation_result("rsi_14", False)

        plan = manager.get_evaluation_plan(["rsi_14", "macd", "combined"])

        # combined should be blocked
        assert "combined" in plan["blocked_features"]
        assert "combined" not in plan["available_features"]

    def test_get_category_features(self):
        manager = FeatureDependencyManager()
        manager.register_feature("rsi_14", ["close"], "oscillator")
        manager.register_feature("ema_20", ["close"], "trend")
        manager.register_feature("macd", ["close"], "oscillator")

        oscillators = manager.get_category_features("oscillator")
        trends = manager.get_category_features("trend")

        assert set(oscillators) == {"rsi_14", "macd"}
        assert set(trends) == {"ema_20"}

    def test_validate_graph(self):
        manager = FeatureDependencyManager()
        manager.register_feature("rsi_14", ["close"], "oscillator")

        validation = manager.validate_graph()

        assert validation["valid"]
        assert validation["total_features"] == 1
        assert validation["total_dependencies"] == 1

    def test_validate_graph_with_issues(self):
        manager = FeatureDependencyManager()
        manager.register_feature("a", ["b"], "test")
        manager.register_feature("b", ["c"], "test")
        manager.register_feature("c", ["a"], "test")  # Creates cycle

        validation = manager.validate_graph()

        assert not validation["valid"]
        assert len(validation["issues"]) > 0
        assert "cycles" in str(validation["issues"])