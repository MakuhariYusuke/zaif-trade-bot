"""
Feature Dependencies Management with DAG (Directed Acyclic Graph).

This module provides dependency graph management for features,
including cycle detection, evaluation order determination, and
failure propagation.
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, deque
import json
from datetime import datetime


class DependencyGraph:
    """Directed Acyclic Graph for feature dependencies"""

    def __init__(self) -> None:
        self.graph: Dict[str, Set[str]] = defaultdict(set)  # feature -> dependencies
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)  # feature -> dependents
        self.failed_features: Set[str] = set()
        self.blocked_features: Set[str] = set()

    def add_dependency(self, feature: str, dependency: str) -> None:
        """Add a dependency relationship: feature depends on dependency"""
        self.graph[feature].add(dependency)
        self.reverse_graph[dependency].add(feature)

    def add_feature_dependencies(self, feature: str, dependencies: List[str]) -> None:
        """Add multiple dependencies for a feature"""
        for dep in dependencies:
            self.add_dependency(feature, dep)

    def remove_feature(self, feature: str) -> None:
        """Remove a feature and all its relationships"""
        # Remove from dependencies
        if feature in self.graph:
            del self.graph[feature]

        # Remove from reverse dependencies
        if feature in self.reverse_graph:
            del self.reverse_graph[feature]

        # Remove feature from other features' dependency lists
        for deps in self.graph.values():
            deps.discard(feature)

        # Remove feature from other features' dependent lists
        for dependents in self.reverse_graph.values():
            dependents.discard(feature)

        # Clean up failed/blocked status
        self.failed_features.discard(feature)
        self.blocked_features.discard(feature)

    def has_cycles(self) -> bool:
        """Check if the graph contains cycles"""
        visited = set()
        rec_stack = set()

        def has_cycle_util(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.graph.get(node, set()):
                if neighbor not in visited:
                    if has_cycle_util(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in self.graph:
            if node not in visited:
                if has_cycle_util(node):
                    return True

        return False

    def get_evaluation_order(self) -> List[str]:
        """
        Get topological sort order for feature evaluation.
        Features with no dependencies come first.
        """
        if self.has_cycles():
            raise ValueError("Dependency graph contains cycles")

        # Kahn's algorithm
        in_degree: Dict[str, int] = defaultdict(int)
        for node in self.graph:
            for dep in self.graph[node]:
                in_degree[dep] += 1

        # Nodes with no incoming edges (no dependencies)
        queue = deque([node for node in self.graph if in_degree[node] == 0])

        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            # For each dependent of this node
            for dependent in self.reverse_graph.get(node, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self.graph):
            raise ValueError("Graph contains cycles or is not fully connected")

        return result

    def mark_feature_failed(self, feature: str) -> None:
        """Mark a feature as failed and propagate blocking to dependents"""
        if feature not in self.failed_features:
            self.failed_features.add(feature)

            # Recursively block all dependents
            self._propagate_blocking(feature)

    def _propagate_blocking(self, feature: str) -> None:
        """Recursively propagate blocking to all dependents"""
        for dependent in self.reverse_graph.get(feature, set()):
            if dependent not in self.blocked_features:
                self.blocked_features.add(dependent)
                # Continue propagation
                self._propagate_blocking(dependent)

    def is_feature_blocked(self, feature: str) -> bool:
        """Check if a feature is blocked due to failed dependencies"""
        return feature in self.blocked_features

    def get_dependency_chain(self, feature: str) -> List[str]:
        """Get the full dependency chain for a feature"""
        chain = []
        visited = set()

        def traverse(node: str) -> None:
            if node in visited:
                return
            visited.add(node)
            chain.append(node)
            for dep in self.graph.get(node, set()):
                traverse(dep)

        traverse(feature)
        return chain

    def get_blocked_children(self, feature: str) -> List[str]:
        """Get all features blocked by this feature's failure"""
        blocked = []
        visited = set()

        def collect_blocked(node: str) -> None:
            if node in visited:
                return
            visited.add(node)

            for dependent in self.reverse_graph.get(node, set()):
                if dependent not in blocked:
                    blocked.append(dependent)
                    collect_blocked(dependent)

        collect_blocked(feature)
        return blocked

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation"""
        return {
            "graph": {k: list(v) for k, v in self.graph.items()},
            "reverse_graph": {k: list(v) for k, v in self.reverse_graph.items()},
            "failed_features": list(self.failed_features),
            "blocked_features": list(self.blocked_features)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DependencyGraph':
        """Create graph from dictionary representation"""
        graph = cls()
        graph.graph = defaultdict(set, {k: set(v) for k, v in data.get('graph', {}).items()})
        graph.reverse_graph = defaultdict(set, {k: set(v) for k, v in data.get('reverse_graph', {}).items()})
        graph.failed_features = set(data.get('failed_features', []))
        graph.blocked_features = set(data.get('blocked_features', []))
        return graph

    def get_evaluation_candidates(self, available_features: List[str]) -> List[str]:
        """
        Get features that can be evaluated (all dependencies are available and not failed)
        """
        candidates = []

        for feature in available_features:
            if feature in self.blocked_features:
                continue

            # Check if all dependencies are available
            deps = self.graph.get(feature, set())
            if all(dep in available_features and dep not in self.failed_features for dep in deps):
                candidates.append(feature)

        return candidates


class FeatureDependencyManager:
    """Manager for feature dependencies and evaluation orchestration"""

    def __init__(self) -> None:
        self.graph = DependencyGraph()
        self.feature_categories: Dict[str, str] = {}  # feature -> category

    def register_feature(self, name: str, dependencies: List[str], category: str = "other") -> None:
        """Register a feature with its dependencies"""
        self.graph.add_feature_dependencies(name, dependencies)
        self.feature_categories[name] = category

    def unregister_feature(self, name: str) -> None:
        """Unregister a feature"""
        self.graph.remove_feature(name)
        self.feature_categories.pop(name, None)

    def mark_evaluation_result(self, feature: str, success: bool, error_details: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Mark the result of feature evaluation"""
        if not success:
            self.graph.mark_feature_failed(feature)

            # Record dependency chain and blocked children for coverage
            dependency_chain = self.graph.get_dependency_chain(feature)
            blocked_children = self.graph.get_blocked_children(feature)

            return {
                "dependency_chain": dependency_chain,
                "blocked_children": blocked_children,
                "error_details": error_details
            }

        return None

    def get_evaluation_plan(self, target_features: List[str]) -> Dict[str, Any]:
        """
        Create an evaluation plan considering dependencies

        Returns:
            {
                "evaluation_order": [...],
                "blocked_features": [...],
                "available_features": [...]
            }
        """
        # Get all features that need to be evaluated (including dependencies)
        all_required = set(target_features)
        for feature in target_features:
            all_required.update(self.graph.get_dependency_chain(feature))

        # Get evaluation order
        try:
            evaluation_order = self.graph.get_evaluation_order()
            # Filter to only required features
            evaluation_order = [f for f in evaluation_order if f in all_required]
        except ValueError:
            # If cycles detected, fall back to simple order
            evaluation_order = list(all_required)

        # Get blocked features
        blocked_features = [f for f in all_required if self.graph.is_feature_blocked(f)]

        # Get available features (not blocked)
        available_features = [f for f in evaluation_order if f not in blocked_features]

        return {
            "evaluation_order": evaluation_order,
            "blocked_features": blocked_features,
            "available_features": available_features,
            "has_cycles": self.graph.has_cycles()
        }

    def get_category_features(self, category: str) -> List[str]:
        """Get all features in a specific category"""
        return [name for name, cat in self.feature_categories.items() if cat == category]

    def validate_graph(self) -> Dict[str, Any]:
        """Validate the dependency graph"""
        issues = []

        if self.graph.has_cycles():
            issues.append("Dependency graph contains cycles")

        # Check for orphaned dependencies
        all_features = set(self.graph.graph.keys())
        for deps in self.graph.reverse_graph.values():
            all_features.update(deps)

        for feature in all_features:
            if feature not in self.feature_categories:
                issues.append(f"Feature '{feature}' is referenced in dependencies but not registered")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_features": len(self.feature_categories),
            "total_dependencies": sum(len(deps) for deps in self.graph.graph.values())
        }


def create_dependency_manager() -> FeatureDependencyManager:
    """Factory function to create dependency manager"""
    return FeatureDependencyManager()