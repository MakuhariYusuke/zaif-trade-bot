#!/usr/bin/env python3
"""
Unit tests for unified_analyze.py

Tests the UnifiedAnalysisSuite and its analyzer components.
"""

import argparse
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.analysis.unified_analyze import (
    UnifiedAnalysisSuite,
    ModelAnalysis,
    DataAnalysis,
    TrainingAnalysis,
    PerformanceAnalysis,
    ComparativeAnalysis,
    PaperTradingAnalysis,
    DiagnosticAnalysis,
    SpecializedAnalysis,
    SessionAnalysis,
    BaseAnalyzer,
    create_parser,
    main,
)


class TestUnifiedAnalysisSuite:
    """Test cases for UnifiedAnalysisSuite class."""

    def test_init(self):
        """Test initialization of UnifiedAnalysisSuite."""
        suite = UnifiedAnalysisSuite()

        assert isinstance(suite.project_root, Path)
        assert suite.timesteps_override is None
        assert isinstance(suite.categories, dict)
        assert len(suite.categories) == 9  # All expected categories

        # Check that all expected categories are present
        expected_categories = [
            "model", "data", "training", "performance",
            "comparative", "paper_trading", "diagnostic",
            "specialized", "session"
        ]
        for category in expected_categories:
            assert category in suite.categories

    def test_run_unknown_category(self):
        """Test run method with unknown category."""
        suite = UnifiedAnalysisSuite()

        # Create mock args with unknown category
        args = argparse.Namespace()
        args.category = "unknown"
        args.tool = "some_tool"

        result = suite.run(args)
        assert result == 1

    def test_run_no_tool_specified_known_category(self):
        """Test run method with known category but no tool specified."""
        suite = UnifiedAnalysisSuite()

        # Create mock args
        args = argparse.Namespace()
        args.category = "model"
        args.tool = None

        with patch('builtins.print') as mock_print:
            result = suite.run(args)
            assert result == 0
            mock_print.assert_called_once()

    def test_run_unknown_tool(self):
        """Test run method with unknown tool."""
        suite = UnifiedAnalysisSuite()

        # Create mock args
        args = argparse.Namespace()
        args.category = "model"
        args.tool = "unknown_tool"

        result = suite.run(args)
        assert result == 1

    @patch('ztb.analysis.unified_analyze.ModelAnalysis')
    def test_run_valid_tool(self, mock_model_analysis):
        """Test run method with valid category and tool."""
        suite = UnifiedAnalysisSuite()

        # Mock the analyzer instance
        mock_analyzer = MagicMock()
        mock_analyzer.run_sac.return_value = 0
        mock_model_analysis.return_value = mock_analyzer

        # Create mock args
        args = argparse.Namespace()
        args.category = "model"
        args.tool = "sac"

        result = suite.run(args)
        assert result == 0
        mock_model_analysis.assert_called_once()
        mock_analyzer.run_sac.assert_called_once_with(args)

    def test_run_with_timesteps_override(self):
        """Test run method with timesteps override."""
        suite = UnifiedAnalysisSuite()

        # Create mock args with timesteps
        args = argparse.Namespace()
        args.category = "model"
        args.tool = "sac"
        args.timesteps = 1000
        args.model = "/path/to/model.zip"

        # Mock the entire run_sac method to avoid import issues
        with patch.object(ModelAnalysis, 'run_sac', return_value=0) as mock_run_sac:
            result = suite.run(args)
            assert result == 0
            assert suite.timesteps_override == 1000
            mock_run_sac.assert_called_once_with(args)

    def test_run_exception_handling(self):
        """Test run method exception handling."""
        suite = UnifiedAnalysisSuite()

        # Create mock args
        args = argparse.Namespace()
        args.category = "model"
        args.tool = "sac"

        with patch('ztb.analysis.unified_analyze.ModelAnalysis') as mock_model_analysis:
            mock_model_analysis.side_effect = Exception("Test error")

            result = suite.run(args)
            assert result == 1


class TestBaseAnalyzer:
    """Test cases for BaseAnalyzer class."""

    def test_init(self):
        """Test BaseAnalyzer initialization."""
        analyzer = BaseAnalyzer()

        # BaseAnalyzer doesn't have project_root attribute by default
        # It's inherited from UnifiedAnalysisSuite's global project_root
        assert hasattr(analyzer, 'get_available_tools')

    def test_get_available_tools(self):
        """Test get_available_tools returns list."""
        analyzer = BaseAnalyzer()

        tools = analyzer.get_available_tools()
        assert isinstance(tools, list)


class TestModelAnalysis:
    """Test cases for ModelAnalysis class."""

    def test_init(self):
        """Test ModelAnalysis initialization."""
        analyzer = ModelAnalysis()

        assert isinstance(analyzer, BaseAnalyzer)
        assert hasattr(analyzer, 'get_available_tools')

    def test_get_available_tools(self):
        """Test get_available_tools returns expected tools."""
        analyzer = ModelAnalysis()

        tools = analyzer.get_available_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    @patch('ztb.analysis.unified_analyze.logger')
    def test_run_sac_no_model_path(self, mock_logger):
        """Test run_sac method with no model path."""
        analyzer = ModelAnalysis()

        args = argparse.Namespace()
        args.model = None

        result = analyzer.run_sac(args)
        assert result == 1
        mock_logger.error.assert_called()

    @patch('os.path.exists')
    @patch('ztb.analysis.unified_analyze.logger')
    def test_run_sac_model_not_found(self, mock_logger, mock_exists):
        """Test run_sac method when model file doesn't exist."""
        analyzer = ModelAnalysis()

        mock_exists.return_value = False

        args = argparse.Namespace()
        args.model = "/path/to/nonexistent/model.zip"

        result = analyzer.run_sac(args)
        assert result == 1
        mock_logger.error.assert_called()


class TestDataAnalysis:
    """Test cases for DataAnalysis class."""

    def test_init(self):
        """Test DataAnalysis initialization."""
        analyzer = DataAnalysis()

        assert isinstance(analyzer, BaseAnalyzer)

    def test_get_available_tools(self):
        """Test get_available_tools returns expected tools."""
        analyzer = DataAnalysis()

        tools = analyzer.get_available_tools()
        assert isinstance(tools, list)
        assert "quality" in tools


class TestTrainingAnalysis:
    """Test cases for TrainingAnalysis class."""

    def test_init(self):
        """Test TrainingAnalysis initialization."""
        analyzer = TrainingAnalysis()

        assert isinstance(analyzer, BaseAnalyzer)

    def test_get_available_tools(self):
        """Test get_available_tools returns expected tools."""
        analyzer = TrainingAnalysis()

        tools = analyzer.get_available_tools()
        assert isinstance(tools, list)
        # Check that it contains some expected tools
        expected_tools = ["progress", "profile", "sac_v423"]
        for tool in expected_tools:
            assert tool in tools


class TestPerformanceAnalysis:
    """Test cases for PerformanceAnalysis class."""

    def test_init(self):
        """Test PerformanceAnalysis initialization."""
        analyzer = PerformanceAnalysis()

        assert isinstance(analyzer, BaseAnalyzer)

    def test_get_available_tools(self):
        """Test get_available_tools returns expected tools."""
        analyzer = PerformanceAnalysis()

        tools = analyzer.get_available_tools()
        assert isinstance(tools, list)


class TestComparativeAnalysis:
    """Test cases for ComparativeAnalysis class."""

    def test_init(self):
        """Test ComparativeAnalysis initialization."""
        analyzer = ComparativeAnalysis()

        assert isinstance(analyzer, BaseAnalyzer)

    def test_get_available_tools(self):
        """Test get_available_tools returns expected tools."""
        analyzer = ComparativeAnalysis()

        tools = analyzer.get_available_tools()
        assert isinstance(tools, list)
        assert "versions" in tools


class TestPaperTradingAnalysis:
    """Test cases for PaperTradingAnalysis class."""

    def test_init(self):
        """Test PaperTradingAnalysis initialization."""
        analyzer = PaperTradingAnalysis()

        assert isinstance(analyzer, BaseAnalyzer)

    def test_get_available_tools(self):
        """Test get_available_tools returns expected tools."""
        analyzer = PaperTradingAnalysis()

        tools = analyzer.get_available_tools()
        assert isinstance(tools, list)


class TestDiagnosticAnalysis:
    """Test cases for DiagnosticAnalysis class."""

    def test_init(self):
        """Test DiagnosticAnalysis initialization."""
        analyzer = DiagnosticAnalysis()

        assert isinstance(analyzer, BaseAnalyzer)

    def test_get_available_tools(self):
        """Test get_available_tools returns expected tools."""
        analyzer = DiagnosticAnalysis()

        tools = analyzer.get_available_tools()
        assert isinstance(tools, list)


class TestSpecializedAnalysis:
    """Test cases for SpecializedAnalysis class."""

    def test_init(self):
        """Test SpecializedAnalysis initialization."""
        analyzer = SpecializedAnalysis()

        assert isinstance(analyzer, BaseAnalyzer)

    def test_get_available_tools(self):
        """Test get_available_tools returns expected tools."""
        analyzer = SpecializedAnalysis()

        tools = analyzer.get_available_tools()
        assert isinstance(tools, list)


class TestSessionAnalysis:
    """Test cases for SessionAnalysis class."""

    def test_init(self):
        """Test SessionAnalysis initialization."""
        analyzer = SessionAnalysis()

        assert isinstance(analyzer, BaseAnalyzer)

    def test_get_available_tools(self):
        """Test get_available_tools returns expected tools."""
        analyzer = SessionAnalysis()

        tools = analyzer.get_available_tools()
        assert isinstance(tools, list)


class TestParserAndMain:
    """Test cases for parser creation and main function."""

    def test_create_parser(self):
        """Test create_parser returns ArgumentParser."""
        parser = create_parser()

        assert isinstance(parser, argparse.ArgumentParser)

        # Check that required arguments are present
        help_text = parser.format_help()
        assert "category" in help_text
        assert "tool" in help_text

    @patch('sys.argv', ['unified_analyze.py', 'model', 'sac', '--model', 'test.zip'])
    @patch('ztb.analysis.unified_analyze.create_parser')
    @patch('ztb.analysis.unified_analyze.UnifiedAnalysisSuite')
    def test_main_success(self, mock_suite_class, mock_create_parser):
        """Test main function with successful execution."""
        mock_parser = MagicMock()
        mock_args = MagicMock()
        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        mock_suite = MagicMock()
        mock_suite.run.return_value = 0
        mock_suite_class.return_value = mock_suite

        with patch('sys.exit') as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)

    @patch('sys.argv', ['unified_analyze.py', 'invalid'])
    @patch('ztb.analysis.unified_analyze.create_parser')
    @patch('ztb.analysis.unified_analyze.UnifiedAnalysisSuite')
    def test_main_failure(self, mock_suite_class, mock_create_parser):
        """Test main function with failed execution."""
        mock_parser = MagicMock()
        mock_args = MagicMock()
        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        mock_suite = MagicMock()
        mock_suite.run.return_value = 1
        mock_suite_class.return_value = mock_suite

        with patch('sys.exit') as mock_exit:
            main()
            mock_exit.assert_called_once_with(1)


if __name__ == "__main__":
    pytest.main([__file__])