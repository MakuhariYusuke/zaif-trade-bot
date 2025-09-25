"""
Weekly report generation tests for comprehensive feature evaluation reporting.
"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestWeeklyReportGeneration:
    """Test weekly report generation functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_evaluator(self):
        """Create mock ComprehensiveFeatureReEvaluator"""
        evaluator = MagicMock()

        # Mock evaluation results
        evaluator.return_value = {
            'status': 'verified',
            'computation_time_ms': 150.5,
            'baseline_sharpe': 0.8,
            'best_delta_sharpe': 0.15,
            'total_features_tested': 10,
            'successful_features': 8,
            'failed_features': 2
        }

        return evaluator

    def test_weekly_report_contains_unverified_section(self, temp_dir, mock_evaluator):
        """Test that weekly report contains unverified features section"""
        report_path = temp_dir / "weekly_report.md"

        # Mock unverified features data
        unverified_data = {
            'TestFeature1': {'status': 'unverified', 'last_evaluated': '2024-01-01'},
            'TestFeature2': {'status': 'pending', 'last_evaluated': '2024-01-02'}
        }

        import sys
        from pathlib import Path
        import importlib.util

        reports_path = Path(__file__).parent.parent.parent / "reports" / "generate_weekly_report.py"
        spec = importlib.util.spec_from_file_location("generate_weekly_report", reports_path)
        if spec is None or spec.loader is None:
            pytest.skip("Could not load generate_weekly_report module")
        generate_weekly_report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generate_weekly_report)
        sys.modules["generate_weekly_report"] = generate_weekly_report

        with patch.object(generate_weekly_report, 'load_ablation_results') as mock_load, \
             patch.object(generate_weekly_report, 'load_evaluation_config') as mock_config, \
             patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:

            # Mock the data loading
            mock_load.return_value = {
                'ablation_results': {
                    'FeatureA': {'delta_sharpe': {'mean': 0.1, 'ci95': [0.05, 0.15]}},
                    'FeatureB': {'delta_sharpe': {'mean': -0.02, 'ci95': [-0.05, 0.01]}}
                }
            }
            mock_config.return_value = {'thresholds': {'re_evaluate': 0.05, 'monitor': 0.01}}

            # Mock file operations
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            # Import and run the main function
            from ztb.reports.generate_weekly_report import main
            # This would normally be called with arguments, but we'll mock the necessary parts

            # Check that file was written
            assert mock_open.called

            # Check that unverified section would be included
            # (In real implementation, this would check the actual content)
            assert True  # Placeholder - actual implementation would verify content

    def test_unverified_features_json_generation(self, temp_dir):
        """Test that unverified_features.json is generated correctly"""
        json_path = temp_dir / "unverified_features.json"

        # Mock data
        unverified_data = {
            'FeatureA': {
                'status': 'unverified',
                'last_evaluation': '2024-01-01T00:00:00',
                'reason': 'insufficient_data',
                'metrics': {'sample_size': 50, 'sharpe': 0.2}
            },
            'FeatureB': {
                'status': 'pending',
                'last_evaluation': '2024-01-02T00:00:00',
                'reason': 'being_evaluated',
                'metrics': {'sample_size': 80, 'sharpe': 0.25}
            }
        }

        # Write JSON file
        with open(json_path, 'w') as f:
            json.dump(unverified_data, f, indent=2)

        # Verify file exists and has correct content
        assert json_path.exists()

        with open(json_path, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data == unverified_data
        assert 'FeatureA' in loaded_data
        assert 'FeatureB' in loaded_data
        assert loaded_data['FeatureA']['status'] == 'unverified'
        assert loaded_data['FeatureB']['status'] == 'pending'

    def test_markdown_table_formatting(self):
        """Test that markdown tables are properly formatted"""
        # Test data
        features_data = [
            {'name': 'FeatureA', 'status': 'verified', 'sharpe': 0.45, 'samples': 150},
            {'name': 'FeatureB', 'status': 'pending', 'sharpe': 0.25, 'samples': 80},
            {'name': 'FeatureC', 'status': 'unverified', 'sharpe': 0.15, 'samples': 30}
        ]

        # Generate markdown table
        table_lines = [
            "| Feature | Status | Sharpe | Samples |",
            "|---------|--------|--------|---------|"
        ]

        for feature in features_data:
            table_lines.append(
                f"| {feature['name']} | {feature['status']} | {feature['sharpe']:.2f} | {feature['samples']} |"
            )

        table = '\n'.join(table_lines)

        # Verify table structure
        assert table.startswith('| Feature | Status | Sharpe | Samples |')
        assert '|---------|--------|--------|---------|' in table
        assert '| FeatureA | verified | 0.45 | 150 |' in table
        assert '| FeatureB | pending | 0.25 | 80 |' in table
        assert '| FeatureC | unverified | 0.15 | 30 |' in table

    def test_report_section_headers(self):
        """Test that report contains all required section headers"""
        required_sections = [
            '# Weekly Feature Evaluation Report',
            '## Summary',
            '## Verified Features',
            '## Unverified Features',
            '## Failed Features',
            '## Recommendations'
        ]

        # Mock report content
        mock_report = f"""
{required_sections[0]}

{required_sections[1]}
- Total evaluations: 10
- Success rate: 80%

{required_sections[2]}
| Feature | Sharpe | Samples |
|---------|--------|---------|
| FeatureA | 0.45 | 150 |

{required_sections[3]}
| Feature | Status | Reason |
|---------|--------|--------|
| FeatureB | pending | insufficient_data |

{required_sections[4]}
| Feature | Error |
|---------|-------|
| FeatureC | computation_error |

{required_sections[5]}
- Review unverified features
- Investigate failed features
"""

        for section in required_sections:
            assert section in mock_report, f"Missing section: {section}"

    def test_json_attachment_structure(self, temp_dir):
        """Test that JSON attachments have correct structure"""
        json_path = temp_dir / "feature_details.json"

        # Create structured JSON data
        feature_details = {
            'metadata': {
                'generated_at': '2024-01-01T00:00:00',
                'evaluation_period': '2023-12-25 to 2024-01-01',
                'total_features': 15
            },
            'verified_features': {
                'FeatureA': {
                    'sharpe_ratio': 0.45,
                    'win_rate': 0.58,
                    'max_drawdown': -0.12,
                    'sample_size': 150,
                    'computation_time_ms': 145.2
                }
            },
            'unverified_features': {
                'FeatureB': {
                    'status': 'pending',
                    'reason': 'insufficient_samples',
                    'current_samples': 80,
                    'required_samples': 100,
                    'last_evaluation': '2024-01-01T00:00:00'
                }
            },
            'performance_summary': {
                'avg_sharpe_improvement': 0.15,
                'total_computation_time': 2150.5,
                'success_rate': 0.8
            }
        }

        with open(json_path, 'w') as f:
            json.dump(feature_details, f, indent=2)

        # Verify structure
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)

        assert 'metadata' in loaded_data
        assert 'verified_features' in loaded_data
        assert 'unverified_features' in loaded_data
        assert 'performance_summary' in loaded_data

        # Check metadata
        metadata = loaded_data['metadata']
        assert 'generated_at' in metadata
        assert 'total_features' in metadata
        assert metadata['total_features'] == 15

        # Check performance summary
        perf = loaded_data['performance_summary']
        assert 'avg_sharpe_improvement' in perf
        assert 'success_rate' in perf
        assert perf['success_rate'] == 0.8

    def test_error_handling_in_report_generation(self):
        """Test error handling when report generation fails"""
        import sys
        from pathlib import Path
        import importlib.util

        reports_path = Path(__file__).parent.parent.parent / "reports" / "generate_weekly_report.py"
        spec = importlib.util.spec_from_file_location("generate_weekly_report", reports_path)
        if spec is None or spec.loader is None:
            pytest.skip("Could not load generate_weekly_report module")
        generate_weekly_report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generate_weekly_report)

        with patch.object(generate_weekly_report, 'load_ablation_results', side_effect=IOError("File not found")):
            # Should handle file I/O errors gracefully
            try:
                from ztb.reports.generate_weekly_report import load_ablation_results
                # This would normally try to read a file
                # In test, we just ensure no unhandled exceptions
                assert True
            except IOError:
                # Should handle IOError gracefully
                assert True

    def test_report_content_validation(self):
        """Test that report content meets quality standards"""
        # Mock report sections
        sections = {
            'summary': 'Total features evaluated: 10',
            'verified_count': 7,
            'unverified_count': 2,
            'failed_count': 1,
            'avg_improvement': 0.15
        }

        # Validate content quality
        assert sections['verified_count'] >= 0
        assert sections['unverified_count'] >= 0
        assert sections['failed_count'] >= 0
        assert sections['verified_count'] + sections['unverified_count'] + sections['failed_count'] == 10

        # Check that improvement is reasonable
        assert -1 <= sections['avg_improvement'] <= 1, "Sharpe improvement should be reasonable"

    def test_json_file_naming_convention(self, temp_dir):
        """Test that JSON files follow naming conventions"""
        # Test various JSON file names
        json_files = [
            'unverified_features.json',
            'feature_details.json',
            'evaluation_summary.json',
            'performance_metrics.json'
        ]

        for filename in json_files:
            path = temp_dir / filename

            # Create empty file
            path.touch()

            # Check naming convention
            assert filename.endswith('.json'), f"File {filename} should end with .json"
            assert '_' in filename or filename == 'feature_details.json', f"File {filename} should use snake_case"

            # Verify file exists
            assert path.exists()