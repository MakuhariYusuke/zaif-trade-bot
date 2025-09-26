"""
Unit tests for ReportGenerator
"""
import tempfile
import os
from pathlib import Path
from ztb.utils.report_generator import ReportGenerator


def test_generate_csv():
    """Test CSV generation"""
    generator = ReportGenerator()
    results = [{"name": "test", "value": 1.0}]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name

    try:
        generator.generate_csv(results, temp_path)
        assert os.path.exists(temp_path)
    finally:
        os.unlink(temp_path)


def test_generate_json():
    """Test JSON generation"""
    generator = ReportGenerator()
    results = [{"name": "test", "value": 1.0}]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        generator.generate_json(results, temp_path)
        assert os.path.exists(temp_path)
    finally:
        os.unlink(temp_path)


def test_generate_markdown():
    """Test Markdown generation"""
    generator = ReportGenerator()
    results = [{"name": "test", "value": 1.0}]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        temp_path = f.name

    try:
        generator.generate_markdown(results, temp_path)
        assert os.path.exists(temp_path)
    finally:
        os.unlink(temp_path)