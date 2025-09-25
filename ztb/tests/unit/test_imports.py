"""
Unit Tests for Import Validation
インポート検証の単体テスト

This module tests that all major modules can be imported without errors,
validating the package structure and dependencies.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path for testing
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TestImports(unittest.TestCase):
    """Test that all major modules can be imported"""

    def test_core_imports(self):
        """Test core algorithm imports"""
        try:
            from rl.core.algorithms.train_ppo import PPOTrainer
            self.assertIsNotNone(PPOTrainer)
            print("✅ Core algorithms import: OK")
        except ImportError as e:
            self.fail(f"Core algorithms import failed: {e}")

    def test_environment_imports(self):
        """Test environment imports"""
        try:
            from rl.envs.heavy_trading_env import HeavyTradingEnv
            self.assertIsNotNone(HeavyTradingEnv)
            print("✅ Environment import: OK")
        except ImportError as e:
            self.fail(f"Environment import failed: {e}")

    def test_data_analysis_imports(self):
        """Test data analysis imports"""
        try:
            from rl.data.analysis.data_quality_check import analyze_feature_distributions
            from rl.data.analysis.outlier_filter import ATROutlierFilter
            from rl.data.etl.pipeline import ETLPipeline
            self.assertIsNotNone(analyze_feature_distributions)
            self.assertIsNotNone(ATROutlierFilter)
            self.assertIsNotNone(ETLPipeline)
            print("✅ Data analysis imports: OK")
        except ImportError as e:
            self.fail(f"Data analysis import failed: {e}")

    def test_notification_imports(self):
        """Test notification imports"""
        try:
            from rl.notify.discord.discord_notifications import DiscordNotifier
            self.assertIsNotNone(DiscordNotifier)
            print("✅ Notification imports: OK")
        except ImportError as e:
            self.fail(f"Notification import failed: {e}")

    def test_utility_imports(self):
        """Test utility imports"""
        try:
            from rl.utils.fee_model import FeeModelFactory
            from rl.utils.resource.process_priority import ProcessPriorityManager
            self.assertIsNotNone(FeeModelFactory)
            self.assertIsNotNone(ProcessPriorityManager)
            print("✅ Utility imports: OK")
        except ImportError as e:
            self.fail(f"Utility import failed: {e}")

    def test_template_imports(self):
        """Test template imports"""
        try:
            from rl.templates.base_trainer import BaseTrainer
            from rl.templates.base_evaluator import BaseEvaluator
            self.assertIsNotNone(BaseTrainer)
            self.assertIsNotNone(BaseEvaluator)
            print("✅ Template imports: OK")
        except ImportError as e:
            self.fail(f"Template import failed: {e}")

    def test_script_imports(self):
        """Test script imports"""
        try:
            from rl.scripts.main import load_config
            from rl.scripts.evaluation.evaluate_model import TradingEvaluator
            from rl.scripts.training.optimize_params import OptunaCallback
            self.assertIsNotNone(load_config)
            # Note: TradingEvaluator and OptunaCallback may not be instantiable without setup
            print("✅ Script imports: OK")
        except ImportError as e:
            self.fail(f"Script import failed: {e}")

    def test_config_imports(self):
        """Test configuration imports"""
        try:
            from rl.config.config_validator import ConfigValidator
            self.assertIsNotNone(ConfigValidator)
            print("✅ Config imports: OK")
        except ImportError as e:
            self.fail(f"Config import failed: {e}")

    def test_all_major_modules_importable(self):
        """Test that all major modules are importable without exceptions"""
        modules_to_test = [
            'rl.core.algorithms.train_ppo',
            'rl.envs.heavy_trading_env',
            'rl.data.analysis.data_quality_check',
            'rl.data.analysis.outlier_filter',
            'rl.data.etl.pipeline',
            'rl.notify.discord.discord_notifications',
            'rl.utils.fee_model',
            'rl.utils.resource.process_priority',
            'rl.templates.base_trainer',
            'rl.templates.base_evaluator',
            'rl.config.config_validator',
        ]

        failed_imports = []

        for module in modules_to_test:
            try:
                __import__(module)
            except ImportError as e:
                failed_imports.append(f"{module}: {e}")
            except Exception as e:
                failed_imports.append(f"{module}: {e}")

        if failed_imports:
            self.fail(f"Failed to import modules: {failed_imports}")

        print(f"✅ All {len(modules_to_test)} major modules imported successfully")

    def test_package_structure(self):
        """Test that package __init__.py files exist"""
        required_init_files = [
            project_root / 'rl' / '__init__.py',
            project_root / 'rl' / 'core' / '__init__.py',
            project_root / 'rl' / 'core' / 'algorithms' / '__init__.py',
            project_root / 'rl' / 'data' / '__init__.py',
            project_root / 'rl' / 'data' / 'analysis' / '__init__.py',
            project_root / 'rl' / 'data' / 'etl' / '__init__.py',
            project_root / 'rl' / 'notify' / '__init__.py',
            project_root / 'rl' / 'notify' / 'discord' / '__init__.py',
            project_root / 'rl' / 'utils' / '__init__.py',
            project_root / 'rl' / 'utils' / 'resource' / '__init__.py',
            project_root / 'rl' / 'templates' / '__init__.py',
            project_root / 'rl' / 'scripts' / '__init__.py',
            project_root / 'rl' / 'tests' / '__init__.py',
            project_root / 'rl' / 'tests' / 'unit' / '__init__.py',
            project_root / 'rl' / 'tests' / 'integration' / '__init__.py',
            project_root / 'rl' / 'config' / '__init__.py',
        ]

        missing_files = []
        for init_file in required_init_files:
            if not init_file.exists():
                missing_files.append(str(init_file))

        if missing_files:
            self.fail(f"Missing __init__.py files: {missing_files}")

        print(f"✅ All {len(required_init_files)} __init__.py files present")

if __name__ == '__main__':
    unittest.main()