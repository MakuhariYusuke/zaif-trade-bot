#!/usr/bin/env python3
"""
統合分析ツール

トレーニング済みモデルに対して包括的な分析を実行します。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.analysis.core.analyzer import UnifiedAnalyzer
from ztb.io.json_io import read_json
from ztb.reporting.generators.analysis_rich import ReportGenerator
from ztb.training.core.config_manager import ConfigManager

class UnifiedAnalysisTool:
    """統合分析ツール"""

    def __init__(self, config_path: str):
        """
        初期化

        Args:
            config_path: 設定ファイルパス
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.config_manager = ConfigManager(self.config)
        self.analyzer = UnifiedAnalyzer()
        self.report_generator = ReportGenerator()

        # ロギング設定
        self._setup_logging()

    def _load_config(self) -> dict[str, Any]:
        """設定ファイルを読み込み"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        return read_json(self.config_path)

    def _setup_logging(self):
        """ロギング設定"""
        log_level = getattr(logging, self.config.get("log_level", "INFO").upper())
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/unified_analysis.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def run_analysis(
        self, model_path: str, output_dir: str | None = None
    ) -> dict[str, Any]:
        """
        分析実行

        Args:
            model_path: モデルファイルパス
            output_dir: 出力ディレクトリ

        Returns:
            分析結果
        """
        self.logger.info(f"Starting unified analysis for model: {model_path}")

        # 出力ディレクトリ設定
        if output_dir is None:
            output_dir = self.config.get("output_dir", "reports/analysis")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # モデル読み込み
            model = self._load_model(model_path)

            # 分析設定取得
            analysis_config = self.config.get("analysis", {})

            # 分析実行
            results = {}

            # パフォーマンス分析
            if analysis_config.get("performance_analysis", True):
                self.logger.info("Running performance analysis...")
                results["performance"] = self.analyzer.analyze_performance(
                    model, **analysis_config.get("performance", {})
                )

            # リスク分析
            if analysis_config.get("risk_analysis", True):
                self.logger.info("Running risk analysis...")
                results["risk"] = self.analyzer.analyze_risk(
                    model, **analysis_config.get("risk", {})
                )

            # 行動分析
            if analysis_config.get("behavioral_analysis", True):
                self.logger.info("Running behavioral analysis...")
                results["behavioral"] = self.analyzer.analyze_behavior(
                    model, **analysis_config.get("behavioral", {})
                )

            # 比較分析
            if analysis_config.get("comparison_analysis", False):
                self.logger.info("Running comparison analysis...")
                results["comparison"] = self.analyzer.analyze_comparison(
                    model, **analysis_config.get("comparison", {})
                )

            # レポート生成
            self.logger.info("Generating reports...")
            report_config = self.config.get("reporting", {})
            report_path = self.report_generator.generate_report(
                results, output_path, **report_config
            )

            results["report_path"] = str(report_path)
            results["status"] = "success"

            self.logger.info(
                f"Analysis completed successfully. Report saved to: {report_path}"
            )
            return results

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise

    def _load_model(self, model_path: str):
        """モデル読み込み"""
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # ここでは仮の実装 - 実際のモデル読み込みロジックを実装
        self.logger.info(f"Loading model from: {model_file}")
        return {"model_path": str(model_file), "loaded": True}

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Unified Analysis Tool")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--model", required=True, help="Model file path")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    try:
        # ツール初期化
        tool = UnifiedAnalysisTool(args.config)

        # 分析実行
        results = tool.run_analysis(args.model, args.output_dir)

        # 結果出力
        print(json.dumps(results, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
