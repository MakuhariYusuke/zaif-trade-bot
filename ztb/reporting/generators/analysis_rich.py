"""
レポート生成器

分析結果から各種レポートを生成します。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

try:
    import markdown

    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

try:
    import jinja2

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


class ReportGenerator:
    """レポート生成器"""

    def __init__(self) -> None:
        """初期化"""
        self.logger = logging.getLogger(__name__)

        # テンプレートディレクトリ
        self.template_dir = Path(__file__).resolve().parents[1] / "templates"
        self.template_dir.mkdir(exist_ok=True)

        # Jinja2環境設定（利用可能な場合）
        if HAS_JINJA2:
            self.jinja_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(self.template_dir)),
                autoescape=jinja2.select_autoescape(["html", "xml"]),
            )
        else:
            self.jinja_env = None

    def generate_report(
        self, results: Dict[str, Any], output_dir: Path, format: str = "json", **kwargs
    ) -> Path:
        """
        レポート生成

        Args:
            results: 分析結果
            output_dir: 出力ディレクトリ
            format: レポートフォーマット ('json', 'html', 'markdown')
            **kwargs: 追加パラメータ

        Returns:
            レポートファイルパス
        """
        self.logger.info(f"Generating {format} report...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "json":
            return self._generate_json_report(results, output_dir, timestamp)
        if format == "html":
            return self._generate_html_report(results, output_dir, timestamp, **kwargs)
        if format == "markdown":
            return self._generate_markdown_report(results, output_dir, timestamp)
        raise ValueError(f"Unsupported format: {format}")

    def _generate_json_report(
        self, results: Dict[str, Any], output_dir: Path, timestamp: str
    ) -> Path:
        """JSONレポート生成"""
        filename = f"analysis_report_{timestamp}.json"
        filepath = output_dir / filename

        report_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "report_type": "unified_analysis",
                "version": "1.0",
            },
            "results": results,
        }

        write_json(filepath, report_data, indent=2, ensure_ascii=False)

        self.logger.info(f"JSON report saved to: {filepath}")
        return filepath

    def _generate_html_report(
        self, results: Dict[str, Any], output_dir: Path, timestamp: str, **kwargs
    ) -> Path:
        """HTMLレポート生成"""
        filename = f"analysis_report_{timestamp}.html"
        filepath = output_dir / filename

        template_path = self.template_dir / "analysis_report.html"
        if HAS_JINJA2 and self.jinja_env and template_path.exists():
            template = self.jinja_env.get_template("analysis_report.html")
            html_content = template.render(
                results=results, timestamp=datetime.now(), **kwargs
            )
        else:
            html_content = self._generate_simple_html(results, timestamp)

        write_text(filepath, html_content)

        self.logger.info(f"HTML report saved to: {filepath}")
        return filepath

    def _generate_markdown_report(
        self, results: Dict[str, Any], output_dir: Path, timestamp: str
    ) -> Path:
        """Markdownレポート生成"""
        filename = f"analysis_report_{timestamp}.md"
        filepath = output_dir / filename

        md_content = self._generate_markdown_content(results, timestamp)

        write_text(filepath, md_content)

        self.logger.info(f"Markdown report saved to: {filepath}")
        return filepath

    def _generate_simple_html(self, results: Dict[str, Any], timestamp: str) -> str:
        """簡易HTMLレポート生成"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>統合分析レポート - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .section {{ margin-bottom: 30px; }}
        .metric {{ background: #f5f5f5; padding: 10px; margin: 5px 0; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>統合分析レポート</h1>
    <p>生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

        if "performance" in results:
            html += """
    <div class="section">
        <h2>📊 パフォーマンス分析</h2>
"""
            for key, value in results["performance"].items():
                html += f"""
        <div class="metric">
            <strong>{key}:</strong> {value}
        </div>"""
            html += "    </div>"

        if "risk" in results:
            html += """
    <div class="section">
        <h2>⚠️ リスク分析</h2>
        <table>
            <tr><th>指標</th><th>値</th></tr>"""
            for key, value in results["risk"].items():
                html += f"""
            <tr><td>{key}</td><td>{value}</td></tr>"""
            html += """
        </table>
    </div>"""

        if "behavioral" in results:
            html += """
    <div class="section">
        <h2>🎯 行動分析</h2>"""
            for key, value in results["behavioral"].items():
                if isinstance(value, dict):
                    html += f"""
        <h3>{key}</h3>
        <ul>"""
                    for sub_key, sub_value in value.items():
                        html += f"""
            <li><strong>{sub_key}:</strong> {sub_value}</li>"""
                    html += """
        </ul>"""
        html += """
</body>
</html>
"""

        return html

    def _generate_markdown_content(self, results: Dict[str, Any], timestamp: str) -> str:
        """Markdownレポート生成（簡易）"""
        md = f"# 統合分析レポート ({timestamp})\n\n"
        for section, values in results.items():
            md += f"## {section}\n"
            if isinstance(values, dict):
                for key, value in values.items():
                    md += f"- **{key}**: {value}\n"
            else:
                md += f"- {values}\n"
            md += "\n"
        return md
from ztb.io.json_io import write_json
from ztb.io.text_io import write_text
