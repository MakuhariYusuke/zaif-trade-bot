#!/usr/bin/env python3
"""
Monthly archival system for coverage data.
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import gzip
from typing import Dict, List
import logging

# Add ztb to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class CoverageArchiver:
    """Handles monthly archival of coverage data"""

    def __init__(self, coverage_file: str = "coverage.json", archive_dir: str = "archive"):
        self.coverage_file = Path(coverage_file)
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    def archive_monthly_coverage(self) -> None:
        """Archive current coverage data with monthly timestamp"""
        if not self.coverage_file.exists():
            logging.warning(f"Coverage file {self.coverage_file} not found, skipping archival")
            return

        # Get current date for archiving
        now = datetime.now()
        archive_filename = f"coverage_{now.strftime('%Y_%m')}.json.gz"
        archive_path = self.archive_dir / archive_filename

        # Compress and archive
        with open(self.coverage_file, 'r', encoding='utf-8') as f:
            coverage_data = json.load(f)

        with gzip.open(archive_path, 'wt', encoding='utf-8') as f:
            json.dump(coverage_data, f, indent=2)

        logging.info(f"Archived coverage data to {archive_path}")

        # Clean up old archives (keep last 12 months)
        self._cleanup_old_archives()

    def _cleanup_old_archives(self) -> None:
        """Remove archives older than 12 months"""
        cutoff_date = datetime.now() - relativedelta(months=12)
        import re
        pattern = re.compile(r'^coverage_(\d{4})_(\d{2})$')
        for archive_file in self.archive_dir.glob("coverage_*.json.gz"):
            match = pattern.match(archive_file.stem)
            if not match:
                continue
            try:
                year, month = int(match.group(1)), int(match.group(2))
                file_date = datetime(year, month, 1)
                if file_date < cutoff_date:
                    archive_file.unlink()
                    logging.info(f"Removed old archive: {archive_file}")
            except ValueError:
                continue
                continue

    def get_archived_coverage(self, year: int, month: int) -> Dict:
        """Retrieve archived coverage data for specific month"""
        archive_filename = f"coverage_{year:04d}_{month:02d}.json.gz"
        archive_path = self.archive_dir / archive_filename

        if not archive_path.exists():
            raise FileNotFoundError(f"Archive {archive_filename} not found")

        # Load and return the archived coverage data
        with gzip.open(archive_path, 'rt', encoding='utf-8') as f:
            return json.load(f)

    def list_available_archives(self) -> List[str]:
        """List all available archived coverage files sorted by date (descending)"""
        files = list(self.archive_dir.glob("coverage_*.json.gz"))
        def extract_date(f):
            try:
                parts = f.stem.split('_')[1:3]
                return datetime.strptime('_'.join(parts), '%Y_%m')
            except Exception:
                return datetime.min
        files.sort(key=extract_date, reverse=True)
        return [f.name for f in files]


def archive_current_month():
    """Archive current month's coverage data"""
    archiver = CoverageArchiver()
    archiver.archive_monthly_coverage()


if __name__ == "__main__":
    archive_current_month()