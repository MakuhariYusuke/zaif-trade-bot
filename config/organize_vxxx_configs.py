#!/usr/bin/env python3
"""
Config Directory Organizer for vXXX Series

This script organizes vXXX series configuration files in the config directory.
It creates version-specific subdirectories and moves files accordingly.
"""

import re
import shutil
from pathlib import Path
from typing import Dict, List, Set


class ConfigOrganizer:
    """Organizes vXXX series configuration files."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.version_pattern = re.compile(r"v(\d+)(?:\.(\d+))?(?:_.*)?\.")
        self.sac_version_pattern = re.compile(r"sac_v(\d+)(?:\.(\d+))?(?:_.*)?\.")
        self.ppo_version_pattern = re.compile(r"ppo_v(\d+)(?:_.*)?\.")
        self.organized_files: Dict[str, List[str]] = {}
        self.errors: List[str] = []

    def extract_version(self, filename: str) -> str:
        """Extract version from filename."""
        # Check SAC pattern first
        sac_match = self.sac_version_pattern.search(filename)
        if sac_match:
            major = sac_match.group(1)
            minor = sac_match.group(2) if sac_match.group(2) else "0"
            # Normalize version: remove .0 and integrate sub-versions
            if minor == "0":
                return f"v{major}"
            else:
                return f"v{major}"

        # Check PPO pattern
        ppo_match = self.ppo_version_pattern.search(filename)
        if ppo_match:
            major = ppo_match.group(1)
            return f"v{major}"

        # Check general vXXX pattern
        v_match = self.version_pattern.search(filename)
        if v_match:
            major = v_match.group(1)
            minor = v_match.group(2) if v_match.group(2) else "0"
            # Normalize version: remove .0 and integrate sub-versions
            if minor == "0":
                return f"v{major}"
            else:
                return f"v{major}"

        return ""

    def scan_files(self) -> Dict[str, List[str]]:
        """Scan config directory for vXXX series files."""
        version_files: Dict[str, List[str]] = {}

        for file_path in self.config_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix == ".json":
                filename = file_path.name
                version = self.extract_version(filename)

                if version:
                    if version not in version_files:
                        version_files[version] = []
                    version_files[version].append(
                        str(file_path.relative_to(self.config_dir))
                    )

        return version_files

    def create_version_directories(self, versions: Set[str]) -> None:
        """Create version-specific directories."""
        for version in sorted(versions):
            version_dir = self.config_dir / version
            if not version_dir.exists():
                version_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {version_dir}")

    def move_files_to_version_dirs(self, version_files: Dict[str, List[str]]) -> None:
        """Move files to their respective version directories."""
        for version, files in version_files.items():
            version_dir = self.config_dir / version

            for file_path_str in files:
                src_path = self.config_dir / file_path_str
                dst_path = version_dir / src_path.name

                try:
                    if src_path != dst_path:
                        shutil.move(str(src_path), str(dst_path))
                        print(f"Moved: {file_path_str} -> {version}/{src_path.name}")
                except Exception as e:
                    error_msg = f"Failed to move {file_path_str}: {str(e)}"
                    self.errors.append(error_msg)
                    print(f"Error: {error_msg}")

    def consolidate_version_dirs(self) -> None:
        """Consolidate version directories by removing .0 and integrating sub-versions."""
        version_dirs: Dict[str, List[Path]] = {}
        dirs_to_remove = []

        # First pass: collect all version directories
        for dir_path in self.config_dir.iterdir():
            if dir_path.is_dir() and dir_path.name.startswith("v"):
                version = dir_path.name
                # Extract major version (remove .0 and sub-versions)
                major_match = re.match(r"v(\d+)(?:\.\d+)?", version)
                if major_match:
                    major_version = f"v{major_match.group(1)}"
                    if major_version not in version_dirs:
                        version_dirs[major_version] = []
                    version_dirs[major_version].append(dir_path)

        # Second pass: consolidate directories
        for major_version, dirs in version_dirs.items():
            if len(dirs) > 1:
                # Multiple directories for same major version - consolidate
                target_dir = self.config_dir / major_version
                if not target_dir.exists():
                    target_dir.mkdir()

                for src_dir in dirs:
                    if src_dir != target_dir:
                        # Move all files from src_dir to target_dir
                        for file_path in src_dir.glob("*"):
                            if file_path.is_file():
                                dst_path = target_dir / file_path.name
                                try:
                                    shutil.move(str(file_path), str(dst_path))
                                    print(
                                        f"Consolidated: {file_path.relative_to(self.config_dir)} -> {major_version}/{file_path.name}"
                                    )
                                except Exception as e:
                                    self.errors.append(
                                        f"Failed to consolidate {file_path}: {str(e)}"
                                    )

                        # Mark for removal
                        dirs_to_remove.append(src_dir)

        # Remove empty consolidated directories
        for dir_path in dirs_to_remove:
            try:
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    print(f"Removed consolidated directory: {dir_path.name}")
            except Exception as e:
                print(f"Failed to remove directory {dir_path}: {str(e)}")

    def generate_summary_report(self, version_files: Dict[str, List[str]]) -> str:
        """Generate a summary report of the organization."""
        report = []
        report.append("# Config Directory Organization Report")
        report.append("")

        total_files = sum(len(files) for files in version_files.values())
        report.append(f"Total vXXX series files organized: {total_files}")
        report.append("")

        for version in sorted(version_files.keys()):
            files = version_files[version]
            report.append(f"## {version}")
            report.append(f"- Files: {len(files)}")
            for file in sorted(files):
                report.append(f"  - {file}")
            report.append("")

        if self.errors:
            report.append("## Errors")
            for error in self.errors:
                report.append(f"- {error}")
            report.append("")

        return "\n".join(report)

    def organize(self, dry_run: bool = False) -> str:
        """Main organization method."""
        print("Scanning config directory for vXXX series files...")

        version_files = self.scan_files()

        if not version_files:
            return "No vXXX series files found to organize."

        print(
            f"Found {sum(len(files) for files in version_files.values())} files in {len(version_files)} versions"
        )

        if dry_run:
            print("\nDRY RUN - Would organize the following:")
            for version, files in sorted(version_files.items()):
                print(f"\n{version}:")
                for file in files:
                    print(f"  {file}")
            return "Dry run completed."

        # Create version directories
        versions = set(version_files.keys())
        self.create_version_directories(versions)

        # Move files
        self.move_files_to_version_dirs(version_files)

        # Consolidate version directories (remove .0 and integrate sub-versions)
        self.consolidate_version_dirs()

        # Generate report
        report = self.generate_summary_report(version_files)

        return report


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Organize vXXX series config files")
    parser.add_argument("--config-dir", default="config", help="Config directory path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    organizer = ConfigOrganizer(args.config_dir)

    try:
        result = organizer.organize(dry_run=args.dry_run)
        print(result)

        if organizer.errors:
            print(f"\nCompleted with {len(organizer.errors)} errors.")
            return 1
        else:
            print("\nOrganization completed successfully!")
            return 0

    except Exception as e:
        print(f"Error during organization: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
