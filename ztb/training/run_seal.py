"""
Run seal for training reproducibility.

Ensures deterministic training runs with proper seed management and environment tracking.
"""

import hashlib
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class EnvironmentSnapshot:
    """Snapshot of the runtime environment."""

    python_version: str
    platform: str
    hostname: str
    user: str
    working_directory: str
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    requirements_hash: Optional[str] = None
    config_hash: Optional[str] = None


@dataclass
class RunSeal:
    """Run seal for reproducibility."""

    run_id: str
    seed: int
    timestamp: str
    environment: EnvironmentSnapshot
    config: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "environment": asdict(self.environment),
            "config": self.config,
            "metadata": self.metadata,
        }


class RunSealManager:
    """Manages run seals for training reproducibility."""

    def __init__(self, seal_dir: str = "run_seals"):
        self.seal_dir = Path(seal_dir)
        self.seal_dir.mkdir(exist_ok=True)

    def create_seal(
        self,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RunSeal:
        """Create a new run seal."""
        run_id = self._generate_run_id()
        if seed is None:
            seed = self._generate_seed()

        timestamp = datetime.utcnow().isoformat()
        environment = self._capture_environment()

        if config is None:
            config = {}
        if metadata is None:
            metadata = {}

        seal = RunSeal(
            run_id=run_id,
            seed=seed,
            timestamp=timestamp,
            environment=environment,
            config=config,
            metadata=metadata,
        )

        # Save seal
        self._save_seal(seal)

        return seal

    def load_seal(self, run_id: str) -> Optional[RunSeal]:
        """Load a run seal by ID."""
        seal_path = self.seal_dir / f"{run_id}.json"
        if not seal_path.exists():
            return None

        import json

        with open(seal_path, "r") as f:
            data = json.load(f)

        environment = EnvironmentSnapshot(**data["environment"])
        return RunSeal(
            run_id=data["run_id"],
            seed=data["seed"],
            timestamp=data["timestamp"],
            environment=environment,
            config=data["config"],
            metadata=data["metadata"],
        )

    def list_seals(self) -> list[str]:
        """List all available run seals."""
        return [f.stem for f in self.seal_dir.glob("*.json")]

    def validate_environment(self, seal: RunSeal) -> Dict[str, bool]:
        """Validate current environment against seal."""
        current = self._capture_environment()

        return {
            "python_version": current.python_version == seal.environment.python_version,
            "platform": current.platform == seal.environment.platform,
            "git_commit": current.git_commit == seal.environment.git_commit,
            "requirements_hash": current.requirements_hash
            == seal.environment.requirements_hash,
            "config_hash": current.config_hash == seal.environment.config_hash,
        }

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        random_part = hashlib.md5(str(id(self)).encode()).hexdigest()[:8]
        return f"run_{timestamp}_{random_part}"

    def _generate_seed(self) -> int:
        """Generate random seed."""
        import random

        return random.randint(0, 2**31 - 1)

    def _capture_environment(self) -> EnvironmentSnapshot:
        """Capture current environment snapshot."""
        env = EnvironmentSnapshot(
            python_version=platform.python_version(),
            platform=platform.platform(),
            hostname=platform.node(),
            user=os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
            working_directory=str(Path.cwd()),
        )

        # Git info
        try:
            env.git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
            env.git_branch = subprocess.check_output(
                ["git", "branch", "--show-current"], text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Requirements hash
        try:
            req_file = Path("requirements.txt")
            if req_file.exists():
                with open(req_file, "rb") as f:
                    env.requirements_hash = hashlib.md5(f.read()).hexdigest()
        except Exception:
            pass

        # Config hash (if trade-config.json exists)
        try:
            config_file = Path("trade-config.json")
            if config_file.exists():
                with open(config_file, "rb") as f:
                    env.config_hash = hashlib.md5(f.read()).hexdigest()
        except Exception:
            pass

        return env

    def _save_seal(self, seal: RunSeal):
        """Save run seal to file."""
        import json

        seal_path = self.seal_dir / f"{seal.run_id}.json"
        with open(seal_path, "w") as f:
            json.dump(seal.to_dict(), f, indent=2, ensure_ascii=False)


# Global instance
_run_seal_manager = RunSealManager()


def get_run_seal_manager() -> RunSealManager:
    """Get global run seal manager."""
    return _run_seal_manager


def create_run_seal(
    seed: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> RunSeal:
    """Convenience function to create a run seal."""
    return _run_seal_manager.create_seal(seed, config, metadata)
