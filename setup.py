from runpy import run_path
from typing import cast
from setuptools import setup, find_packages

# Read version from package
def get_version() -> str:
    namespace = run_path("ztb/__version__.py")
    return cast(str, namespace["__version__"])

setup(
    name="zaif-trade-bot",
    version=get_version(),
    packages=find_packages(),
    install_requires=[
        "rich>=13.0.0,<14.0",
        "numpy>=1.24.4,<2.0",
        "scikit-learn>=1.3.2,<2.0",
        "requests>=2.31.0,<3.0",
        "APScheduler>=3.10.0,<4.0",
        "prometheus_client>=0.19.0,<1.0",
        "PyYAML>=6.0.1,<7.0",
        "pandas>=2.0.0,<3.0",
        "matplotlib>=3.7.0,<4.0",
        "seaborn>=0.12.0,<1.0",
        "jupyterlab>=4.0.0,<5.0",
        "optuna>=3.0.0,<4.0",
        "gymnasium>=0.28.1,<1.0",
        "stable-baselines3[extra]>=1.8.0,<2.0",
        "pyarrow>=12.0.0,<17.0",
        "psutil>=5.9.0,<6.0",
    ],
)
