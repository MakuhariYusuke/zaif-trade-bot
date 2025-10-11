from setuptools import setup, find_packages

# Read version from package
def get_version():
    with open("ztb/__version__.py", "r") as f:
        exec(f.read())
        return locals()["__version__"]

setup(
    name="zaif-trade-bot",
    version=get_version(),
    packages=find_packages(),
    install_requires=[
        "rich>=13.0.0",
        "numpy==2.3.3",
        "scikit-learn==1.7.2",
        "requests>=2.31.0",
        "APScheduler>=3.10.0",
        "prometheus_client>=0.19.0",
        "PyYAML>=6.0.0",
        "pandas>=2.0.0",
        # "pyarrow>=14.0.0",  # Python 3.14 not yet supported
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "jupyterlab>=4.0.0",
        "optuna>=3.0.0",
        "gymnasium>=0.29.0",
        "psutil>=5.9.0",
    ],
)