# Zaif Trade Bot - Developer Makefile
#
# Common development tasks and shortcuts
#
# Usage:
#   make help          - Show this help message
#   make test          - Run all tests
#   make unit          - Run unit tests only
#   make integration   - Run integration tests only
#   make smoke         - Run smoke tests
#   make lint          - Run linting
#   make typecheck     - Run type checking
#   make format        - Format code with black and isort
#   make docs          - Generate documentation
#   make clean         - Clean up temporary files
#   make setup         - Set up development environment

.PHONY: help test unit integration smoke lint typecheck format docs clean setup

# Default target
help:
	@echo "Zaif Trade Bot - Developer Makefile"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

# Testing targets
test: unit integration  ## Run all tests (unit + integration)

unit:  ## Run unit tests
	npm run test:unit

integration:  ## Run integration tests
	npm run test:int-fast

smoke:  ## Run smoke tests
	export PYTHONPATH="${PYTHONPATH}:$(pwd)" && \
	export ZTB_MEM_PROFILE=1 && \
	python ztb/experiments/smoke_test.py --steps 1000 --dataset synthetic

# Code quality targets
lint:  ## Run linting (flake8)
	python -m flake8 ztb/ --max-line-length=120 --extend-ignore=E203,W503

typecheck:  ## Run type checking (mypy)
	mypy ztb/ --config-file mypy.ini

format:  ## Format code with black and isort
	black ztb/ tests/ scripts/
	isort --profile black ztb/ tests/ scripts/

audit:  ## Run security audit (pip-audit)
	pip-audit

# Documentation
docs:  ## Generate documentation (placeholder)
	@echo "Documentation generation not yet implemented"
	@echo "See docs/ directory for current documentation"

# Development setup
setup:  ## Set up development environment
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install types-requests types-psutil
	pre-commit install
	npm install

# Cleanup
clean:  ## Clean up temporary files and caches
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "node_modules" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete

# Development workflow shortcuts
check: lint typecheck audit test  ## Run full code quality check (lint + typecheck + audit + test)

ci: check smoke  ## Run CI-equivalent checks locally

# Python environment info
info:  ## Show Python environment information
	@echo "Python version: $$(python --version)"
	@echo "Pip version: $$(pip --version)"
	@echo "Node version: $$(node --version)"
	@echo "NPM version: $$(npm --version)"

# Quick development cycle
dev: format lint typecheck audit unit  ## Quick development cycle (format + lint + typecheck + audit + unit tests)

# 1M Training orchestration targets
1m-start:  ## Start 1M training session (requires CORR variable)
	@if [ -z "$(CORR)" ]; then \
		echo "Error: CORR variable must be set (e.g., CORR=$$(date -u +%Y%m%dT%H%M%SZ))"; \
		exit 1; \
	fi
	python -m ztb.training.supervise_1m --correlation-id $(CORR)

1m-watch:  ## Watch running 1M training session (requires CORR variable)
	@if [ -z "$(CORR)" ]; then \
		echo "Error: CORR variable must be set"; \
		exit 1; \
	fi
	ZTB_WATCH_STALL_MIN=10 ZTB_WATCH_RSS_MB=2048 ZTB_WATCH_VRAM_MB=4096 \
	python -m ztb.training.watch_1m --correlation-id $(CORR)

1m-rollup:  ## Generate rollup artifacts for 1M training session (requires CORR variable)
	@if [ -z "$(CORR)" ]; then \
		echo "Error: CORR variable must be set"; \
		exit 1; \
	fi
	python -m ztb.training.rollup_artifacts --correlation-id $(CORR) --interval-minutes 5

1m-stop:  ## Stop 1M training session
	touch ztb.stop
	@echo "Stop signal sent. Training will stop gracefully."