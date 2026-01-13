# AIG-Kernel-Eval Makefile
# ========================
# Shortcuts for environment setup and common operations.

.PHONY: help install install-dev clean test lint format run-analyze run-compare docker-build docker-run

# Default target
help:
	@echo "AIG-Kernel-Eval - GPU Kernel Evaluation Framework"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make venv          - Create virtual environment"
	@echo ""
	@echo "Development Commands:"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linter"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean build artifacts"
	@echo ""
	@echo "Evaluation Commands:"
	@echo "  make run-analyze KERNEL=<path>  - Analyze a kernel"
	@echo "  make run-compare K1=<p1> K2=<p2> - Compare two kernels"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run evaluation in Docker"

# Python and virtualenv settings
PYTHON ?= python3
VENV_DIR ?= .venv
PIP := $(VENV_DIR)/bin/pip
PYTHON_VENV := $(VENV_DIR)/bin/python

# Create virtual environment
venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	@echo "Virtual environment created. Activate with: source $(VENV_DIR)/bin/activate"

# Install production dependencies
install: venv
	$(PIP) install -r requirements.txt

# Install development dependencies
install-dev: install
	$(PIP) install -r requirements-dev.txt 2>/dev/null || \
		$(PIP) install pytest pytest-cov black isort mypy ruff

# Run linter
lint:
	$(PYTHON_VENV) -m ruff check src/ main.py
	$(PYTHON_VENV) -m mypy src/ main.py --ignore-missing-imports

# Format code
format:
	$(PYTHON_VENV) -m black src/ main.py
	$(PYTHON_VENV) -m isort src/ main.py

# Clean build artifacts
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf src/__pycache__ src/**/__pycache__
	rm -rf *.egg-info build dist
	rm -rf results/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Run analyze mode
run-analyze:
ifndef KERNEL
	@echo "Error: KERNEL is not set. Usage: make run-analyze KERNEL=path/to/kernel.py"
	@exit 1
endif
	$(PYTHON_VENV) main.py analyze $(KERNEL)

# Run compare mode
run-compare:
ifndef K1
	@echo "Error: K1 is not set. Usage: make run-compare K1=kernel1.py K2=kernel2.py"
	@exit 1
endif
ifndef K2
	@echo "Error: K2 is not set. Usage: make run-compare K1=kernel1.py K2=kernel2.py"
	@exit 1
endif
	$(PYTHON_VENV) main.py compare $(K1) $(K2)

# Docker configuration
DOCKER_IMAGE ?= aig-kernel-eval
DOCKER_TAG ?= latest

# Build Docker image
docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

# Run in Docker
docker-run:
	docker run --gpus all -it --rm \
		-v $(PWD):/workspace \
		$(DOCKER_IMAGE):$(DOCKER_TAG) \
		$(ARGS)

# Verify installation
verify:
	$(PYTHON_VENV) -c "from src.core import Executor, Scheduler; print('Core modules OK')"
	$(PYTHON_VENV) -c "from src.modes import AnalyzeEvaluator, CompareEvaluator; print('Mode modules OK')"
	@echo "Installation verified successfully!"
