# Magpie Makefile
# ========================
# Shortcuts for environment setup and common operations.

.PHONY: help install install-dev clean test lint format run-analyze run-compare docker-build docker-run

# Default target
help:
	@echo "Magpie - GPU Kernel Evaluation Framework"
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
	$(PYTHON_VENV) -m ruff check Magpie/
	$(PYTHON_VENV) -m mypy Magpie/ --ignore-missing-imports

# Format code
format:
	$(PYTHON_VENV) -m black Magpie/
	$(PYTHON_VENV) -m isort Magpie/

# Clean build artifacts
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf Magpie/__pycache__ Magpie/**/__pycache__
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
	$(PYTHON_VENV) -m Magpie analyze $(KERNEL)

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
	$(PYTHON_VENV) -m Magpie compare $(K1) $(K2)

# Docker configuration
DOCKER_IMAGE ?= magpie
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
	$(PYTHON_VENV) -c "from Magpie.core import Executor, Scheduler; print('Core modules OK')"
	$(PYTHON_VENV) -c "from Magpie.modes import AnalyzeMode, CompareMode; print('Mode modules OK')"
	@echo "Installation verified successfully!"
