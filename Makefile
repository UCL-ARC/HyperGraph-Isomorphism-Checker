# =============================================================================
# Configuration
# =============================================================================
PROJECT_NAME := IsomorphismChecker_python_serial
VENV_DIR := .venv
UV_INSTALL_URL := https://astral.sh/uv/install.sh

# Tool commands
UV_RUN := uv run
PIP_INSTALL := $(VENV_DIR)/bin/pip install

# Python dependencies
DEV_DEPS := pytest pytest-cov pre-commit black ruff mypy

# =============================================================================
# PHONY targets
# =============================================================================
.PHONY: help setup install test lint format pre-commit clean
.PHONY: check-uv install-uv install-pip test-ci pre-commit-install
.PHONY: docs docs-serve docs-build docs-deploy

# =============================================================================
# Help and Setup
# =============================================================================
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: install-system install pre-commit-install ## Complete project setup (system deps + install + pre-commit)

# =============================================================================
# Environment Management
# =============================================================================
check-uv: ## Check if uv is installed, prompt to install if missing
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "❌ uv is not installed."; \
		echo ""; \
		echo "uv is a fast Python package manager that improves development experience."; \
		echo "Would you like to install it now? [y/N]"; \
		read -r response; \
		if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
			$(MAKE) install-uv; \
		else \
			echo ""; \
			echo "To install uv later, run: make install-uv"; \
			echo "Or install manually: curl -LsSf $(UV_INSTALL_URL) | sh"; \
			echo ""; \
			echo "Alternative: Use traditional Python tools with 'make install-pip'"; \
		fi; \
		exit 1; \
	else \
		echo "✅ uv is already installed"; \
	fi

install-uv: ## Install uv package manager
	@echo "📦 Installing uv..."
	@curl -LsSf $(UV_INSTALL_URL) | sh
	@echo "✅ uv installed! Please restart your terminal or run: source ~/.cargo/env"

# =============================================================================
# Dependency Installation
# =============================================================================
install-system: ## Install system dependencies (graphviz)
	@echo "📦 Installing system dependencies..."
	@if command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get update && sudo apt-get install -y graphviz; \
	elif command -v yum >/dev/null 2>&1; then \
		sudo yum install -y graphviz; \
	elif command -v dnf >/dev/null 2>&1; then \
		sudo dnf install -y graphviz; \
	elif command -v brew >/dev/null 2>&1; then \
		brew install graphviz; \
	elif command -v pacman >/dev/null 2>&1; then \
		sudo pacman -S graphviz; \
	else \
		echo "❌ Package manager not detected. Please install graphviz manually."; \
		echo "   Ubuntu/Debian: sudo apt-get install graphviz"; \
		echo "   RHEL/CentOS: sudo yum install graphviz"; \
		echo "   Fedora: sudo dnf install graphviz"; \
		echo "   macOS: brew install graphviz"; \
		echo "   Arch: sudo pacman -S graphviz"; \
		exit 1; \
	fi
	@echo "✅ System dependencies installed!"

install: check-uv ## Install dependencies with uv
	@echo "📦 Installing dependencies with uv..."
	@uv sync --dev
	@echo "✅ Dependencies installed!"

install-pip: ## Install dependencies using traditional pip (alternative to uv)
	@echo "📦 Installing with pip..."
	@$(MAKE) _create-venv
	@$(MAKE) _install-deps-pip
	@echo "✅ Dependencies installed!"
	@echo "💡 Activate virtual environment with: source $(VENV_DIR)/bin/activate"

_create-venv: ## Internal: Create virtual environment if it doesn't exist
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		python -m venv $(VENV_DIR); \
	fi

_install-deps-pip: ## Internal: Install dependencies with pip
	@echo "Installing dependencies..."
	@$(PIP_INSTALL) -e .
	@$(PIP_INSTALL) $(DEV_DEPS)

# =============================================================================
# Testing
# =============================================================================
test: ## Run tests with coverage
	@echo "🧪 Running tests..."
	@$(UV_RUN) pytest --cov=$(PROJECT_NAME) --cov-report=term-missing

test-ci: ## Run tests with XML coverage (for CI)
	@echo "🧪 Running tests for CI..."
	@$(UV_RUN) pytest --cov=$(PROJECT_NAME) --cov-report=xml --cov-report=term-missing

# =============================================================================
# Code Quality
# =============================================================================
lint: ## Run linting (ruff and mypy)
	@echo "🔍 Running linters..."
	@$(UV_RUN) ruff check .
	@$(UV_RUN) mypy .

format: ## Format code with black and ruff
	@echo "🎨 Formatting code..."
	@$(UV_RUN) black .
	@$(UV_RUN) ruff check --fix .

# =============================================================================
# Pre-commit
# =============================================================================
pre-commit: ## Run all pre-commit hooks
	@echo "🔧 Running pre-commit hooks..."
	@$(UV_RUN) pre-commit run --all-files

pre-commit-install: ## Install pre-commit hooks
	@echo "🔧 Installing pre-commit hooks..."
	@$(UV_RUN) pre-commit install
	@$(UV_RUN) pre-commit install --hook-type pre-push
	@echo "✅ Pre-commit hooks installed!"

# =============================================================================
# Documentation
# =============================================================================
docs-serve: ## 📚 Serve documentation locally with live reload
	@echo "📚 Starting documentation server..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run --group docs mkdocs serve; \
	else \
		echo "⚠️  uv not found. Using system mkdocs..."; \
		mkdocs serve; \
	fi

docs-build: ## 📚 Build documentation
	@echo "📚 Building documentation..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run --group docs mkdocs build; \
	else \
		echo "⚠️  uv not found. Using system mkdocs..."; \
		mkdocs build; \
	fi

docs-deploy: ## 📚 Deploy documentation to GitHub Pages
	@echo "📚 Deploying documentation to GitHub Pages..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run --group docs mkdocs gh-deploy; \
	else \
		echo "⚠️  uv not found. Using system mkdocs..."; \
		mkdocs gh-deploy; \
	fi

docs: docs-serve ## 📚 Alias for docs-serve

# =============================================================================
# Cleanup
# =============================================================================
clean: ## Clean up cache files
	@echo "🧹 Cleaning up cache files..."
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf .ruff_cache/
	@rm -rf __pycache__/
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cache files cleaned!"
