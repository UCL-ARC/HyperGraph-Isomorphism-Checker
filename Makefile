.PHONY: help install test lint format pre-commit clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync --dev

test: ## Run tests with coverage
	uv run pytest --cov=proof_checker --cov-report=term-missing

test-ci: ## Run tests with XML coverage (for CI)
	uv run pytest --cov=proof_checker --cov-report=xml --cov-report=term-missing

lint: ## Run linting (ruff and mypy)
	uv run ruff check .
	uv run mypy .

format: ## Format code with black and ruff
	uv run black .
	uv run ruff check --fix .

pre-commit: ## Run all pre-commit hooks
	uv run pre-commit run --all-files

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push

clean: ## Clean up cache files
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
