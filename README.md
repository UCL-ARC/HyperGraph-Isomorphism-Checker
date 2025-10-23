
# HyperGraph Isomorphism Checker

[![CI](https://github.com/UCL-ARC/HyperGraph-Isomorphism-Checker/actions/workflows/ci.yml/badge.svg)](https://github.com/UCL-ARC/HyperGraph-Isomorphism-Checker/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/UCL-ARC/HyperGraph-Isomorphism-Checker/branch/main/graph/badge.svg)](https://codecov.io/gh/UCL-ARC/HyperGraph-Isomorphism-Checker)

The project description
The plan of the milestones agreed with ARIA. See attached documents

1. https://doi.org/10.48550/arXiv.2305.08768
2. https://arxiv.org/abs/2305.01041

## Development Setup

### Quick Start

**Option 1: With uv (recommended)**
```bash
# Install dependencies (will prompt to install uv if needed)
make install

# Install pre-commit hooks
make pre-commit-install
```

**Option 2: Traditional Python setup**
```bash
# Install dependencies with pip
make install-pip

# Activate virtual environment
source .venv/bin/activate

# Install pre-commit hooks
python -m pre_commit install
python -m pre_commit install --hook-type pre-push
```

### Development Commands
```bash
# Install dependencies (will check for uv first)
make install

# Install pre-commit hooks (runs automatically before commits)
make pre-commit-install

# Run tests
make test

# Run linting and formatting
make format
make lint

# Run all pre-commit checks
make pre-commit
```

### Available Commands
Run `make help` to see all available commands.

### CI/CD
This project uses GitHub Actions for continuous integration:
- **Pre-commit hooks**: Run automatically before each commit (formatting, linting)
- **Pre-push hooks**: Run tests before pushing
- **GitHub Actions**: Run full test suite and pre-commit checks on PRs and pushes

**Links to the repositories that might be relevant for the project:**
https://github.com/statusfailed/open-hypergraphs
https://github.com/statusfailed/catgrad

https://github.com/hellas-ai/open-hypergraphs
https://github.com/yarrow-id/diagrams
https://pypi.org/project/yarrow-polycirc/
https://github.com/hellas-ai/catgrad
