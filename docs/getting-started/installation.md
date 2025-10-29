# Installation

## Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- Graphviz (for diagram rendering)

## Install with uv (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/UCL-ARC/HyperGraph-Isomorphism-Checker
cd HyperGraph-Isomorphism-Checker
```

2. Install system dependencies:
```bash
make install-system
```

3. Install Python dependencies:
```bash
make install
```

## Install with pip

If you prefer using pip and virtualenv:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
pip install -e ".[dev]"
```

## Install Graphviz

### Ubuntu/Debian
```bash
sudo apt-get install graphviz
```

### macOS
```bash
brew install graphviz
```

### Windows
Download and install from [graphviz.org](https://graphviz.org/download/)

## Verify Installation

```python
from IsomorphismChecker_python_serial.node import Node

# Create a simple node
node = Node(index=0, label="test")
print(f"Node created: {node.label}")
```

## Development Setup

For development with pre-commit hooks:

```bash
make setup
```

This will:
- Install all dependencies
- Set up pre-commit hooks
- Configure the development environment

## Running Tests

```bash
make test
```

Or with coverage:

```bash
pytest --cov=IsomorphismChecker_python_serial
```
