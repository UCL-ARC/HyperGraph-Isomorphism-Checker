# Contributing

We welcome contributions to the Data Parallel Proof Checker project!

## Getting Started

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/yourusername/Data-Parallel-Proof-Checker-1368.git
cd Data-Parallel-Proof-Checker-1368
```

3. Set up development environment:
```bash
make setup
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, well-documented code
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write docstrings for all public functions and classes

### 3. Add Tests

All new features should include tests:

```python
# tests/test_your_feature.py
import pytest
from IsomorphismChecker_python_serial.your_module import YourClass

def test_your_feature():
    obj = YourClass(param=value)
    assert obj.method() == expected_result
```

### 4. Run Tests

```bash
make test
```

### 5. Check Code Quality

```bash
make lint
make format
```

### 6. Commit Changes

Pre-commit hooks will automatically run:

```bash
git add .
git commit -m "feat: add your feature description"
```

We follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions or changes
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

### 7. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style

### Python Style

- Follow PEP 8
- Use Black for formatting
- Use Ruff for linting
- Add type hints (we use mypy)

### Documentation Style

- Use Google-style docstrings
- Include examples in docstrings when helpful
- Keep line length to 88 characters

Example:

```python
def my_function(param1: str, param2: int) -> bool:
    """Short description of function.

    Longer description if needed, explaining the purpose
    and behavior of the function.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative

    Example:
        >>> my_function("test", 42)
        True
    """
    if param2 < 0:
        raise ValueError("param2 must be non-negative")
    return True
```

## Testing Guidelines

### Test Structure

```python
class TestYourFeature:
    """Test suite for YourFeature."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for tests."""
        return {...}

    def test_basic_functionality(self, sample_data):
        """Test basic functionality."""
        # Arrange
        obj = YourClass(sample_data)

        # Act
        result = obj.method()

        # Assert
        assert result == expected

    def test_edge_case(self):
        """Test edge case behavior."""
        with pytest.raises(ValueError):
            YourClass(invalid_data)
```

### Coverage

- Aim for >90% test coverage
- Test edge cases and error conditions
- Include integration tests where appropriate

## Documentation

### Updating Docs

Documentation is in the `docs/` directory. To preview locally:

```bash
mkdocs serve
```

Then visit http://127.0.0.1:8000

### Building Docs

```bash
mkdocs build
```

## Questions?

- Open an issue for bug reports
- Start a discussion for feature requests
- Check existing issues before creating new ones

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help maintain a positive community

Thank you for contributing! 🎉
