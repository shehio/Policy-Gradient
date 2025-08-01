# Testing Guide

This directory contains comprehensive tests for the Policy-Gradient project.

## Test Structure

```
tests/
├── __init__.py                    # Tests package
├── test_atari_baselines.py        # Tests for Stable Baselines3 implementation
├── test_dqn.py                    # Tests for DQN implementation
├── test_policy_gradient.py        # Tests for Policy Gradient implementation
└── README.md                      # This file
```

## Quick Start

### Run All Tests
```bash
# From project root
python run_tests.py
```

### Run Tests with pytest
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=atari --cov-report=html

# Run specific test file
pytest tests/test_atari_baselines.py -v

# Run specific test class
pytest tests/test_dqn.py::TestDuelingCNN -v

# Run specific test method
pytest tests/test_policy_gradient.py::TestMLP::test_mlp_initialization -v
```

## Test Categories

### 1. Unit Tests
- **Model Tests**: Test individual neural network models
- **Agent Tests**: Test RL agent implementations
- **Utility Tests**: Test helper functions and utilities

### 2. Integration Tests
- **Import Tests**: Verify all modules can be imported
- **Configuration Tests**: Test configuration loading
- **Script Tests**: Test command-line scripts

### 3. Code Quality Tests
- **Linting**: Flake8 for code style and syntax
- **Formatting**: Black for code formatting
- **Import Sorting**: isort for import organization

### 4. Security Tests
- **Bandit**: Security vulnerability scanning
- **Safety**: Dependency vulnerability checking

## Test Configuration

### pytest.ini
- Test discovery patterns
- Coverage settings
- Warning filters
- Custom markers

### GitHub Actions
- Automated testing on every push/PR
- Multiple Python versions (3.9, 3.10, 3.11, 3.12)
- Parallel job execution
- Coverage reporting

## Coverage Reports

### HTML Coverage
```bash
pytest --cov=atari --cov-report=html
# Open htmlcov/index.html in browser
```

### Terminal Coverage
```bash
pytest --cov=atari --cov-report=term-missing
```

### XML Coverage (for CI)
```bash
pytest --cov=atari --cov-report=xml
```

## Test Markers

### Available Markers
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests

### Using Markers
```bash
# Run only fast tests
pytest -m "not slow"

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration
```

## Debugging Tests

### Verbose Output
```bash
pytest -v -s --tb=long
```

### Debug Specific Test
```bash
# Add breakpoint() in test code
pytest tests/test_dqn.py::TestDuelingCNN::test_model_initialization -s
```

### Mock Debugging
```bash
# Show mock calls
pytest --tb=short -s
```

## Writing Tests

### Test Naming Convention
- Files: `test_*.py`
- Classes: `Test*`
- Methods: `test_*`

### Example Test Structure
```python
class TestMyClass:
    """Test the MyClass functionality."""
    
    def test_initialization(self):
        """Test that the class can be initialized."""
        obj = MyClass()
        assert obj is not None
    
    def test_method_behavior(self):
        """Test specific method behavior."""
        obj = MyClass()
        result = obj.my_method()
        assert result == expected_value
```

### Mocking Guidelines
```python
# Mock external dependencies
with patch('module.external_function') as mock_func:
    mock_func.return_value = expected_value
    result = my_function()
    assert result == expected_value
```

## Continuous Integration

### GitHub Actions Workflow
- **Triggers**: Push to main/master, Pull Requests
- **Environments**: Ubuntu with Python 3.9-3.12
- **Jobs**:
  - Unit tests with coverage
  - Code quality checks
  - Security scanning
  - Import validation

### Local CI Simulation
```bash
# Run the same checks as CI
python run_tests.py
```

## Coverage Goals

### Current Coverage
- **Atari Baselines**: ~80%
- **DQN Implementation**: ~75%
- **Policy Gradient**: ~70%

### Coverage Targets
- **Unit Tests**: >90%
- **Integration Tests**: >80%
- **Overall**: >85%

## Common Issues

### Import Errors
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Missing Dependencies
```bash
# Install test dependencies
pip install -r requirements.txt
```

### Mock Issues
```python
# Use correct import path for mocking
with patch('module.submodule.Class') as mock_class:
    # Mock implementation
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-mock Documentation](https://pytest-mock.readthedocs.io/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [Black Documentation](https://black.readthedocs.io/) 