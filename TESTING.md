# Simple Testing Guide

This guide shows you how to run tests for the Policy-Gradient project.

## Quick Start

### Run All Tests Locally
```bash
# From the project root directory
python run_tests.py
```

This will:
- Install test dependencies (pytest, flake8, black)
- Run all unit tests
- Check code style
- Give you a summary

### Run Tests Manually
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Check code style
flake8 . --count --exit-zero --max-line-length=88
black --check .
```

## What Gets Tested

### 1. Unit Tests
- **Atari Baselines**: Argument parsing, environment handling
- **DQN**: Model initialization, agent functionality
- **Policy Gradient**: MLP models, memory management

### 2. Code Style
- **Flake8**: Python style guide compliance
- **Black**: Code formatting consistency

### 3. Imports
- **Import Validation**: Make sure all modules can be imported

## GitHub Actions (CI)

When you push code or create a pull request, GitHub Actions automatically:

1. **Sets up Python 3.11**
2. **Installs dependencies**
3. **Runs all tests**
4. **Checks code style**
5. **Tests imports**

If any step fails, the CI will show you exactly what went wrong.

## Common Issues

### Import Errors
```bash
# Make sure you're in the project root
pwd
# Should show: /path/to/Policy-Gradient
```

### Missing Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt
```

### Permission Issues
```bash
# Make test runner executable
chmod +x run_tests.py
```

## Test Results

- **Green**: Tests passed
- **Red**: Tests failed - check the error message
- **Yellow**: Style warnings - fix to improve code quality

## Simple Workflow

1. **Write code**
2. **Run tests locally**: `python run_tests.py`
3. **Fix any issues**
4. **Commit and push**
5. **GitHub Actions runs automatically**

That's it! The testing setup is now much simpler and easier to use.
