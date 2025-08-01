#!/usr/bin/env python3
"""
Simple test runner for the Policy-Gradient project.
"""
import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and print the result."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")

    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print("✅ SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Error: {e}")
        if e.stderr:
            print("Error output:")
            print(e.stderr)
        return False


def main():
    """Run all tests."""
    print("🧪 Policy-Gradient Test Suite")
    print("=" * 50)

    # Check if we're in the right directory
    if not os.path.exists("tests"):
        print("❌ Error: tests directory not found. Please run from the project root.")
        sys.exit(1)

    # Install test dependencies
    print("\n📦 Installing test dependencies...")
    run_command("pip install pytest flake8 black", "Installing test dependencies")

    # Run tests
    print("\n🧪 Running tests...")
    test_success = run_command("python -m pytest tests/ -v", "Running pytest")

    # Run code style checks
    print("\n🔍 Running code style checks...")
    style_success = run_command(
        "flake8 . --count --exit-zero --max-line-length=88", "Flake8 style check"
    )
    run_command("black --check .", "Black formatting check")

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    if test_success and style_success:
        print("✅ All tests passed!")
        print("🎉 Your code is ready!")
        return 0
    else:
        print("❌ Some tests failed.")
        print("🔧 Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
