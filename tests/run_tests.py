#!/usr/bin/env python
"""
Test runner for the Credit Risk Model Development and Validation Framework.
"""

import os
import sys
import unittest
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def discover_and_run_tests(test_pattern=None, verbosity=2):
    """
    Discover and run tests matching the pattern.
    
    Args:
        test_pattern: Optional pattern to filter test modules
        verbosity: Verbosity level for test output
    
    Returns:
        TestResult object
    """
    start_dir = os.path.dirname(os.path.abspath(__file__))
    
    if test_pattern:
        pattern = f"test_{test_pattern}.py"
        logger.info(f"Running tests matching pattern: {pattern}")
    else:
        pattern = "test_*.py"
        logger.info("Running all tests")
    
    test_suite = unittest.defaultTestLoader.discover(start_dir, pattern=pattern)
    test_runner = unittest.TextTestRunner(verbosity=verbosity)
    
    return test_runner.run(test_suite)


def main():
    """Main function to run tests."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run tests for the Credit Risk Model Framework")
    parser.add_argument(
        "--module", "-m",
        help="Specific module to test (e.g., 'models', 'validator', 'monitor')"
    )
    parser.add_argument(
        "--verbosity", "-v", type=int, default=2,
        help="Verbosity level (1-3)"
    )
    args = parser.parse_args()
    
    # Run tests
    result = discover_and_run_tests(args.module, args.verbosity)
    
    # Print summary
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    
    # Return error code if tests failed
    return 1 if (result.failures or result.errors) else 0


if __name__ == "__main__":
    sys.exit(main()) 