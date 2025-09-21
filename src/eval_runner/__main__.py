"""
Entry point for running eval_runner as a module.

Usage: python -m eval_runner.cli --tests path/to/test.yaml
"""

from .cli import main

if __name__ == '__main__':
    main()
