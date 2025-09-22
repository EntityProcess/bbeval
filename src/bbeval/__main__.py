"""
Entry point for running bbeval as a module.

Usage: python -m bbeval.cli --tests path/to/test.yaml
"""

from .cli import main

if __name__ == '__main__':
    main()
