"""
Entry point for running speceval as a module.

Usage: python -m speceval.cli --tests path/to/test.yaml
"""

from .cli import main

if __name__ == '__main__':
    main()
