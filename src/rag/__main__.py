"""Allow `python -m rag.cli ...` after `pip install -e .` or PYTHONPATH=src."""

from rag.cli import main

if __name__ == "__main__":
    main()
