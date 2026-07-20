"""Run every ``test_*.py`` module in this directory from the repository root."""

from pathlib import Path
import sys
import unittest


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
