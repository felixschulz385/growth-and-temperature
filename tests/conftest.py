"""
Run pytest from the project root (``pytest`` or ``pytest tests/``) so
``from src.xxx import yyy`` imports resolve — there is no pytest.ini/
pyproject.toml, so this relies on pytest's default rootdir detection.

Mirror src/'s package layout when adding new tests: a test for
src/data/download/sources/glass.py belongs at
tests/data/download/sources/test_glass.py. No __init__.py files anywhere
in this tree — pytest imports each file by its (unique) basename.
"""
