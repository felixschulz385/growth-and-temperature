# glass/tests/conftest.py
import sys
import os

# Add the project root (where `download/`, `gcs/`, and `glass/` live)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
