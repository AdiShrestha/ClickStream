"""
conftest.py at the repo root makes pytest add the repo root to sys.path
automatically, so "from src.X import Y" works in all test files without
any per-file sys.path hacks.
"""
