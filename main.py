# main.py
"""
Entry point.
"""
import logging, sys
from app.api import app  # noqa

root = logging.getLogger()
if not root.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(h)
    root.setLevel(logging.INFO)
