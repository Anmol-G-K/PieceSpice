"""Pytest configuration, HTML report metadata, and path bootstrap."""

import sys
from pathlib import Path

# Bootstrap: make the repo root importable regardless of the invocation cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_html_report_title(report):
    report.title = "PieceSpice Test & Validation Report"


def pytest_configure(config):
    if hasattr(config, "_metadata"):
        config._metadata["Project"] = "PieceSpice"
        config._metadata["Engine"] = "LTspice 26 + spicelib"
        config._metadata["Environment"] = "uv (Python 3.13)"
