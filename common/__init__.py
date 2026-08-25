"""Common utilities, types, and plotting helpers."""


# Suppress duplicate ngspice console warnings at the logging layer
# (vendor models repeat identical warnings hundreds of times per
# sweep; see common.diagnostics).
from common.diagnostics import install_log_dedup as _install_dedup
_install_dedup()
from common.diagnostics import install_console_suppressor as _install_suppressor
_install_suppressor()


# Notebook-safe ngspice console suppression MUST be installed before the
# first NgSpiceShared instance binds its SendChar callback. common is
# imported by every entrypoint (engine, testbenches, CLI, notebooks), so
# this guarantees ordering.
try:
    from common.diagnostics import install_notebook_quiet_console as _iqc
    _iqc()
except Exception:  # pragma: no cover - never block package import
    pass
