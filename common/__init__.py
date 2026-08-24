"""Common utilities, types, and plotting helpers."""


# Suppress duplicate ngspice console warnings at the logging layer
# (vendor models repeat identical warnings hundreds of times per
# sweep; see common.diagnostics).
from common.diagnostics import install_log_dedup as _install_dedup
_install_dedup()
from common.diagnostics import install_console_suppressor as _install_suppressor
_install_suppressor()
