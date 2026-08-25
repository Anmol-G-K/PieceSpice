"""Common utilities, types, and plotting helpers."""


# Diagnostics install: legacy ngspice suppressors are OPTIONAL during the
# spicelib migration (T1.1). They import PySpice internally, so once pyspice
# is uninstalled these calls must not break package import. The LTspice
# layer (diagnostics_ltspice) needs no import-time wiring - out-of-process
# runs carry their own logs.
from common.diagnostics import install_log_dedup as _install_dedup
try:
    _install_dedup()
except Exception:  # pragma: no cover - pyspice gone
    pass

try:
    from common.diagnostics import install_console_suppressor as _install_suppressor
    _install_suppressor()
except Exception:  # pragma: no cover - pyspice gone
    pass

try:
    from common.diagnostics import install_notebook_quiet_console as _iqc
    _iqc()
except Exception:  # pragma: no cover - pyspice gone
    pass
