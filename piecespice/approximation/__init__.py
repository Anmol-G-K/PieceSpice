"""Piecewise linear approximation and charge modeling algorithms."""

from piecespice.approximation.pwlf_fit import fit_piecewise_linear
from piecespice.approximation.rdp_compress import compress_rdp
from piecespice.approximation.charge_model import compute_charge_qoss

__all__ = ["fit_piecewise_linear", "compress_rdp", "compute_charge_qoss"]
