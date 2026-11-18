"""Tier-3 nonlinear reduced-order modelling: TPWL + POD-DEIM.

Official algorithm sources:

* M. Rewienski, J. White, "A Trajectory Piecewise-Linear Approach to Model
  Order Reduction and Fast Simulation of Nonlinear Circuits and
  Micromachined Devices," IEEE TCAD 22(2), Feb 2003.
  https://www2.ee.washington.edu/research/mscad/workspace/mor/public/biblio/04d_3.pdf
* S. Chaturantabut, D. C. Sorensen, "Nonlinear Model Reduction via Discrete
  Empirical Interpolation," SIAM J. Sci. Comput. 32(5), 2010.
  DEIM Algorithm 1 as reprinted in Sorensen & Embree, arXiv:1407.5516.

No physical constants are invented here: every model parameter comes from
the configured DPT conditions, the simulated trajectory, or the algorithm
constants printed in the papers.
"""
