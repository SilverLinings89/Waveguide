"""Modes of waveguides with a cross-section that does not change along the direction of propagation z.

A mode is a field E(x, y) exp(i beta z). This package computes beta and the field

- analytically for step-index fibres (circular core), slab waveguides and hollow metallic waveguides with a
  rectangular or circular cross-section (analytic.py),
- approximately for rectangular dielectric cores with the method of Marcatili (analytic.py),
- numerically for any distribution of the permittivity over the cross-section with a full-vector finite difference
  method (fd.py); structures.py describes rectangular, circular and elliptical cores, rib and strip waveguides on
  a substrate, slot waveguides and coupled cores,

and writes mode files that the solver reads (modefile.py, case parameter "mode file").

Conventions: lengths in micrometers, time dependence exp(-i omega t), so a mode travelling in the +z direction is
E(x, y) exp(i beta z). Without losses the transverse components E_x and E_y are real and E_z is imaginary; a Mode
stores the real numbers E_x, E_y and e_z with E_z = i e_z.

    python -m modesolver --help
"""
from .structures import (Structure, rectangular, circular, elliptical, slab, rib, strip, slot, coupled, homogeneous)
from .mode import Mode
from .fd import solve as solve_fd
from .analytic import (fibre_modes, slab_modes, slab_mode_table, metallic_rectangular_modes, metallic_circular_modes,
                       marcatili_modes)
from .modefile import write_mode_file, read_mode_file

__all__ = ["Structure", "rectangular", "circular", "elliptical", "slab", "rib", "strip", "slot", "coupled",
           "homogeneous", "Mode", "solve_fd", "fibre_modes", "slab_modes", "slab_mode_table",
           "metallic_rectangular_modes", "metallic_circular_modes", "marcatili_modes", "write_mode_file",
           "read_mode_file"]
