#!/usr/bin/env python3
"""Tapering function of the signal coupling.

Dissertation: figure "phi(z) and (1 - phi(z)) Re(E_I,x)" (section Tapered Coupling).
Produces: signal_taper.png

No simulation is involved. phi(z) = 2 t^3 - 3 t^2 + 1 with t = (z - z_min) / (z_max - z_min) ramps from 1 to 0.
The solver computes the field (1 - phi) E, i.e. the mode is switched on over the taper, and the part phi E is
replaced by a source term. The wave is plotted with the propagation constant of the mode of the solver.
"""
import numpy as np
import common


def main():
    arguments = common.argument_parser(__doc__).parse_args()
    common.prepare(arguments, "signal_taper")
    plt = common.pyplot()
    z_min, z_max = 0.0, 1.0
    beta = 2 * np.pi * np.sqrt(2.1588449) / 1.55
    z = np.linspace(-0.35, 1.3, 1000)
    t = np.clip((z - z_min) / (z_max - z_min), 0, 1)
    phi = 2 * t ** 3 - 3 * t ** 2 + 1
    wave = np.cos(beta * z)

    figure, axis = plt.subplots(figsize=(8, 3.8))
    axis.plot(z, phi, color="tab:blue", label=r"$\phi(z)$")
    axis.plot(z, phi * wave, color="tab:blue", alpha=0.3, label=r"$\phi\,\mathrm{Re}(E_{I,x})$ (replaced by the source term)")
    axis.plot(z, 1 - phi, color="tab:red", linestyle=":", label=r"$1 - \phi(z)$")
    axis.plot(z, (1 - phi) * wave, color="tab:red", label=r"$(1-\phi)\,\mathrm{Re}(E_{I,x})$ (computed field)")
    axis.set_xlabel(r"z [$\mu$m], taper from $z_{\min} = 0$ to $z_{\max} = 1$")
    axis.legend(loc="lower left", fontsize=9)
    common.save(figure, arguments, "signal_taper.png")


if __name__ == "__main__":
    main()
