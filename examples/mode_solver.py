#!/usr/bin/env python3
"""Modes of several cross-sections, computed by the mode solver, and their use as input of the solver.

Not a figure of the dissertation (its mode file Modes/output_mode.dat came from an external solver).
Produces: mode_solver_gallery.png, mode_solver_accuracy.png, mode_solver.dat, mode files in the output folder, and
          the HTML report of a run with a circular core (report.html in the output folder of the run)

1. Gallery: the fundamental mode of a rectangular core, a circular core (analytical), an elliptical core, two
   coupled cores, a strip, a rib and a slot waveguide (silicon on silica) and a hollow metallic waveguide
   (analytical).
2. Accuracy: the error of the finite difference solver against the analytical solutions for the circular core and
   the metallic waveguide, for decreasing grid spacing.
3. Use in the solver: every dielectric structure of the gallery with a homogeneous cladding is sent through a
   straight waveguide with the permittivity taken from the mode file; the transmitted signal has to be close to 1.
   The same modes are also sent through the rectangular core of the dissertation, where they are not modes.

laptop: 2 processes, 16 x 16 x 20 cells each, about 2 minutes.
paper:  2 processes, 24 x 24 x 30 cells each.
"""
import os
import sys
import numpy as np
import common

sys.path.insert(0, common.REPOSITORY)
import modesolver as ms
from modesolver import __main__ as command_line

WAVELENGTH = 1.55
SILICON, SILICA = 12.11, 2.085


def main():
    arguments = common.argument_parser(__doc__).parse_args()
    output = common.prepare(arguments, "mode_solver")
    plt = common.pyplot()
    data_file = os.path.join(output, "mode_solver.dat")

    # name, structure, window (x, y), grid spacing, analytical modes or None
    wide = ((-3.0, 3.0), (-2.5, 2.5))
    small = ((-1.5, 1.5), (-1.0, 1.0))
    gallery = [
        ("rectangular core", ms.rectangular(), wide, 0.05, None),
        ("circular core (analytical)", ms.circular(2.0), wide, 0.05, ms.fibre_modes(2.0)),
        ("elliptical core", ms.elliptical(2.4, 1.4), wide, 0.05, None),
        ("two coupled cores", ms.coupled(1.0, 1.0, 0.5), wide, 0.05, None),
        ("strip, silicon on silica", ms.strip(0.5, 0.22, SILICON, SILICA, 1.0), small, 0.02, None),
        ("rib, silicon on silica", ms.rib(0.5, 0.22, 0.09, SILICON, SILICA, 1.0), small, 0.02, None),
        ("slot, silicon in silica", ms.slot(0.2, 0.1, 0.22, SILICON, SILICA), small, 0.01, None),
        ("metallic waveguide (analytical)", ms.homogeneous(1.0), ((-1.0, 1.0), (-0.5, 0.5)), 0.05, ms.metallic_rectangular_modes(2.0, 1.0)),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(15, 6.6), constrained_layout=True)
    modes = {}
    for axis, (name, structure, window, step, analytical) in zip(axes.flat, gallery):
        mode = analytical[0] if analytical else ms.solve_fd(structure, WAVELENGTH, window[0], window[1], step, n_modes=1)[0]
        modes[name] = mode
        x, y = np.linspace(window[0][0], window[0][1], 241), np.linspace(window[1][0], window[1][1], 201)
        ex, ey, ez = mode.sample(x, y)
        dominant = ex if np.abs(ex).max() >= np.abs(ey).max() else ey
        limit = np.abs(dominant).max()
        axis.pcolormesh(x, y, dominant.T, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto", rasterized=True)
        X, Y = np.meshgrid(x, y, indexing="ij")
        eps = structure.epsilon(X, Y)
        if eps.max() > eps.min():
            axis.contour(x, y, eps.T, levels=np.unique(eps)[:-1] + 1e-9, colors="k", linewidths=0.6)
        axis.set_title("%s\n$n_{eff}$ = %.5f" % (name, mode.n_eff), fontsize=10)
        axis.set_aspect("equal")
        axis.set_xlabel(r"x [$\mu$m]")
        print("  %-32s %-10s n_eff = %.6f" % (name, mode.label, mode.n_eff))
    axes[0][0].set_ylabel(r"y [$\mu$m]")
    axes[1][0].set_ylabel(r"y [$\mu$m]")
    figure.suptitle("Dominant transverse component of the fundamental mode at %g um" % WAVELENGTH, fontsize=11)
    common.save(figure, arguments, "mode_solver_gallery.png")

    # Accuracy of the finite differences against the analytical solutions.
    steps = [0.2, 0.1, 0.05, 0.025] if arguments.scale == "laptop" else [0.2, 0.1, 0.05, 0.025, 0.0125]
    exact_fibre, exact_metal = ms.fibre_modes(2.0)[0].n_eff, ms.metallic_rectangular_modes(2.0, 1.0)[0].n_eff
    errors = []
    for step in steps:
        fibre = ms.solve_fd(ms.circular(2.0), WAVELENGTH, (-4, 4), (-4, 4), step, n_modes=1)[0]
        metal = ms.solve_fd(ms.homogeneous(1.0), WAVELENGTH, (-1, 1), (-0.5, 0.5), step, n_modes=1, guided_only=False)[0]
        errors.append([step, abs(fibre.n_eff - exact_fibre), abs(metal.n_eff - exact_metal)])
    errors = np.array(errors)
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.loglog(errors[:, 0], errors[:, 1], "o-", label="circular core, HE11")
    axis.loglog(errors[:, 0], errors[:, 2], "s-", label="metallic waveguide, TE10")
    axis.loglog(errors[:, 0], errors[0, 2] * (errors[:, 0] / errors[0, 0]) ** 2, "k:", label="second order")
    axis.set_xticks(errors[:, 0])
    axis.set_xticklabels(["%g" % step for step in errors[:, 0]])
    axis.minorticks_off()
    axis.set_xlabel(r"grid spacing [$\mu$m]")
    axis.set_ylabel("error of the effective index")
    axis.legend(fontsize=9)
    common.save(figure, arguments, "mode_solver_accuracy.png")

    # Use in the solver: dielectric structures with a homogeneous cladding and the dimensions of the default domain.
    rows = []
    if not arguments.plot_only:
        cells = (16, 16, 20) if arguments.scale == "laptop" else (24, 24, 30)
        run = {"sweeping level": 1, "processes in x": 1, "processes in y": 1, "processes in z": 2,
               "cell count x": cells[0], "cell count y": cells[1], "cell count z": cells[2], "solver precision": 1e-8}
        base = dict(common.STRAIGHT_WAVEGUIDE, **{"geometry size z": 6.0, "Input Signal Method": "Dirichlet"})
        for index, (name, structure, window, step, analytical) in enumerate(gallery[:4]):
            path = os.path.join(output, "mode_%s.dat" % structure.name)
            ms.write_mode_file(modes[name], path, window[0], window[1], 0.05, structure)
            own = common.run_solver(arguments, "%s_%s" % (arguments.scale, structure.name),
                                    dict(base, **{"mode file": path, "waveguide cross section": "mode file"}), run, report=(structure.name == "circular"))
            other = common.run_solver(arguments, "%s_%s_in_rectangular_core" % (arguments.scale, structure.name), dict(base, **{"mode file": path}), run)
            rows.append([index, modes[name].n_eff, own["signal_strength_abs"], other["signal_strength_abs"]])
            print("  %-28s signal %.4f in its own structure, %.4f in the rectangular core" % (name, own["signal_strength_abs"], other["signal_strength_abs"]))
        common.write_table(data_file, ["structure_index", "n_eff", "signal_in_own_structure", "signal_in_rectangular_core"], rows)


if __name__ == "__main__":
    main()
