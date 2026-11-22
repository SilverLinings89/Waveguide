#!/usr/bin/env python3
"""Convergence of the discretization for two element orders.

Dissertation: experiment "Convergence Analysis", figure "Convergence plot of the numerical error" and the table of
its results (section Validation of the Method).
Produces: convergence_study.png, convergence_study.dat

A straight waveguide (4 x 3.6 x 4 um) with Dirichlet data, truncated by 6 cell layers of PML (sigma_max 10, order
3), solved with 1 x 2 x 2 processes and two sweeping levels. Each process has n x n x n cells. The solution on the
finest mesh is the reference; the error of the coarser meshes is evaluated at the cell centers of the reference
mesh by the solver itself (ConvergenceRun). "Order 1" and "order 2" of the dissertation are "fem order" 0 and 1.

Note that the coarse meshes are far from resolving the wavelength (h = 2 um for n = 2 at a wavelength of 1 um), so
the first points are not in the asymptotic regime.

laptop: n = 2, 4, 8 with n = 8 as reference.
paper:  n = 2, 4, 8, 16 with n = 16 as reference (the order 2 reference has 1.2 million unknowns).
"""
import os
import re
import numpy as np
import common

SEQUENCES = {"laptop": [2, 4, 8], "paper": [2, 4, 8, 16]}


def main():
    arguments = common.argument_parser(__doc__).parse_args()
    output = common.prepare(arguments, "convergence_study")
    data_file = os.path.join(output, "convergence_study.dat")
    sequence = SEQUENCES[arguments.scale]
    if not arguments.plot_only:
        rows = []
        for fem_order in (0, 1):
            case = dict(common.STRAIGHT_WAVEGUIDE, **{
                "fem order": fem_order, "geometry size z": 4.0, "PML n layers": 6, "PML thickness": 1.0,
                "PML sigma max": 10, "perform convergence test": True,
                "convergence sequence cell count": ",".join(str(n) for n in sequence)})
            run = {"sweeping level": 2, "processes in x": 1, "processes in y": 2, "processes in z": 2,
                   "cell count x": sequence[-1], "cell count y": sequence[-1], "cell count z": sequence[-1],
                   "solver precision": 1e-6}
            result = common.run_solver(arguments, "fem_order_%d" % fem_order, case, run)
            errors = [float(value) for value in re.findall(r"Result: \d+ found numerical error ([\d.eE+-]+?)and", result["stdout"])]
            h_values = [float(value) for value in re.search(r"maximal values of h were: ([\d. ]+)", result["stdout"]).group(1).split()]
            dofs = [float(value) for value in re.search(r"number of dofs were: ([\d ]+)", result["stdout"]).group(1).split()]
            for h, error, n_dofs in zip(h_values, errors, dofs):
                rows.append([fem_order + 1, h, error, n_dofs])
        common.write_table(data_file, ["order", "h", "error_against_finest_mesh", "dofs"], rows)
    rows = np.array(common.read_table(data_file))

    plt = common.pyplot()
    figure, axis = plt.subplots(figsize=(6.5, 4.8))
    for order, color in ((1, "tab:purple"), (2, "tab:green")):
        selected = rows[rows[:, 0] == order]
        axis.loglog(selected[:, 1], selected[:, 2], "o-", color=color, label="order %d" % order)
        # reference slope through the point of the finest mesh
        h = selected[:, 1]
        axis.loglog(h, selected[-1, 2] * (h / h[-1]) ** order, ":", color=color, label=r"$\propto h^%d$" % order)
    axis.set_xlabel(r"mesh size h [$\mu$m]")
    axis.set_ylabel("error against the solution on the finest mesh")
    axis.legend()
    common.save(figure, arguments, "convergence_study.png")


if __name__ == "__main__":
    main()
