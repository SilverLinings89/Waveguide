#!/usr/bin/env python3
"""Hierarchical sweeping with three levels: residual histories of the GMRES solvers of every level.

Dissertation: experiment "Waveguide with vertical displacement" and the figures "Convergence of the GMRES run(s) on
level 1 / 2 / 3" (section Numeric Results).
Produces: hierarchical_sweeping.png (three panels), hierarchical_sweeping.dat (dofs, solves, timings)

The global problem is solved by GMRES with a sweep in z (level 1). Every subdomain solve of that sweep is a GMRES
solve with a sweep in y (level 2), whose subdomain solves use a sweep in x (level 3) with direct solvers on the
lowest level. The inner solves only reduce their residual by a factor of 1e-2 relative to the residual of the level
above. The plots show the histories that process 0 took part in.

laptop: a waveguide of 8 um with a hump of 0.5 um described by shape functions, 2 x 2 x 4 = 16 processes with
        8 x 8 x 10 cells each, tapered coupling.
paper:  predefined shape number 3 of the project partners (100 um), 3 x 3 x 39 = 351 processes with 16 x 16 x 40
        cells each, 10 PML layers. This ran for 39.5 hours on 351 cores and needs about 1 TB of memory.
"""
import os
import numpy as np
import common


def main():
    arguments = common.argument_parser(__doc__).parse_args()
    output = common.prepare(arguments, "hierarchical_sweeping")
    marker = os.path.join(output, "hierarchical_sweeping.dat")
    if not arguments.plot_only:
        if arguments.scale == "laptop":
            case = dict(common.STRAIGHT_WAVEGUIDE, **{
                "geometry size z": 8.0, "PML n layers": 6, "PML thickness": 0.5, "PML sigma max": 10,
                "PML skaling order": 2, "Input Signal Method": "Taper", "vertical waveguide displacement": 0.5})
            run = {"sweeping level": 3, "processes in x": 2, "processes in y": 2, "processes in z": 4,
                   "cell count x": 8, "cell count y": 8, "cell count z": 10}
        else:
            case = dict(common.STRAIGHT_WAVEGUIDE, **{
                "geometry size x": 6.0, "geometry size y": 5.4, "geometry size z": 100, "PML n layers": 10,
                "PML thickness": 0.5, "PML sigma max": 20, "PML skaling order": 3,
                "Use predefined shape": True, "Predefined case number": 3})
            run = {"sweeping level": 3, "processes in x": 3, "processes in y": 3, "processes in z": 39,
                   "cell count x": 16, "cell count y": 16, "cell count z": 40}
        run.update({"use relative convergence criterion": True, "relative convergence criterion": 1e-2,
                    "solver precision": 1e-6, "GMRES maximum steps": 30})
        result = common.run_solver(arguments, arguments.scale, case, run)
        common.write_table(marker, ["dofs", "gmres_steps_level_1", "subdomain_solves", "wall_time_s", "signal_strength"],
                           [[result["n_dofs"], result["solver_iterations"], result["n_local_solves"], result["wall_time"], result["signal_strength_abs"]]])
    run_directory = os.path.join(output, "runs", arguments.scale, "out", "run0")

    plt = common.pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    # Sweeping level 1 (z) is the top level, i.e. hierarchy level 3 of the solver for a three level run.
    for axis, (solver_level, direction) in zip(axes, ((3, "z"), (2, "y"), (1, "x"))):
        histories = common.read_residual_histories(run_directory, solver_level)
        for history in histories:
            axis.semilogy(range(1, len(history) + 1), history, "-", marker=".", linewidth=0.8)
        axis.set_title("GMRES with sweep in %s: %d solve%s" % (direction, len(histories), "" if len(histories) == 1 else "s"))
        axis.set_xlabel("step")
        axis.set_ylabel("residual")
    common.save(figure, arguments, "hierarchical_sweeping.png")


if __name__ == "__main__":
    main()
