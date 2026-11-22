#!/usr/bin/env python3
"""Scaling of the sweeping preconditioner with the number of subdomains.

Dissertation: experiment "Sweeping scaling" and figure "Results of the sweeping scaling experiment" (section
Problems with Sweeping Preconditioners).
Produces: sweeping_scaling.png, sweeping_scaling.dat

A straight waveguide is extended by one subdomain (2 um, 20 x 20 x 20 cells, one process) at a time. The mode is
prescribed as Dirichlet data, the domain is wrapped in 10 cell layers of PML (thickness 0.5, sigma_max 30, order 3)
and the system is solved by GMRES with a sweep in z as preconditioner. Plotted are the number of GMRES steps and
the wall time of the whole run against the number of subdomains.

laptop: 2 to 12 subdomains (12 cores and about 1.5 GB of memory per process; with less memory the machine
        starts to swap and the wall times become meaningless).
paper:  the list of the dissertation up to 256 subdomains (cluster).
"""
import os
import numpy as np
import common

SUBDOMAINS = {
    "laptop": [2, 3, 4, 6, 8, 12],
    "paper": list(range(2, 17)) + [20, 24, 28, 32, 64, 128, 256],
}


def main():
    arguments = common.argument_parser(__doc__).parse_args()
    output = common.prepare(arguments, "sweeping_scaling")
    data_file = os.path.join(output, "sweeping_scaling.dat")
    if not arguments.plot_only:
        rows = []
        for n in SUBDOMAINS[arguments.scale]:
            case = dict(common.STRAIGHT_WAVEGUIDE, **{"geometry size z": 2.0 * n})
            if arguments.scale == "paper":
                case["PML n layers"] = common.PAPER_PML_LAYERS
            run = {"sweeping level": 1, "processes in x": 1, "processes in y": 1, "processes in z": n,
                   "cell count x": 20, "cell count y": 20, "cell count z": 20, "solver precision": 1e-6}
            result = common.run_solver(arguments, "subdomains_%03d" % n, case, run)
            rows.append([n, result["solver_iterations"], result["wall_time"], result["signal_strength_abs"]])
        common.write_table(data_file, ["subdomains", "gmres_steps", "wall_time_s", "signal_strength"], rows)
    rows = np.array(common.read_table(data_file))
    n, steps, seconds = rows[:, 0], rows[:, 1], rows[:, 2]

    plt = common.pyplot()
    # Two measures with different units: two panels with a shared x-axis instead of two y-axes.
    figure, (axis, time_axis) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    axis.plot(n, steps, "o", color="tab:blue", label="GMRES steps")
    linear = np.polyfit(n, steps, 1)
    axis.plot(n, np.polyval(linear, n), color="tab:blue", linewidth=1, alpha=0.6, label="fit g(n) = %.3f n + %.2f" % tuple(linear))
    axis.set_ylabel("GMRES steps")
    axis.set_ylim(bottom=0)
    axis.legend(fontsize=9)
    time_axis.plot(n, seconds, "o", color="tab:orange", label="wall time of the run")
    if len(n) >= 3:
        quadratic = np.polyfit(n, seconds, 2)
        fine = np.linspace(n.min(), n.max(), 200)
        time_axis.plot(fine, np.polyval(quadratic, fine), color="tab:orange", linewidth=1, alpha=0.6, label="fit f(n) = %.3f n² + %.2f n + %.1f" % tuple(quadratic))
    time_axis.set_ylabel("wall time [s]")
    time_axis.set_xlabel("number of subdomains n (one process each)")
    time_axis.set_ylim(bottom=0)
    time_axis.legend(fontsize=9)
    common.save(figure, arguments, "sweeping_scaling.png")


if __name__ == "__main__":
    main()
