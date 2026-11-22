#!/usr/bin/env python3
"""Sweeping on a stretched domain: what happens when the mesh no longer resolves the wave.

Dissertation: experiment "Sweeping on a stretched domain" and figure "Required steps and runtime of a GMRES solver
with a 1D sweeping preconditioner for an increasing length of the computational domain" (section Description of the
Method).
Produces: domain_stretching.png, domain_stretching.dat

8 processes with 20 x 20 x 20 cells each (160 cells in z in total). Only the length of the waveguide changes, so the
cells get longer and longer. The wavelength in the core is about 1 um, so the mesh has 10 cells per wavelength at a
length of 16 um and 5 at 32 um.

laptop: lengths 8, 16, ..., 64 um.
paper:  lengths 8, 10, ..., 68 um.
"""
import os
import numpy as np
import common

LENGTHS = {"laptop": list(range(8, 65, 8)), "paper": list(range(8, 69, 2))}


def main():
    arguments = common.argument_parser(__doc__).parse_args()
    output = common.prepare(arguments, "domain_stretching")
    data_file = os.path.join(output, "domain_stretching.dat")
    if not arguments.plot_only:
        rows = []
        for length in LENGTHS[arguments.scale]:
            case = dict(common.STRAIGHT_WAVEGUIDE, **{"geometry size z": length, "PML thickness": 1.0, "PML sigma max": 20})
            if arguments.scale == "paper":
                case["PML n layers"] = common.PAPER_PML_LAYERS
            run = {"sweeping level": 1, "processes in x": 1, "processes in y": 1, "processes in z": 8,
                   "cell count x": 20, "cell count y": 20, "cell count z": 20, "solver precision": 1e-6,
                   "GMRES maximum steps": 100}
            result = common.run_solver(arguments, "length_%02d" % length, case, run)
            rows.append([length, result["solver_iterations"], result["wall_time"], result["signal_strength_abs"]])
        common.write_table(data_file, ["length_um", "gmres_steps", "wall_time_s", "signal_strength"], rows)
    rows = np.array(common.read_table(data_file))

    plt = common.pyplot()
    figure, (axis, signal_axis) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    axis.plot(rows[:, 0], rows[:, 1], "o-", color="tab:blue")
    axis.set_ylabel("GMRES steps")
    axis.set_ylim(bottom=0)
    cells_axis = axis.secondary_xaxis("top", functions=(lambda length: 160.0 * 1.013 / np.maximum(length, 1e-9), lambda cells: 160.0 * 1.013 / np.maximum(cells, 1e-9)))
    cells_axis.set_xlabel("cells per wavelength in the core")
    cells_axis.set_xticks([20, 10, 7, 5, 4, 3])
    signal_axis.plot(rows[:, 0], rows[:, 3], "o-", color="tab:orange")
    signal_axis.set_ylabel("signal at the output (1 = lossless)")
    signal_axis.set_xlabel(r"length of the waveguide [$\mu$m], always 160 cells")
    common.save(figure, arguments, "domain_stretching.png")


if __name__ == "__main__":
    main()
