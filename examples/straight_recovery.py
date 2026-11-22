#!/usr/bin/env python3
"""Shape optimization with a known optimum: a perturbed waveguide has to become straight again.

Not a figure of the dissertation. The test checks the optimization, because the best shape is known.
Produces: straight_recovery_convergence.png, straight_recovery_shapes.png, straight_recovery.dat and the HTML
          report of the run (report.html in the output folder of the run, see run_report.py)

Input and output of the 6 um long waveguide are at the same height, so the straight waveguide (displacement
f(z) = 0, all three shape parameters 0) transmits the mode without loss and is the optimal shape. The optimization
starts from a waveguide with two bumps of 0.36 um (shape parameters 0.3, 0.3, 0). The plot shows how far the shape
of every evaluation is away from the straight one, and how far its signal is below the signal of the straight
waveguide on the same mesh (computed first as a reference; it is below 1 because of the discretization error).

The loss tolerance is set to 1e-9 instead of the default 1e-5, so that the optimization does not stop as soon as
the signal is good: close to the optimum the signal depends only weakly on the shape (a displacement of 0.013 um
costs 2e-5), and the default would stop there.

laptop: 4 processes with 16 x 16 x 12 cells each, at most 40 evaluations, about 5 minutes.
paper:  the same with 24 x 24 x 16 cells per process.
"""
import os
import shutil
import numpy as np
import common

LENGTH, SECTORS = 6.0, 5
INITIAL_PARAMETERS = [0.3, 0.3, 0.0]


def main():
    parser = common.argument_parser(__doc__)
    parser.add_argument("--initial", default=",".join(str(value) for value in INITIAL_PARAMETERS),
                        help="initial shape parameters (three slopes, comma separated)")
    arguments = parser.parse_args()
    output = common.prepare(arguments, "straight_recovery")
    data_file = os.path.join(output, "straight_recovery.dat")
    run_directory = os.path.join(output, "runs", arguments.scale, "out", "run0")
    if not arguments.plot_only:
        straight = dict(common.STRAIGHT_WAVEGUIDE, **{
            "geometry size z": LENGTH, "Input Signal Method": "Taper",
            "vertical waveguide displacement": 0.0, "Number of shape sectors": SECTORS,
            "PML n layers": 6, "PML thickness": 0.5, "PML sigma max": 10, "PML skaling order": 2})
        case = dict(straight, **{
            "perform optimization": True, "Initial shape parameters": arguments.initial,
            "optimization maximum evaluations": 40, "optimization maximum steps": 40,
            "optimization gradient tolerance": 1e-4, "optimization loss tolerance": 1e-9})
        cells = (16, 16, 12) if arguments.scale == "laptop" else (24, 24, 16)
        run = {"sweeping level": 1, "processes in x": 1, "processes in y": 1, "processes in z": 4,
               "cell count x": cells[0], "cell count y": cells[1], "cell count z": cells[2],
               "solver precision": 1e-8, "GMRES maximum steps": 60}
        reference = common.run_solver(arguments, arguments.scale + "_straight", straight, run)
        result = common.run_solver(arguments, arguments.scale, case, run, report=True)
        history = np.array(common.read_table(os.path.join(result["run_directory"], "optimization_history.txt")))
        shapes = np.array(common.read_table(os.path.join(result["run_directory"], "optimization_shapes.txt")))
        z, shapes = shapes[0], shapes[1:]
        # Distance to the optimal shape f = 0: largest displacement and root mean square over z.
        rows = [[evaluation, -history[evaluation, 1], history[evaluation, 2], np.max(np.abs(shapes[evaluation])),
                 np.sqrt(np.mean(shapes[evaluation] ** 2)), reference["signal_strength_abs"]] + list(history[evaluation, 3:])
                for evaluation in range(len(history))]
        common.write_table(data_file, ["evaluation", "signal", "gradient_norm", "max_displacement_um", "rms_displacement_um",
                                       "signal_of_the_straight_waveguide", "parameter_1", "parameter_2", "parameter_3"], rows)
        shutil.copy(os.path.join(result["run_directory"], "optimization_shapes.txt"), os.path.join(output, "straight_recovery_shapes.dat"))
    rows = np.array(common.read_table(data_file))
    shapes = np.array(common.read_table(os.path.join(output, "straight_recovery_shapes.dat")))
    z, shapes = shapes[0], shapes[1:]
    evaluations, signal, gradient_norm, max_displacement, straight_signal = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], rows[0, 5]
    # The best shape so far is what the optimization would return if it stopped at this evaluation.
    best_so_far = [int(np.argmax(signal[:index + 1])) for index in range(len(signal))]

    plt = common.pyplot()
    figure, (axis, loss_axis, gradient_axis) = plt.subplots(3, 1, figsize=(7.8, 9.0), sharex=True)
    axis.semilogy(evaluations, max_displacement, "o-", color="tab:blue", label="evaluated shape")
    axis.semilogy(evaluations, max_displacement[best_so_far], "-", color="tab:orange", linewidth=2, label="best shape so far")
    axis.set_title("Distance to the optimal (straight) waveguide", fontsize=11)
    axis.set_ylabel(r"max |f(z)| [$\mu$m]")
    axis.legend(fontsize=9)
    gap = np.maximum(straight_signal - signal, 1e-16)
    loss_axis.semilogy(evaluations, gap, "o-", color="tab:blue", label="evaluated shape")
    loss_axis.semilogy(evaluations, gap[best_so_far], "-", color="tab:orange", linewidth=2, label="best shape so far")
    loss_axis.set_title("Signal below the signal of the straight waveguide (%.5f on this mesh)" % straight_signal, fontsize=11)
    loss_axis.set_ylabel(r"$|s_{straight}| - |s|$")
    gradient_axis.semilogy(evaluations, gradient_norm, "o-", color="tab:blue")
    gradient_axis.set_title("Norm of the shape gradient", fontsize=11)
    gradient_axis.set_ylabel("|gradient|")
    gradient_axis.set_xlabel("evaluation (one forward and one adjoint solve)")
    figure.align_ylabels()
    common.save(figure, arguments, "straight_recovery_convergence.png")

    figure, axis = plt.subplots(figsize=(7.8, 4.2))
    colors = plt.get_cmap("Blues")(np.linspace(0.3, 0.9, len(shapes)))
    for index, shape in enumerate(shapes):
        axis.plot(z, shape, color=colors[index], linewidth=1.0)
    best = int(np.argmax(signal))
    axis.plot(z, shapes[0], color="0.25", linestyle="--", linewidth=1.8, label="initial shape, |s| = %.4f" % signal[0])
    axis.plot(z, shapes[best], color="tab:orange", linewidth=2.2, label="best shape (evaluation %d), |s| = %.4f" % (best, signal[best]))
    axis.axhline(0, color="0.4", linewidth=0.8, linestyle=":", label="optimum: straight waveguide")
    axis.set_xlabel(r"z [$\mu$m]")
    axis.set_ylabel(r"vertical displacement f(z) [$\mu$m]")
    axis.legend(fontsize=9)
    common.save(figure, arguments, "straight_recovery_shapes.png")
    print("  initial: max |f| = %.4f um, signal %.6f; best (evaluation %d): max |f| = %.2e um, signal %.6f" % (
        max_displacement[0], signal[0], best, max_displacement[best], signal[best]))
    if os.path.exists(os.path.join(run_directory, "report.html")):
        print("  report of the run:", os.path.join(run_directory, "report.html"))


if __name__ == "__main__":
    main()
