#!/usr/bin/env python3
"""Adjoint based shape optimization of a waveguide with a vertical displacement.

Dissertation: experiment "Optimization" and the figures "Results from the optimization experiment" (shapes),
"signal loss and step width" and the GMRES convergence histories (section Numerical Results of the chapter
Electromagnetic Design).
Produces: shape_optimization_shapes.png, shape_optimization_history.png, shape_optimization_gmres.png,
          shape_optimization.dat (copy of optimization_history.txt of the solver)

The waveguide is 6 um long and has to rise by a prescribed height in the middle (a hump, symmetric to z = 3). The
shape of its axis is a piecewise quadratic function f(z) whose slopes at 0.6, 1.2 and 1.8 um are the three shape
parameters. BFGS minimizes the loss -|s|, where s is the overlap of the field with the mode at the output. The
gradient comes from one adjoint solve per evaluation.

laptop: hump of 0.5 um, 4 processes with 16 x 16 x 12 cells each, at most 14 evaluations.
paper:  hump of 1.5 um, 4 processes with 12 x 12 x 12 cells, PML of 10 layers and thickness 1.5 as in the
        dissertation. Its transverse resolution (three cells per wavelength) is too coarse for reliable numbers.
"""
import os
import shutil
import numpy as np
import common

LENGTH, SECTORS = 6.0, 5


def shape(z, slopes, height):
    """The shape function of the solver (Code/Optimization/ShapeFunction.cpp): f(0) = 0, f'(0) = 0, free slopes at
    the inner nodes, f'(L/2) = 0, f(L/2) = height; the last inner slope is determined by the height."""
    half, h = LENGTH / 2, LENGTH / (2 * SECTORS)
    nodes = np.arange(SECTORS + 1) * h

    def integrate(derivatives, position):
        value = 0.0
        for i in range(SECTORS):
            width = min(max(position - nodes[i], 0.0), h)
            value += derivatives[i] * width + 0.5 * (derivatives[i + 1] - derivatives[i]) * width ** 2 / h
        return value

    derivatives = np.array([0.0] + list(slopes) + [0.0, 0.0])
    derivatives[SECTORS - 1] = (height - integrate(derivatives, nodes[SECTORS - 2])) / h - 0.5 * (derivatives[SECTORS - 2] + derivatives[SECTORS])
    mirrored = np.where(z > half, LENGTH - z, z)
    return np.array([integrate(derivatives, position) for position in mirrored])


def main():
    arguments = common.argument_parser(__doc__).parse_args()
    output = common.prepare(arguments, "shape_optimization")
    height = 0.5 if arguments.scale == "laptop" else 1.5
    history_file = os.path.join(output, "shape_optimization.dat")
    if not arguments.plot_only:
        case = dict(common.STRAIGHT_WAVEGUIDE, **{
            "geometry size z": LENGTH, "Input Signal Method": "Taper", "perform optimization": True,
            "vertical waveguide displacement": height, "Number of shape sectors": SECTORS})
        if arguments.scale == "laptop":
            case.update({"PML n layers": 6, "PML thickness": 0.5, "PML sigma max": 10, "PML skaling order": 2,
                         "optimization maximum evaluations": 14})
            cells = (16, 16, 12)
        else:
            case.update({"PML n layers": 10, "PML thickness": 1.5, "PML sigma max": 20, "PML skaling order": 3})
            cells = (12, 12, 12)
        run = {"sweeping level": 1, "processes in x": 1, "processes in y": 1, "processes in z": 4,
               "cell count x": cells[0], "cell count y": cells[1], "cell count z": cells[2],
               "solver precision": 1e-8, "GMRES maximum steps": 60}
        result = common.run_solver(arguments, arguments.scale, case, run, report=True)
        shutil.copy(os.path.join(result["run_directory"], "optimization_history.txt"), history_file)
    run_directory = os.path.join(output, "runs", arguments.scale, "out", "run0")
    history = np.array(common.read_table(history_file))
    losses, gradient_norms, parameters = history[:, 1], history[:, 2], history[:, 3:]
    best = int(np.argmin(losses))

    plt = common.pyplot()
    z = np.linspace(0, LENGTH, 601)
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    selection = sorted(set([0, 1, best]))
    for evaluation in selection:
        label = "evaluation %d, signal %.4f" % (evaluation, -losses[evaluation]) + (" (best)" if evaluation == best else "")
        axis.plot(z, shape(z, parameters[evaluation], height), label=label)
    axis.set_xlabel(r"z [$\mu$m]")
    axis.set_ylabel(r"vertical displacement f(z) [$\mu$m]")
    axis.legend(fontsize=9)
    common.save(figure, arguments, "shape_optimization_shapes.png")

    figure, (axis, gradient_axis) = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True)
    evaluations = np.arange(len(losses))
    axis.semilogy(evaluations, 1 + losses, "o-", color="tab:blue")
    axis.set_ylabel("signal loss 1 − |s|")
    gradient_axis.semilogy(evaluations, gradient_norms, "o-", color="tab:orange")
    gradient_axis.set_ylabel("norm of the shape gradient")
    gradient_axis.set_xlabel("evaluation (one forward and one adjoint solve)")
    common.save(figure, arguments, "shape_optimization_history.png")

    # Every evaluation consists of a forward and an adjoint solve; the forward solves are the even series.
    histories = common.read_residual_histories(run_directory, 1)
    figure, axis = plt.subplots(figsize=(7.5, 4.4))
    for evaluation, residuals in enumerate(histories[0:2 * len(losses):2]):
        axis.semilogy(range(1, len(residuals) + 1), residuals, marker=".", linewidth=0.9, label="evaluation %d" % evaluation)
    axis.set_xlabel("GMRES step")
    axis.set_ylabel("residual of the forward solve")
    axis.legend(fontsize=7, ncol=2)
    common.save(figure, arguments, "shape_optimization_gmres.png")


if __name__ == "__main__":
    main()
