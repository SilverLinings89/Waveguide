#!/usr/bin/env python3
"""Profile of the fundamental mode that is used as input signal.

Dissertation: figures "Real part of E_x" and "Imaginary part of E_z" of the experiment on the variational effective
index approximation (section Modal Theory).
Produces: mode_profile_Ex.png, mode_profile_Ez.png

The data is the file Modes/output_mode.dat of the repository (computed with the EIMS solver for the
2.0 um x 1.8 um waveguide at 1550 nm). The script also prints two consistency checks of the profile: the residual of
the Helmholtz equation in the core and the divergence of the field as it is imported by the solver.
"""
import os
import numpy as np
import common

WIDTH, HEIGHT = 2.0, 1.8
EPSILON_CORE, EPSILON_EFFECTIVE, WAVELENGTH = 2.3409, 2.1588449, 1.55


def main():
    arguments = common.argument_parser(__doc__).parse_args()
    common.prepare(arguments, "mode_profile")
    plt = common.pyplot()
    data = np.loadtxt(os.path.join(common.REPOSITORY, "Modes", "output_mode.dat"))
    x, y = np.unique(data[:, 0]), np.unique(data[:, 1])
    grid = lambda column: data[:, column].reshape(len(y), len(x))  # x is the fast index of the file
    scale = np.abs(data[:, 2:5]).max()
    Ex, Ez = grid(2) / scale, grid(4) / scale

    beta = 2 * np.pi * np.sqrt(EPSILON_EFFECTIVE) / WAVELENGTH
    k0 = 2 * np.pi / WAVELENGTH
    core = np.ix_(np.abs(y) < 0.7, np.abs(x) < 0.8)
    laplace = np.gradient(np.gradient(Ex, x, axis=1), x, axis=1) + np.gradient(np.gradient(Ex, y, axis=0), y, axis=0)
    helmholtz = laplace + (k0 ** 2 * EPSILON_CORE - beta ** 2) * Ex
    # The solver imports E_z = -i * (stored value) for a wave exp(i beta z): div E = dEx/dx + i beta E_z.
    divergence = np.gradient(Ex, x, axis=1) + beta * Ez
    print("  beta = %.4f / um" % beta)
    print("  relative residual of the Helmholtz equation in the core: %.1e" % (np.linalg.norm(helmholtz[core]) / np.linalg.norm(beta ** 2 * Ex[core])))
    print("  relative divergence in the core:                         %.1e" % (np.linalg.norm(divergence[core]) / np.linalg.norm(beta * Ez[core])))

    for values, name, title in ((Ex, "Ex", r"$\mathrm{Re}(E_x)$"), (Ez, "Ez", r"$\mathrm{Im}(E_z)$ as stored (the solver uses $E_z = -\mathrm{i}\,\cdot$ this)")):
        figure, axis = plt.subplots(figsize=(7, 4.6))
        mesh = axis.pcolormesh(x, y, values, shading="auto", cmap="viridis" if name == "Ex" else "RdBu_r")
        axis.add_patch(plt.Rectangle((-WIDTH / 2, -HEIGHT / 2), WIDTH, HEIGHT, fill=False, edgecolor="black"))
        axis.set_xlim(-4, 4)
        axis.set_ylim(-4, 4)
        axis.set_aspect("equal")
        axis.grid(False)
        axis.set_xlabel(r"x [$\mu$m]")
        axis.set_ylabel(r"y [$\mu$m]")
        axis.set_title(title + ", normalized; the rectangle is the core")
        figure.colorbar(mesh, ax=axis)
        common.save(figure, arguments, "mode_profile_%s.png" % name)


if __name__ == "__main__":
    main()
