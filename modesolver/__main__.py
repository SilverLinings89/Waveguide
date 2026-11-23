"""Command line of the mode solver: python -m modesolver <structure> [options]   (run in the root of the repository)

Examples
    python -m modesolver rectangular --output Modes/rectangular.dat
    python -m modesolver circular --diameter 2 --output Modes/circular.dat --plot circular.png
    python -m modesolver rib --width 0.5 --height 0.22 --slab-height 0.09 --window 4 3 --step 0.02 --modes 4
    python -m modesolver circular --method fd      # numerical instead of the analytical solution
"""
import argparse
import sys
import numpy as np
from . import structures, analytic, fd, modefile

STRUCTURES = ["rectangular", "circular", "elliptical", "slab", "strip", "rib", "slot", "coupled",
              "metallic-rectangular", "metallic-circular"]
ANALYTIC = {"circular", "slab", "metallic-rectangular", "metallic-circular"}


def build_structure(a):
    """The Structure for the command line arguments, with the defaults of the structure for values not given."""
    def given(**names):
        return {key: getattr(a, argument) for key, argument in names.items() if getattr(a, argument) is not None}
    core = given(eps_core="eps_core", eps_cladding="eps_cladding")
    layered = given(eps_core="eps_core", eps_substrate="eps_substrate", eps_cover="eps_cover")
    if a.structure == "rectangular":
        return structures.rectangular(**given(width="width", height="height"), **core)
    if a.structure == "circular":
        return structures.circular(**given(diameter="diameter"), **core)
    if a.structure == "elliptical":
        return structures.elliptical(**given(width="width", height="height"), **core)
    if a.structure == "slab":
        return structures.slab(**given(thickness="thickness", eps_core="eps_core", eps_top="eps_cover", eps_bottom="eps_substrate"))
    if a.structure == "strip":
        return structures.strip(**given(width="width", height="height"), **layered)
    if a.structure == "rib":
        return structures.rib(**given(width="width", height="height", slab_height="slab_height"), **layered)
    if a.structure == "slot":
        return structures.slot(**given(rail_width="rail_width", gap="gap", height="height"), **core)
    if a.structure == "coupled":
        return structures.coupled(**given(width="width", height="height", gap="gap"), **core)
    return structures.homogeneous(a.eps_core if a.eps_core is not None else 1.0)


def compute(a):
    """Returns (structure, modes, window) for the command line arguments."""
    structure = build_structure(a)
    method = a.method
    if method == "auto":
        method = "analytic" if a.structure in ANALYTIC else "fd"
    if a.structure.startswith("metallic"):
        # The window of a metallic waveguide is the waveguide itself.
        width = a.width or 2.0
        height = (a.height or 1.0) if a.structure == "metallic-rectangular" else width
        if a.structure == "metallic-circular":
            width = height = a.diameter or 2.0
        window = ((-width / 2, width / 2), (-height / 2, height / 2))
        if method != "analytic":
            if a.structure == "metallic-circular":
                sys.exit("The finite difference solver has a rectangular conducting boundary; use the analytical solution for the circular metallic waveguide.")
            return structure, fd.solve(structure, a.wavelength, window[0], window[1], a.step, a.modes, guided_only=False), window
        if a.structure == "metallic-rectangular":
            return structure, analytic.metallic_rectangular_modes(width, height, a.wavelength, structure.background)[:a.modes], window
        return structure, analytic.metallic_circular_modes(width, a.wavelength, structure.background)[:a.modes], window
    window = ((-a.window[0] / 2, a.window[0] / 2), (-a.window[1] / 2, a.window[1] / 2))
    p = structure.parameters
    if method == "analytic":
        if a.structure == "circular":
            modes = analytic.fibre_modes(p["diameter"], p["eps_core"], p["eps_cladding"], a.wavelength)
        elif a.structure == "slab":
            modes = analytic.slab_modes(p["thickness"], p["eps_core"], p["eps_top"], p["eps_bottom"], a.wavelength)
        else:
            sys.exit("There is no analytical solution for the structure '%s'. Use --method fd%s." % (
                a.structure, " or --method marcatili" if a.structure == "rectangular" else ""))
    elif method == "marcatili":
        if a.structure != "rectangular":
            sys.exit("Marcatili's approximation is for the rectangular structure.")
        modes = analytic.marcatili_modes(p["width"], p["height"], p["eps_core"], p["eps_cladding"], a.wavelength)
    else:
        modes = fd.solve(structure, a.wavelength, window[0], window[1], a.step, a.modes)
    return structure, modes[:a.modes], window


def plot(mode, structure, window, path):
    """Writes a figure with the three field components and the outline of the structure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.linspace(window[0][0], window[0][1], 241)
    y = np.linspace(window[1][0], window[1][1], 241)
    components = mode.sample(x, y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    eps = structure.epsilon(X, Y)
    limit = max(np.abs(component).max() for component in components)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for axis, component, name in zip(axes, components, ("$E_x$", "$E_y$", "$E_z / i$")):
        image = axis.pcolormesh(x, y, component.T, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto")
        if eps.max() > eps.min():
            axis.contour(x, y, eps.T, levels=np.unique(eps)[:-1] + 1e-9, colors="k", linewidths=0.6)
        axis.set_title(name)
        axis.set_xlabel("x [um]")
        axis.set_aspect("equal")
    axes[0].set_ylabel("y [um]")
    figure.colorbar(image, ax=axes, shrink=0.85)
    figure.suptitle("%s: %s, n_eff = %.6f (%s)" % (structure.name, mode.label, mode.n_eff, mode.method), fontsize=11)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main(arguments=None):
    parser = argparse.ArgumentParser(prog="python -m modesolver", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("structure", choices=STRUCTURES)
    for name, text in (("width", "width of the core, strip or rib (x)"), ("height", "height of the core, strip, rib or rails (y)"),
                       ("diameter", "diameter of the circular core"), ("thickness", "thickness of the slab"),
                       ("slab-height", "height of the layer beside the rib"), ("gap", "gap of the slot or between coupled cores"),
                       ("rail-width", "width of one rail of the slot waveguide"),
                       ("eps-core", "relative permittivity of the core (or of the filling of a metallic waveguide)"),
                       ("eps-cladding", "relative permittivity of a homogeneous cladding"),
                       ("eps-substrate", "relative permittivity below the core"), ("eps-cover", "relative permittivity above the core")):
        parser.add_argument("--" + name, type=float, help=text + " (default: the default of the structure)")
    parser.add_argument("--wavelength", type=float, default=1.55, help="vacuum wavelength in um (default 1.55)")
    parser.add_argument("--method", choices=["auto", "analytic", "fd", "marcatili"], default="auto",
                        help="auto: the analytical solution where one exists, finite differences otherwise")
    parser.add_argument("--window", type=float, nargs=2, default=[10.0, 9.0], metavar=("X", "Y"),
                        help="size of the computational window and of the written grid (default 10 9); it has to "
                             "contain the domain of the solver with its absorbing layers")
    parser.add_argument("--step", type=float, default=0.05, help="grid spacing of the finite difference solver and of the written file (default 0.05)")
    parser.add_argument("--modes", type=int, default=6, help="number of modes to compute (default 6)")
    parser.add_argument("--mode", type=int, default=None, help="index of the mode to write or plot (default: the first one, see --polarization)")
    parser.add_argument("--polarization", choices=["x", "y"], default="x", help="without --mode: take the first mode polarized mainly in this direction (default x)")
    parser.add_argument("--output", help="write the selected mode as a mode file for the solver")
    parser.add_argument("--plot", help="write a figure of the selected mode (PNG)")
    a = parser.parse_args(arguments)

    structure, modes, window = compute(a)
    if not modes:
        sys.exit("No guided mode was found for: %s" % structure.description)
    x = np.linspace(window[0][0], window[0][1], 81)
    y = np.linspace(window[1][0], window[1][1], 81)
    fractions = [mode.polarization_fraction_x(x, y) for mode in modes]
    print("Structure: %s, wavelength %g um" % (structure.description, a.wavelength))
    print("  index  mode             n_eff        epsilon effective  beta [1/um]   E_x share  method")
    for index, (mode, fraction) in enumerate(zip(modes, fractions)):
        print("  %-5d  %-15s  %.8f   %.8f         %.8f    %.2f       %s" % (
            index, mode.label, mode.n_eff, mode.epsilon_effective, mode.beta, fraction, mode.method))
    selected = a.mode
    if selected is None:
        wanted = [index for index, fraction in enumerate(fractions) if (fraction > 0.5) == (a.polarization == "x")]
        selected = wanted[0] if wanted else 0
    if selected >= len(modes):
        sys.exit("There is no mode with index %d." % selected)
    mode = modes[selected]
    if a.plot:
        plot(mode, structure, window, a.plot)
        print("Figure of mode %d: %s" % (selected, a.plot))
    if a.output:
        modefile.write_mode_file(mode, a.output, window[0], window[1], a.step, structure)
        print("Mode %d (%s) written to %s. Case parameters for the solver:" % (selected, mode.label, a.output))
        print("    set mode file = %s" % a.output)
        print("    set waveguide cross section = mode file")
    return modes


if __name__ == "__main__":
    main()
