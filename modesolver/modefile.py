"""Mode files: the input of the solver for the case parameter "mode file".

A mode file is a text file. Lines that start with '#' form the header with entries "# key = value"; the solver
uses beta (propagation constant in 1 / um) and wavelength. Every other line is one point of a uniform grid,

    x y E_x E_y s epsilon

with x running fastest. s is the stored longitudinal component: the solver uses E_z = -i s (the file
Modes/output_mode.dat of the dissertation was computed for the opposite sign of the phase). The last column is the
relative permittivity of the cross-section; the solver uses it as material for "waveguide cross section = mode
file". Files without header and without the last column (the old format) are still valid.
"""
import numpy as np


def write_mode_file(mode, path, x_range=(-5.0, 5.0), y_range=(-4.5, 4.5), step=0.05, structure=None):
    """Samples a Mode on a uniform grid and writes it. The grid has to cover the computational domain of the solver
    including its absorbing layers. Returns the dictionary that read_mode_file would return."""
    structure = structure if structure is not None else mode.structure
    nx = int(round((x_range[1] - x_range[0]) / step)) + 1
    ny = int(round((y_range[1] - y_range[0]) / step)) + 1
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    ex, ey, ez = mode.sample(x, y)
    scale = 1.0 / np.sqrt(ex ** 2 + ey ** 2 + ez ** 2).max()
    X, Y = np.meshgrid(x, y, indexing="ij")
    eps = structure.epsilon(X, Y) if structure is not None else np.ones_like(X)
    header = [("format", "waveguide mode file 2"), ("columns", "x y E_x E_y s epsilon (E_z = -i s, x runs fastest)"),
              ("beta", "%.12g" % mode.beta), ("n_eff", "%.12g" % mode.n_eff), ("wavelength", "%.12g" % mode.wavelength),
              ("mode", mode.label), ("method", mode.method),
              ("structure", structure.description if structure is not None else "unknown"),
              ("structure_type", structure.name if structure is not None else "unknown")]
    if structure is not None:
        header += [("structure_" + key, "%.12g" % value) for key, value in structure.parameters.items()]
    with open(path, "w") as f:
        for key, value in header:
            f.write("# %s = %s\n" % (key, value))
        for j in range(ny):
            for i in range(nx):
                f.write("%.10g %.10g %.9e %.9e %.9e %.9g\n" % (x[i], y[j], scale * ex[i, j], scale * ey[i, j], -scale * ez[i, j], eps[i, j]))
    return read_mode_file(path)


def read_mode_file(path):
    """Reads a mode file (old or new format): {"header": {...}, "x", "y", "ex", "ey", "s", "epsilon" (or None)};
    the arrays have the index [i_x, i_y]."""
    header, rows = {}, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "=" in line:
                    key, value = line[1:].split("=", 1)
                    header[key.strip()] = value.strip()
                continue
            rows.append([float(entry) for entry in line.split()])
    data = np.array(rows)
    x, y = np.unique(data[:, 0]), np.unique(data[:, 1])
    shape = (len(y), len(x))
    table = lambda column: data[:, column].reshape(shape).T
    return {"header": header, "x": x, "y": y, "ex": table(2), "ey": table(3), "s": table(4),
            "epsilon": table(5) if data.shape[1] > 5 else None}
