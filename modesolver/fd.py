"""Full-vector finite difference mode solver for an arbitrary permittivity over the cross-section.

With fields proportional to exp(i beta z - i omega t) and h = Z0 H, Maxwell's equations are

    curl E = i k0 h,    curl h = -i k0 eps E.

Eliminating E_z and h_z gives two first order systems for the transverse fields,

    beta (E_x, E_y) = P (h_x, h_y),    beta (h_x, h_y) = Q (E_x, E_y),

    P = 1/k0 [ -dx eps_z^-1 dy          dx eps_z^-1 dx + k0^2 ]      Q = 1/k0 [  dx dy              -dx dx - k0^2 eps_y ]
             [ -dy eps_z^-1 dy - k0^2   dy eps_z^-1 dx        ]               [  dy dy + k0^2 eps_x  -dy dx             ]

and therefore the eigenvalue problem P Q (E_x, E_y) = beta^2 (E_x, E_y) with real matrices. The derivatives are
discretized on a staggered (Yee) grid, on which every difference is centered:

    E_x, h_y at (i + 1/2, j)    E_y, h_x at (i, j + 1/2)    E_z at (i, j)    h_z at (i + 1/2, j + 1/2).

The outer boundary of the computational window is a perfect electric conductor. For guided modes the window has
to be large enough for the field to have decayed; for a hollow metallic waveguide the window is the waveguide.

At material interfaces the permittivity of a grid cell is averaged over sub-samples: arithmetically along an
interface and harmonically across it for the component normal to it, which keeps the second order of the scheme
for interfaces parallel to the axes.
"""
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
from scipy.interpolate import RegularGridInterpolator
from .mode import Mode


def _difference_matrices(n, d):
    """Forward and backward difference of a grid function with n values that is zero outside."""
    forward = sparse.diags([-np.ones(n), np.ones(n - 1)], [0, 1]) / d
    backward = sparse.diags([np.ones(n), -np.ones(n - 1)], [0, -1]) / d
    return forward.tocsr(), backward.tocsr()


def _averaged_permittivity(structure, x, y, dx, dy, samples, harmonic_axis):
    """Permittivity of the cells of size dx x dy around the points (x[i], y[j]).

    harmonic_axis: None for the arithmetic mean, 0 or 1 for the harmonic mean in that direction of the arithmetic
    means in the other direction.
    """
    offsets = (np.arange(samples) + 0.5) / samples - 0.5
    xs = (x[:, None] + offsets[None, :] * dx).reshape(-1)
    ys = (y[:, None] + offsets[None, :] * dy).reshape(-1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    eps = structure.epsilon(X, Y).reshape(len(x), samples, len(y), samples)
    if harmonic_axis is None:
        return eps.mean(axis=(1, 3))
    if harmonic_axis == 0:
        return 1.0 / (1.0 / eps.mean(axis=3)).mean(axis=1)
    return 1.0 / (1.0 / eps.mean(axis=1)).mean(axis=2)


def solve(structure, wavelength, x_range=(-5.0, 5.0), y_range=(-4.5, 4.5), step=0.05, n_modes=4, samples=8,
          guided_only=True, n_eff_guess=None):
    """Computes the modes with the largest propagation constants.

    structure:    a Structure (or any object with epsilon(x, y), epsilon_max).
    wavelength:   vacuum wavelength.
    x_range, y_range: the computational window; its boundary is a perfect electric conductor.
    step:         grid spacing (a number or a pair for x and y); the window is divided into a whole number of cells.
    n_modes:      number of eigenvalues to compute.
    samples:      sub-samples per direction for the averaging of the permittivity.
    guided_only:  drop modes that do not decay towards the boundary of the window (modes of the box or of the
                  layers that reach the boundary, not of the waveguide), see _edge_limit().
    n_eff_guess:  search modes near this effective index instead of near the largest index of the structure.

    Returns a list of Mode objects ordered by decreasing beta.
    """
    step_x, step_y = (step, step) if np.isscalar(step) else step
    nx = max(int(round((x_range[1] - x_range[0]) / step_x)), 4)
    ny = max(int(round((y_range[1] - y_range[0]) / step_y)), 4)
    dx = (x_range[1] - x_range[0]) / nx
    dy = (y_range[1] - y_range[0]) / ny
    k0 = 2 * np.pi / wavelength
    x_nodes = x_range[0] + dx * np.arange(nx)
    y_nodes = y_range[0] + dy * np.arange(ny)
    x_half, y_half = x_nodes + dx / 2, y_nodes + dy / 2
    n = nx * ny

    eps_x = _averaged_permittivity(structure, x_half, y_nodes, dx, dy, samples, 0)
    eps_y = _averaged_permittivity(structure, x_nodes, y_half, dx, dy, samples, 1)
    eps_z = _averaged_permittivity(structure, x_nodes, y_nodes, dx, dy, samples, None)

    # Tangential E is zero on the conductor at the lower ends of the window (nodes i = 0 and j = 0); at the upper
    # ends this follows from the zero continuation of the differences.
    keep_x = np.ones((nx, ny), dtype=bool)
    keep_x[:, 0] = False
    keep_y = np.ones((nx, ny), dtype=bool)
    keep_y[0, :] = False
    keep_z = keep_x & keep_y

    identity_x, identity_y = sparse.identity(nx, format="csr"), sparse.identity(ny, format="csr")
    forward_x, backward_x = _difference_matrices(nx, dx)
    forward_y, backward_y = _difference_matrices(ny, dy)
    Dxf, Dxb = sparse.kron(forward_x, identity_y, format="csr"), sparse.kron(backward_x, identity_y, format="csr")
    Dyf, Dyb = sparse.kron(identity_x, forward_y, format="csr"), sparse.kron(identity_x, backward_y, format="csr")
    inverse_eps_z = sparse.diags(np.where(keep_z, 1.0 / eps_z, 0.0).reshape(-1))
    Ex_diagonal, Ey_diagonal = sparse.diags(eps_x.reshape(-1)), sparse.diags(eps_y.reshape(-1))
    unit = sparse.identity(n, format="csr")

    P = sparse.bmat([[-Dxf @ inverse_eps_z @ Dyb, Dxf @ inverse_eps_z @ Dxb + k0 ** 2 * unit],
                     [-Dyf @ inverse_eps_z @ Dyb - k0 ** 2 * unit, Dyf @ inverse_eps_z @ Dxb]], format="csr") / k0
    Q = sparse.bmat([[Dxb @ Dyf, -Dxb @ Dxf - k0 ** 2 * Ey_diagonal],
                     [Dyb @ Dyf + k0 ** 2 * Ex_diagonal, -Dyb @ Dxf]], format="csr") / k0
    kept = np.flatnonzero(np.concatenate([keep_x.reshape(-1), keep_y.reshape(-1)]))
    selection = sparse.csr_matrix((np.ones(len(kept)), (kept, np.arange(len(kept)))), shape=(2 * n, len(kept)))
    operator = (selection.T @ P @ Q @ selection).tocsc()

    n_guess = np.sqrt(structure.epsilon_max) if n_eff_guess is None else n_eff_guess
    n_requested = min(n_modes + (4 if guided_only else 0), len(kept) - 2)
    values, vectors = scipy.sparse.linalg.eigs(operator, k=n_requested, sigma=(k0 * n_guess) ** 2, which="LM")
    order = np.argsort(-values.real)

    threshold = max(_edge_limit(eps_z[0, :], dy, k0), _edge_limit(eps_z[-1, :], dy, k0),
                    _edge_limit(eps_z[:, 0], dx, k0), _edge_limit(eps_z[:, -1], dx, k0))
    modes = []
    for index in order:
        beta_squared = values[index].real
        if beta_squared <= 0 or abs(values[index].imag) > 1e-6 * abs(beta_squared):
            continue
        if guided_only and beta_squared <= threshold * (1 + 1e-9):
            continue
        beta = np.sqrt(beta_squared)
        e_t = selection @ vectors[:, index].real
        h_t = (Q @ e_t) / beta
        e_z = (inverse_eps_z @ (Dxb @ h_t[n:] - Dyb @ h_t[:n])) / k0
        ex, ey, ez = e_t[:n].reshape(nx, ny), e_t[n:].reshape(nx, ny), e_z.reshape(nx, ny)
        modes.append(Mode(beta, wavelength, _grid_field(ex, ey, ez, x_nodes, y_nodes, x_half, y_half), "mode %d" % len(modes),
                          "finite differences, step %g x %g um" % (dx, dy), structure))
        if len(modes) == n_modes:
            break
    return modes


def _edge_limit(eps, d, k0):
    """Largest beta^2 of a field that does not decay towards an edge of the window with the given permittivity
    profile: k0^2 eps for a homogeneous edge, otherwise the fundamental mode of the layers that reach the edge
    (for example the layer beside a rib). Modes of the waveguide have a larger beta^2."""
    if eps.max() - eps.min() < 1e-12:
        return k0 ** 2 * eps.max()
    n = len(eps)
    operator = (np.diag(k0 ** 2 * eps - 2.0 / d ** 2) + np.diag(np.ones(n - 1), 1) / d ** 2 + np.diag(np.ones(n - 1), -1) / d ** 2)
    return max(np.linalg.eigvalsh(operator)[-1], k0 ** 2 * eps.min())


def _grid_field(ex, ey, ez, x_nodes, y_nodes, x_half, y_half):
    """Field function that interpolates the three components linearly from their staggered grids."""
    norm = np.sqrt(np.maximum.reduce([np.abs(ex), np.abs(ey), np.abs(ez)]).max() ** 2)
    index = np.unravel_index(np.argmax(np.abs(ex) + np.abs(ey)), ex.shape)
    dominant = ex[index] if abs(ex[index]) >= abs(ey[index]) else ey[index]
    scale = (1.0 if dominant >= 0 else -1.0) / norm
    options = dict(method="linear", bounds_error=False, fill_value=0.0)
    interpolants = (RegularGridInterpolator((x_half, y_nodes), scale * ex, **options),
                    RegularGridInterpolator((x_nodes, y_half), scale * ey, **options),
                    RegularGridInterpolator((x_nodes, y_nodes), scale * ez, **options))

    def field(x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        points = np.stack([x.reshape(-1), y.reshape(-1)], axis=-1)
        return tuple(interpolant(points).reshape(x.shape) for interpolant in interpolants)
    return field
