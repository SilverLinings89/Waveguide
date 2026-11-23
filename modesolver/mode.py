"""The result of every mode computation of this package."""
import numpy as np


class Mode:
    """A mode E(x, y) exp(i beta z) with real E_x, E_y and imaginary E_z = i e_z.

    field(x, y) returns (E_x, E_y, e_z) for arrays x and y of the same shape. The field is scaled such that the
    largest Euclidean norm is about 1 and the dominant component is positive where it is largest.
    """

    def __init__(self, beta, wavelength, field, label, method, structure=None, exact=False):
        self.beta = float(beta)              # propagation constant in 1 / um
        self.wavelength = float(wavelength)  # vacuum wavelength in um
        self.field = field                   # callable (x, y) -> (E_x, E_y, e_z)
        self.label = label                   # e.g. "HE11 (x)", "TE0", "mode 0"
        self.method = method                 # "analytic", "finite differences", "Marcatili"
        self.structure = structure
        self.exact = exact                   # True for analytical solutions without approximation

    @property
    def n_eff(self):
        """Effective index beta / k0."""
        return self.beta * self.wavelength / (2 * np.pi)

    @property
    def epsilon_effective(self):
        """n_eff squared, the value of the case parameter "epsilon effective"."""
        return self.n_eff ** 2

    def sample(self, x, y):
        """The field on the grid of the coordinate vectors x and y: three arrays with index [i_x, i_y]."""
        X, Y = np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float), indexing="ij")
        return tuple(np.asarray(component, dtype=float) for component in self.field(X, Y))

    def polarization_fraction_x(self, x, y):
        """Share of E_x in the transverse field energy on the given grid (1: polarized in x, 0: in y)."""
        ex, ey, _ = self.sample(x, y)
        return float(np.sum(ex ** 2) / max(np.sum(ex ** 2 + ey ** 2), 1e-300))

    def __repr__(self):
        return "Mode(%s, n_eff = %.6f, %s)" % (self.label, self.n_eff, self.method)


def overlap(mode_a, mode_b, x, y):
    """Normalized overlap |<a, b>| / (|a| |b|) of two fields on a grid (all three components). 1 for equal fields."""
    a, b = mode_a.sample(x, y), mode_b.sample(x, y)
    inner = sum(np.sum(p * q) for p, q in zip(a, b))
    return float(abs(inner) / np.sqrt(sum(np.sum(p * p) for p in a) * sum(np.sum(q * q) for q in b)))


def normalized(field_function, x_extent, y_extent):
    """Wraps a field function such that the largest norm on a coarse grid is 1 and the dominant component is positive there."""
    x = np.linspace(-x_extent, x_extent, 121)
    y = np.linspace(-y_extent, y_extent, 121)
    X, Y = np.meshgrid(x, y, indexing="ij")
    ex, ey, ez = field_function(X, Y)
    norm = np.sqrt(ex ** 2 + ey ** 2 + ez ** 2)
    index = np.unravel_index(np.nanargmax(norm), norm.shape)
    dominant = ex[index] if abs(ex[index]) >= abs(ey[index]) else ey[index]
    scale = (1.0 if dominant >= 0 else -1.0) / norm[index]

    def scaled(x, y):
        return tuple(scale * component for component in field_function(x, y))
    return scaled
