"""Cross-sections: the relative permittivity as a function of x and y.

A Structure is a background permittivity with shapes painted on top of it in the given order (a later shape
replaces an earlier one). The core of every predefined structure is centered at the origin, as in the solver.
"""
import numpy as np


class Structure:
    """Piecewise constant permittivity over the cross-section."""

    def __init__(self, name, background, shapes=(), description="", parameters=None):
        self.name = name                    # short name, e.g. "circular"
        self.background = float(background)  # permittivity where no shape is
        self.shapes = list(shapes)          # list of (kind, numbers..., permittivity), see epsilon()
        self.description = description      # one line for file headers and reports
        self.parameters = dict(parameters or {})  # the numbers the structure was built from

    def epsilon(self, x, y):
        """Relative permittivity at the points (x, y) (arrays of the same shape or numbers)."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        eps = np.full(x.shape, self.background)
        for shape in self.shapes:
            kind, value = shape[0], shape[-1]
            if kind == "rectangle":
                x0, x1, y0, y1 = shape[1:5]
                inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
            elif kind == "ellipse":
                cx, cy, rx, ry = shape[1:5]
                inside = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
            else:
                raise ValueError("Unknown shape '%s'." % kind)
            eps = np.where(inside, value, eps)
        return eps

    @property
    def epsilon_max(self):
        return max([self.background] + [shape[-1] for shape in self.shapes])

    @property
    def epsilon_min(self):
        return min([self.background] + [shape[-1] for shape in self.shapes])

    def __repr__(self):
        return "Structure(%s: %s)" % (self.name, self.description)


INFINITE = 1e9  # stands for "to the end of the domain" in the extents of rectangles


def homogeneous(epsilon=1.0):
    """One material everywhere (the inside of a hollow metallic waveguide)."""
    return Structure("homogeneous", epsilon, [], "homogeneous, permittivity %g" % epsilon, {"epsilon": epsilon})


def rectangular(width=2.0, height=1.8, eps_core=2.3409, eps_cladding=1.8496):
    """Rectangular core in a homogeneous cladding (the waveguide of the dissertation)."""
    return Structure("rectangular", eps_cladding, [("rectangle", -width / 2, width / 2, -height / 2, height / 2, eps_core)],
                     "rectangular core %g x %g um, permittivity %g in %g" % (width, height, eps_core, eps_cladding),
                     {"width": width, "height": height, "eps_core": eps_core, "eps_cladding": eps_cladding})


def circular(diameter=2.0, eps_core=2.3409, eps_cladding=1.8496):
    """Circular core in a homogeneous cladding (step-index fibre)."""
    return Structure("circular", eps_cladding, [("ellipse", 0.0, 0.0, diameter / 2, diameter / 2, eps_core)],
                     "circular core, diameter %g um, permittivity %g in %g" % (diameter, eps_core, eps_cladding),
                     {"diameter": diameter, "eps_core": eps_core, "eps_cladding": eps_cladding})


def elliptical(width=2.0, height=1.2, eps_core=2.3409, eps_cladding=1.8496):
    """Elliptical core in a homogeneous cladding."""
    return Structure("elliptical", eps_cladding, [("ellipse", 0.0, 0.0, width / 2, height / 2, eps_core)],
                     "elliptical core %g x %g um, permittivity %g in %g" % (width, height, eps_core, eps_cladding),
                     {"width": width, "height": height, "eps_core": eps_core, "eps_cladding": eps_cladding})


def slab(thickness=1.0, eps_core=2.3409, eps_top=1.8496, eps_bottom=None):
    """Layer of the given thickness, infinite in x, between two half spaces."""
    eps_bottom = eps_top if eps_bottom is None else eps_bottom
    return Structure("slab", eps_top, [("rectangle", -INFINITE, INFINITE, -INFINITE, -thickness / 2, eps_bottom),
                                       ("rectangle", -INFINITE, INFINITE, -thickness / 2, thickness / 2, eps_core)],
                     "slab of thickness %g um, permittivity %g between %g (top) and %g (bottom)" % (thickness, eps_core, eps_top, eps_bottom),
                     {"thickness": thickness, "eps_core": eps_core, "eps_top": eps_top, "eps_bottom": eps_bottom})


def strip(width=0.5, height=0.22, eps_core=12.11, eps_substrate=2.085, eps_cover=1.0):
    """Strip (wire) on a substrate with a different material above, e.g. silicon on silica in air. The core is
    centered at the origin; the substrate fills the half space below it."""
    return Structure("strip", eps_cover, [("rectangle", -INFINITE, INFINITE, -INFINITE, -height / 2, eps_substrate),
                                          ("rectangle", -width / 2, width / 2, -height / 2, height / 2, eps_core)],
                     "strip %g x %g um, permittivity %g on %g, cover %g" % (width, height, eps_core, eps_substrate, eps_cover),
                     {"width": width, "height": height, "eps_core": eps_core, "eps_substrate": eps_substrate, "eps_cover": eps_cover})


def rib(width=0.5, height=0.22, slab_height=0.09, eps_core=12.11, eps_substrate=2.085, eps_cover=1.0):
    """Rib waveguide: a strip on a thinner layer of the same material, on a substrate. The rib (width x height,
    measured from the substrate) is centered at the origin."""
    bottom = -height / 2
    return Structure("rib", eps_cover, [("rectangle", -INFINITE, INFINITE, -INFINITE, bottom, eps_substrate),
                                        ("rectangle", -INFINITE, INFINITE, bottom, bottom + slab_height, eps_core),
                                        ("rectangle", -width / 2, width / 2, bottom, height / 2, eps_core)],
                     "rib %g x %g um on a layer of %g um, permittivity %g on %g, cover %g" % (width, height, slab_height, eps_core, eps_substrate, eps_cover),
                     {"width": width, "height": height, "slab_height": slab_height, "eps_core": eps_core,
                      "eps_substrate": eps_substrate, "eps_cover": eps_cover})


def slot(rail_width=0.2, gap=0.1, height=0.22, eps_core=12.11, eps_cladding=2.085):
    """Slot waveguide: two rails separated by a narrow gap, in a homogeneous cladding."""
    return Structure("slot", eps_cladding, [("rectangle", -gap / 2 - rail_width, -gap / 2, -height / 2, height / 2, eps_core),
                                            ("rectangle", gap / 2, gap / 2 + rail_width, -height / 2, height / 2, eps_core)],
                     "slot: two rails %g x %g um, gap %g um, permittivity %g in %g" % (rail_width, height, gap, eps_core, eps_cladding),
                     {"rail_width": rail_width, "gap": gap, "height": height, "eps_core": eps_core, "eps_cladding": eps_cladding})


def coupled(width=2.0, height=1.8, gap=1.0, eps_core=2.3409, eps_cladding=1.8496):
    """Two equal rectangular cores side by side (directional coupler); its modes are the even and the odd supermode."""
    return Structure("coupled", eps_cladding, [("rectangle", -gap / 2 - width, -gap / 2, -height / 2, height / 2, eps_core),
                                               ("rectangle", gap / 2, gap / 2 + width, -height / 2, height / 2, eps_core)],
                     "two cores %g x %g um, gap %g um, permittivity %g in %g" % (width, height, gap, eps_core, eps_cladding),
                     {"width": width, "height": height, "gap": gap, "eps_core": eps_core, "eps_cladding": eps_cladding})
