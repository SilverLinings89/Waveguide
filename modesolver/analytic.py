"""Analytical modes: slab waveguides, step-index fibres, hollow metallic waveguides, and Marcatili's approximation
for rectangular dielectric cores.

All fields follow from the longitudinal components. With exp(i beta z - i omega t), h = Z0 H and
kc^2 = k0^2 eps - beta^2,

    E_t = i / kc^2 (beta grad_t E_z - k0 z x grad_t h_z),
    h_t = i / kc^2 (beta grad_t h_z + k0 eps z x grad_t E_z).

The functions return Mode objects with the fields multiplied by -i, so that E_x and E_y are real and E_z = i e_z.
"""
import numpy as np
from scipy import optimize, special
from .mode import Mode, normalized
from . import structures


# ----------------------------------------------------------------------------------------------------------------
# Slab waveguides
# ----------------------------------------------------------------------------------------------------------------

def _slab_solutions(eps_core, eps_top, eps_bottom, thickness, k0, tm):
    """Solutions of the dispersion relation of a three layer slab.

    kappa d - atan(xi_b gamma_b / kappa) - atan(xi_t gamma_t / kappa) = m pi, with xi = 1 for TE and
    xi = eps_core / eps_cladding for TM. Returns a list of (beta, kappa, gamma_bottom, gamma_top, phase_bottom, m).
    """
    lower = k0 * np.sqrt(max(eps_top, eps_bottom))
    upper = k0 * np.sqrt(eps_core)
    xi_b = eps_core / eps_bottom if tm else 1.0
    xi_t = eps_core / eps_top if tm else 1.0

    def parts(beta):
        kappa = np.sqrt(max(k0 ** 2 * eps_core - beta ** 2, 0.0))
        gamma_b = np.sqrt(max(beta ** 2 - k0 ** 2 * eps_bottom, 0.0))
        gamma_t = np.sqrt(max(beta ** 2 - k0 ** 2 * eps_top, 0.0))
        return kappa, gamma_b, gamma_t

    def residual(beta, m):
        kappa, gamma_b, gamma_t = parts(beta)
        return kappa * thickness - np.arctan2(xi_b * gamma_b, kappa) - np.arctan2(xi_t * gamma_t, kappa) - m * np.pi

    solutions = []
    span = upper - lower
    for m in range(1000):
        a, b = lower + 1e-13 * span, upper - 1e-13 * span
        if residual(a, m) <= 0:
            break
        beta = optimize.brentq(residual, a, b, args=(m,), xtol=1e-15, rtol=1e-15)
        kappa, gamma_b, gamma_t = parts(beta)
        solutions.append((beta, kappa, gamma_b, gamma_t, np.arctan2(xi_b * gamma_b, kappa), m))
    return solutions


def _slab_profile(solution, thickness):
    """The continuous field of a slab solution (E for TE, h for TM) and its derivative as functions of the
    coordinate across the layer, which is centered at 0."""
    _, kappa, gamma_b, gamma_t, phase, _ = solution
    half = thickness / 2
    top_value = np.cos(kappa * thickness - phase)

    def profile(s):
        s = np.asarray(s, dtype=float)
        core, below = np.abs(s) <= half, s < -half
        argument = kappa * (np.clip(s, -half, half) + half) - phase
        lower_decay = np.exp(gamma_b * np.minimum(s + half, 0.0))
        upper_decay = np.exp(-gamma_t * np.maximum(s - half, 0.0))
        f = np.where(core, np.cos(argument), np.where(below, np.cos(phase) * lower_decay, top_value * upper_decay))
        df = np.where(core, -kappa * np.sin(argument),
                      np.where(below, gamma_b * np.cos(phase) * lower_decay, -gamma_t * top_value * upper_decay))
        return f, df
    return profile


def slab_mode_table(eps_core, eps_top, eps_bottom, thickness, wavelength):
    """Effective indices of all guided modes of a slab: {"TE": [...], "TM": [...]}, ordered by mode number."""
    k0 = 2 * np.pi / wavelength
    return {name: [solution[0] / k0 for solution in _slab_solutions(eps_core, eps_top, eps_bottom, thickness, k0, tm)]
            for name, tm in (("TE", False), ("TM", True))}


def slab_modes(thickness=1.0, eps_core=2.3409, eps_top=1.8496, eps_bottom=None, wavelength=1.55):
    """Guided modes of a layer between two half spaces (exact). The layer is normal to y and infinite in x.

    TE modes have the electric field along x, TM modes have E_y and E_z. Returns all modes ordered by decreasing beta.
    """
    eps_bottom = eps_top if eps_bottom is None else eps_bottom
    k0 = 2 * np.pi / wavelength
    structure = structures.slab(thickness, eps_core, eps_top, eps_bottom)
    half = thickness / 2
    modes = []
    for tm in (False, True):
        for solution in _slab_solutions(eps_core, eps_top, eps_bottom, thickness, k0, tm):
            beta, profile = solution[0], _slab_profile(solution, thickness)

            def field(x, y, beta=beta, profile=profile, tm=tm):
                y = np.asarray(y, dtype=float) + 0 * np.asarray(x, dtype=float)
                f, df = profile(y)
                if not tm:
                    return f, np.zeros_like(f), np.zeros_like(f)
                eps = np.where(np.abs(y) <= half, eps_core, np.where(y < -half, eps_bottom, eps_top))
                # f is h_x: E_y = -beta h_x / (k0 eps), E_z = -i dh_x/dy / (k0 eps).
                return np.zeros_like(f), -beta * f / (k0 * eps), -df / (k0 * eps)
            modes.append(Mode(beta, wavelength, normalized(field, thickness, 2 * thickness), "%s%d" % ("TM" if tm else "TE", solution[5]),
                              "analytic", structure, exact=True))
    return sorted(modes, key=lambda mode: -mode.beta)


# ----------------------------------------------------------------------------------------------------------------
# Fields in cylindrical coordinates
# ----------------------------------------------------------------------------------------------------------------

def _cylindrical_field(order, angle, beta, k0, regions):
    """Field function for E_z = A F(r) cos(order (phi - angle)) and h_z = B F(r) sin(order (phi - angle)).

    regions(r) returns (A F, A F', B F, B F', kc^2, mask): the radial functions with their amplitudes, the squared
    transverse wavenumber of the region of every point, and where the field is not zero. For order 0 the angular
    functions are 1.
    """
    def field(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        r = np.maximum(np.hypot(x, y), 1e-12)
        phi = np.arctan2(y, x)
        c = np.cos(order * (phi - angle)) if order > 0 else np.ones_like(r)
        s = np.sin(order * (phi - angle)) if order > 0 else np.ones_like(r)
        a_f, a_df, b_f, b_df, kc2, mask = regions(r)
        e_r = (beta * a_df + k0 * order * b_f / r) * c / kc2
        e_phi = (-beta * order * a_f / r - k0 * b_df) * s / kc2
        e_x = e_r * np.cos(phi) - e_phi * np.sin(phi)
        e_y = e_r * np.sin(phi) + e_phi * np.cos(phi)
        e_z = -a_f * c
        return np.where(mask, e_x, 0.0), np.where(mask, e_y, 0.0), np.where(mask, e_z, 0.0)
    return field


# ----------------------------------------------------------------------------------------------------------------
# Step-index fibre
# ----------------------------------------------------------------------------------------------------------------

def _fibre_terms(beta, order, radius, eps_core, eps_cladding, k0):
    """Quantities of the boundary conditions at the core radius, with the poles at the zeros of J removed.

    Continuity of E_phi and h_phi at r = a for E_z = A F cos, h_z = B F sin gives
        [ -s            -k0 t_te / J ] [A]   0
        [ k0 t_tm / J    s           ] [B] = 0,   s = beta order / a (1 / kc1^2 - 1 / kc2^2).
    """
    u = radius * np.sqrt(k0 ** 2 * eps_core - beta ** 2)
    w = radius * np.sqrt(beta ** 2 - k0 ** 2 * eps_cladding)
    kc1, kc2 = (u / radius) ** 2, -(w / radius) ** 2
    bessel = special.jv(order, u)
    g1 = u / radius * special.jvp(order, u)                                    # J times the derivative of F in the core
    f2 = w / radius * special.kvp(order, w) / special.kv(order, w)             # the derivative of F in the cladding
    t_te = g1 / kc1 - bessel * f2 / kc2
    t_tm = eps_core * g1 / kc1 - eps_cladding * bessel * f2 / kc2
    s = beta * order / radius * (1 / kc1 - 1 / kc2)
    return u, w, bessel, t_te, t_tm, s


def _roots(function, lower, upper, samples=4000):
    """All sign changes of a function on an interval, refined with Brent's method, in decreasing order."""
    grid = np.linspace(lower, upper, samples)
    values = np.array([function(value) for value in grid])
    roots = []
    for i in range(samples - 1):
        if np.isfinite(values[i]) and np.isfinite(values[i + 1]) and values[i] * values[i + 1] < 0:
            roots.append(optimize.brentq(function, grid[i], grid[i + 1], xtol=1e-15, rtol=1e-15))
    return sorted(roots, reverse=True)


def fibre_modes(diameter=2.0, eps_core=2.3409, eps_cladding=1.8496, wavelength=1.55, max_order=3):
    """Guided modes of a step-index fibre (exact vector solution): TE0m, TM0m and the hybrid modes HE and EH.

    Every hybrid mode exists in two polarizations, rotated by a quarter of its angular period. For the fundamental
    mode HE11 these are the polarizations along x and y. Returns all modes with azimuthal order up to max_order,
    ordered by decreasing beta.
    """
    radius = diameter / 2
    k0 = 2 * np.pi / wavelength
    lower, upper = k0 * np.sqrt(eps_cladding), k0 * np.sqrt(eps_core)
    margin = 1e-9 * (upper - lower)
    structure = structures.circular(diameter, eps_core, eps_cladding)
    modes = []

    def add(beta, order, amplitude_e, amplitude_h, label, angles):
        u, w = _fibre_terms(beta, order, radius, eps_core, eps_cladding, k0)[:2]
        j_u, k_w = special.jv(order, u), special.kv(order, w)

        def regions(r):
            inside = r <= radius
            f = np.where(inside, special.jv(order, u * np.minimum(r, radius) / radius) / j_u,
                         special.kv(order, w * np.maximum(r, radius) / radius) / k_w)
            df = np.where(inside, u / radius * special.jvp(order, u * np.minimum(r, radius) / radius) / j_u,
                          w / radius * special.kvp(order, w * np.maximum(r, radius) / radius) / k_w)
            kc2 = np.where(inside, (u / radius) ** 2, -(w / radius) ** 2)
            return amplitude_e * f, amplitude_e * df, amplitude_h * f, amplitude_h * df, kc2, np.ones_like(r, dtype=bool)

        for angle in angles:
            field = normalized(_cylindrical_field(order, angle, beta, k0, regions), 2 * radius, 2 * radius)
            mode = Mode(beta, wavelength, field, label, "analytic", structure, exact=True)
            if order == 1:
                grid = np.linspace(-2 * radius, 2 * radius, 41)
                mode.label += " (x)" if mode.polarization_fraction_x(grid, grid) > 0.5 else " (y)"
            elif order > 1:
                mode.label += " (even)" if angle == 0 else " (odd)"
            modes.append(mode)

    # Order 0: the conditions decouple into TE (h_z only) and TM (E_z only).
    for index, name, amplitudes in ((3, "TE", (0.0, 1.0)), (4, "TM", (1.0, 0.0))):
        function = lambda beta, index=index: _fibre_terms(beta, 0, radius, eps_core, eps_cladding, k0)[index]
        for number, beta in enumerate(_roots(function, lower + margin, upper - margin)):
            add(beta, 0, amplitudes[0], amplitudes[1], "%s0%d" % (name, number + 1), [0.0])
    # Hybrid modes: determinant of the 2 x 2 system, multiplied by J^2.
    for order in range(1, max_order + 1):
        def determinant(beta, order=order):
            _, _, bessel, t_te, t_tm, s = _fibre_terms(beta, order, radius, eps_core, eps_cladding, k0)
            return k0 ** 2 * t_te * t_tm - (s * bessel) ** 2
        for number, beta in enumerate(_roots(determinant, lower + margin, upper - margin)):
            _, _, bessel, t_te, t_tm, s = _fibre_terms(beta, order, radius, eps_core, eps_cladding, k0)
            # Second row of the system: k0 t_tm / J A + s B = 0.
            amplitude_h = -k0 * t_tm / (bessel * s)
            label = "%s%d%d" % ("HE" if number % 2 == 0 else "EH", order, number // 2 + 1)
            add(beta, order, 1.0, amplitude_h, label, [0.0, np.pi / (2 * order)])
    return sorted(modes, key=lambda mode: -mode.beta)


# ----------------------------------------------------------------------------------------------------------------
# Hollow metallic waveguides
# ----------------------------------------------------------------------------------------------------------------

def metallic_rectangular_modes(width=2.0, height=1.0, wavelength=1.55, epsilon=1.0, max_index=3):
    """Propagating TE_mn and TM_mn modes of a rectangular waveguide with perfectly conducting walls (exact).
    The cross-section is centered at the origin. Ordered by decreasing beta."""
    k0 = 2 * np.pi / wavelength
    structure = structures.homogeneous(epsilon)
    modes = []
    for m in range(max_index + 1):
        for n in range(max_index + 1):
            kx, ky = m * np.pi / width, n * np.pi / height
            kc2 = kx ** 2 + ky ** 2
            if kc2 == 0 or k0 ** 2 * epsilon <= kc2:
                continue
            beta = np.sqrt(k0 ** 2 * epsilon - kc2)
            for kind in ("TE", "TM"):
                if kind == "TM" and (m == 0 or n == 0):
                    continue

                def field(x, y, kx=kx, ky=ky, kc2=kc2, beta=beta, kind=kind):
                    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
                    inside = (np.abs(x) <= width / 2) & (np.abs(y) <= height / 2)
                    X, Y = x + width / 2, y + height / 2
                    if kind == "TE":   # h_z = cos cos, E_t = k0 / kc^2 (d_y h_z, -d_x h_z)
                        e_x = -k0 * ky / kc2 * np.cos(kx * X) * np.sin(ky * Y)
                        e_y = k0 * kx / kc2 * np.sin(kx * X) * np.cos(ky * Y)
                        e_z = np.zeros_like(e_x)
                    else:              # E_z = sin sin, E_t = beta / kc^2 grad E_z
                        e_x = beta * kx / kc2 * np.cos(kx * X) * np.sin(ky * Y)
                        e_y = beta * ky / kc2 * np.sin(kx * X) * np.cos(ky * Y)
                        e_z = -np.sin(kx * X) * np.sin(ky * Y)
                    return np.where(inside, e_x, 0.0), np.where(inside, e_y, 0.0), np.where(inside, e_z, 0.0)
                modes.append(Mode(beta, wavelength, normalized(field, width / 2, height / 2), "%s%d%d" % (kind, m, n), "analytic",
                                  structure, exact=True))
    return sorted(modes, key=lambda mode: -mode.beta)


def metallic_circular_modes(diameter=2.0, wavelength=1.55, epsilon=1.0, max_order=3, max_radial=3):
    """Propagating TE_nm and TM_nm modes of a circular waveguide with a perfectly conducting wall (exact).
    Modes with azimuthal order above 0 exist in two polarizations. Ordered by decreasing beta."""
    k0 = 2 * np.pi / wavelength
    radius = diameter / 2
    structure = structures.homogeneous(epsilon)
    modes = []
    for order in range(max_order + 1):
        for kind, zeros in (("TE", special.jnp_zeros(order, max_radial)), ("TM", special.jn_zeros(order, max_radial))):
            for number, zero in enumerate(zeros):
                kc = zero / radius
                if k0 ** 2 * epsilon <= kc ** 2:
                    continue
                beta = np.sqrt(k0 ** 2 * epsilon - kc ** 2)

                def regions(r, order=order, kc=kc, kind=kind):
                    f, df = special.jv(order, kc * r), kc * special.jvp(order, kc * r)
                    zero_array = np.zeros_like(r)
                    if kind == "TE":
                        return zero_array, zero_array, f, df, kc ** 2, r <= radius
                    return f, df, zero_array, zero_array, kc ** 2, r <= radius
                for angle in ([0.0] if order == 0 else [0.0, np.pi / (2 * order)]):
                    label = "%s%d%d" % (kind, order, number + 1) + ("" if order == 0 else (" (even)" if angle == 0 else " (odd)"))
                    field = normalized(_cylindrical_field(order, angle, beta, k0, regions), radius, radius)
                    modes.append(Mode(beta, wavelength, field, label, "analytic", structure, exact=True))
    return sorted(modes, key=lambda mode: -mode.beta)


# ----------------------------------------------------------------------------------------------------------------
# Marcatili's approximation for a rectangular dielectric core
# ----------------------------------------------------------------------------------------------------------------

def marcatili_modes(width=2.0, height=1.8, eps_core=2.3409, eps_cladding=1.8496, wavelength=1.55):
    """Approximate modes of a rectangular core (Marcatili 1969): the field is a product of two slab profiles and
    the corner regions are neglected. Good far from cut-off; use the finite difference solver for accurate
    results. Returns the modes polarized along x and along y, ordered by decreasing beta."""
    k0 = 2 * np.pi / wavelength
    structure = structures.rectangular(width, height, eps_core, eps_cladding)
    modes = []
    for polarization in ("x", "y"):
        # The field is normal to the interfaces in the direction of its polarization (TM-like condition) and
        # tangential to the others (TE-like condition).
        along_x = _slab_solutions(eps_core, eps_cladding, eps_cladding, width, k0, polarization == "x")
        along_y = _slab_solutions(eps_core, eps_cladding, eps_cladding, height, k0, polarization == "y")
        for solution_x in along_x:
            for solution_y in along_y:
                beta_squared = k0 ** 2 * eps_core - solution_x[1] ** 2 - solution_y[1] ** 2
                if beta_squared <= k0 ** 2 * eps_cladding:
                    continue
                beta = np.sqrt(beta_squared)
                profile_x, profile_y = _slab_profile(solution_x, width), _slab_profile(solution_y, height)

                def field(x, y, beta=beta, profile_x=profile_x, profile_y=profile_y, polarization=polarization):
                    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
                    fx, dfx = profile_x(x)
                    fy, dfy = profile_y(y)
                    if polarization == "x":
                        jump = np.where(np.abs(x) <= width / 2, 1.0, eps_core / eps_cladding)
                        e = jump * fx * fy
                        return e, np.zeros_like(e), jump * dfx * fy / beta
                    jump = np.where(np.abs(y) <= height / 2, 1.0, eps_core / eps_cladding)
                    e = jump * fx * fy
                    return np.zeros_like(e), e, jump * fx * dfy / beta
                modes.append(Mode(beta, wavelength, normalized(field, width, height),
                                  "E%s%d%d" % (polarization, solution_x[5] + 1, solution_y[5] + 1), "Marcatili", structure))
    return sorted(modes, key=lambda mode: -mode.beta)
