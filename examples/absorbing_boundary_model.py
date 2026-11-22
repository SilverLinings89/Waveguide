"""One-dimensional reference model of the two absorbing boundary methods.

For a wave exp(i beta z) the truncation of the half line z > L by a PML or by Hardy space infinite elements (HSIE)
can be written down in a few lines. This is the problem every transverse mode of a waveguide or every plane wave
component of a scattered field sees in the direction normal to the boundary (beta = k cos(angle of incidence)).
The model is used by pml_vs_hsie.py to compare the accuracy of the two methods independently of the 3D solver.

PML:  complex stretching s(z) = 1 + i sigma_max t^p over a number of cells, the first of which is not stretched
      (as in the solver), with a PEC condition at the end. Weak form: int u'v'/s - beta^2 s u v.
HSIE: the exterior field is represented by its value u0 at the boundary and the coefficients a_0 ... a_N of a
      polynomial on the unit disc (Hohage and Nannen). With the operators T+ and T- of the dissertation
      (section HSIE) the exterior integrals are
          int u'v' = -2 i kappa0 (T+ u) . (T+ v),    int u v = (2 i / kappa0) (T- u) . (T- v).
"""
import numpy as np

def gauss(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1), 0.5 * w

def element_matrices(order, h, coefficient_stiffness, coefficient_mass, z0):
    """Local matrices of int c1(z) u'v' and int c2(z) u v on [z0, z0+h] for Lagrange elements of the given order."""
    nodes = np.linspace(0, 1, order + 1)
    q, w = gauss(order + 3)
    K = np.zeros((order + 1, order + 1), dtype=complex); M = np.zeros_like(K)
    for xi, wi in zip(q, w):
        phi = np.ones(order + 1); dphi = np.zeros(order + 1)
        for i in range(order + 1):
            for j in range(order + 1):
                if j != i:
                    phi[i] *= (xi - nodes[j]) / (nodes[i] - nodes[j])
            for m in range(order + 1):
                if m != i:
                    term = 1.0 / (nodes[i] - nodes[m])
                    for j in range(order + 1):
                        if j != i and j != m:
                            term *= (xi - nodes[j]) / (nodes[i] - nodes[j])
                    dphi[i] += term
        z = z0 + xi * h
        K += wi * coefficient_stiffness(z) * np.outer(dphi, dphi) / h
        M += wi * coefficient_mass(z) * np.outer(phi, phi) * h
    return K, M

def solve(beta, length, n, order, method, **p):
    """u'' + beta^2 u = 0 on [0, length], u(0) = 1, absorbing boundary at z = length. Returns (relative L2 error, extra unknowns)."""
    h = length / n
    cells = [(i * h, h) for i in range(n)]
    one = lambda z: 1.0
    s = one
    if method == "pml":
        layers, sigma, exponent = p["layers"], p["sigma"], p.get("exponent", 2)
        thickness = p.get("thickness", layers * h)
        hp = thickness / layers
        # as in the solver: the first cell layer of the PML is not stretched
        def s(z):
            t = (z - length - hp) / max(thickness - hp, 1e-300)
            return 1.0 + 1j * sigma * max(t, 0.0) ** exponent if layers > 1 else 1.0 + 1j * sigma
        cells += [(length + i * hp, hp) for i in range(layers)]
    n_fe = len(cells) * order + 1
    n_extra = 0
    if method == "hsie":
        n_extra = p["degree"] + 1
    A = np.zeros((n_fe + n_extra, n_fe + n_extra), dtype=complex)
    for c, (z0, hc) in enumerate(cells):
        K, M = element_matrices(order, hc, lambda z: 1.0 / s(z), lambda z: s(z), z0)
        idx = np.arange(c * order, c * order + order + 1)
        A[np.ix_(idx, idx)] += K - beta ** 2 * M
    if method == "hsie":
        N, kappa0 = p["degree"], p["kappa0"]
        Tp = np.zeros((N + 2, N + 2), dtype=complex); Tm = np.zeros_like(Tp)
        for j in range(N + 2):
            # c_j = (previous +/- a_j) / 2 with previous = u0 for j = 0 and a_{j-1} otherwise, a_{N+1} = 0
            Tp[j, j] += 0.5; Tm[j, j] += 0.5
            if j + 1 <= N + 1:
                Tp[j, j + 1] += 0.5; Tm[j, j + 1] -= 0.5
        E = -2j * kappa0 * (Tp.T @ Tp) - beta ** 2 * (2j / kappa0) * (Tm.T @ Tm)
        idx = np.concatenate(([n * order], np.arange(n_fe, n_fe + n_extra)))
        A[np.ix_(idx, idx)] += E
    b = np.zeros(A.shape[0], dtype=complex)
    free = np.ones(A.shape[0], dtype=bool); free[0] = False
    if method == "pml":
        free[n_fe - 1] = False          # PEC at the end of the PML
        n_extra = len(cells) * order - n * order - 1
    u = np.zeros(A.shape[0], dtype=complex); u[0] = 1.0
    b -= A[:, 0] * 1.0
    u[free] = np.linalg.solve(A[np.ix_(free, free)], b[free])
    # L2 error in the interior
    q, w = gauss(order + 3); nodes = np.linspace(0, 1, order + 1)
    err = 0.0; norm = 0.0
    for c in range(n):
        idx = np.arange(c * order, c * order + order + 1)
        for xi, wi in zip(q, w):
            phi = np.array([np.prod([(xi - nodes[j]) / (nodes[i] - nodes[j]) for j in range(order + 1) if j != i]) for i in range(order + 1)])
            z = (c + xi) * h
            exact = np.exp(1j * beta * z)
            err += wi * h * abs(u[idx] @ phi - exact) ** 2; norm += wi * h * abs(exact) ** 2
    return np.sqrt(err / norm), n_extra
