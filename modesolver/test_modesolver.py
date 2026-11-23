"""Tests of the mode solver: python3 modesolver/test_modesolver.py (does not need the solver executable).

The analytical solutions are checked against Maxwell's equations and known limits, the finite difference solver
against the analytical solutions, and both against the mode file of the dissertation.
"""
import os
import sys
import tempfile
import unittest

try:
    import numpy as np
    import scipy  # noqa: F401
except ImportError:
    print("numpy and scipy are needed for the mode solver; the tests are skipped.")
    sys.exit(77)

REPOSITORY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPOSITORY)
import modesolver as ms
from modesolver.mode import overlap
from modesolver import __main__ as command_line

WAVELENGTH = 1.55
K0 = 2 * np.pi / WAVELENGTH
EPS_CORE, EPS_CLADDING = 2.3409, 1.8496


def maxwell_residuals(mode, epsilon, x0, y0, h=1e-3):
    """Residuals of two consequences of Maxwell's equations at a point of a homogeneous region, by central
    differences: div E = 0 and the Helmholtz equation for E_x and E_y, relative to the size of the field. The
    differences have a truncation error of about 1e-6; a wrong sign or factor in a field gives a residual of order 1."""
    def components(x, y):
        return np.array([float(c) for c in mode.field(np.array(x), np.array(y))])
    center = components(x0, y0)
    dx = (components(x0 + h, y0) - components(x0 - h, y0)) / (2 * h)
    dy = (components(x0, y0 + h) - components(x0, y0 - h)) / (2 * h)
    laplace = (components(x0 + h, y0) + components(x0 - h, y0) + components(x0, y0 + h) + components(x0, y0 - h) - 4 * center) / h ** 2
    size = np.linalg.norm(center)
    # E_z = i e_z, so div E = d_x E_x + d_y E_y + i beta (i e_z).
    divergence = dx[0] + dy[1] - mode.beta * center[2]
    helmholtz = laplace[:2] + (K0 ** 2 * epsilon - mode.beta ** 2) * center[:2]
    return abs(divergence) / (mode.beta * size), np.linalg.norm(helmholtz) / (K0 ** 2 * size)


class SlabTests(unittest.TestCase):
    def test_dispersion_relation_of_the_symmetric_slab(self):
        # Even TE modes of a symmetric slab: kappa tan(kappa d / 2) = gamma.
        table = ms.slab_mode_table(EPS_CORE, EPS_CLADDING, EPS_CLADDING, 1.0, WAVELENGTH)
        beta = table["TE"][0] * K0
        kappa, gamma = np.sqrt(K0 ** 2 * EPS_CORE - beta ** 2), np.sqrt(beta ** 2 - K0 ** 2 * EPS_CLADDING)
        self.assertAlmostEqual(kappa * np.tan(kappa * 0.5), gamma, places=9)
        beta = table["TM"][0] * K0
        kappa, gamma = np.sqrt(K0 ** 2 * EPS_CORE - beta ** 2), np.sqrt(beta ** 2 - K0 ** 2 * EPS_CLADDING)
        self.assertAlmostEqual(kappa * np.tan(kappa * 0.5), gamma * EPS_CORE / EPS_CLADDING, places=9)

    def test_number_of_modes_follows_the_normalized_frequency(self):
        for thickness in (0.5, 2.0, 5.0):
            v = K0 * thickness * np.sqrt(EPS_CORE - EPS_CLADDING)
            table = ms.slab_mode_table(EPS_CORE, EPS_CLADDING, EPS_CLADDING, thickness, WAVELENGTH)
            self.assertEqual(len(table["TE"]), int(np.floor(v / np.pi)) + 1)
            self.assertEqual(len(table["TM"]), int(np.floor(v / np.pi)) + 1)

    def test_asymmetric_slab_has_a_cut_off(self):
        self.assertEqual(ms.slab_mode_table(EPS_CORE, 1.0, EPS_CLADDING, 0.1, WAVELENGTH)["TE"], [])
        self.assertEqual(len(ms.slab_mode_table(EPS_CORE, 1.0, EPS_CLADDING, 1.0, WAVELENGTH)["TE"]), 1)

    def test_fields_fulfil_maxwell_in_every_layer(self):
        for mode in ms.slab_modes(1.5, EPS_CORE, 1.0, EPS_CLADDING, WAVELENGTH):
            for y, epsilon in ((0.2, EPS_CORE), (-1.1, EPS_CLADDING), (1.0, 1.0)):
                divergence, helmholtz = maxwell_residuals(mode, epsilon, 0.3, y)
                self.assertLess(divergence, 2e-5, mode.label)
                self.assertLess(helmholtz, 1e-4, mode.label)

    def test_tm_field_is_continuous_in_the_normal_flux(self):
        mode = [m for m in ms.slab_modes(1.0, EPS_CORE, EPS_CLADDING, None, WAVELENGTH) if m.label == "TM0"][0]
        inside = mode.field(np.array(0.0), np.array(0.5 - 1e-9))[1]
        outside = mode.field(np.array(0.0), np.array(0.5 + 1e-9))[1]
        self.assertAlmostEqual(float(EPS_CORE * inside), float(EPS_CLADDING * outside), places=6)

    def test_finite_differences_reproduce_the_te_mode(self):
        # A TE slab mode is uniform in x and has only E_x, which is compatible with conducting walls at the sides.
        exact = ms.slab_mode_table(EPS_CORE, EPS_CLADDING, EPS_CLADDING, 1.0, WAVELENGTH)["TE"][0]
        modes = ms.solve_fd(ms.slab(1.0), WAVELENGTH, (-0.5, 0.5), (-4.0, 4.0), step=(0.1, 0.02), n_modes=1, guided_only=False)
        self.assertAlmostEqual(modes[0].n_eff, exact, delta=2e-5)


class FibreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modes = ms.fibre_modes(2.0, EPS_CORE, EPS_CLADDING, WAVELENGTH)

    def test_mode_set_and_order(self):
        labels = [mode.label for mode in self.modes]
        self.assertEqual(labels[:2], ["HE11 (x)", "HE11 (y)"])
        self.assertIn("TE01", labels)
        self.assertIn("TM01", labels)
        n_eff = [mode.n_eff for mode in self.modes]
        self.assertEqual(n_eff, sorted(n_eff, reverse=True))
        self.assertTrue(all(np.sqrt(EPS_CLADDING) < value < np.sqrt(EPS_CORE) for value in n_eff))

    def test_single_mode_below_the_cut_off_frequency(self):
        # V = k0 a sqrt(eps1 - eps2) < 2.405: only the two polarizations of HE11 are guided.
        diameter = 2 * 2.3 / (K0 * np.sqrt(EPS_CORE - EPS_CLADDING))
        self.assertEqual([mode.label for mode in ms.fibre_modes(diameter, EPS_CORE, EPS_CLADDING, WAVELENGTH)], ["HE11 (x)", "HE11 (y)"])

    def test_weakly_guiding_limit(self):
        # For a small index contrast HE11 becomes LP01: u J1(u) / J0(u) = w K1(w) / K0(w).
        from scipy import special
        eps_core, eps_cladding, diameter = 1.4505 ** 2, 1.45 ** 2, 20.0
        mode = ms.fibre_modes(diameter, eps_core, eps_cladding, WAVELENGTH)[0]
        u = diameter / 2 * np.sqrt(K0 ** 2 * eps_core - mode.beta ** 2)
        w = diameter / 2 * np.sqrt(mode.beta ** 2 - K0 ** 2 * eps_cladding)
        self.assertAlmostEqual(u * special.jv(1, u) / special.jv(0, u), w * special.kv(1, w) / special.kv(0, w), delta=2e-3)

    def test_fields_fulfil_maxwell_in_core_and_cladding(self):
        for mode in self.modes:
            for (x, y), epsilon in (((0.3, 0.2), EPS_CORE), ((1.1, 0.9), EPS_CLADDING)):
                divergence, helmholtz = maxwell_residuals(mode, epsilon, x, y)
                self.assertLess(divergence, 2e-5, mode.label)
                self.assertLess(helmholtz, 1e-4, mode.label)

    def test_interface_conditions(self):
        # At the core boundary on the x-axis: E_y and E_z (tangential) and eps E_x (normal flux) are continuous.
        for mode in self.modes:
            inside = [float(c) for c in mode.field(np.array(1.0 - 1e-9), np.array(0.0))]
            outside = [float(c) for c in mode.field(np.array(1.0 + 1e-9), np.array(0.0))]
            self.assertAlmostEqual(inside[1], outside[1], places=6, msg=mode.label)
            self.assertAlmostEqual(inside[2], outside[2], places=6, msg=mode.label)
            self.assertAlmostEqual(EPS_CORE * inside[0], EPS_CLADDING * outside[0], places=6, msg=mode.label)

    def test_polarizations_are_orthogonal(self):
        grid = np.linspace(-4, 4, 161)
        self.assertLess(overlap(self.modes[0], self.modes[1], grid, grid), 1e-9)
        self.assertGreater(self.modes[0].polarization_fraction_x(grid, grid), 0.99)
        self.assertLess(self.modes[1].polarization_fraction_x(grid, grid), 0.01)

    def test_finite_differences_agree_with_the_exact_modes(self):
        numerical = ms.solve_fd(ms.circular(2.0), WAVELENGTH, step=0.05, n_modes=4)
        grid = np.linspace(-4, 4, 161)
        self.assertAlmostEqual(numerical[0].n_eff, self.modes[0].n_eff, delta=1e-4)
        self.assertAlmostEqual(numerical[1].n_eff, self.modes[1].n_eff, delta=1e-4)
        # The two polarizations are degenerate; the solver may return any combination of them.
        total = sum(overlap(numerical[0], exact, grid, grid) ** 2 for exact in self.modes[:2])
        self.assertGreater(total, 0.999)
        exact = {mode.label: mode.n_eff for mode in self.modes}
        self.assertAlmostEqual(numerical[2].n_eff, exact["TE01"], delta=3e-4)
        self.assertAlmostEqual(numerical[3].n_eff, exact["TM01"], delta=3e-4)


class MetallicWaveguideTests(unittest.TestCase):
    def test_rectangular_propagation_constants(self):
        modes = {mode.label: mode for mode in ms.metallic_rectangular_modes(2.0, 1.0, WAVELENGTH)}
        self.assertAlmostEqual(modes["TE10"].beta, np.sqrt(K0 ** 2 - (np.pi / 2.0) ** 2), places=12)
        self.assertAlmostEqual(modes["TE11"].beta, modes["TM11"].beta, places=12)
        self.assertNotIn("TM10", modes)

    def test_rectangular_fields_fulfil_maxwell_and_the_wall_condition(self):
        for mode in ms.metallic_rectangular_modes(2.0, 1.0, WAVELENGTH):
            divergence, helmholtz = maxwell_residuals(mode, 1.0, 0.31, 0.17)
            self.assertLess(divergence, 2e-5, mode.label)
            self.assertLess(helmholtz, 1e-4, mode.label)
            # Tangential E on the wall x = width / 2 is E_y and E_z, on y = height / 2 it is E_x and E_z.
            side = [float(c) for c in mode.field(np.array(1.0 - 1e-12), np.array(0.2))]
            top = [float(c) for c in mode.field(np.array(0.3), np.array(0.5 - 1e-12))]
            self.assertLess(max(abs(side[1]), abs(side[2]), abs(top[0]), abs(top[2])), 1e-9, mode.label)

    def test_circular_propagation_constants_and_wall_condition(self):
        from scipy import special
        modes = ms.metallic_circular_modes(2.0, WAVELENGTH)
        self.assertEqual(modes[0].label, "TE11 (even)")   # the fundamental mode of the circular waveguide
        self.assertAlmostEqual(modes[0].beta, np.sqrt(K0 ** 2 - special.jnp_zeros(1, 1)[0] ** 2), places=12)
        for mode in modes:
            divergence, helmholtz = maxwell_residuals(mode, 1.0, 0.31, 0.17)
            self.assertLess(divergence, 2e-5, mode.label)
            self.assertLess(helmholtz, 1e-4, mode.label)
            # On the wall at (radius, 0) the tangential components are E_y and E_z.
            wall = [float(c) for c in mode.field(np.array(1.0 - 1e-12), np.array(0.0))]
            self.assertLess(max(abs(wall[1]), abs(wall[2])), 1e-9, mode.label)

    def test_finite_differences_reproduce_the_rectangular_waveguide(self):
        exact = ms.metallic_rectangular_modes(2.0, 1.0, WAVELENGTH)
        coarse = ms.solve_fd(ms.homogeneous(1.0), WAVELENGTH, (-1, 1), (-0.5, 0.5), step=0.05, n_modes=1, guided_only=False)[0]
        fine = ms.solve_fd(ms.homogeneous(1.0), WAVELENGTH, (-1, 1), (-0.5, 0.5), step=0.025, n_modes=1, guided_only=False)[0]
        error_coarse, error_fine = abs(coarse.n_eff - exact[0].n_eff), abs(fine.n_eff - exact[0].n_eff)
        self.assertLess(error_fine, 2e-5)
        self.assertAlmostEqual(error_coarse / error_fine, 4.0, delta=0.3)   # second order
        x, y = np.linspace(-0.95, 0.95, 39), np.linspace(-0.45, 0.45, 19)
        self.assertGreater(overlap(fine, exact[0], x, y), 0.9999)


class RectangularCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modes = ms.solve_fd(ms.rectangular(), WAVELENGTH, step=0.05, n_modes=4)
        cls.reference = ms.read_mode_file(os.path.join(REPOSITORY, "Modes", "output_mode.dat"))

    def test_fundamental_modes(self):
        grid = np.linspace(-3, 3, 61)
        self.assertGreater(self.modes[0].polarization_fraction_x(grid, grid), 0.99)   # the core is wider than high
        self.assertLess(self.modes[1].polarization_fraction_x(grid, grid), 0.01)
        self.assertGreater(self.modes[0].n_eff, self.modes[1].n_eff)

    def test_effective_index_of_the_dissertation(self):
        # "epsilon effective" = 2.1588449 belongs to the mode file of the dissertation.
        self.assertAlmostEqual(self.modes[0].n_eff, np.sqrt(2.1588449), delta=5e-4)

    def test_field_agrees_with_the_mode_file_of_the_dissertation(self):
        reference = self.reference
        ex, ey, ez = self.modes[0].sample(reference["x"], reference["y"])
        # The stored longitudinal component is -e_z.
        inner = np.sum(ex * reference["ex"]) + np.sum(ey * reference["ey"]) + np.sum(ez * -reference["s"])
        norm = np.sqrt(np.sum(ex ** 2 + ey ** 2 + ez ** 2) * np.sum(reference["ex"] ** 2 + reference["ey"] ** 2 + reference["s"] ** 2))
        self.assertGreater(inner / norm, 0.998)
        self.assertGreater(np.sum(ez * -reference["s"]), 0)   # the sign of the longitudinal component

    def test_grid_convergence(self):
        coarse = ms.solve_fd(ms.rectangular(), WAVELENGTH, step=0.1, n_modes=1)[0]
        self.assertAlmostEqual(coarse.n_eff, self.modes[0].n_eff, delta=5e-4)

    def test_marcatili_is_close_far_from_cut_off(self):
        approximate = ms.marcatili_modes()
        self.assertEqual(approximate[0].label, "Ex11")
        self.assertAlmostEqual(approximate[0].n_eff, self.modes[0].n_eff, delta=2e-3)
        grid = np.linspace(-3, 3, 121)
        self.assertGreater(overlap(approximate[0], self.modes[0], grid, grid), 0.99)

    def test_square_core_is_degenerate(self):
        modes = ms.solve_fd(ms.rectangular(1.5, 1.5), WAVELENGTH, (-4, 4), (-4, 4), step=0.05, n_modes=2)
        self.assertAlmostEqual(modes[0].n_eff, modes[1].n_eff, delta=1e-7)


class OtherStructureTests(unittest.TestCase):
    def test_structures_have_the_expected_permittivity(self):
        self.assertEqual(float(ms.circular(2.0).epsilon(0.99, 0.0)), EPS_CORE)
        self.assertEqual(float(ms.circular(2.0).epsilon(0.8, 0.8)), EPS_CLADDING)
        rib = ms.rib(0.5, 0.22, 0.09)
        self.assertEqual([float(rib.epsilon(x, y)) for x, y in ((0, 0.1), (1, 0.1), (1, -0.05), (1, -0.2))], [12.11, 1.0, 12.11, 2.085])
        slot = ms.slot(0.2, 0.1, 0.22)
        self.assertEqual([float(slot.epsilon(x, 0)) for x in (0.0, 0.1, 0.3)], [2.085, 12.11, 2.085])

    def test_coupled_cores_have_an_even_and_an_odd_supermode(self):
        modes = [m for m in ms.solve_fd(ms.coupled(1.0, 1.0, 0.5), WAVELENGTH, (-3, 3), (-2.5, 2.5), step=0.05, n_modes=4)]
        grid_x, grid_y = np.linspace(-3, 3, 121), np.linspace(-2.5, 2.5, 101)
        polarized_x = [m for m in modes if m.polarization_fraction_x(grid_x, grid_y) > 0.5]
        even, odd = polarized_x[0].sample(grid_x, grid_y)[0], polarized_x[1].sample(grid_x, grid_y)[0]
        self.assertLess(np.abs(even - even[::-1, :]).max(), 1e-6 * np.abs(even).max() + 1e-9)
        self.assertLess(np.abs(odd + odd[::-1, :]).max(), 1e-6 * np.abs(odd).max() + 1e-9)
        single = ms.solve_fd(ms.rectangular(1.0, 1.0), WAVELENGTH, (-3, 3), (-2.5, 2.5), step=0.05, n_modes=1)[0]
        self.assertGreater(polarized_x[0].n_eff, single.n_eff)
        self.assertLess(polarized_x[1].n_eff, single.n_eff)

    def test_strip_and_rib_on_a_substrate(self):
        strip = ms.solve_fd(ms.strip(), WAVELENGTH, (-1.5, 1.5), (-1.0, 1.0), step=0.02, n_modes=2)
        rib = ms.solve_fd(ms.rib(), WAVELENGTH, (-2.0, 2.0), (-1.5, 1.5), step=0.02, n_modes=2)
        self.assertTrue(np.sqrt(2.085) < strip[0].n_eff < np.sqrt(12.11))
        # The layer beside the rib adds material, so the mode has a larger index than the mode of the strip, and
        # it has to be above the index of the mode of the layer alone (otherwise it would leak into the layer).
        self.assertGreater(rib[0].n_eff, strip[0].n_eff)
        layer = ms.slab_mode_table(12.11, 1.0, 2.085, 0.09, WAVELENGTH)["TE"][0]
        self.assertTrue(all(mode.n_eff > layer for mode in rib))

    def test_slot_mode_is_concentrated_in_the_gap(self):
        mode = ms.solve_fd(ms.slot(), WAVELENGTH, (-1.5, 1.5), (-1.0, 1.0), step=0.01, n_modes=1)[0]
        in_gap = abs(float(mode.field(np.array(0.0), np.array(0.0))[0]))
        in_rail = abs(float(mode.field(np.array(0.15), np.array(0.0))[0]))
        self.assertGreater(in_gap, 2 * in_rail)


class ModeFileTests(unittest.TestCase):
    def test_round_trip_and_header(self):
        mode = ms.fibre_modes(2.0, EPS_CORE, EPS_CLADDING, WAVELENGTH)[0]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "mode.dat")
            data = ms.write_mode_file(mode, path, (-3, 3), (-2, 2), 0.1)
            with open(path) as f:
                first_data_line = [line for line in f if not line.startswith("#")][1]
        self.assertAlmostEqual(float(data["header"]["beta"]), mode.beta, places=9)
        self.assertEqual(float(data["header"]["wavelength"]), WAVELENGTH)
        self.assertEqual(data["header"]["structure_type"], "circular")
        self.assertEqual((len(data["x"]), len(data["y"])), (61, 41))
        self.assertAlmostEqual(float(first_data_line.split()[0]), -2.9)   # x runs fastest
        ex, ey, ez = mode.sample(data["x"], data["y"])
        scale = data["ex"][30, 20] / ex[30, 20]
        # Grid points on the core boundary, where E_x jumps, can fall on either side after rounding.
        X, Y = np.meshgrid(data["x"], data["y"], indexing="ij")
        away = np.abs(np.hypot(X, Y) - 1.0) > 1e-6
        self.assertLess(np.abs(data["ex"] - scale * ex)[away].max(), 1e-8)
        self.assertLess(np.abs(data["s"] + scale * ez)[away].max(), 1e-8)   # stored value = -e_z
        self.assertAlmostEqual(np.sqrt(data["ex"] ** 2 + data["ey"] ** 2 + data["s"] ** 2).max(), 1.0, places=8)
        self.assertEqual(data["epsilon"][30, 20], EPS_CORE)
        self.assertEqual(data["epsilon"][0, 0], EPS_CLADDING)

    def test_old_format_is_read(self):
        data = ms.read_mode_file(os.path.join(REPOSITORY, "Modes", "output_mode.dat"))
        self.assertEqual(data["header"], {})
        self.assertIsNone(data["epsilon"])
        self.assertEqual(data["ex"].shape, (len(data["x"]), len(data["y"])))

    def test_command_line(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "mode.dat")
            modes = command_line.main(["circular", "--diameter", "2", "--polarization", "y", "--output", path, "--window", "6", "6", "--step", "0.1"])
            self.assertEqual(ms.read_mode_file(path)["header"]["mode"], "HE11 (y)")
            self.assertEqual(modes[0].label, "HE11 (x)")
            with self.assertRaises(SystemExit):
                command_line.main(["rib", "--method", "analytic"])


if __name__ == "__main__":
    unittest.main()
