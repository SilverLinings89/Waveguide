"""Tests of the frontend server that do not need the solver: configuration handling and output parsing."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server


class ConfigurationTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        config = server.validate({})
        self.assertEqual(config, server.DEFAULTS)

    def test_unknown_and_invalid_settings_are_rejected(self):
        for raw in ({"no_such_setting": 1}, {"length": "long"}, {"length": -1}, {"core_width": 9},
                    {"epsilon_core": 1.5, "epsilon_cladding": 2.0}, {"shape": "spiral"},
                    {"boundary_method": "HSIE", "processes_z": 2}):
            with self.assertRaises(server.ConfigurationError):
                server.validate(raw)

    def test_sweeping_level_follows_the_process_layout(self):
        for processes, level in (((1, 1, 1), 0), ((1, 1, 4), 1), ((1, 2, 4), 2), ((2, 2, 4), 3)):
            config = server.validate({"processes_x": processes[0], "processes_y": processes[1], "processes_z": processes[2]})
            self.assertEqual(server.derive(config)["sweeping_level"], level)

    def test_mesh_reaches_the_requested_resolution(self):
        config = server.validate({"cells_per_wavelength": 10, "processes_z": 3})
        derived = server.derive(config)
        self.assertTrue(all(value >= 10 - 1e-9 for value in derived["cells_per_wavelength"]))
        self.assertEqual(derived["cells"][2], derived["cells_per_process"][2] * 3)

    def test_parameter_files(self):
        config = server.validate({"shape": "hump", "hump_height": 0.4, "processes_y": 2, "processes_z": 2, "coupling": "Taper"})
        case, run = server.parameter_files(config)
        self.assertIn("set vertical waveguide displacement = 0.4", case)
        self.assertIn("set Input Signal Method = Taper", case)
        self.assertIn("set sweeping level = 2", run)
        self.assertIn("set use relative convergence criterion = true", run)
        self.assertTrue(case.startswith("subsection Case parameters") and case.rstrip().endswith("end"))

    def test_predefined_shapes_have_a_fixed_length(self):
        self.assertEqual(server.validate({"shape": "predefined", "length": 5})["length"], 100.0)


class OutputParsingTests(unittest.TestCase):
    def test_progress_separates_solver_levels(self):
        stdout = "\n".join([
            "x: Mesh Preparation finished. System has 10 degrees of freedom.",
            "Residual in step 1 : 6.000000",
            "\t[0]: Residual in step 1 : 1.000000",
            "\t[0]: Residual in step 2 : 0.01 Factor: 0.01",
            "Residual in step 2 : 0.5 Factor: 0.08",
            "\t[0]: Residual in step 1 : 0.900000",
        ])
        progress = server.parse_progress(stdout)
        self.assertEqual(progress["phase"], "Solving")
        self.assertEqual(progress["levels"][0]["histories"], [[6.0, 0.5]])
        self.assertEqual(progress["levels"][1]["histories"], [[1.0, 0.01], [0.9]])
        self.assertEqual(progress["levels"][1]["solves"], 2)

    def test_field_slices_of_two_processes_are_merged(self):
        with tempfile.TemporaryDirectory() as directory:
            header = "# x y z x_physical y_physical z_physical re_Ex im_Ex re_Ey im_Ey re_Ez im_Ez in_core\n"
            for rank, z in ((0, 0.5), (1, 1.5)):
                with open(os.path.join(directory, "field_slice_x%d.txt" % rank), "w") as f:
                    f.write(header)
                    for y in (-0.5, 0.5):
                        f.write("0.1 %g %g 0.1 %g %g %g 0 0 0 0 0 1\n" % (y, z, y + 0.25, z, 10 * z + y))
            field = server.read_field(directory, "x")
        self.assertEqual(field["axes"], ["y", "z"])
        self.assertEqual(field["u"], [-0.5, 0.5])
        self.assertEqual(field["v"], [0.5, 1.5])
        self.assertEqual(field["field"]["re_x"], [[4.5, 5.5], [14.5, 15.5]])
        self.assertEqual(field["physical_u"][0], [-0.25, 0.75])
        self.assertIsNone(server.read_field(directory, "y"))


if __name__ == "__main__":
    unittest.main()
