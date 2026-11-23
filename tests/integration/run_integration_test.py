#!/usr/bin/env python3
"""Integration tests for the waveguide solver.

Every test runs the real executable with MPI on a small problem and checks the numbers in the
file results.txt, which the solver writes into its output folder.

Usage:
    run_integration_test.py --main <path to main> --mpiexec <path to mpiexec> <test name>
    run_integration_test.py --list
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

REPOSITORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STRAIGHT_CASE = """subsection Case parameters
    set source type = 3
    set fem order = 0
    set geometry size x = 4.0
    set geometry size y = 3.6
    set geometry size z = 6.0
    set epsilon in = 2.3409
    set epsilon out = 1.8496
    set Boundary Method = PML
    set PML n layers = 6
    set PML thickness = 0.5
    set PML sigma max = 10
    set PML skaling order = 2
    set Input Signal Method = {coupling}
{extra_case}end
"""
RUN = """subsection Run parameters
    set sweeping level = {level}
    set processes in x = {px}
    set processes in y = {py}
    set processes in z = {pz}
    set cell count x = {cx}
    set cell count y = {cy}
    set cell count z = {cz}
    set solve directly = {direct}
    set local solver compression tolerance = {compression}
    set solver precision = {precision}
    set Logging Level = Production One
end
"""


class Runner:
    def __init__(self, main, mpiexec, keep):
        self.main = main
        self.mpiexec = mpiexec
        self.keep = keep
        self.directory = tempfile.mkdtemp(prefix="waveguide_test_")
        self.counter = 0

    def solve(self, coupling="Dirichlet", level=1, processes=(1, 1, 2), cells=(16, 16, 8), direct=False,
              compression=0, precision=1e-6, extra_case="", results_file="results.txt"):
        """Runs the solver and returns the content of results.txt as a dictionary."""
        self.counter += 1
        work = os.path.join(self.directory, "solve%d" % self.counter)
        os.makedirs(work)
        case_file = os.path.join(work, "case.prm")
        run_file = os.path.join(work, "run.prm")
        with open(case_file, "w") as f:
            f.write(STRAIGHT_CASE.format(coupling=coupling, extra_case=extra_case))
        with open(run_file, "w") as f:
            f.write(RUN.format(level=level, px=processes[0], py=processes[1], pz=processes[2], cx=cells[0],
                               cy=cells[1], cz=cells[2], direct="true" if direct else "false",
                               compression=compression, precision=precision))
        environment = dict(os.environ, WAVEGUIDE_ROOT=REPOSITORY, WAVEGUIDE_OUTPUT_DIR=os.path.join(work, "out"))
        n_processes = processes[0] * processes[1] * processes[2]
        command = [self.mpiexec, "-np", str(n_processes), self.main, "--case", case_file, "--run", run_file]
        completed = subprocess.run(command, env=environment, cwd=work, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, universal_newlines=True)
        if completed.returncode != 0:
            print(completed.stdout[-4000:])
            fail("The solver exited with code %d." % completed.returncode)
        results = {}
        self.last_output = os.path.join(work, "out", "run0")
        with open(os.path.join(work, "out", "run0", results_file)) as f:
            for line in f:
                key, value = line.split("=")
                results[key.strip()] = float(value)
        print("   ", " ".join(command[1:3]), "level", level, "direct" if direct else "sweeping", "->", results)
        return results

    def cleanup(self):
        if self.keep:
            print("Outputs were kept in", self.directory)
        else:
            shutil.rmtree(self.directory, ignore_errors=True)


def fail(message):
    print("FAILED:", message)
    sys.exit(1)


def check(condition, message):
    if not condition:
        fail(message)


def signal(results):
    return complex(results["signal_strength_real"], results["signal_strength_imag"])


def check_same_solution(a, b, tolerance, what):
    difference = abs(signal(a) - signal(b))
    check(difference < tolerance, "%s: the transmitted amplitudes differ by %.3e (allowed %.1e)." % (what, difference, tolerance))


# ---------------------------------------------------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------------------------------------------------

def test_straight_waveguide_dirichlet(runner):
    """The fundamental mode has to pass a straight waveguide without loss (Dirichlet coupling)."""
    r = runner.solve(coupling="Dirichlet", cells=(20, 20, 20))
    check(0.95 < r["signal_strength_abs"] < 1.05, "Signal strength %.4f is not close to 1." % r["signal_strength_abs"])
    check(r["solver_iterations"] <= 8, "GMRES needed %d steps (expected at most 8)." % r["solver_iterations"])


def test_straight_waveguide_taper(runner):
    """The same for the tapered right hand side. This is the configuration that is sensitive to the sign of the
    longitudinal component of the mode and to the correct source term."""
    r = runner.solve(coupling="Taper", cells=(20, 20, 20))
    check(0.90 < r["signal_strength_abs"] < 1.05, "Signal strength %.4f is not close to 1." % r["signal_strength_abs"])
    check(r["solver_iterations"] <= 8, "GMRES needed %d steps (expected at most 8)." % r["solver_iterations"])


def test_mesh_refinement_improves_transmission(runner):
    """With a finer mesh the transmitted amplitude has to approach 1 and the error has to decrease."""
    coarse = runner.solve(coupling="Taper", cells=(12, 12, 12))
    fine = runner.solve(coupling="Taper", cells=(20, 20, 20))
    check(abs(1 - fine["signal_strength_abs"]) < abs(1 - coarse["signal_strength_abs"]), "No improvement of the signal strength.")
    check(fine["error_l2"] < 0.6 * coarse["error_l2"], "The L2 error did not decrease as expected.")


def test_reproducibility(runner):
    """Two identical runs have to give identical results (no dependence on uninitialized or random data)."""
    a = runner.solve(coupling="Taper", cells=(12, 12, 12))
    b = runner.solve(coupling="Taper", cells=(12, 12, 12))
    for key in a:
        if "time" in key:
            continue  # wall times differ between runs
        check(a[key] == b[key], "Two identical runs gave different values for %s: %r and %r." % (key, a[key], b[key]))


def test_sweeping_matches_direct_solver(runner):
    """GMRES with the sweeping preconditioner has to converge to the solution of the global linear system."""
    direct = runner.solve(direct=True)
    sweeping = runner.solve(direct=False)
    check_same_solution(direct, sweeping, 1e-6, "level 1 sweeping against direct solver")
    check(sweeping["solver_iterations"] <= 8, "GMRES needed %d steps (expected at most 8)." % sweeping["solver_iterations"])


def test_four_subdomains(runner):
    """More subdomains in the sweeping direction must not change the solution."""
    reference = runner.solve(processes=(1, 1, 2), cells=(16, 16, 8))
    four = runner.solve(processes=(1, 1, 4), cells=(16, 16, 4))
    check_same_solution(reference, four, 1e-6, "2 against 4 subdomains")


def test_hierarchical_level_2(runner):
    """Sweeping in z with a y-sweep as subdomain solver has to give the solution of the same global system."""
    direct = runner.solve(direct=True)
    level2 = runner.solve(level=2, processes=(1, 2, 2), cells=(16, 8, 8))
    check_same_solution(direct, level2, 1e-6, "level 2 sweeping against direct solver")


def test_hierarchical_level_3(runner):
    """All three levels (z, y and x sweeps)."""
    direct = runner.solve(direct=True)
    level3 = runner.solve(level=3, processes=(2, 2, 2), cells=(8, 8, 8))
    check_same_solution(direct, level3, 1e-6, "level 3 sweeping against direct solver")


def test_partition_independence(runner):
    """The assembled global system must not depend on the distribution to processes."""
    a = runner.solve(direct=True, processes=(1, 1, 2), cells=(16, 16, 8))
    b = runner.solve(direct=True, level=2, processes=(1, 2, 2), cells=(16, 8, 8))
    check_same_solution(a, b, 1e-8, "direct solves for two partitions")
    check(a["n_dofs"] == b["n_dofs"], "The number of degrees of freedom depends on the partition.")
    check(abs(a["error_l2"] - b["error_l2"]) < 1e-6 * a["error_l2"], "The L2 error depends on the partition.")


def test_block_low_rank_compression(runner):
    """Compressed (BLR) subdomain factorizations must give the same solution with fewer stored factor entries."""
    exact = runner.solve(coupling="Taper", cells=(20, 20, 20))
    compressed = runner.solve(coupling="Taper", cells=(20, 20, 20), compression=1e-4)
    check_same_solution(exact, compressed, 1e-6, "exact against compressed subdomain factorizations")
    check(exact["local_factor_entries_stored"] == exact["local_factor_entries_full_rank"], "Compression is active without being requested.")
    check(compressed["local_factor_entries_stored"] < 0.9 * compressed["local_factor_entries_full_rank"], "The factors were not compressed.")
    check(compressed["solver_iterations"] <= exact["solver_iterations"] + 2, "Compression costs too many GMRES steps.")


def test_shape_gradient(runner):
    """The adjoint based shape gradient has to agree with central finite differences of the loss functional."""
    case = ("    set perform optimization = true\n    set perform gradient check = true\n"
            "    set gradient check step width = 0.02\n    set vertical waveguide displacement = 0.5\n")
    r = runner.solve(coupling="Taper", processes=(1, 1, 4), cells=(16, 16, 12), precision=1e-8, extra_case=case,
                     results_file="gradient_check.txt")
    n_parameters = len([key for key in r if key.startswith("adjoint_gradient_")])
    check(n_parameters == 3, "Expected 3 shape parameters, found %d." % n_parameters)
    for i in range(n_parameters):
        adjoint, difference = r["adjoint_gradient_%d" % i], r["finite_difference_%d" % i]
        check(abs(difference) > 1e-3, "The finite difference %d is too small to be meaningful." % i)
        check(abs(adjoint / difference - 1) < 0.01, "Shape parameter %d: adjoint gradient %.5f, finite difference %.5f." % (i, adjoint, difference))


def test_optimization_recovers_straight_waveguide(runner):
    """Starting from a perturbed waveguide, the shape optimization has to find the straight one (the known optimum)."""
    straight = runner.solve(coupling="Taper", processes=(1, 1, 4), cells=(12, 12, 12), precision=1e-8)
    case = ("    set perform optimization = true\n    set vertical waveguide displacement = 0\n"
            "    set Initial shape parameters = 0.3, 0.3, 0\n    set optimization maximum evaluations = 25\n"
            "    set optimization maximum steps = 25\n    set optimization gradient tolerance = 1e-3\n"
            "    set optimization loss tolerance = 1e-9\n")
    r = runner.solve(coupling="Taper", processes=(1, 1, 4), cells=(12, 12, 12), precision=1e-8, extra_case=case)
    with open(os.path.join(runner.last_output, "optimization_shapes.txt")) as f:
        shapes = [[float(value) for value in line.split()] for line in f if not line.startswith("#")][1:]
    with open(os.path.join(runner.last_output, "optimization_history.txt")) as f:
        losses = [float(line.split()[1]) for line in f if not line.startswith("#")]
    displacements = [max(abs(value) for value in shape) for shape in shapes]
    best = losses.index(min(losses))
    print("    evaluations: %d, largest displacement: initial %.3f, best shape %.2e" % (len(losses), displacements[0], displacements[best]))
    check(abs(displacements[0] - 0.36) < 1e-6, "The initial shape is not the prescribed one (largest displacement %.4f)." % displacements[0])
    check(len(losses) < 25, "The optimization did not converge within 25 evaluations.")
    check(displacements[best] < 5e-3, "The best shape is %.2e um away from the straight waveguide." % displacements[best])
    check(r["signal_strength_abs"] > straight["signal_strength_abs"] - 1e-5,
          "Signal of the best shape %.6f, of the straight waveguide %.6f." % (r["signal_strength_abs"], straight["signal_strength_abs"]))
    check(r["signal_strength_abs"] < straight["signal_strength_abs"] + 1e-6, "The optimized shape is better than the straight waveguide.")


def mode_solver(runner):
    """The Python package modesolver and a folder for mode files. The test is skipped if numpy or scipy is missing."""
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        print("SKIPPED: numpy and scipy are needed for the mode solver.")
        sys.exit(77)
    sys.path.insert(0, REPOSITORY)
    import modesolver
    return modesolver, runner.directory


def test_mode_solver_circular_waveguide(runner):
    """Modes computed analytically for a circular core (step-index fibre) are used as input of a straight waveguide
    with a circular cross-section: the fundamental mode HE11 and the higher mode TE01 have to pass without loss,
    and the propagation constant has to come from the mode file."""
    modesolver, folder = mode_solver(runner)
    modes = {mode.label: mode for mode in modesolver.fibre_modes(3.0)}
    for label, wrong_label in (("HE11 (x)", "TE01"), ("TE01", "HE11 (x)")):
        path = os.path.join(folder, label.split()[0] + ".dat")
        modesolver.write_mode_file(modes[label], path, (-3, 3), (-3, 3), 0.05)
        case = "    set mode file = %s\n    set waveguide cross section = circular\n    set width of waveguide = 3.0\n" % path
        r = runner.solve(coupling="Dirichlet", cells=(16, 16, 20), extra_case=case)
        check(0.97 < r["signal_strength_abs"] < 1.03, "%s: signal strength %.4f is not close to 1." % (label, r["signal_strength_abs"]))
        # "epsilon effective" of the wrong mode: the file has to overrule it.
        wrong = "    set epsilon effective = %.8f\n" % modes[wrong_label].epsilon_effective
        check_same_solution(r, runner.solve(coupling="Dirichlet", cells=(16, 16, 20), extra_case=case + wrong), 1e-12,
                            "%s with and without a wrong 'epsilon effective'" % label)
    # The cross-section matters: TE01 of the fibre is not a mode of the rectangular core.
    rectangular = runner.solve(coupling="Dirichlet", cells=(16, 16, 20), extra_case="    set mode file = %s\n" % path)
    check(rectangular["signal_strength_abs"] < r["signal_strength_abs"] - 0.03,
          "TE01 of the fibre passed the rectangular core as well as the circular one (%.4f, %.4f)." % (rectangular["signal_strength_abs"], r["signal_strength_abs"]))


def test_mode_solver_file_cross_section(runner):
    """A mode computed numerically for a structure that the solver does not know (two coupled cores) is used
    together with the permittivity stored in the mode file. It has to pass without loss; in the default
    rectangular core it must not. For the rectangular core the permittivity from the file has to give the same
    result as the built-in cross-section."""
    modesolver, folder = mode_solver(runner)
    structure = modesolver.coupled(1.0, 1.0, 0.5)
    mode = modesolver.solve_fd(structure, 1.55, (-3, 3), (-2.5, 2.5), step=0.05, n_modes=1)[0]
    path = os.path.join(folder, "coupled.dat")
    modesolver.write_mode_file(mode, path, (-3, 3), (-2.5, 2.5), 0.05)
    case = "    set mode file = %s\n" % path
    from_file = runner.solve(coupling="Dirichlet", cells=(16, 16, 20), extra_case=case + "    set waveguide cross section = mode file\n")
    check(0.97 < from_file["signal_strength_abs"] < 1.03, "Coupled cores: signal strength %.4f is not close to 1." % from_file["signal_strength_abs"])
    rectangular = runner.solve(coupling="Dirichlet", cells=(16, 16, 20), extra_case=case)
    check(rectangular["signal_strength_abs"] < 0.9, "The supermode of two cores passed a single rectangular core (%.4f)." % rectangular["signal_strength_abs"])

    mode = modesolver.solve_fd(modesolver.rectangular(), 1.55, (-3, 3), (-2.5, 2.5), step=0.05, n_modes=1)[0]
    path = os.path.join(folder, "rectangular.dat")
    modesolver.write_mode_file(mode, path, (-3, 3), (-2.5, 2.5), 0.05)
    case = "    set mode file = %s\n" % path
    built_in = runner.solve(coupling="Taper", cells=(16, 16, 20), extra_case=case)
    from_file = runner.solve(coupling="Taper", cells=(16, 16, 20), extra_case=case + "    set waveguide cross section = mode file\n")
    check_same_solution(built_in, from_file, 1e-10, "rectangular core: built-in cross-section against the permittivity of the mode file")
    # The computed mode has to do as well as the mode file of the dissertation.
    shipped = runner.solve(coupling="Taper", cells=(16, 16, 20))
    check(abs(1 - built_in["signal_strength_abs"]) < abs(1 - shipped["signal_strength_abs"]) + 0.01,
          "The computed mode (%.4f) transmits worse than the mode file of the dissertation (%.4f)." % (built_in["signal_strength_abs"], shipped["signal_strength_abs"]))


def test_run_report(runner):
    """The run report (text and JSON) has to exist and to be consistent with results.txt and the field outputs."""
    import glob
    import json
    r = runner.solve(coupling="Taper", level=2, processes=(1, 2, 2), cells=(12, 6, 8))
    output = runner.last_output
    with open(os.path.join(output, "run_report.txt")) as f:
        text = f.read()
    with open(os.path.join(output, "run_report.json")) as f:
        report = json.load(f)
    sections = {section["title"]: section for section in report["sections"]}
    for title in ("Run", "Geometry", "Mesh", "Unknowns", "Solution process", "Result"):
        check(title in sections, "The section '%s' is missing in run_report.json." % title)
        check(title in text, "The section '%s' is missing in run_report.txt." % title)
    # Unknowns: the table of the hierarchy has one row per level, and interior + boundary = all for the global problem.
    levels = sections["Unknowns"]["tables"][0]["rows"]
    check([row[0] for row in levels] == ["2", "1", "0"], "Expected the levels 2, 1, 0 in the table of unknowns.")
    total, interior, boundary = (int(levels[0][column].replace(",", "")) for column in (4, 5, 6))
    check(total == r["n_dofs"], "The report states %d unknowns, results.txt %d." % (total, r["n_dofs"]))
    check(interior > 0 and boundary > 0 and interior + boundary == total, "Interior and boundary unknowns do not add up.")
    sides = sections["Unknowns"]["tables"][1]["rows"]
    check(sum(int(row[2].replace(",", "")) for row in sides) == boundary, "The unknowns by side do not add up to the boundary unknowns.")
    # Solver statistics: one row per sweeping level and the residual history of the global solve.
    solvers = sections["Solution process"]["tables"][0]["rows"]
    check([row[0] for row in solvers] == ["2", "1"], "Expected the solver statistics of the levels 2 and 1.")
    top = [series["values"] for series in report["series"] if series["name"] == "residuals_level_2"]
    check(len(top) == 1 and len(top[0]) == r["solver_iterations"] + 1, "The residual history of the global solve is incomplete.")
    check(abs(report["data"]["signal_strength_abs"] - r["signal_strength_abs"]) < 1e-9, "The signal in the report differs from results.txt.")
    # Signal along z: summed over the processes, the value on the output side is the signal of results.txt.
    planes = {}
    for path in glob.glob(os.path.join(output, "signal_along_z*.txt")):
        with open(path) as f:
            for line in f:
                if not line.startswith("#"):
                    values = [float(value) for value in line.split()]
                    sums = planes.setdefault(round(values[0], 6), [0.0, 0.0, 0.0])
                    for i in range(3):
                        sums[i] += values[1 + i]
    check(len(planes) == 2 * 8 + 1, "Expected 17 planes in signal_along_z, found %d." % len(planes))
    last = planes[max(planes)]
    at_output = abs(complex(last[0], last[1]) / last[2])
    check(abs(at_output - r["signal_strength_abs"]) < 2e-3, "Signal along z at the output %.5f, results.txt %.5f." % (at_output, r["signal_strength_abs"]))


METALLIC_WAVEGUIDE = """    set source type = 4
    set Lateral boundary = PEC
    set geometry size x = 2.0
    set geometry size y = 1.0
    set geometry size z = 2.0
    set epsilon in = 1.0
    set epsilon out = 1.0
    set Boundary Method = HSIE
    set Kappa 0 Imaginary = 0
"""


def test_hsie_single_face(runner):
    """Hardy space infinite elements on the output face must absorb the TE10 mode of a metallic waveguide (exact
    solution known): the error has to fall with the mesh width and with the order of the elements. With kappa_0 equal
    to the wavenumber a low degree of the infinite elements suffices; with a poor kappa_0 the degree has to matter."""
    def error(order, cells, degree, kappa_0=4.0537):
        extra = METALLIC_WAVEGUIDE + "    set fem order = %d\n    set HSIE polynomial degree = %d\n    set Kappa 0 Real = %g\n" % (order, degree, kappa_0)
        return runner.solve(level=0, processes=(1, 1, 1), cells=(cells,) * 3, extra_case=extra)["error_l2_relative"]
    coarse, fine = error(0, 8, 4), error(0, 16, 4)
    check(fine < 0.06, "Lowest order elements, 16 cells: relative error %.3f (expected about 0.044)." % fine)
    check(fine < 0.4 * coarse, "The error does not fall with the mesh width (%.3f -> %.3f)." % (coarse, fine))
    second_order = error(1, 8, 4)
    check(second_order < 0.006, "Second order elements, 8 cells: relative error %.4f (expected about 0.0046)." % second_order)
    check(abs(error(1, 8, 8) - second_order) < 1e-4, "The error depends on the degree of the infinite elements.")
    low, high = error(0, 16, 0, kappa_0=1.6), error(0, 16, 8, kappa_0=1.6)
    check(low > fine + 0.005, "Degree 0 with a poor kappa_0 should be visibly less accurate (%.4f against %.4f)." % (low, fine))
    check(abs(high - fine) < 1e-4, "Degree 8 with a poor kappa_0 should reach the accuracy of the mesh (%.4f against %.4f)." % (high, fine))


def test_hsie_matches_pml(runner):
    """Dielectric waveguide with absorbing boundaries on five faces: infinite elements and PML must give the same
    error against the exact mode, also for second order elements (where the functions on the side faces differ from
    those of the surface mesh by more than a sign)."""
    hsie = "    set Boundary Method = HSIE\n    set HSIE polynomial degree = 4\n    set Kappa 0 Real = 4.0537\n    set Kappa 0 Imaginary = 0\n"
    for order, cells, tolerance in ((0, (16, 16, 24), 0.02), (1, (8, 8, 12), 0.02)):
        fem = "    set fem order = %d\n" % order
        pml = runner.solve(level=0, processes=(1, 1, 1), cells=cells, extra_case=fem)["error_l2_relative"]
        infinite = runner.solve(level=0, processes=(1, 1, 1), cells=cells, extra_case=fem + hsie)["error_l2_relative"]
        check(abs(pml - infinite) < tolerance, "Element order %d: error %.4f with PML but %.4f with infinite elements." % (order + 1, pml, infinite))


TESTS = {name[5:]: function for name, function in sorted(globals().items()) if name.startswith("test_")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test", nargs="?", help="name of the test to run")
    parser.add_argument("--main", help="path of the solver executable")
    parser.add_argument("--mpiexec", default="mpiexec", help="MPI launcher")
    parser.add_argument("--keep", action="store_true", help="keep the output folders")
    parser.add_argument("--list", action="store_true", help="list the available tests")
    arguments = parser.parse_args()
    if arguments.list or arguments.test is None:
        for name, function in TESTS.items():
            print("%-40s %s" % (name, function.__doc__.split("\n")[0]))
        return
    if arguments.test not in TESTS:
        fail("Unknown test '%s'." % arguments.test)
    runner = Runner(os.path.abspath(arguments.main), arguments.mpiexec, arguments.keep)
    try:
        print("Running", arguments.test)
        TESTS[arguments.test](runner)
    finally:
        runner.cleanup()
    print("PASSED")


if __name__ == "__main__":
    main()
