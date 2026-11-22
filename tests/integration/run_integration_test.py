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
