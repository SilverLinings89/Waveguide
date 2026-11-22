"""Shared helpers of the examples: running the solver, reading its outputs and saving figures."""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

REPOSITORY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Parameters of the straight rectangular waveguide used in most experiments (lengths in micrometers).
STRAIGHT_WAVEGUIDE = {
    "source type": 3,
    "fem order": 0,
    "geometry size x": 4.0,
    "geometry size y": 3.6,
    "geometry size z": 4.0,
    "epsilon in": 2.3409,
    "epsilon out": 1.8496,
    "Boundary Method": "PML",
    "PML n layers": 10,
    "PML thickness": 0.5,
    "PML sigma max": 30,
    "PML skaling order": 3,
    "Input Signal Method": "Dirichlet",
}


def argument_parser(description):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", choices=["laptop", "paper"], default="laptop",
                        help="laptop: reduced set of runs that finishes on a workstation (default). "
                             "paper: the parameters of the dissertation; several examples then need a cluster.")
    parser.add_argument("--main", default=None, help="solver executable (default: build/Main/main or build-release/Main/main)")
    parser.add_argument("--mpiexec", default="mpiexec", help="MPI launcher")
    parser.add_argument("--output", default=None, help="output folder (default: examples/output/<name of the example>)")
    parser.add_argument("--plot-only", action="store_true", help="do not run the solver, only plot the data of a previous run")
    return parser


def prepare(arguments, name):
    """Resolves the executable and the output folder. Returns the output folder."""
    if arguments.output is None:
        arguments.output = os.path.join(REPOSITORY, "examples", "output", name)
    os.makedirs(arguments.output, exist_ok=True)
    if arguments.main is None:
        for candidate in ("build/Main/main", "build-release/Main/main"):
            path = os.path.join(REPOSITORY, candidate)
            if os.path.exists(path):
                arguments.main = path
        if arguments.main is None and not arguments.plot_only:
            sys.exit("The solver executable was not found. Build the code first (see Readme.md) or pass --main.")
    return arguments.output


def write_parameter_file(path, section, parameters):
    with open(path, "w") as f:
        f.write("subsection %s\n" % section)
        for key, value in parameters.items():
            if isinstance(value, bool):
                value = "true" if value else "false"
            f.write("  set %s = %s\n" % (key, value))
        f.write("end\n")


def run_solver(arguments, label, case, run):
    """Runs the solver for the given case and run parameters.

    Returns a dictionary with the entries of results.txt (if the run wrote one) and additionally
    "wall_time" (seconds, whole run), "run_directory" and "stdout".
    """
    work = os.path.join(arguments.output, "runs", label)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    run = dict(run)
    run.setdefault("Logging Level", "Production One")
    write_parameter_file(os.path.join(work, "case.prm"), "Case parameters", case)
    write_parameter_file(os.path.join(work, "run.prm"), "Run parameters", run)
    n_processes = run.get("processes in x", 1) * run.get("processes in y", 1) * run.get("processes in z", 1)
    command = [arguments.mpiexec, "-np", str(n_processes), arguments.main,
               "--case", os.path.join(work, "case.prm"), "--run", os.path.join(work, "run.prm")]
    environment = dict(os.environ, WAVEGUIDE_ROOT=REPOSITORY, WAVEGUIDE_OUTPUT_DIR=os.path.join(work, "out"))
    print("  running %-28s (%d processes) ..." % (label, n_processes), end="", flush=True)
    start = time.time()
    completed = subprocess.run(command, env=environment, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, errors="replace")
    wall_time = time.time() - start
    with open(os.path.join(work, "stdout.txt"), "w") as f:
        f.write(completed.stdout)
    if completed.returncode != 0:
        print(" failed")
        sys.exit("The solver exited with code %d. See %s" % (completed.returncode, os.path.join(work, "stdout.txt")))
    results = {"wall_time": wall_time, "run_directory": os.path.join(work, "out", "run0"), "stdout": completed.stdout}
    results_file = os.path.join(results["run_directory"], "results.txt")
    if os.path.exists(results_file):
        results.update(read_key_value_file(results_file))
    print(" %.0f s" % wall_time)
    return results


def read_key_value_file(path):
    values = {}
    with open(path) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = float(value)
    return values


def read_residual_histories(run_directory, level, rank=0):
    """Residual histories of all GMRES solves of one sweeping level on one process (a list of lists)."""
    path = os.path.join(run_directory, "ConvergenceHistoryLevel%d%d.plt" % (level, rank))
    series = {}
    with open(path) as f:
        for line in f:
            match = re.match(r"series(\d+)\[(\d+)\] = (\S+)", line)
            if match:
                series.setdefault(int(match.group(1)), []).append(float(match.group(3)))
    return [series[key] for key in sorted(series)]


def write_table(path, header, rows):
    with open(path, "w") as f:
        f.write("# " + " ".join(header) + "\n")
        for row in rows:
            f.write(" ".join("%.10g" % value for value in row) + "\n")


def read_table(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.startswith("#") and line.strip():
                rows.append([float(value) for value in line.split()])
    return rows


def pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("The examples need matplotlib and numpy: pip install -r examples/requirements.txt")
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
    return plt


def save(figure, arguments, filename):
    path = os.path.join(arguments.output, filename)
    figure.savefig(path, bbox_inches="tight")
    print("  wrote", path)
