#!/usr/bin/env python3
"""Web frontend of the waveguide solver.

Starts a small web server (Python standard library only) that serves the browser application in frontend/static
and an API to configure, start and inspect solver runs.

    python3 frontend/server.py                     # then open http://localhost:8750
    python3 frontend/server.py --port 9000 --main build-release/Main/main

Every run gets a folder frontend/runs/<id>/ with the generated parameter files (case.prm, run.prm), the console
output (stdout.txt), the configuration (config.json) and the output folder of the solver (out/run0).

API
    GET  /api/info                 defaults, limits of this machine, state of the executable
    GET  /api/runs                 all runs, newest first
    POST /api/runs                 body: configuration (JSON); starts the solver, returns the run
    GET  /api/runs/<id>            status, progress (residual histories), results
    GET  /api/runs/<id>/field?plane=x|y|z
                                   the field on a plane through the domain as a regular grid
    GET  /api/runs/<id>/log        the last part of the console output
    POST /api/runs/<id>/cancel     stops a running solver
    DELETE /api/runs/<id>          removes a finished run
"""
import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

FRONTEND = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.dirname(FRONTEND)
STATIC = os.path.join(FRONTEND, "static")
RUNS = os.path.join(FRONTEND, "runs")

WAVELENGTH = 1.55  # vacuum wavelength in micrometers, fixed in the solver

DEFAULTS = {
    "name": "",
    # waveguide
    "shape": "straight",              # straight | hump | predefined
    "length": 6.0,                    # geometry size z
    "domain_width": 4.0,              # geometry size x
    "domain_height": 3.6,             # geometry size y
    "core_width": 2.0,
    "core_height": 1.8,
    "epsilon_core": 2.3409,
    "epsilon_cladding": 1.8496,
    "hump_height": 0.5,               # vertical waveguide displacement (shape = hump)
    "predefined_case": 3,             # shape = predefined
    # discretization
    "cells_per_wavelength": 8,
    "fem_order": 0,
    "processes_x": 1,
    "processes_y": 1,
    "processes_z": 2,
    # boundary method
    "boundary_method": "PML",         # PML | HSIE
    "pml_layers": 6,
    "pml_thickness": 0.5,
    "pml_sigma_max": 10.0,
    "pml_order": 2,
    "hsie_degree": 5,
    # signal
    "coupling": "Dirichlet",          # Dirichlet | Taper
    # solver
    "solver_precision": 1e-6,
    "gmres_max_steps": 60,
    "compression_tolerance": 0.0,
    "inner_relative_tolerance": 1e-2,
}

LIMITS = {
    "length": (1.0, 200.0), "domain_width": (2.5, 20.0), "domain_height": (2.5, 20.0),
    "core_width": (0.2, 10.0), "core_height": (0.2, 10.0), "epsilon_core": (1.0, 20.0),
    "epsilon_cladding": (1.0, 20.0), "hump_height": (0.0, 5.0), "predefined_case": (0, 35),
    "cells_per_wavelength": (3, 40), "fem_order": (0, 2), "processes_x": (1, 16), "processes_y": (1, 16),
    "processes_z": (1, 64), "pml_layers": (2, 30), "pml_thickness": (0.05, 5.0), "pml_sigma_max": (0.1, 200.0),
    "pml_order": (0, 6), "hsie_degree": (1, 20), "solver_precision": (1e-12, 1e-1), "gmres_max_steps": (1, 500),
    "compression_tolerance": (0.0, 1e-1), "inner_relative_tolerance": (1e-8, 0.5),
}


class ConfigurationError(Exception):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# Configuration -> parameter files
# ----------------------------------------------------------------------------------------------------------------------

def validate(raw):
    """Returns a complete, type-checked configuration. Raises ConfigurationError with a message for the user."""
    config = dict(DEFAULTS)
    for key, value in (raw or {}).items():
        if key not in DEFAULTS:
            raise ConfigurationError("Unknown setting '%s'." % key)
        default = DEFAULTS[key]
        try:
            if isinstance(default, str):
                value = str(value)
            elif isinstance(default, int):
                value = int(value)
            else:
                value = float(value)
        except (TypeError, ValueError):
            raise ConfigurationError("The value of '%s' is not a number." % key)
        config[key] = value
    for key, (low, high) in LIMITS.items():
        if not (low <= config[key] <= high) or (isinstance(config[key], float) and math.isnan(config[key])):
            raise ConfigurationError("'%s' has to be between %g and %g." % (key.replace("_", " "), low, high))
    if config["shape"] not in ("straight", "hump", "predefined"):
        raise ConfigurationError("Unknown shape '%s'." % config["shape"])
    if config["boundary_method"] not in ("PML", "HSIE"):
        raise ConfigurationError("Unknown boundary method '%s'." % config["boundary_method"])
    if config["coupling"] not in ("Dirichlet", "Taper"):
        raise ConfigurationError("Unknown signal coupling '%s'." % config["coupling"])
    if config["core_width"] >= config["domain_width"] or config["core_height"] >= config["domain_height"]:
        raise ConfigurationError("The core has to be smaller than the domain.")
    if config["epsilon_core"] <= config["epsilon_cladding"]:
        raise ConfigurationError("The permittivity of the core has to be larger than that of the cladding, otherwise the waveguide does not guide.")
    if config["shape"] == "predefined":
        config["length"] = 100.0  # the predefined shapes of the project partners are 100 micrometers long
    if config["shape"] == "hump" and config["hump_height"] > 0.5 * config["length"]:
        raise ConfigurationError("The hump is too high for this length: its slope would exceed 1.")
    if config["boundary_method"] == "HSIE" and config["processes_x"] * config["processes_y"] * config["processes_z"] > 1:
        raise ConfigurationError("Hardy space infinite elements cannot be used with more than one process (no sweeping).")
    if config["coupling"] == "Taper" and config["length"] / config["processes_z"] < 1.0:
        raise ConfigurationError("With tapered coupling the first subdomain has to be at least 1 µm long (the taper region).")
    return config


def derive(config):
    """Quantities that follow from a configuration: mesh, unknowns, memory estimate, warnings."""
    wavelength_core = WAVELENGTH / math.sqrt(config["epsilon_core"])
    target = wavelength_core / config["cells_per_wavelength"]
    processes = (config["processes_x"], config["processes_y"], config["processes_z"])
    sizes = (config["domain_width"], config["domain_height"], config["length"])
    cells_per_process = [max(2, int(math.ceil(size / n_proc / target))) for size, n_proc in zip(sizes, processes)]
    cells = [c * p for c, p in zip(cells_per_process, processes)]
    h = [size / c for size, c in zip(sizes, cells)]
    n_processes = processes[0] * processes[1] * processes[2]
    # sweeping level = number of directions that are split, in the order z, y, x
    if processes[0] > 1:
        level = 3
    elif processes[1] > 1:
        level = 2
    elif processes[2] > 1:
        level = 1
    else:
        level = 0
    order_factor = (config["fem_order"] + 1) ** 3

    def edge_unknowns(nx, ny, nz):
        return (nx * (ny + 1) * (nz + 1) + (nx + 1) * ny * (nz + 1) + (nx + 1) * (ny + 1) * nz) * order_factor

    layers = config["pml_layers"] if config["boundary_method"] == "PML" else 0
    # the global problem: PML on all sides except the input side for Dirichlet coupling
    lower_z = layers if config["coupling"] == "Taper" else 0
    unknowns = edge_unknowns(cells[0] + 2 * layers, cells[1] + 2 * layers, cells[2] + layers + lower_z)
    # the largest subdomain problem of the lowest level: one process with PML (or neighbors' PML) on all sides
    local = edge_unknowns(cells_per_process[0] + 2 * layers, cells_per_process[1] + 2 * layers, cells_per_process[2] + 2 * layers)
    # Memory of a sparse LDL^T factorization of a 3D problem grows like N^(4/3). The constant is fitted to
    # measurements with MUMPS for lowest order elements (1 GB at 150 000 unknowns, 7 GB at 390 000).
    memory_per_process_gb = 1.0 * (local / 150000.0) ** (4.0 / 3.0) * (1.6 if config["fem_order"] > 0 else 1.0)
    warnings = []
    cells_per_wavelength = [wavelength_core / value for value in h]
    if min(cells_per_wavelength) < 5 and config["fem_order"] == 0:
        warnings.append("Fewer than 5 cells per wavelength: the wave is not resolved, expect large errors and slow convergence.")
    cores = os.cpu_count() or 1
    if n_processes > cores:
        warnings.append("%d processes on a machine with %d cores." % (n_processes, cores))
    if config["boundary_method"] == "HSIE":
        warnings.append("The Hardy space infinite elements of the solver do currently not absorb outgoing waves (see MODERNIZATION.md). Results will be wrong.")
    if config["shape"] != "straight" and config["coupling"] == "Dirichlet":
        warnings.append("A bent waveguide reflects. Dirichlet coupling sends reflections back into the domain; tapered coupling absorbs them.")
    if config["boundary_method"] == "PML" and config["pml_thickness"] / config["pml_layers"] > 2.5 * max(h):
        warnings.append("The PML cells are much larger than the cells of the interior.")
    return {
        "wavelength_core": wavelength_core,
        "cells_per_process": cells_per_process,
        "cells": cells,
        "cell_size": h,
        "cells_per_wavelength": cells_per_wavelength,
        "processes": n_processes,
        "sweeping_level": level,
        "unknowns": unknowns,
        "unknowns_local": local,
        "memory_per_process_gb": memory_per_process_gb,
        "memory_total_gb": memory_per_process_gb * n_processes,
        "warnings": warnings,
    }


def parameter_files(config):
    """Returns the contents of the case file and the run file for a validated configuration."""
    derived = derive(config)
    case = {
        "source type": 3,
        "fem order": config["fem_order"],
        "geometry size x": config["domain_width"],
        "geometry size y": config["domain_height"],
        "geometry size z": config["length"],
        "width of waveguide": config["core_width"],
        "height of waveguide": config["core_height"],
        "epsilon in": config["epsilon_core"],
        "epsilon out": config["epsilon_cladding"],
        "Boundary Method": config["boundary_method"],
        "PML n layers": config["pml_layers"],
        "PML thickness": config["pml_thickness"],
        "PML sigma max": config["pml_sigma_max"],
        "PML skaling order": config["pml_order"],
        "HSIE polynomial degree": config["hsie_degree"],
        "Kappa 0 Real": round(2 * math.pi / WAVELENGTH * math.sqrt(config["epsilon_cladding"]), 6),
        "Kappa 0 Imaginary": 0,
        "Input Signal Method": config["coupling"],
    }
    if config["shape"] == "hump":
        case["vertical waveguide displacement"] = config["hump_height"]
    if config["shape"] == "predefined":
        case["Use predefined shape"] = "true"
        case["Predefined case number"] = config["predefined_case"]
    run = {
        "sweeping level": derived["sweeping_level"],
        "processes in x": config["processes_x"],
        "processes in y": config["processes_y"],
        "processes in z": config["processes_z"],
        "cell count x": derived["cells_per_process"][0],
        "cell count y": derived["cells_per_process"][1],
        "cell count z": derived["cells_per_process"][2],
        "solver precision": config["solver_precision"],
        "GMRES maximum steps": config["gmres_max_steps"],
        "local solver compression tolerance": config["compression_tolerance"],
        "Logging Level": "Debug One",
    }
    if derived["sweeping_level"] > 1:
        run["use relative convergence criterion"] = "true"
        run["relative convergence criterion"] = config["inner_relative_tolerance"]

    def render(section, values):
        lines = ["subsection %s" % section] + ["  set %s = %s" % item for item in values.items()] + ["end", ""]
        return "\n".join(lines)

    return render("Case parameters", case), render("Run parameters", run)


# ----------------------------------------------------------------------------------------------------------------------
# Reading the outputs of the solver
# ----------------------------------------------------------------------------------------------------------------------

RESIDUAL_LINE = re.compile(r"^(\t*)(?:\[(\d+)\]: )?Residual in step (\d+) : ([0-9.eE+-]+)")


def parse_progress(stdout):
    """Extracts what can be shown while the solver runs: the phase and the residual histories of the GMRES solvers.

    The solver indents the residual lines of the inner levels with tabs. Level 0 is the outermost solver.
    Returns {"phase": ..., "levels": [{"solves": n, "current": [residuals], "histories": [[...], ...]}, ...]}.
    Only the first and the most recent histories of the inner levels are kept; there can be thousands.
    """
    levels = {}
    phase = "Setting up"
    for line in stdout.splitlines():
        match = RESIDUAL_LINE.match(line)
        if match:
            depth = len(match.group(1))
            step, value = int(match.group(3)), float(match.group(4))
            level = levels.setdefault(depth, {"solves": 0, "histories": []})
            if step == 1:
                level["solves"] += 1
                level["histories"].append([])
                if len(level["histories"]) > 24:
                    del level["histories"][12]  # keep the first 12 and the latest 12
            if level["histories"]:
                level["histories"][-1].append(value)
            phase = "Solving"
            continue
        if "Mesh Preparation finished" in line:
            phase = "Building meshes"
        elif "Begin assembly" in line or "NonLocalProblem::assemble" in line:
            phase = "Assembling"
        elif "Begin solver factorization" in line:
            phase = "Factorizing subdomains"
        elif "Start output results" in line:
            phase = "Writing results"
    ordered = [levels[depth] for depth in sorted(levels)]
    return {"phase": phase, "levels": ordered}


def read_key_value_file(path):
    values = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    try:
                        values[key.strip()] = float(value)
                    except ValueError:
                        pass
    return values


def read_field(output_directory, plane):
    """Merges the slice files of all processes for one plane into a regular grid.

    Returns the two in-plane axes (cell centers), the physical position of the grid points, the complex field
    components and the core indicator. The grid is rectangular because the meshes are axis-parallel.
    """
    axes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[plane]
    rows = []
    if os.path.isdir(output_directory):
        for name in sorted(os.listdir(output_directory)):
            if name.startswith("field_slice_%s" % plane) and name.endswith(".txt"):
                with open(os.path.join(output_directory, name)) as f:
                    for line in f:
                        if not line.startswith("#"):
                            parts = line.split()
                            if len(parts) == 13:
                                rows.append([float(value) for value in parts])
    if not rows:
        return None

    def grid_axis(index):
        values = sorted(set(round(row[index], 9) for row in rows))
        return values, {value: i for i, value in enumerate(values)}

    u_values, u_index = grid_axis(axes[0])
    v_values, v_index = grid_axis(axes[1])
    n_u, n_v = len(u_values), len(v_values)
    empty = lambda: [[None] * n_u for _ in range(n_v)]
    field = {name: empty() for name in ("re_x", "im_x", "re_y", "im_y", "re_z", "im_z")}
    core = empty()
    physical_u, physical_v = empty(), empty()
    for row in rows:
        i, j = u_index[round(row[axes[0]], 9)], v_index[round(row[axes[1]], 9)]
        physical_u[j][i], physical_v[j][i] = row[3 + axes[0]], row[3 + axes[1]]
        for k, name in enumerate(("re_x", "im_x", "re_y", "im_y", "re_z", "im_z")):
            field[name][j][i] = row[6 + k]
        core[j][i] = int(row[12])
    return {"plane": plane, "axes": ["xyz"[axes[0]], "xyz"[axes[1]]], "u": u_values, "v": v_values,
            "physical_u": physical_u, "physical_v": physical_v, "field": field, "core": core}


def read_residual_histories(output_directory):
    """All residual histories of process 0 after the run, per solver level (outermost first)."""
    levels = {}
    if os.path.isdir(output_directory):
        for name in os.listdir(output_directory):
            match = re.match(r"ConvergenceHistoryLevel(\d)0\.plt$", name)
            if not match:
                continue
            series = {}
            with open(os.path.join(output_directory, name)) as f:
                for line in f:
                    entry = re.match(r"series(\d+)\[(\d+)\] = (\S+)", line)
                    if entry:
                        series.setdefault(int(entry.group(1)), []).append(float(entry.group(3)))
            levels[int(match.group(1))] = [series[key] for key in sorted(series)]
    return [levels[key] for key in sorted(levels, reverse=True)]


# ----------------------------------------------------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------------------------------------------------

class RunManager:
    def __init__(self, main, mpiexec):
        self.main = main
        self.mpiexec = mpiexec
        self.processes = {}
        self.lock = threading.Lock()
        os.makedirs(RUNS, exist_ok=True)

    def directory(self, run_id):
        if not re.match(r"^[0-9]{8}-[0-9]{6}(-[0-9]+)?$", run_id):
            raise KeyError(run_id)
        path = os.path.join(RUNS, run_id)
        if not os.path.isdir(path):
            raise KeyError(run_id)
        return path

    def start(self, raw_config):
        config = validate(raw_config)
        if not (self.main and os.path.exists(self.main)):
            raise ConfigurationError("The solver executable was not found. Build the code first (see Readme.md) and restart this server, or pass --main.")
        with self.lock:
            run_id = time.strftime("%Y%m%d-%H%M%S")
            suffix = 1
            while os.path.exists(os.path.join(RUNS, run_id)):
                suffix += 1
                run_id = time.strftime("%Y%m%d-%H%M%S") + "-%d" % suffix
            path = os.path.join(RUNS, run_id)
            os.makedirs(path)
        case, run = parameter_files(config)
        derived = derive(config)
        for name, content in (("case.prm", case), ("run.prm", run)):
            with open(os.path.join(path, name), "w") as f:
                f.write(content)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"config": config, "derived": derived, "started": time.time()}, f, indent=1)
        command = [self.mpiexec, "-np", str(derived["processes"]), self.main,
                   "--case", os.path.join(path, "case.prm"), "--run", os.path.join(path, "run.prm")]
        environment = dict(os.environ, WAVEGUIDE_ROOT=REPOSITORY, WAVEGUIDE_OUTPUT_DIR=os.path.join(path, "out"))
        stdout = open(os.path.join(path, "stdout.txt"), "w")
        process = subprocess.Popen(command, cwd=path, env=environment, stdout=stdout, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        self.processes[run_id] = process
        threading.Thread(target=self._wait, args=(run_id, process, stdout), daemon=True).start()
        return self.describe(run_id)

    def _wait(self, run_id, process, stdout):
        code = process.wait()
        stdout.close()
        with open(os.path.join(RUNS, run_id, "exit.json"), "w") as f:
            json.dump({"code": code, "finished": time.time()}, f)

    def cancel(self, run_id):
        self.directory(run_id)
        process = self.processes.get(run_id)
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            with open(os.path.join(RUNS, run_id, "cancelled"), "w") as f:
                f.write("cancelled by the user\n")
        return self.describe(run_id)

    def delete(self, run_id):
        path = self.directory(run_id)
        process = self.processes.get(run_id)
        if process is not None and process.poll() is None:
            raise ConfigurationError("The run is still active. Cancel it first.")
        shutil.rmtree(path)

    def describe(self, run_id, details=False):
        path = self.directory(run_id)
        with open(os.path.join(path, "config.json")) as f:
            stored = json.load(f)
        description = {"id": run_id, "config": stored["config"], "derived": stored["derived"], "started": stored["started"]}
        exit_file = os.path.join(path, "exit.json")
        if os.path.exists(exit_file):
            with open(exit_file) as f:
                exit_data = json.load(f)
            description["elapsed"] = exit_data["finished"] - stored["started"]
            if os.path.exists(os.path.join(path, "cancelled")):
                description["status"] = "cancelled"
            else:
                description["status"] = "finished" if exit_data["code"] == 0 else "failed"
        elif run_id in self.processes:
            description["status"] = "running"
            description["elapsed"] = time.time() - stored["started"]
        else:
            # the server was restarted while this run was active
            description["status"] = "interrupted"
            description["elapsed"] = None
        output = os.path.join(path, "out", "run0")
        results = read_key_value_file(os.path.join(output, "results.txt"))
        description["signal_strength"] = results.get("signal_strength_abs")
        if details:
            stdout = ""
            stdout_file = os.path.join(path, "stdout.txt")
            if os.path.exists(stdout_file):
                with open(stdout_file, errors="replace") as f:
                    stdout = f.read()
            description["progress"] = parse_progress(stdout)
            description["results"] = results
            if description["status"] == "finished":
                description["histories"] = read_residual_histories(output)
                description["planes"] = [plane for plane in "xyz" if os.path.exists(os.path.join(output, "field_slice_%s0.txt" % plane))]
            if description["status"] == "failed":
                description["error"] = failure_message(stdout)
        return description

    def list(self):
        runs = []
        for name in sorted(os.listdir(RUNS), reverse=True):
            try:
                runs.append(self.describe(name))
            except (KeyError, OSError, ValueError):
                continue
        return runs

    def log(self, run_id, n_lines=200):
        path = os.path.join(self.directory(run_id), "stdout.txt")
        if not os.path.exists(path):
            return ""
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
        cleaned = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in lines[-n_lines:]]
        return "\n".join(cleaned)


def failure_message(stdout):
    """The most informative line of the output of a failed run."""
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in stdout.splitlines()]
    for pattern in ("could not be set up", "runtime_error", "The run file requests", "PETSC ERROR", "Additional information", "violated condition"):
        for i, line in enumerate(lines):
            if pattern in line:
                return " ".join(part.strip() for part in lines[i:i + 3])[:600]
    if any("Killed" in line or "signal 9" in line for line in lines):
        return "The solver was killed, most likely because it ran out of memory."
    return "The solver stopped with an error. The log has the details."


# ----------------------------------------------------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------------------------------------------------

CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".json": "application/json"}


def make_handler(manager):
    class Handler(BaseHTTPRequestHandler):
        server_version = "WaveguideFrontend/1.0"

        def log_message(self, format, *args):  # quiet: the polling of the browser would flood the terminal
            pass

        def send_json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_error_message(self, status, message):
            self.send_json({"error": message}, status)

        def read_body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length == 0:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode())
            except ValueError:
                raise ConfigurationError("The request is not valid JSON.")

        def route(self, method):
            url = urlparse(self.path)
            parts = [part for part in url.path.split("/") if part]
            query = parse_qs(url.query)
            try:
                if parts[:1] != ["api"]:
                    if method == "GET":
                        return self.send_static(url.path)
                    return self.send_error_message(405, "Method not allowed.")
                if parts == ["api", "info"] and method == "GET":
                    return self.send_json({"defaults": DEFAULTS, "limits": LIMITS, "cores": os.cpu_count(),
                                           "wavelength": WAVELENGTH, "solver_found": bool(manager.main and os.path.exists(manager.main)),
                                           "solver": manager.main})
                if parts == ["api", "estimate"] and method == "POST":
                    config = validate(self.read_body())
                    return self.send_json({"config": config, "derived": derive(config)})
                if parts == ["api", "runs"] and method == "GET":
                    return self.send_json(manager.list())
                if parts == ["api", "runs"] and method == "POST":
                    return self.send_json(manager.start(self.read_body()), 201)
                if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
                    run_id = parts[2]
                    if len(parts) == 3 and method == "GET":
                        return self.send_json(manager.describe(run_id, details=True))
                    if len(parts) == 3 and method == "DELETE":
                        manager.delete(run_id)
                        return self.send_json({"deleted": run_id})
                    if parts[3:] == ["cancel"] and method == "POST":
                        return self.send_json(manager.cancel(run_id))
                    if parts[3:] == ["log"] and method == "GET":
                        return self.send_json({"log": manager.log(run_id)})
                    if parts[3:] == ["field"] and method == "GET":
                        plane = (query.get("plane") or ["x"])[0]
                        if plane not in ("x", "y", "z"):
                            raise ConfigurationError("The plane has to be x, y or z.")
                        field = read_field(os.path.join(manager.directory(run_id), "out", "run0"), plane)
                        if field is None:
                            return self.send_error_message(404, "This run has no field data for the plane.")
                        return self.send_json(field)
                return self.send_error_message(404, "Not found.")
            except ConfigurationError as error:
                return self.send_error_message(400, str(error))
            except KeyError:
                return self.send_error_message(404, "This run does not exist.")

        def send_static(self, path):
            if path in ("", "/"):
                path = "/index.html"
            full = os.path.normpath(os.path.join(STATIC, path.lstrip("/")))
            if not full.startswith(STATIC) or not os.path.isfile(full):
                return self.send_error_message(404, "Not found.")
            with open(full, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPES.get(os.path.splitext(full)[1], "application/octet-stream"))
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.route("GET")

        def do_POST(self):
            self.route("POST")

        def do_DELETE(self):
            self.route("DELETE")

    return Handler


def find_solver():
    for candidate in ("build-release/Main/main", "build/Main/main"):
        path = os.path.join(REPOSITORY, candidate)
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8750)
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default: only this machine)")
    parser.add_argument("--main", default=None, help="solver executable (default: build-release/Main/main or build/Main/main)")
    parser.add_argument("--mpiexec", default="mpiexec")
    arguments = parser.parse_args()
    solver = os.path.abspath(arguments.main) if arguments.main else find_solver()
    manager = RunManager(solver, arguments.mpiexec)
    server = ThreadingHTTPServer((arguments.host, arguments.port), make_handler(manager))
    print("Waveguide frontend: http://%s:%d" % ("localhost" if arguments.host == "127.0.0.1" else arguments.host, arguments.port))
    print("Solver: %s" % (solver or "not found - build the code first"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
