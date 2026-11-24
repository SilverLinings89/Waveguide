#!/usr/bin/env python3
"""The photonic wire bonds of the project partners: this solver against the measurements and the other methods.

Not a figure of the dissertation. Modes/reference_transmission.csv (copied from Modes/Results.ods) contains the
measured transmission of 17 wire bonds with humps of 5.5 to 16.2 um height over a length of 100 um, and the results
of three other methods (columns FMA, TO, CST_FULL) and of this solver in 2022 (column HSP; whether that value is a power or an amplitude is not
recorded, so both conversions to dB are shown). The shapes of the bonds
are the predefined shapes of the solver (Modes/test.csv, bond = case + 37).

Produces: hump_comparison.dat (one row per case), hump_comparison.txt (overview table with the deviations from the
measurement), hump_comparison.png, and one run folder with report.html per case in runs/.

The solver computes the amplitude s of the mode at the output for an input of amplitude 1 (signal_strength_abs).
The transmitted power is |s|^2, in dB 20 log10 |s| (transmission_db in results.txt).

Every case is one run with one sweeping level and slabs in z. A case that has a result is not run again, so the
script can be restarted after an interruption; --rerun computes everything again.

laptop: 16 x 16 cells in the cross-section, 780 in z (39 slabs), factors of the subdomains on disk. About 4 minutes
        and 17 GB per case on a laptop. The result is NOT converged at this resolution; it shows the workflow.
paper:  48 x 48 x 1560 cells, the mesh of the dissertation: about 540 GB (300 GB with --factors-on-disk).
Other resolutions: --cells-xy, --cells-z-per-slab, --slabs.

One existing run can be compared without running anything:
    python3 examples/hump_comparison.py --compare-run <run folder with results.txt> --case 3
"""
import csv
import math
import os
import shutil
import tempfile

import numpy as np

import common

RESOLUTIONS = {"laptop": (16, 20, 39), "paper": (48, 40, 39)}  # cells in x and y, cells in z per slab, slabs
METHODS = [("measured_db", "measured"), ("solver_db", "this solver"), ("fma_db", "FMA"), ("to_db", "TO"),
           ("cst_full_db", "CST_FULL"), ("hsp_db", "HSP 2022, 10log"), ("hsp_amplitude_db", "HSP 2022, 20log")]


def read_reference():
    """Rows of Modes/reference_transmission.csv as dictionaries {case: {...}}; missing values are nan."""
    reference = {}
    with open(os.path.join(common.REPOSITORY, "Modes", "reference_transmission.csv")) as f:
        lines = [line for line in f if not line.startswith("#")]
    for row in csv.DictReader(lines):
        values = {key: (float(value) if value else math.nan) for key, value in row.items()}
        # The file converts the value of this solver of 2022 with 10 log10, i.e. as a power. If it was the signal
        # strength |s| that the solver writes (an amplitude), 20 log10 is the transmission. Both are listed.
        values["hsp_db"] = 10 * math.log10(values["hsp_value"]) if values["hsp_value"] > 0 else math.nan
        values["hsp_amplitude_db"] = 2 * values["hsp_db"]
        reference[int(values["case"])] = values
    return reference


def hump_height(case):
    """Largest displacement of the nodes of a predefined shape in micrometers."""
    with open(os.path.join(common.REPOSITORY, "Modes", "test.csv")) as f:
        fields = f.read().splitlines()[case].split(",")
    n = int(fields[0])
    return max(float(value) for value in fields[1 + n:1 + 2 * n])


def case_parameters(case, pml_layers):
    return {"source type": 3, "fem order": 0, "geometry size x": 6.0, "geometry size y": 5.4, "geometry size z": 100.0,
            "epsilon in": 2.3409, "epsilon out": 1.8496, "Boundary Method": "PML", "PML n layers": pml_layers,
            "PML thickness": 0.5, "PML sigma max": 20, "PML skaling order": 3, "Input Signal Method": "Taper",
            "vertical waveguide displacement": 0.0, "Use predefined shape": True, "Predefined case number": case}


def run_parameters(cells_xy, cells_z, slabs):
    return {"sweeping level": 1, "processes in x": 1, "processes in y": 1, "processes in z": slabs,
            "cell count x": cells_xy, "cell count y": cells_xy, "cell count z": cells_z,
            "use relative convergence criterion": True, "relative convergence criterion": 0.01,
            "solver precision": 1e-6, "GMRES maximum steps": 60}


def transmission_db(result):
    return 20 * math.log10(max(result["signal_strength_abs"], 1e-30))


def run_case(arguments, case, label, cells_xy, cells_z, slabs, run=True):
    """Runs one case unless its result exists. Returns the entries of results.txt or None if the run failed.

    With run=False the case is not run: the result is read if it exists, otherwise None is returned."""
    run_directory = os.path.join(arguments.output, "runs", label, "out", "run0")
    results_file = os.path.join(run_directory, "results.txt")
    if os.path.exists(results_file) and not arguments.rerun:
        print("  %-28s result exists, not run again" % label)
        result = common.read_key_value_file(results_file)
        wall_time_file = os.path.join(arguments.output, "runs", label, "wall_time.txt")
        result["wall_time"] = float(open(wall_time_file).read()) if os.path.exists(wall_time_file) else math.nan
        return result
    if not run:
        return None
    try:
        result = common.run_solver(arguments, label, case_parameters(case, arguments.pml_layers),
                                   run_parameters(cells_xy, cells_z, slabs), report=True)
    except SystemExit as error:
        print("  case %d failed: %s" % (case, error))
        return None
    with open(os.path.join(arguments.output, "runs", label, "wall_time.txt"), "w") as f:
        f.write("%.1f\n" % result["wall_time"])
    return result


def format_db(value):
    return "%8.2f" % value if not math.isnan(value) else "       -"


def overview(rows, resolution_text):
    """Text table: one line per case and the deviations of every method from the measurement."""
    lines = ["Transmission of the wire bonds in dB (0 = lossless). " + resolution_text, "",
             "bond case  hump[um] " + " ".join("%15s" % title for _, title in METHODS)]
    for row in rows:
        lines.append("%4d %4d %9.2f " % (row["bond"], row["case"], row["hump_um"])
                     + " ".join("%15s" % format_db(row[key]).strip() for key, _ in METHODS))
    lines += ["", "Deviation from the measurement in dB over the cases for which both values exist:",
              "%-18s %6s %10s %10s %10s" % ("method", "cases", "mean", "mean abs", "largest")]
    for key, title in METHODS[1:]:
        differences = np.array([row[key] - row["measured_db"] for row in rows
                                if not math.isnan(row[key]) and not math.isnan(row["measured_db"])])
        if len(differences) == 0:
            lines.append("%-18s %6d" % (title, 0))
            continue
        lines.append("%-18s %6d %+10.2f %10.2f %10.2f" % (title, len(differences), differences.mean(),
                                                          np.abs(differences).mean(), np.abs(differences).max()))
    lines += ["", "mean > 0: the method predicts less loss than was measured.",
              "HSP 2022 is the column HSP of Modes/Results.ods, the result of this solver in 2022: converted as in that",
              "file (10 log10, value taken as a power) and as an amplitude like signal_strength_abs (20 log10)."]
    return "\n".join(lines) + "\n"


def plot(arguments, rows, resolution_text):
    plt = common.pyplot()
    # Fixed order of the series; the measurement is drawn in the text color and on top.
    styles = {"solver_db": ("#2a78d6", "o", 2.2), "fma_db": ("#eb6834", "s", 1.4), "to_db": ("#1baf7a", "^", 1.4),
              "cst_full_db": ("#eda100", "D", 1.4), "hsp_db": ("#e87ba4", "v", 1.4), "hsp_amplitude_db": ("#008300", "P", 1.4)}
    heights = np.array([row["hump_um"] for row in rows])
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for key, title in METHODS[1:]:
        values = np.array([row[key] for row in rows])
        valid = ~np.isnan(values)
        if valid.any():
            color, marker, width = styles[key]
            axis.plot(heights[valid], values[valid], marker=marker, color=color, linewidth=width, markersize=5, label=title)
    measured = np.array([row["measured_db"] for row in rows])
    axis.plot(heights, measured, marker="o", color="#222222", linewidth=2.2, markersize=6, label="measured", zorder=5)
    axis.set_xlabel(r"height of the hump [$\mu$m]")
    axis.set_ylabel("transmission [dB]")
    axis.set_title("Wire bonds of 100 $\\mu$m length: simulations and measurement\n" + resolution_text, fontsize=10)
    bond_axis = axis.secondary_xaxis("top")
    bond_axis.set_xticks(heights[::2])
    bond_axis.set_xticklabels(["%d" % row["bond"] for row in rows[::2]])
    bond_axis.set_xlabel("bond number")
    axis.legend(loc="lower center", ncol=2, frameon=False)
    common.save(figure, arguments, "hump_comparison.png")


def compare_run(arguments, reference):
    """Prints the result of one existing run next to the reference values of its case."""
    result = common.read_key_value_file(os.path.join(arguments.compare_run, "results.txt"))
    if arguments.case is None:
        raise SystemExit("--compare-run needs --case <predefined case number of that run>.")
    values = reference.get(arguments.case)
    print("Case %d (bond %d, hump %.2f um): |s| = %.4f, transmitted power %.4f, %.2f dB"
          % (arguments.case, arguments.case + 37 if arguments.case > 2 else arguments.case, hump_height(arguments.case),
             result["signal_strength_abs"], result["signal_strength_abs"] ** 2, transmission_db(result)))
    if values is None:
        print("There is no measurement for this case (measured: cases 3, 5, ..., 35).")
        return
    for key, title in METHODS:
        if key != "solver_db" and not math.isnan(values[key]):
            print("  %-18s %7.2f dB   (this run %+.2f dB)" % (title, values[key], transmission_db(result) - values[key]))


def main():
    parser = common.argument_parser(__doc__)
    parser.add_argument("--cases", type=int, nargs="+", default=None, help="predefined case numbers (default: all measured cases 3, 5, ..., 35)")
    parser.add_argument("--cells-xy", type=int, default=None, help="cells in x and in y (interior)")
    parser.add_argument("--cells-z-per-slab", type=int, default=None, help="cells in z per slab")
    parser.add_argument("--slabs", type=int, default=None, help="number of slabs in z = number of MPI processes")
    parser.add_argument("--pml-layers", type=int, default=7, help="cell layers of the PML (default 7)")
    parser.add_argument("--factors-on-disk", action="store_true", help="MUMPS writes the factors to disk (default for --scale laptop)")
    parser.add_argument("--scratch", default=None, help="folder for the factors on disk (default: a temporary folder)")
    parser.add_argument("--collect", action="store_true", help="run nothing: build the overview from the results that exist (after one job per case)")
    parser.add_argument("--rerun", action="store_true", help="run cases again that already have a result")
    parser.add_argument("--compare-run", default=None, help="only compare this existing run folder (needs --case)")
    parser.add_argument("--case", type=int, default=None, help="case number of the run given with --compare-run")
    arguments = parser.parse_args()
    reference = read_reference()
    if arguments.compare_run is not None:
        compare_run(arguments, reference)
        return

    output = common.prepare(arguments, "hump_comparison")
    cells_xy, cells_z, slabs = RESOLUTIONS[arguments.scale]
    cells_xy = arguments.cells_xy or cells_xy
    cells_z = arguments.cells_z_per_slab or cells_z
    slabs = arguments.slabs or slabs
    cases = [] if arguments.collect else (arguments.cases or sorted(reference))
    resolution_text = "This solver: %d x %d x %d cells, %d PML layers." % (cells_xy, cells_xy, cells_z * slabs, arguments.pml_layers)
    data_file = os.path.join(output, "hump_comparison.dat")

    if not arguments.plot_only:
        factors_on_disk = arguments.factors_on_disk or arguments.scale == "laptop"
        if factors_on_disk:
            os.environ["PETSC_OPTIONS"] = (os.environ.get("PETSC_OPTIONS", "") + " -local_mat_mumps_icntl_22 1").strip()
        table = []
        scratch = None
        try:
            # Cases that are not requested are read if they have a result, so that the overview is complete when
            # the cases are run by separate jobs.
            for case in sorted(set(reference) | set(cases)):
                label = "case%02d_n%d_z%d" % (case, cells_xy, cells_z * slabs)
                requested = case in cases
                if factors_on_disk and requested:
                    # One folder per case, removed after the case: factor files that a run leaves behind would
                    # otherwise fill the disk for the following cases.
                    scratch = tempfile.mkdtemp(prefix="mumps_", dir=arguments.scratch)
                    os.environ["MUMPS_OOC_TMPDIR"] = scratch
                    print("  scratch %s: %.0f GB free" % (scratch, shutil.disk_usage(scratch).free / 1e9), flush=True)
                result = run_case(arguments, case, label, cells_xy, cells_z, slabs, run=requested)
                if scratch is not None:
                    shutil.rmtree(scratch, ignore_errors=True)
                    scratch = None
                if result is None:
                    continue
                table.append([case, result["signal_strength_abs"], transmission_db(result), result["solver_iterations"],
                              result["n_dofs"], result["wall_time"]])
                print("  case %2d: |s| = %.4f, %.2f dB, %d GMRES steps" % (case, table[-1][1], table[-1][2], table[-1][3]))
                common.write_table(data_file, ["case", "signal_strength_abs", "transmission_db", "gmres_steps", "n_dofs", "wall_time_s"], table)
        finally:
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)

    solver = {int(row[0]): row for row in common.read_table(data_file)} if os.path.exists(data_file) else {}
    rows = []
    for case in sorted(set(reference) | set(solver)):
        row = dict(reference.get(case, {key: math.nan for key, _ in METHODS}))
        row.update({"case": case, "bond": case + 37 if case > 2 else case, "hump_um": hump_height(case),
                    "solver_db": solver[case][2] if case in solver else math.nan})
        rows.append(row)
    text = overview(rows, resolution_text)
    with open(os.path.join(output, "hump_comparison.txt"), "w") as f:
        f.write(text)
    print()
    print(text)
    plot(arguments, rows, resolution_text)


if __name__ == "__main__":
    main()
