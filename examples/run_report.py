#!/usr/bin/env python3
"""Builds report.html for one run of the solver.

    python3 examples/run_report.py <output folder of the run, e.g. .../out/run0>

The solver writes run_report.json (setup, unknowns, solver statistics, results), the field at the cell centers of
the three middle planes (field_slice_*), the overlap with the mode along z (signal_along_z*) and, for optimization
runs, the history and the shapes. This script adds the figures and puts everything into one HTML file that can be
opened in a browser without any other file:

  - the field in the computational (transformed) and in the physical coordinates,
  - the signal transmission along the waveguide,
  - the residual histories of the solvers on every sweeping level,
  - for optimization runs the shapes and the history of the loss.

The figures are also stored as PNG files in the subfolder report_figures.
"""
import base64
import glob
import html
import json
import os
import sys

import numpy as np

import common

COMPONENTS = ["E_x", "E_y", "E_z"]


def read_slices(run_directory, plane):
    """All processes' values of one plane: array with the columns of the file, or None."""
    rows = []
    for path in sorted(glob.glob(os.path.join(run_directory, "field_slice_%s*.txt" % plane))):
        rows += common.read_table(path)
    return np.array(rows) if rows else None


def structured(rows, first, second):
    """Sorts the cell centers of a plane into a grid. Returns the grid of row indices (first x second) or None."""
    first_values = np.unique(np.round(rows[:, first], 9))
    second_values = np.unique(np.round(rows[:, second], 9))
    if len(first_values) * len(second_values) != len(rows):
        return None
    order = np.lexsort((np.round(rows[:, second], 9), np.round(rows[:, first], 9)))
    return order.reshape(len(first_values), len(second_values))


def edges(centers):
    """Cell edges from cell centers (per row of a 2D array, along the last axis)."""
    centers = np.asarray(centers, dtype=float)
    middle = 0.5 * (centers[..., 1:] + centers[..., :-1])
    first = centers[..., :1] - (middle[..., :1] - centers[..., :1])
    last = centers[..., -1:] + (centers[..., -1:] - middle[..., -1:])
    return np.concatenate([first, middle, last], axis=-1)


def corner_grid(a, b):
    """Corner coordinates for pcolormesh from the cell center coordinates a, b (both n x m)."""
    def expand(values):
        values = edges(values)                      # n x (m + 1)
        return edges(values.T).T                    # (n + 1) x (m + 1)
    return expand(a), expand(b)


def complex_field(rows):
    return rows[:, 6:12:2] + 1j * rows[:, 7:12:2]


def dominant_component(rows):
    return int(np.argmax(np.mean(np.abs(complex_field(rows)), axis=0)))


def figure_field(plt, run_directory, data, figures):
    """The plane x = middle (y over z): it shows the vertical displacement of the waveguide."""
    rows = read_slices(run_directory, "x")
    if rows is None:
        return
    grid = structured(rows, 2, 1)
    if grid is None:
        return
    field = complex_field(rows)
    component = dominant_component(rows)
    real_part = field[:, component].real[grid]
    magnitude = np.sqrt(np.sum(np.abs(field) ** 2, axis=1))[grid]
    limit = np.max(np.abs(real_part)) or 1.0
    # The outline of the core is known for the rectangular and the circular cross-section only.
    half_height = 0.5 * data.get("core_height", 0.0) if data.get("cross_section", 0.0) != 2.0 else 0.0
    z = rows[:, 2][grid]
    displacement = (rows[:, 4] - rows[:, 1])[grid][:, 0]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 6.4), sharex=True, sharey=True, constrained_layout=True)
    for row, (label, y_column) in enumerate([("computational (transformed) coordinates", 1), ("physical coordinates", 4)]):
        y = rows[:, y_column][grid]
        zc, yc = corner_grid(z, y)
        shift = displacement if y_column == 4 else np.zeros_like(displacement)
        for column, (values, colormap, limits, title) in enumerate([
                (real_part, "RdBu_r", (-limit, limit), "Re(%s)" % COMPONENTS[component]),
                (magnitude, "Blues", (0, np.max(magnitude) or 1.0), "|E|")]):
            axis = axes[row][column]
            mesh = axis.pcolormesh(zc, yc, values, cmap=colormap, vmin=limits[0], vmax=limits[1], shading="flat", rasterized=True)
            if data.get("source_type") in (0.0, 3.0) and half_height > 0:
                for sign in (-1, 1):
                    axis.plot(z[:, 0], sign * half_height + shift, color="0.15", linewidth=0.9, linestyle="--",
                              label="core boundary" if sign == -1 else None)
            axis.set_title("%s, %s" % (title, label), fontsize=10)
            axis.set_aspect("equal")
            axis.grid(False)
            if row == 1:
                axis.set_xlabel(r"z [$\mu$m]")
            if column == 0:
                axis.set_ylabel(r"y [$\mu$m]")
            if row == 0 and column == 0 and data.get("source_type") in (0.0, 3.0):
                axis.legend(fontsize=8, loc="upper right")
            figure.colorbar(mesh, ax=axis, shrink=0.85, pad=0.01)
    figures.append(("field_side_view", figure, "Field in the plane x = middle of the domain (side view). Top: the "
                    "coordinates the solver works in, where the waveguide is straight. Bottom: physical coordinates, "
                    "y shifted by the displacement f(z) of the waveguide axis. Values at the cell centers of the "
                    "interior; the absorbing layers are not shown."))


def figure_other_planes(plt, run_directory, data, figures):
    top = read_slices(run_directory, "y")
    cross = read_slices(run_directory, "z")
    if top is None or cross is None:
        return
    top_grid, cross_grid = structured(top, 2, 0), structured(cross, 0, 4)
    if top_grid is None:
        return
    figure, (axis, cross_axis) = plt.subplots(1, 2, figsize=(12.5, 3.6), constrained_layout=True,
                                              gridspec_kw={"width_ratios": [2.2, 1]})
    component = dominant_component(top)
    values = complex_field(top)[:, component].real[top_grid]
    limit = np.max(np.abs(values)) or 1.0
    zc, xc = corner_grid(top[:, 2][top_grid], top[:, 0][top_grid])
    mesh = axis.pcolormesh(zc, xc, values, cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True)
    if data.get("source_type") in (0.0, 3.0) and data.get("cross_section", 0.0) != 2.0:
        for sign in (-1, 1):
            axis.axhline(sign * 0.5 * data.get("core_width", 0.0), color="0.15", linewidth=0.9, linestyle="--")
    axis.set_title("Re(%s), plane y = middle (top view, computational coordinates)" % COMPONENTS[component], fontsize=10)
    axis.set_xlabel(r"z [$\mu$m]")
    axis.set_ylabel(r"x [$\mu$m]")
    axis.set_aspect("equal")
    axis.grid(False)
    figure.colorbar(mesh, ax=axis, shrink=0.85, pad=0.01)
    magnitude = np.sqrt(np.sum(np.abs(complex_field(cross)) ** 2, axis=1))
    if cross_grid is not None:
        xc, yc = corner_grid(cross[:, 3][cross_grid], cross[:, 4][cross_grid])
        mesh = cross_axis.pcolormesh(xc, yc, magnitude[cross_grid], cmap="Blues", vmin=0, rasterized=True)
    else:
        mesh = cross_axis.tripcolor(cross[:, 3], cross[:, 4], magnitude, cmap="Blues", vmin=0, rasterized=True)
    if data.get("source_type") in (0.0, 3.0) and data.get("cross_section", 0.0) != 2.0:
        shift = float(np.mean(cross[:, 4] - cross[:, 1]))
        w, h = 0.5 * data.get("core_width", 0.0), 0.5 * data.get("core_height", 0.0)
        if data.get("cross_section", 0.0) == 1.0:
            angle = np.linspace(0, 2 * np.pi, 181)
            cross_axis.plot(w * np.cos(angle), shift + w * np.sin(angle), color="0.15", linewidth=0.9, linestyle="--")
        else:
            cross_axis.plot([-w, w, w, -w, -w], [shift - h, shift - h, shift + h, shift + h, shift - h], color="0.15", linewidth=0.9, linestyle="--")
    cross_axis.set_title("|E|, cross-section z = %.3g (physical)" % float(np.mean(cross[:, 2])), fontsize=10)
    cross_axis.set_xlabel(r"x [$\mu$m]")
    cross_axis.set_ylabel(r"y [$\mu$m]")
    cross_axis.set_aspect("equal")
    cross_axis.grid(False)
    figure.colorbar(mesh, ax=cross_axis, shrink=0.85, pad=0.02)
    figures.append(("field_top_view_and_cross_section", figure, "Left: top view of the field. Right: cross-section in "
                    "the middle of the domain. Dashed: boundary of the core."))


def read_signal_along_z(run_directory):
    rows = []
    for path in sorted(glob.glob(os.path.join(run_directory, "signal_along_z*.txt"))):
        rows += common.read_table(path)
    if not rows:
        return None
    rows = np.array(rows)
    keys = np.round(rows[:, 0], 6)
    z = np.unique(keys)
    sums = np.array([rows[keys == value, 1:].sum(axis=0) for value in z])
    signal = (sums[:, 0] + 1j * sums[:, 1]) / sums[:, 2]
    return z, signal, sums[:, 3], sums[:, 4]


def figure_signal(plt, run_directory, data, figures):
    result = read_signal_along_z(run_directory)
    if result is None or data.get("source_type") not in (0.0, 3.0, 4.0):
        return
    z, signal, core_energy, total_energy = result
    figure, (axis, energy_axis) = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True, constrained_layout=True)
    axis.plot(z, np.abs(signal), color="tab:blue", linewidth=1.8, label="|s(z)|, overlap with the mode")
    axis.axhline(1.0, color="0.4", linewidth=0.8, linestyle=":")
    if "signal_strength_abs" in data:
        axis.plot([data["z_max"]], [data["signal_strength_abs"]], "o", color="tab:orange", markersize=7,
                  label="|s| at the output = %.4f" % data["signal_strength_abs"])
    axis.set_ylabel("signal |s(z)|")
    axis.set_ylim(0, max(1.08, 1.05 * float(np.max(np.abs(signal)))))
    energy_axis.plot(z, core_energy / np.max(total_energy), color="tab:blue", linewidth=1.8, label="in the core")
    energy_axis.plot(z, total_energy / np.max(total_energy), color="tab:orange", linewidth=1.8, label="whole cross-section")
    energy_axis.set_ylabel(r"$\int |E|^2$ per plane (relative to its maximum)")
    energy_axis.set_xlabel(r"z [$\mu$m]")
    energy_axis.set_ylim(bottom=0)
    if data.get("tapered_input"):
        for one_axis in (axis, energy_axis):
            one_axis.axvspan(data["tapering_min_z"], data["tapering_max_z"], color="0.85", zorder=0, label="input taper")
    axis.legend(fontsize=9, loc="lower right")
    energy_axis.legend(fontsize=9, loc="lower right")
    figures.append(("signal_along_z", figure, "Signal transmission along the waveguide. Top: s(z) is the overlap of "
                    "the field with the mode of the straight waveguide in every plane of cell faces z = const, divided by the overlap "
                    "of the mode with itself; 1 means that the field is the undisturbed mode. Bottom: the energy of "
                    "the electric field per plane. A drop along z is radiated or reflected power, a ripple is a "
                    "standing wave caused by reflections. With a tapered input the signal is built up inside the "
                    "grey zone."))


def figure_residuals(plt, report, data, figures):
    by_level = {}
    for series in report.get("series", []):
        if series["name"].startswith("residuals_level_") and len(series["values"]) > 0:
            by_level.setdefault(int(series["name"].rsplit("_", 1)[1]), []).append(series["values"])
    if not by_level:
        return
    levels = sorted(by_level, reverse=True)
    figure, axes = plt.subplots(1, len(levels), figsize=(5.2 * len(levels), 4.0), constrained_layout=True, squeeze=False)
    for axis, level in zip(axes[0], levels):
        histories = by_level[level]
        colors = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, len(histories))) if len(histories) > 1 else ["tab:blue"]
        for index, values in enumerate(histories):
            axis.semilogy(range(len(values)), values, color=colors[index], linewidth=1.0 if len(histories) > 4 else 1.8,
                          marker="." if len(histories) <= 12 else None)
        if level == levels[0] and "solver_precision" in data:
            axis.axhline(data["solver_precision"], color="tab:orange", linewidth=1.0, linestyle="--", label="tolerance")
            axis.legend(fontsize=9)
        axis.set_title("level %d: %d solve%s%s" % (level, len(histories), "" if len(histories) == 1 else "s",
                                                  " (light to dark: first to last)" if len(histories) > 1 else ""), fontsize=10)
        axis.set_xlabel("GMRES step")
        axis.set_ylabel("residual norm")
        axis.xaxis.get_major_locator().set_params(integer=True)
    figures.append(("solver_convergence", figure, "Residual histories of the GMRES solvers. The highest level is the "
                    "global problem; on the lower levels the first 40 solves that process 0 took part in are shown. "
                    "Their tolerance follows the residual of the level above, so later solves start and stop lower."))


def figure_optimization(plt, run_directory, figures):
    history_file = os.path.join(run_directory, "optimization_history.txt")
    shapes_file = os.path.join(run_directory, "optimization_shapes.txt")
    if not (os.path.exists(history_file) and os.path.exists(shapes_file)):
        return
    history = np.array(common.read_table(history_file))
    shapes = np.array(common.read_table(shapes_file))
    z, shapes = shapes[0], shapes[1:]
    losses, gradient_norms = history[:, 1], history[:, 2]
    best = int(np.argmin(losses))
    figure, axis = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    colors = plt.get_cmap("Blues")(np.linspace(0.3, 0.9, len(shapes)))
    for index, shape in enumerate(shapes):
        axis.plot(z, shape, color=colors[index], linewidth=1.0)
    axis.plot(z, shapes[0], color="0.25", linewidth=1.8, linestyle="--", label="initial shape, |s| = %.4f" % -losses[0])
    axis.plot(z, shapes[best], color="tab:orange", linewidth=2.2, label="best shape (evaluation %d), |s| = %.4f" % (best, -losses[best]))
    axis.set_xlabel(r"z [$\mu$m]")
    axis.set_ylabel(r"vertical displacement f(z) [$\mu$m]")
    axis.legend(fontsize=9)
    figures.append(("optimization_shapes", figure, "Vertical displacement of the waveguide axis for all evaluated "
                    "shapes (light to dark blue: first to last evaluation)."))
    figure, (axis, gradient_axis) = plt.subplots(2, 1, figsize=(9.5, 5.6), sharex=True, constrained_layout=True)
    evaluations = np.arange(len(losses))
    axis.semilogy(evaluations, np.maximum(1 + losses, 1e-16), "o-", color="tab:blue")
    axis.set_ylabel("signal loss 1 − |s|")
    gradient_axis.semilogy(evaluations, gradient_norms, "o-", color="tab:blue")
    gradient_axis.set_ylabel("norm of the shape gradient")
    gradient_axis.set_xlabel("evaluation (one forward and one adjoint solve)")
    figures.append(("optimization_history", figure, "Signal loss and norm of the shape gradient for every evaluation "
                    "of the optimization. The line search of BFGS also evaluates shapes that are worse than the best "
                    "one so far."))


STYLE = """
:root { --ink: #1c2530; --muted: #5d6b7a; --line: #d5dce4; --paper: #ffffff; --band: #f3f6f9; --accent: #1f5f99; }
@media (prefers-color-scheme: dark) {
  :root { --ink: #e6ebf0; --muted: #9aa8b6; --line: #37424e; --paper: #151a20; --band: #1d242c; --accent: #7fb6e6; }
  img { filter: brightness(0.92); }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 28px 16px 60px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 40px 0 10px; padding-bottom: 6px; border-bottom: 2px solid var(--line); }
h3 { font-size: 15px; margin: 22px 0 6px; }
p.lead { color: var(--muted); margin: 0 0 18px; }
nav { margin: 14px 0 0; font-size: 14px; }
nav a { color: var(--accent); margin-right: 14px; white-space: nowrap; }
.numbers { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0 0; }
.numbers div { background: var(--band); border-radius: 6px; padding: 10px 12px; }
.numbers b { display: block; font-size: 21px; font-variant-numeric: tabular-nums; }
.numbers span { color: var(--muted); font-size: 13px; }
table { border-collapse: collapse; margin: 4px 0 8px; font-size: 14px; }
td, th { text-align: left; vertical-align: top; padding: 4px 14px 4px 0; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; }
table.entries td:first-child { color: var(--muted); white-space: nowrap; }
.scroll { overflow-x: auto; }
.note { color: var(--muted); font-size: 14px; margin: 8px 0; max-width: 78ch; }
figure { margin: 18px 0 26px; }
figure img { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 4px; background: #fff; }
figcaption { color: var(--muted); font-size: 14px; max-width: 86ch; margin-top: 6px; }
"""


def section_html(section):
    parts = []
    if section["entries"]:
        parts.append('<div class="scroll"><table class="entries">')
        for name, value in section["entries"]:
            parts.append("<tr><td>%s</td><td>%s</td></tr>" % (html.escape(name), html.escape(value)))
        parts.append("</table></div>")
    for table in section["tables"]:
        parts.append("<h3>%s</h3><div class=\"scroll\"><table><tr>%s</tr>" % (
            html.escape(table["title"]), "".join("<th>%s</th>" % html.escape(cell) for cell in table["header"])))
        for row in table["rows"]:
            parts.append("<tr>%s</tr>" % "".join("<td>%s</td>" % html.escape(cell) for cell in row))
        parts.append("</table></div>")
    for note in section["notes"]:
        parts.append('<p class="note">%s</p>' % html.escape(note))
    return "\n".join(parts)


def figures_html(figures, names):
    parts = []
    for name, encoded, caption in figures:
        if name in names:
            parts.append('<figure><img alt="%s" src="data:image/png;base64,%s"><figcaption>%s</figcaption></figure>' % (
                html.escape(caption), encoded, html.escape(caption)))
    return "\n".join(parts)


def entry(report, section_title, name):
    for section in report["sections"]:
        if section["title"] == section_title:
            for key, value in section["entries"]:
                if key == name:
                    return value
    return None


def build(run_directory, title=None):
    """Writes report.html into the run directory and returns its path."""
    run_directory = os.path.abspath(run_directory)
    report_file = os.path.join(run_directory, "run_report.json")
    if not os.path.exists(report_file):
        sys.exit("%s does not exist. The folder has to be the output folder of a run of a current build of the solver." % report_file)
    with open(report_file) as f:
        report = json.load(f)
    data = report.get("data", {})
    plt = common.pyplot()
    pending = []
    figure_field(plt, run_directory, data, pending)
    figure_other_planes(plt, run_directory, data, pending)
    figure_signal(plt, run_directory, data, pending)
    figure_residuals(plt, report, data, pending)
    figure_optimization(plt, run_directory, pending)
    figure_directory = os.path.join(run_directory, "report_figures")
    os.makedirs(figure_directory, exist_ok=True)
    figures = []
    for name, figure, caption in pending:
        path = os.path.join(figure_directory, name + ".png")
        figure.savefig(path, dpi=140, facecolor="white")
        plt.close(figure)
        with open(path, "rb") as f:
            figures.append((name, base64.b64encode(f.read()).decode("ascii"), caption))

    figure_places = {
        "Result": ["field_side_view", "field_top_view_and_cross_section", "signal_along_z"],
        "Solution process": ["solver_convergence"],
        "Optimization": ["optimization_shapes", "optimization_history"],
    }
    order = ["Run", "Geometry", "Mesh", "Unknowns", "Solution process", "Optimization", "Result"]
    sections = sorted(report["sections"], key=lambda s: order.index(s["title"]) if s["title"] in order else len(order))
    tiles = [
        ("signal transmission |s|", entry(report, "Result", "Signal transmission |s|")),
        ("unknowns", entry(report, "Unknowns", "Unknowns of the global problem")),
        ("processes", "%d" % data.get("n_processes", 0)),
        ("sweeping levels", "%d" % data.get("sweeping_level", 0)),
        ("subdomain solves", entry(report, "Solution process", "Subdomain solves (direct)")),
        ("wall time", entry(report, "Run", "Wall time until the outputs")),
    ]
    title = title or "Run report: %s" % (entry(report, "Run", "Type of run") or "solver run")
    body = ["<h1>%s</h1>" % html.escape(title),
            '<p class="lead">%s. Folder: %s</p>' % (html.escape(entry(report, "Run", "Problem") or ""), html.escape(run_directory)),
            '<div class="numbers">%s</div>' % "".join("<div><b>%s</b><span>%s</span></div>" % (html.escape(str(value)), html.escape(label))
                                                       for label, value in tiles if value),
            "<nav>%s</nav>" % "".join('<a href="#s%d">%s</a>' % (index, html.escape(section["title"])) for index, section in enumerate(sections))]
    for index, section in enumerate(sections):
        body.append('<h2 id="s%d">%s</h2>' % (index, html.escape(section["title"])))
        if section["title"] == "Result":
            body.append(section_html(dict(section, tables=[])))
            body.append(figures_html(figures, figure_places["Result"]))
            body.append(section_html(dict(section, entries=[], notes=[])))
        else:
            body.append(section_html(section))
            body.append(figures_html(figures, figure_places.get(section["title"], [])))
    page = ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>%s</title><style>%s</style></head><body><main>\n%s\n</main></body></html>\n" % (
                html.escape(title), STYLE, "\n".join(body)))
    path = os.path.join(run_directory, "report.html")
    with open(path, "w") as f:
        f.write(page)
    return path


def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    print("wrote", build(sys.argv[1]))


if __name__ == "__main__":
    main()
