#!/usr/bin/env python3
"""Accuracy against memory for the two absorbing boundary methods: PML and Hardy space infinite elements (HSIE).

Not a figure of the dissertation; it answers the question raised there (section Sweeping Preconditioners with HSIE)
which of the two methods gives the better accuracy for a given amount of memory.
Produces: pml_vs_hsie_solver.dat, pml_vs_hsie_model.dat, pml_vs_hsie_solver_tradeoff.txt, pml_vs_hsie_tradeoff.txt
          and the figures pml_vs_hsie_solver.png, pml_vs_hsie_model.png, pml_vs_hsie_memory.png

Test problem with an exact solution: the TE10 mode of a rectangular waveguide with perfectly conducting walls,
E = (0, cos(pi x / a), 0) exp(i beta z), in a homogeneous medium. The domain is 2 x 1 x 2 um (a = 2 um, about 1.2
wavelengths long), the mode is prescribed as Dirichlet data at z = 0 and the only absorbing boundary is the output
face. One process, direct solver.

Three parts:
1. 3D solver: relative L2 error against the exact solution and memory of the factorization for PML with 2 to 12
   cell layers (cells of the size of the interior cells; the best sigma_max of a small set is used) and for HSIE
   of degree 1 to 12, on several meshes. pml_vs_hsie_solver_tradeoff.txt lists for every mesh the cheapest
   configuration of each method that reaches the accuracy the mesh allows.
2. Reference model (absorbing_boundary_model.py): the same problem reduces to u'' + beta^2 u = 0 on a line. The
   model gives the error of the boundary method alone (without the error of the interior mesh) as a function of
   layers / degree, for the mode of the test problem and for the worst case over a range of angles of incidence.
3. Trade-off: layers / degree needed for a target accuracy (part 2) and the memory this costs in 3D (part 1).

laptop: lowest order elements, n = 8, 12, 16, 24 cells per direction.
paper:  additionally n = 32 and elements of the next order with n = 4, 8, 12, 16.
"""
import os
import numpy as np
import common

WAVELENGTH = 1.55
PML_LAYERS = [2, 3, 4, 6, 8, 12]
PML_SIGMAS = [5, 10, 20, 40]
HSIE_DEGREES = [1, 2, 3, 5, 8, 12]
LENGTH_Z = 2.0


def base_case(fem_order):
    return {"source type": 4, "Lateral boundary": "PEC", "fem order": fem_order, "geometry size x": 2.0,
            "geometry size y": 1.0, "geometry size z": LENGTH_Z, "epsilon in": 1.0, "epsilon out": 1.0,
            "Input Signal Method": "Dirichlet"}


def main():
    arguments = common.argument_parser(__doc__).parse_args()
    output = common.prepare(arguments, "pml_vs_hsie")
    data_file = os.path.join(output, "pml_vs_hsie_solver.dat")
    meshes = {0: [8, 12, 16, 24]}
    if arguments.scale == "paper":
        meshes = {0: [8, 12, 16, 24, 32], 1: [4, 8, 12, 16]}
    if not arguments.plot_only:
        rows = []  # method (0 = PML, 1 = HSIE), fem order, n, layers or degree, sigma, error, dofs, nonzeros, factor entries, memory
        for fem_order, sizes in meshes.items():
            for n in sizes:
                run = {"sweeping level": 0, "processes in x": 1, "processes in y": 1, "processes in z": 1,
                       "cell count x": n, "cell count y": n, "cell count z": n}
                h = LENGTH_Z / n
                for layers in PML_LAYERS:
                    for sigma in PML_SIGMAS:
                        case = dict(base_case(fem_order), **{"Boundary Method": "PML", "PML n layers": layers,
                                    "PML thickness": layers * h, "PML sigma max": sigma, "PML skaling order": 2})
                        r = common.run_solver(arguments, "pml_o%d_n%02d_l%02d_s%02d" % (fem_order, n, layers, sigma), case, run)
                        rows.append([0, fem_order, n, layers, sigma, r["error_l2_relative"], r["n_dofs"], r["matrix_nonzeros"],
                                     r["local_factor_entries_full_rank"], r["local_factor_memory_mb_max"]])
                for degree in HSIE_DEGREES:
                    kappa = 2 * np.pi / WAVELENGTH
                    case = dict(base_case(fem_order), **{"Boundary Method": "HSIE", "HSIE polynomial degree": degree,
                                "Kappa 0 Real": kappa, "Kappa 0 Imaginary": 0})
                    r = common.run_solver(arguments, "hsie_o%d_n%02d_d%02d" % (fem_order, n, degree), case, run)
                    rows.append([1, fem_order, n, degree, 0, r["error_l2_relative"], r["n_dofs"], r["matrix_nonzeros"],
                                 r["local_factor_entries_full_rank"], r["local_factor_memory_mb_max"]])
        common.write_table(data_file, ["method(0=PML,1=HSIE)", "fem_order", "n", "layers_or_degree", "sigma_max",
                                       "relative_l2_error", "dofs", "matrix_nonzeros", "factor_entries", "factor_memory_mb"], rows)
    rows = np.array(common.read_table(data_file))
    plt = common.pyplot()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    megabytes = lambda entries: entries * 16 / 1e6   # complex double entries of the factors
    fem_orders = sorted(set(rows[:, 1]))

    def best_pml(mesh_rows):
        pml = mesh_rows[mesh_rows[:, 0] == 0]
        return np.array([min(pml[pml[:, 3] == layers], key=lambda row: row[5]) for layers in sorted(set(pml[:, 3]))])

    # ---- part 1a: solver, error against memory for PML -------------------------------------------------------------
    figure, axes = plt.subplots(1, len(fem_orders), figsize=(7.5 * len(fem_orders), 5), squeeze=False)
    for axis, fem_order in zip(axes[0], fem_orders):
        selected = rows[rows[:, 1] == fem_order]
        for color, n in zip(colors, sorted(set(selected[:, 2]))):
            best = best_pml(selected[selected[:, 2] == n])
            axis.loglog(megabytes(best[:, 8]), best[:, 5], "o-", color=color, label="PML, n = %d (2 to 12 layers)" % n)
            hsie = selected[(selected[:, 2] == n) & (selected[:, 0] == 1)]
            axis.loglog(megabytes(hsie[:, 8]), hsie[:, 5], "s--", color=color, markerfacecolor="none", label="HSIE, n = %d (degree 1 to 12)" % n)
        axis.set_xlabel("memory of the factors [MB]")
        axis.set_ylabel("relative L2 error against the exact solution")
        axis.set_title("3D solver, element order %d" % (fem_order + 1))
        axis.legend(fontsize=8, ncol=2)
    common.save(figure, arguments, "pml_vs_hsie_solver.png")

    # cheapest configuration of each method that comes within 15 % of the best error reached on the mesh
    lines = ["3D solver: cheapest configuration within 15 % of the smallest error reached on the mesh (either method).", "",
             "%-5s %-4s %-11s | %-26s | %-26s | %s" % ("order", "n", "error", "PML", "HSIE", "HSIE / PML memory"), "-" * 100]
    for fem_order in fem_orders:
        selected = rows[rows[:, 1] == fem_order]
        for n in sorted(set(selected[:, 2])):
            mesh = selected[selected[:, 2] == n]
            target = 1.15 * min(mesh[:, 5])
            entry, memory = [], []
            for method, unit in ((0, "layers"), (1, "degree")):
                good = [row for row in mesh[mesh[:, 0] == method] if row[5] <= target]
                if not good:
                    entry.append("not reached")
                    memory.append(float("nan"))
                    continue
                cheapest = min(good, key=lambda row: row[8])
                entry.append("%s %2d: %6.0f MB" % (unit, cheapest[3], megabytes(cheapest[8])))
                memory.append(megabytes(cheapest[8]))
            lines.append("%-5d %-4d %-11.3e | %-26s | %-26s | %.2f" % (fem_order + 1, n, min(mesh[:, 5]), entry[0], entry[1], memory[1] / memory[0]))
    with open(os.path.join(output, "pml_vs_hsie_solver_tradeoff.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join("  " + line for line in lines))

    # ---- part 1b: solver, memory of both methods ---------------------------------------------------------------------
    figure, axes = plt.subplots(1, len(fem_orders), figsize=(7.5 * len(fem_orders), 5), squeeze=False)
    for axis, fem_order in zip(axes[0], fem_orders):
        selected = rows[rows[:, 1] == fem_order]
        for color, n in zip(colors, sorted(set(selected[:, 2]))):
            mesh = selected[selected[:, 2] == n]
            pml, hsie = best_pml(mesh), mesh[mesh[:, 0] == 1]
            axis.semilogy(pml[:, 3], megabytes(pml[:, 8]), "o-", color=color, label="PML, n = %d" % n)
            axis.semilogy(hsie[:, 3], megabytes(hsie[:, 8]), "s--", color=color, markerfacecolor="none", label="HSIE, n = %d" % n)
        axis.set_xlabel("PML cell layers / HSIE polynomial degree")
        axis.set_ylabel("memory of the factors [MB]")
        axis.set_title("3D solver, element order %d" % (fem_order + 1))
        axis.legend(fontsize=8, ncol=2)
    common.save(figure, arguments, "pml_vs_hsie_memory.png")

    # ---- part 2: reference model ---------------------------------------------------------------------------------------
    import absorbing_boundary_model as model
    k = 2 * np.pi / WAVELENGTH
    beta_mode = np.sqrt(k ** 2 - (np.pi / 2.0) ** 2)
    angles = [0, 30, 45, 60, 75]                       # angles of incidence for the worst case
    betas = [k * np.cos(np.radians(angle)) for angle in angles]
    model_file = os.path.join(output, "pml_vs_hsie_model.dat")
    model_rows = []   # method, order, n, layers or degree, extra unknowns, error (mode), error (worst angle)
    cells = 400       # fine interior mesh: isolates the error of the boundary method
    for order in (1, 2):
        for layers in PML_LAYERS + [16]:
            errors = {}
            for sigma in PML_SIGMAS:
                thickness = layers * LENGTH_Z / 16   # layers of the size of the cells of a mesh with n = 16
                mode_error, extra = model.solve(beta_mode, LENGTH_Z, cells, order, "pml", layers=layers, sigma=sigma, thickness=thickness)
                worst = max(model.solve(beta, LENGTH_Z, cells, order, "pml", layers=layers, sigma=sigma, thickness=thickness)[0] for beta in betas)
                errors[sigma] = (mode_error, worst, extra)
            model_rows.append([0, order, cells, layers, errors[PML_SIGMAS[0]][2], min(e[0] for e in errors.values()), min(e[1] for e in errors.values())])
        for degree in [0] + HSIE_DEGREES:
            mode_error, extra = model.solve(beta_mode, LENGTH_Z, cells, order, "hsie", degree=degree, kappa0=k)
            worst = max(model.solve(beta, LENGTH_Z, cells, order, "hsie", degree=degree, kappa0=k)[0] for beta in betas)
            model_rows.append([1, order, cells, degree, extra, mode_error, worst])
    common.write_table(model_file, ["method(0=PML,1=HSIE)", "order", "n", "layers_or_degree", "extra_unknowns_per_line",
                                    "error_mode", "error_worst_angle"], model_rows)
    model_rows = np.array(model_rows)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, column, title in ((axes[0], 5, "mode of the test problem (beta = %.2f k)" % (beta_mode / k)),
                                (axes[1], 6, "worst case over angles of incidence 0° to 75°")):
        for order, color in ((1, colors[0]), (2, colors[1])):
            selected = model_rows[model_rows[:, 1] == order]
            pml, hsie = selected[selected[:, 0] == 0], selected[selected[:, 0] == 1]
            axis.semilogy(pml[:, 4], pml[:, column], "o-", color=color, label="PML, elements of order %d (2 to 16 layers)" % order)
            axis.semilogy(hsie[:, 4], hsie[:, column], "s--", color=color, markerfacecolor="none", label="HSIE (degree 0 to 12), interior order %d" % order)
        axis.set_xlabel("additional unknowns per unknown on the boundary")
        axis.set_ylabel("relative L2 error")
        axis.set_title("1D model, " + title)
        axis.legend(fontsize=8)
    common.save(figure, arguments, "pml_vs_hsie_model.png")

    # ---- part 3: trade-off -----------------------------------------------------------------------------------------------
    finest = max(rows[rows[:, 1] == 0][:, 2])
    mesh = rows[(rows[:, 1] == 0) & (rows[:, 2] == finest)]
    pml_memory = {int(row[3]): megabytes(row[8]) for row in best_pml(mesh)}
    hsie_memory = {int(row[3]): megabytes(row[8]) for row in mesh[mesh[:, 0] == 1]}
    interior_only = min(pml_memory.values())
    lines = ["Layers / degree needed to push the error of the boundary method below a target (1D model, order 2,",
             "worst case over angles of incidence 0 to 75 degrees) and the memory of the factors in the 3D solver",
             "(lowest order elements, n = %d, one absorbing face of %d x %d cells)." % (finest, finest, finest), "",
             "%-8s | %-28s | %-28s" % ("target", "PML", "HSIE"), "-" * 70]
    selected = model_rows[model_rows[:, 1] == 2]
    for target in (1e-1, 1e-2, 1e-3, 1e-4):
        entry = []
        for method, memory, unit in ((0, pml_memory, "layers"), (1, hsie_memory, "degree")):
            candidates = [row for row in selected[selected[:, 0] == method] if row[6] <= target]
            if not candidates:
                entry.append("not reached")
                continue
            needed = int(min(candidates, key=lambda row: row[3])[3])
            available = [key for key in sorted(memory) if key >= needed]
            entry.append("%s %2d: %s" % (unit, needed, ("%.0f MB" % memory[available[0]]) + ("" if available[0] == needed else " (measured for %d)" % available[0]) if available else "not measured"))
        lines.append("%-8g | %-28s | %-28s" % (target, entry[0], entry[1]))
    with open(os.path.join(output, "pml_vs_hsie_tradeoff.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join("  " + line for line in lines))


if __name__ == "__main__":
    main()
