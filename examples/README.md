# Examples: one script per plot of the dissertation

Every graph of the dissertation is produced by one script in this folder. A script runs the solver (if the plot needs
a simulation), stores the numbers as a plain text table and writes the figure. Run them from anywhere:

```bash
pip install -r examples/requirements.txt        # matplotlib and numpy
python3 examples/sweeping_scaling.py            # laptop scale (default)
python3 examples/sweeping_scaling.py --scale paper
python3 examples/sweeping_scaling.py --plot-only  # only redo the figure from the stored table
python3 examples/sweeping_scaling.py --help     # describes the experiment and its parameters
```

The solver has to be built first (`build/Main/main` or `build-release/Main/main`, see `../Readme.md`; another path
can be passed with `--main`). Outputs go to `examples/output/<name>/`: the figure(s), a `.dat` table with the
plotted numbers and, in `runs/`, the parameter files, the console output and the complete output folder of every
solver run.

`--scale laptop` is a reduced set of runs that finishes on a workstation with 16 cores and 32 GB of memory.
`--scale paper` uses the parameters of the dissertation; where that needs a cluster it is noted below.

| Dissertation | Script | What it runs | Laptop scale | Paper scale |
| --- | --- | --- | --- | --- |
| Fig. "Comparison of constant σ and σ(x) = 5x³" (PML) | `pml_profile.py` | nothing, analytic | seconds | same |
| Figs. "Real part of E_x", "Imaginary part of E_z" (input mode) | `mode_profile.py` | nothing, plots `Modes/output_mode.dat` and checks it | seconds | same |
| Fig. "φ(z) and (1 − φ(z)) Re(E_I,x)" (tapered coupling) | `signal_taper.py` | nothing, analytic | seconds | same |
| Fig. "Results of the sweeping scaling experiment" | `sweeping_scaling.py` | straight waveguide, 1 subdomain per process, growing length | 2 to 12 subdomains, 4 min, 1.5 GB per process | up to 256 subdomains: cluster |
| Fig. "Required steps and runtime ... for an increasing length" | `domain_stretching.py` | 8 processes, fixed mesh, growing length | 8 lengths, 15 min | 31 lengths, about 1 h |
| Fig. "Convergence plot of the numerical error" and its table | `convergence_study.py` | 1 x 2 x 2 processes, element order 1 and 2, mesh sequence | n = 2, 4, 8: 3 min | n = 2 ... 16: the order 2 reference has 1.2 million unknowns |
| Figs. "Convergence of the GMRES run(s) on level 1, 2, 3" | `hierarchical_sweeping.py` | three sweeping levels | 16 processes, 8 µm waveguide with a hump: 1 min | 351 processes, 100 µm predefined shape, 39.5 h: cluster |
| Figs. "Results from the optimization experiment" (shapes), "signal loss and step width", GMRES histories | `shape_optimization.py` | BFGS with adjoint gradients, 4 processes | hump 0.5 µm, at most 14 evaluations: 5 min | hump 1.5 µm on the mesh of the dissertation |

Two more scripts are not tied to a figure of the dissertation:

| Script | What it does |
| --- | --- |
| `pml_vs_hsie.py` | Accuracy against memory for PML and Hardy space infinite elements on a test problem with an exact solution (TE10 mode of a metallic waveguide): 3D solver runs, a 1D reference model of both methods (`absorbing_boundary_model.py`) and a trade-off table. Laptop scale: 120 runs of a few seconds. Results are discussed in `../MODERNIZATION.md`, section 10 |
| `absorbing_boundary_model.py` | The 1D reference model (imported by `pml_vs_hsie.py`) |

Images of the dissertation that are not graphs (fields rendered with ParaView, sketches, mesh pictures) have no
script. The fields can be reproduced by opening `_solution.pvtu` of the corresponding run in ParaView.

Notes:

- The times are for an Apple M5 Max (36 GB) with an optimized build.
- Plots with two different quantities use two panels with a shared x-axis instead of two y-axes.
- The numbers of the dissertation were computed in 2022 with a version of the code that contained the errors listed
  in `../MODERNIZATION.md` (section 2). The examples run the corrected code, so their results differ from the
  printed figures: GMRES needs fewer steps, and the optimization behaves differently.
- `common.py` contains the shared code (writing parameter files, starting the solver, reading `results.txt` and
  the residual histories).
