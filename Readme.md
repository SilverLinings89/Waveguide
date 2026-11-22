# The 3D Maxwell Project

A parallel finite element solver for the time-harmonic Maxwell equations on large 3D domains, written by
Pascal Kraft as part of the dissertation *A Hierarchical Solver for Time-Harmonic Maxwell's Equations* (KIT, 2022).

- Discretization: Nédélec elements (`FE_NedelecSZ`) on axis-parallel hexahedral meshes, complex arithmetic.
- Domain truncation: PML (Hardy space infinite elements are implemented but cannot be used for sweeping).
- Solver: GMRES with a **hierarchical sweeping preconditioner**. The domain is split into blocks in z (level 1), the
  blocks can be split again in y (level 2) and x (level 3). On the lowest level each MPI process factorizes its
  subdomain once with a sparse direct solver (MUMPS by default).
- Applications: propagation of a mode through a (bent) dielectric waveguide described by transformation optics, a
  Hertzian dipole in free space, adjoint based shape optimization.

`MODERNIZATION.md` documents the state of the code after the 2026 migration to deal.II 9.8: what was changed, which
errors were found and fixed, what is still open, and recommendations (tests, usability, GPUs, a deal.II tutorial).

## Quick start

```bash
# 1. Dependencies (PETSc with complex scalars and MUMPS, deal.II 9.8). Takes about 45 minutes.
#    The header of the script lists the packages that have to be installed with brew / apt first.
./install_scripts/install_dependencies.sh            # installs to ../deps

# 2. Build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$PWD/../deps/install/dealii
cmake --build build

# 3. Run the default case (a short straight waveguide, 2 processes) and the tests
mpirun -np 2 build/Main/main
ctest --test-dir build                               # all tests (about 4 minutes in a release build)
ctest --test-dir build -L unit                       # only the fast unit tests
```

Use `-DCMAKE_BUILD_TYPE=Debug` while developing: deal.II then checks array bounds, vector states and many other
preconditions. Use `Release` for production runs.

## Running simulations

```
mpirun -np <N> build/Main/main [--case <case file>] [--run <run file>] [--override "key=value;..."] [PETSc options]
build/Main/main --help
```

A simulation is described by two parameter files:

| File | Describes | Examples |
| --- | --- | --- |
| case file (`Parameters/Case/...`) | **What** is computed: size of the domain, materials, input signal, shape of the waveguide, boundary method and PML parameters, optimization | `straight_waveguide/lowest_order_with_pml.prm`, `hump_examples/predefined_case_with_pml.prm`, `point_source/hertz.prm`, `optimization/base.prm` |
| run file (`Parameters/Run/...`) | **How** it is computed: number of processes in x, y and z, cells per process, sweeping level, solver tolerances, logging | `2-1-1.prm` ... `3-3-39.prm` (the names are processes in x-y-z) |

The files are deal.II `ParameterHandler` files. All parameters, their defaults and their documentation are declared in
`Code/Helpers/ParameterReader.cpp`. The most important rules:

- `N` must be equal to `processes in x * processes in y * processes in z`.
- `sweeping level` is the number of directions that are split: 1 = z only, 2 = z and y, 3 = z, y and x. A direction
  with more than one process requires the corresponding level (so `2 x 2 x 2` processes need level 3). With one
  process in total, use `sweeping level = 0`; the problem is then solved with the direct solver only.
- The mesh has `cell count` cells **per process** in each direction; the size of the domain of interest is
  `geometry size x/y/z` (in micrometers). PML layers are added around it automatically.
- Rule of thumb for the resolution: at least 10 cells per wavelength in the material (the wavelength is fixed at
  1.55 µm, i.e. about 1 µm in the core) for lowest order elements, fewer for `fem order = 1`.
- `local solver compression tolerance = 1e-4` (run file) compresses the subdomain factorizations (block low rank):
  about half the memory for the factors and half the time per subdomain solve for large subdomains, without
  additional GMRES steps. 0 (default) switches it off.
- `perform gradient check = true` (case file, together with `perform optimization = true`) compares the shape
  gradient with finite differences instead of optimizing. An optimization run stops according to
  `optimization gradient tolerance`, `optimization loss tolerance`, `optimization maximum evaluations` and
  `optimization maximum steps` and writes `optimization_history.txt`.
- `Input Signal Method = Dirichlet` prescribes the mode on the input face. `Taper` uses a volume source and a PML
  behind the input, which also absorbs reflections that travel backwards.
- `Boundary Method = HSIE` replaces the PML by Hardy space infinite elements (`HSIE polynomial degree`, typically 3
  to 5; `Kappa 0 Real` and `Kappa 0 Imaginary` default to 4.05 + 2i, the wavenumber plus an imaginary part for the
  decaying field in the cladding). They need far fewer unknowns than a PML and
  reach the same accuracy for waveguides, but only with one process (`sweeping level = 0`): the sweeping
  preconditioner converges poorly with them, and fields that radiate through the edges of the domain are not
  absorbed (see `MODERNIZATION.md`, section 10.1).

Single parameters can be changed from the command line with `--override` (see `--help` for the supported keys).
Options that are not known to the program are passed on to PETSc. The solver of the subdomains has the option prefix
`local_`:

```bash
# print MUMPS statistics for the subdomain factorizations
mpirun -np 2 build/Main/main -local_mat_mumps_icntl_4 2
# any other MUMPS control, e.g. the UCFS variant of the block low rank compression
mpirun -np 2 build/Main/main -local_mat_mumps_icntl_36 1
```

### Outputs

Every run creates a new folder `Solutions/run<i>` (or `$WAVEGUIDE_OUTPUT_DIR/run<i>`, or `$WORK/Solutions/run<i>`):

| File | Content |
| --- | --- |
| `results.txt` | Summary as `key = value`: transmitted signal strength (overlap with the mode at the output, 1 = lossless), errors against the input signal, GMRES iterations, number of unknowns and subdomain solves, factorization time, time per subdomain solve, size of the factors |
| `run_description.txt` | The most important parameters and the git commit of the code |
| `_solution.pvtu` | Open this in ParaView. It references the `.vtu` files of all processes (interior and PML domains) |
| `ConvergenceHistoryLevel*.gnuplot/.dat` | Residual histories of the GMRES solvers of every level (a plot is created if `gnuplot` is installed) |
| `timer_output*.dat`, `main<rank>.log` | Timings per level and the log of every process |

### Environment variables

| Variable | Meaning |
| --- | --- |
| `WAVEGUIDE_ROOT` | Root of the repository (to find `Modes/` and `Parameters/`). Default: the source folder the code was configured from |
| `WAVEGUIDE_OUTPUT_DIR` | Base folder for the outputs |

## Examples: the plots of the dissertation

The folder `examples/` contains one script per graph of the dissertation. Each script runs the solver with the
parameters of the experiment, stores the plotted numbers and writes the figure; `examples/README.md` maps figures
to scripts.

```bash
pip install -r examples/requirements.txt
python3 examples/sweeping_scaling.py            # reduced variant for a workstation
python3 examples/sweeping_scaling.py --help     # description of the experiment and of --scale paper
```

## Web frontend

```bash
python3 frontend/server.py          # then open http://localhost:8750
```

The page shows a drawing of the configured waveguide (core, absorbing layers, mesh, process boundaries) that follows
every setting, an estimate of unknowns and memory, and a button to start the solver. While the solver runs, the
residuals appear in a chart; afterwards the field can be viewed on three planes, together with the numbers of
`results.txt`. Runs are kept in `frontend/runs/<id>/` with their parameter files, so each one can be repeated
from the command line, and a run can be linked as `http://localhost:8750/?run=<id>`. The server needs only the
Python standard library and listens on this machine only (use `--host` to change that).

## Dependencies

| Library | Tested version | Required configuration |
| --- | --- | --- |
| deal.II | 9.8.0 | MPI, PETSc, `DEAL_II_WITH_COMPLEX_VALUES=ON` (off by default since 9.7), SymEngine |
| PETSc | 3.26.1 | `--with-scalar-type=complex`, MUMPS (`--download-mumps --download-scalapack`) |
| MPI, CMake >= 3.16, a C++17 compiler, Python 3 (integration tests), GoogleTest (downloaded automatically) | | |

deal.II 9.8.0 needs two workarounds that `install_dependencies.sh` applies (see `MODERNIZATION.md`): a patch to make it
compile with a complex PETSc and two headers that are not installed. The modification of
`vector_tools_project.templates.h` that older versions of this Readme asked for is **no longer required**.

## Tests

- `tests/*.cpp` (GoogleTest, executable `build/tests/waveguide_tests`): unit tests for Hardy space polynomials,
  Laguerre functions, mesh sorting, shape functions, the import of the mode profile, the signal taper, the space
  transformation and the process layout.
- `tests/integration/run_integration_test.py`: runs the solver with 2 to 8 MPI processes on small problems and
  checks `results.txt`: lossless transmission through a straight waveguide for both coupling methods, improvement
  under mesh refinement, reproducibility, agreement of level 1, 2 and 3 sweeping with a direct solve of the global
  system, and (one process) the infinite elements against an exact solution and against PML. `python3 tests/integration/run_integration_test.py --list` lists them.

## Structure of the repository

| Folder | Content |
| --- | --- |
| `Main/` | `main.cpp`: parses the arguments and starts one of the runners |
| `Code/Runners/` | `SweepingRun` (default), `SingleCoreRun`, `ConvergenceRun`, `ParameterSweep`, `OptimizationRun` |
| `Code/Hierarchy/` | `NonLocalProblem` (one sweeping level: GMRES + sweep), `LocalProblem` (subdomain with direct solver) |
| `Code/Core/` | `InnerDomain` (mesh, assembly and evaluation in the domain of interest) |
| `Code/BoundaryCondition/` | `PMLSurface`, `HSIESurface`, `DirichletSurface`, `NeighborSurface` (interface to another process), `EmptySurface` (PEC) |
| `Code/SpaceTransformations/`, `Code/Optimization/` | Transformation optics for the waveguide shapes and the shape functions |
| `Code/Solutions/` | The input signal: imported mode profile (`ExactSolution`), Hertzian dipole (`PointSourceField` in `Helpers`) |
| `Code/GlobalObjects/` | Global state: parameters, geometry and process layout, output and timers |
| `Modes/` | Mode profile of the 2.0 µm x 1.8 µm waveguide at 1550 nm (`output_mode.dat`) and the predefined shapes |
| `Parameters/`, `run/` | Parameter files and the Slurm scripts of the numerical experiments of the dissertation |
| `examples/` | One script per plot of the dissertation (see above) |
| `frontend/` | Browser application to configure, run and inspect simulations |
| `tests/`, `install_scripts/` | Tests and the installation script for the dependencies |

The global objects (`GlobalParams`, `Geometry`, `GlobalMPI`, ...) are defined in `Main/main.cpp` and used everywhere.
`Progress.md` is the work diary of the original development.

## Naming conventions

| Expression |Meaning     |
|------------|------------------------------------------------------------------------|
| Sector     | A sector is an expression used in shape modeling and refers to a subdomain between the connectors. |
| Layer      | used in the implementation and represents the part of the triangulation owned by one Process.  |
| Connector  | The structure in the two halfspaces towards z->\infty and z->-\infty  outside of the computational domain.  |
| ... | ... |

### Levels in Hierarchical Sweeping

There are 2 level types: global and local.

#### Global

The global level describes how many hierarchical levels there are in total.

- 0: purely local computation, no sweeping.
- 1: there is only sweeping in the z direction.
- 2: there is sweeping in the z direction and the blocks are solved by sweeping in the y direction
- 3: there is sweeping in the z direction and the blocks are solved by sweeping in the y direction. The blocks required in the y-sweep are solved by sweeping the x-direction.

#### Local

It makes a difference for which of the sweeps a matrix is assebled and what the global sweeping level is. The local level describes for which scenario the block is assembled. It counts from 0 up. 0 means that we assemble a local block, i.e. a direct solver for the local problem with sweeping in only one direction. 
If local is 0 and global is 1, there is only sweeping in the z-direction. That means the local block is assembled by cutting off the x- and y- directions.
If local is 0 and global is 2, the lowest order sweeping is in the y-direction, so the direct solver is assembled for a domain with boundary conditions in x- and z-direction but sweeping in y (similar for local 0 and global 3, only for x).
If local is 1 or higher, the block is not assembled for a direct solver but a matrix for GMRES is built. This means that not all surfaces are either boundaries with HSIE or sweeping block interfaces (Dirichlet boundaries), they can now be internal.

## Rotations for surface extraction

For boundary id:
| b_{id} | x | y |
| --- | --- | ---|
| 0 | z | y |
| 1 | z | y |
| 2 | x | z |
| 3 | x | -z |
| 4 | -x | y |
| 5 | x | y |

## Predefined waveguide shapes

This code comes with some example cases called the waveguide hump examples. These examples are available as hdf5 and obj files in the folder Modes -> PWB_Bond_Trajectories. The first three cases are numbered 0,1 and 2 and then continue on at 40, 41, ... , 72. In total there are 3 + 33 cases, so 36. To run these, a parameter in the case file can be used. Be careful, however, because these numbers start at 0 and count to 35. The table below lists how these numbers correlate: 
| PWB Bond index | Predefined case number | Approximate signal loss in % | Approximate signal loss in db |
| --- | --- | --- | --- |
| 0 | 0 |  |  |
| 1 | 1 |  |  |
| 2 | 2 |  |  |
| 40 | 3 |  |  |
| 41 | 4 |  |  |
| 42 | 5 |  |  |
| 43 | 6 |  |  |
| 44 | 7 |  |  |
| 45 | 8 |  |  |
| 46 | 9 |  |  |
| 47 | 10 |  |  |
| 48 | 11 |  |  |
| 49 | 12 |  |  |
| 50 | 13 |  |  |
| 51 | 14 |  |  |
| 52 | 15 |  |  |
| 53 | 16 |  |  |
| 54 | 17 |  |  |
| 55 | 18 |  |  |
| 56 | 19 |  |  |
| 57 | 20 |  |  |
| 58 | 21 |  |  |
| 59 | 22 |  |  |
| 60 | 23 |  |  |
| 61 | 24 |  |  |
| 62 | 25 |  |  |
| 63 | 26 |  |  |
| 64 | 27 |  |  |
| 65 | 28 |  |  |
| 66 | 29 |  |  |
| 67 | 30 |  |  |
| 68 | 31 |  |  |
| 69 | 32 |  |  |
| 70 | 33 |  |  |
| 71 | 34 |  |  |
| 72 | 35 |  |  |


## Thanks

My thanks go to the CRC 1173 which funded this research. I also thank the team behind the deal.II library that has
been the basis for my implementations.
