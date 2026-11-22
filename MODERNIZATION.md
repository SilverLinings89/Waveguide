# State of the code after the 2026 modernization

This document records what was done when the code was picked up again in November 2026: the migration to current
libraries, the errors that were found and fixed, what is still open, and recommendations for the next steps.
`Readme.md` explains how to build and use the code.

Tested configuration: deal.II 9.8.0, PETSc 3.26.1 (complex scalars) with MUMPS, Open MPI 5.0.11, Apple clang 21,
macOS on Apple Silicon (M5 Max). Nothing was tested on Linux or on an HPC system.

## 1. Migration

The code compiled against deal.II 9.8.0 after very few changes:

| Change | Reason |
| --- | --- |
| `PETScWrappers::SparseDirectMUMPS(control)` without communicator | Constructor changed |
| `DataOut<3>::type_cell_data` | `DataOut_DoFData<DoFHandler<...>>` no longer exists |
| `ParameterReader` no longer derives from `Subscriptor` | Deprecated (renamed to `EnableObserverPointer`), and it was not needed |
| C++17, CMake targets, GoogleTest via `find_package`/`FetchContent` | deal.II requires C++17; the git submodule was not initialized |
| `DEAL_II_WITH_COMPLEX_VALUES=ON` when building deal.II | Off by default since deal.II 9.7 |

The hand-edit of `vector_tools_project.templates.h` inside the deal.II installation that the old Readme asked for is
no longer necessary.

Three problems are in the libraries, not in this code. They are worth reporting upstream:

1. **deal.II 9.8.0 does not compile with a complex PETSc.** `source/lac/matrix_scaling.cc` (new in 9.8) instantiates
   `MatrixScaling` for `PETScWrappers::MPI::SparseMatrix`, which only compiles for real scalars. Workaround:
   `install_scripts/patches/dealii-9.8.0-complex-petsc-matrix-scaling.patch`.
2. **deal.II 9.8.0 does not install `optimization/solver_bfgs.h` and `optimization/line_minimization.h`** when it is
   built with CMake >= 3.28 (headers are then installed through a file set that misses these two). The install
   script copies them.
3. Homebrew's SymEngine bottle records a versioned LLVM include path that does not exist after an LLVM update; it ends
   up in `deal.IITargets.cmake`. The install script rewrites it.

## 2. Errors that were found and fixed

The numbers below are for a straight waveguide (4 x 3.6 x 6 µm, lowest order elements, 2 processes, PML), where the
fundamental mode should arrive at the output with a signal strength of 1.

### 2.1 Signal coupling (the "coupling in of the modes")

| # | Error | Effect |
| --- | --- | --- |
| 1 | **Wrong sign of the longitudinal component of the mode.** `Modes/output_mode.dat` was computed for the convention `exp(i(ωt − βz))`, the code uses `exp(i(βz − ωt))`. `E_z` was imported as `+i·h` instead of `−i·h`. | The imported field was not divergence free, i.e. not a mode travelling in +z. It affects the tapered source, the comparison with the "exact solution" and the signal strength (the overlap integral subtracted `\|E_z\|²` instead of adding it). Dirichlet data are tangential, so `E_z` did not enter there. |
| 2 | **Copy-paste error in `ExactSolution::get_derivative_terms`**: all eight shifted points modified the same variable. | The second derivatives `∂²f/∂y²`, `∂²h/∂x²`, `∂²h/∂y²` in the tapered source were always exactly 0, `∂²f/∂x∂y` was arbitrary. |
| 3 | **Missing factor `φ(z)`** in the terms `ω²ε φ f` and `ω²ε φ h` of the source term (the same slip was in the formulas for `F_x` and `F_z` in the section on tapered coupling of the dissertation, now corrected there). | A spurious source proportional to `(1 − φ)` over the whole taper. |
| 4 | Finite differences of the bilinearly interpolated profile used a step (0.01) smaller than the data grid (0.05). | Derivatives measured the kinks of the interpolant. |
| 5 | The derivative of the C1 ramp in `ExactSolutionRamped` lacked the factor `1/(z_max − z_min)`. | Wrong for any taper length other than 1. |

Fix: the source term is now assembled in a weak form that needs only first derivatives of the profile (see the
comment at `evaluate_taper_source_at` in `Code/Core/InnerDomain.cpp`):
`F = L(ψE) = ∇ψ × (μ⁻¹ ∇×E) + ∇×(μ⁻¹ (∇ψ × E))` with `ψ = 1 − φ`, tested with `v` and integrated by parts once. It
is equivalent to the formula of the dissertation for an exact mode, and it stays valid across the jump of ε at the
core surface. The curl of the mode is evaluated analytically in z.

The stored mode profile itself is consistent: in the core it fulfills the Helmholtz equation for ε = 2.3409 and
β = 5.956/µm to 1e-4, and it belongs to a core of 2.0 µm (x) by 1.8 µm (y), which is what the parameter files use.

### 2.2 Waveguide shape (the "waveguide solutions")

| # | Error | Effect |
| --- | --- | --- |
| 6 | **Random shape for straight waveguides.** `WaveguideTransformation` passed `Vertical_displacement < 2.0` as the "random initialization" flag of the shape function, and `ShapeFunction::initialize` then drew random values (seeded with the time) whenever the displacement was 0. | Every run of a "straight" waveguide used a different, slightly sheared material tensor in the PML regions. Results were not reproducible: the same input needed 8, 10 or 13 GMRES steps and the system matrix changed from run to run. |
| 7 | **The interior only applied the space transformation for predefined shapes** (`if(Use_Predefined_Shape)`), the PML regions always applied it. | For shapes described by shape functions (all optimization runs and "vertical waveguide displacement") the interior was computed for a straight waveguide while the PML was transformed. |
| 8 | `WaveguideTransformation::get_J`: operator precedence (`y*dh/h*h`) and missing diagonal entries `1/h`, `1/w`; `waveguide_width` was never initialized. | Wrong Jacobian for variable height / width. |

Fix: random initialization only with the parameter `Initialize Shape Dofs Randomly`; the transformation tensor is
applied everywhere; the Jacobian is corrected and tested against finite differences.

### 2.3 Solver

| # | Error | Effect |
| --- | --- | --- |
| 9 | **GMRES with a variable preconditioner.** On levels 2 and 3 the sweep contains iterative solves, so the preconditioner changes slightly between applications. Plain GMRES then reports a residual that is not the true residual. | Level 2/3 results differed from the solution of the linear system in the 4th digit although GMRES reported 1e-6. The upper levels now use flexible GMRES (FGMRES). Level 2 and 3 now agree with a direct solve to 9 digits and need fewer subdomain solves (3 levels, 8 processes: 1134 instead of 1746 solves, 24 s instead of 36 s). |
| 10 | **Stale factorization.** deal.II's `SparseDirectMUMPS` wrapper calls `KSPSetReusePreconditioner(true)`. After the matrix was assembled again (next shape in an optimization) the factorization of the first shape was still used. | In a small optimization run (4 processes) the second and third shape needed 31 and 62 (= iteration limit) GMRES steps for the forward and adjoint solve together; with a fresh factorization they need 23 and 28. |
| 11 | **Slow ordering.** The same wrapper forces the AMF ordering in MUMPS (`ICNTL(7)=2`). | The factorization of a subdomain with 250 000 unknowns took 49.8 s; with MUMPS' own choice (METIS) 17.6 s, with identical results. |
| 12 | The L2 error of the processes was summed instead of combined as root of the sum of squares. | The reported error grew with the number of processes (factor √2 from 2 to 4, 2 from 2 to 8). |
| 13 | `MPICommunicator::get_neighbor_for_interface` used the number of processes in y as the rank offset in y (correct: in x). | Wrong for layouts with different counts in x and y. The function is currently not called. |

Items 10 and 11 are fixed by replacing the wrapper with an own PETSc `KSP` (`LocalProblem::solve`), which can be
configured with options that carry the prefix `local_`.

### 2.4 Robustness

- Crash at exit (`MPI_Comm_size after MPI_FINALIZE`): the global timer objects were destroyed after MPI.
- Crash of single process runs: `HierarchicalProblem::residual_output` was deleted without being initialized.
- A wrong number of MPI processes ended the program silently with exit code 0. It is an error message now.
- The default parameter files contained entries that no longer exist; running without arguments aborted.
- The program only worked when started from a folder one level below the repository root, and `Solutions/` had to
  exist. Paths are now resolved relative to the repository (`WAVEGUIDE_ROOT`, `WAVEGUIDE_OUTPUT_DIR`).

### 2.5 Measured effect

Signal strength at the output of the straight waveguide (1 = perfect), 20 cells per process and direction:

| | Before | After |
| --- | --- | --- |
| Tapered coupling | 1.561 (12 GMRES steps, varying) | 0.950 (4 steps) |
| Dirichlet coupling | 0.941 (16 steps, varying) | 0.990 (5 steps) |

With refinement (12, 20, 32 cells per process and direction) the tapered case now gives 0.858, 0.950, 0.988 and the
Dirichlet case 0.915, 0.990, 1.020; the L2 error against the mode drops 0.75, 0.32, 0.15. The remaining error on
these meshes is numerical dispersion (the phase of the transmitted signal lags), which is expected for lowest order
elements at 3 to 8 cells per wavelength.

## 3. Open issues

These were found but not fixed. They are ordered by importance.

1. **Shape gradient: fixed in a second step, see section 8.** What remains open there: bounds for the shape
   parameters and the handling of field solves that did not converge.
2. **The optimization does not check whether the field solves converged.** In the same small run the line search
   stepped to strongly bent shapes (slopes above 1), for which GMRES reached its iteration limit on the coarse mesh
   even with a fresh factorization. The loss and gradient of these steps are computed from unconverged fields and
   are meaningless (one step reported a loss of −0.98). A failed solve should end the step (return a large loss to
   the line search), and the shape parameters need bounds.
3. **Consequences for the dissertation.** Items 6, 7 and 10 were present when the experiments ran. The straight
   waveguide experiments are affected by 6: they had a randomly perturbed PML. The optimization experiment of
   chapter 4 is affected by 7 and 10: the interior of the domain did not depend on the shape parameters, and the
   preconditioner was built from the factorization of the first shape, which explains part of the decaying GMRES
   convergence shown there. Its numbers (loss 0.73 % to 0.48 %) should be treated as unverified until the experiment
   is repeated. The hierarchical runs used the relative inner tolerance 1e-2 with plain GMRES (item 9), so their
   final accuracy may be lower than the residual plots suggest. None of this was re-run at the original scale.
4. `solve directly = true` with 4 processes in z returned NaN in one configuration (`1 x 1 x 4`, 4 cells in z per
   process) while sweeping on the same system works. Not investigated; it only concerns the reference solver.
5. `RectangularMode` (an unfinished mode solver) is still called by `SweepingRun` for `source type = 0` and would
   dereference null pointers. `ModeManager` (cylindrical modes) is unused. Both should be removed. There is no mode
   solver in the code: the profile is imported from EIMS, and only for one cross-section and wavelength.
6. The "exact solution" used for the error with tapered coupling is the plain mode, not `ψ·E`; the error therefore
   includes the taper region. `ExactSolutionRamped` ramps in the opposite direction and is unused.
7. `ConvergenceRun` reports a mean of pointwise errors at cell centers (an L1-type quantity), which the dissertation
   labels as an L2 norm.
8. `Parameters::check_validity` iterates with unsigned integers over coordinates and never detects anything.
   The taper region is fixed to z in [0, 1] and cannot be set in the case file. `Kappa 0` is declared as double and
   read as integer.
9. Not run to completion after the migration: HSIE cases, the Hertz dipole convergence study and `fem order = 1/2`.
   A predefined hump shape was only run on a mesh far too coarse to be meaningful (it runs and GMRES converges).
   Memory is leaked at shutdown (children of `NonLocalProblem`, geometry levels).

## 4. Tests

Implemented (`ctest`, 12 targets, about 4 minutes in a release build on 8 cores):

- 39 unit tests. New are the tests in `tests/ModeAndGeometryTests.cpp`; each pins one of the errors above: the mode
  is divergence free in the core, it fulfills the Helmholtz equation for the configured β and ε, the core size of the
  parameters matches the profile, the analytic curl and the ramp derivative match finite differences, a straight
  waveguide is deterministic and has the identity as transformation, the Jacobian matches finite differences, and
  the neighbor ranks of a 2 x 3 x 2 layout.
- 11 integration tests (`tests/integration/run_integration_test.py`; the last two were added with sections 8 and 9:
  compressed subdomain factorizations reproduce the exact ones, and the adjoint gradient agrees with finite
  differences within 1 %): transmission through a straight waveguide is
  close to 1 for both coupling methods, refinement improves it, two identical runs give identical numbers, level 1
  sweeping with 2 and 4 subdomains, level 2 and level 3 sweeping all reproduce the direct solution of the global
  system to 1e-6, and the global system does not depend on the partition.

Recommended next, in this order:

1. ~~Gradient check~~ (done, `integration_shape_gradient`). Extend it to variable height and width and to a
   predefined shape.
2. **Manufactured solution for the transformation**: a plane wave in a homogeneous medium, sheared by
   `AngleWaveguideTransformation`, has an exact solution. It tests transformation optics in the interior and the PML
   together, and gives a real convergence rate (order `k` in H(curl)) for `fem order` 0, 1, 2.
3. **Hertz dipole convergence** as a slow test (`ctest -L long`), with the error measured in a proper L2 norm.
4. **PML reflection test**: one subdomain, a mode entering the PML; the reflected amplitude as a function of
   `PML sigma max`, layers and order. It turns the parameter study of the dissertation into a regression test and
   tells you the accuracy floor of the sweeping preconditioner.
5. **Iteration count regression**: GMRES steps for 2, 4, 8 subdomains with fixed subdomain size must stay below a
   bound (the scaling experiment of the dissertation in small).
6. **HSIE**: at least one single process test, since the code path is otherwise untested.
7. Unit tests for `PMLSurface::get_pml_tensor` (identity at the interface, symmetric, positive imaginary part) and
   for the dof bookkeeping of `NeighborSurface` (every interface dof is owned by exactly one process).
8. Run the unit tests and two integration tests in CI in debug mode; debug mode is where deal.II reports misuse.

## 5. GPUs

Short version: there is now a workable route on NVIDIA hardware, it is not yet in a PETSc release, the expected gain
for this algorithm is moderate, and nothing exists for the GPU of an Apple Silicon machine. The cheaper gains are on
the CPU. The statements about external software in this section come from a literature and documentation survey in
November 2026 and were not tested here.

What the preconditioner needs: one factorization per subdomain, then very many forward/backward substitutions with a
single right hand side. The solve phase dominates long runs (the 3-level run in the dissertation: 216 960 solves).

| Option | Factorization on GPU | Triangular solve on GPU | Complex | Reachable from this code |
| --- | --- | --- | --- | --- |
| **NVIDIA cuDSS 1.0** | yes | yes | yes, incl. complex symmetric LDLᵀ | PETSc `MATSOLVERCUDSS` exists on PETSc `main` only (not in 3.26): `-local_pc_factor_mat_solver_type cudss` with CUDA vectors. Sequential matrices only, which is exactly the subdomain case. Proprietary, NVIDIA only |
| STRUMPACK 8 (+MAGMA) | yes (CUDA/HIP/SYCL) | yes with MAGMA | to be confirmed for a complex PETSc build | PETSc `-local_pc_factor_mat_solver_type strumpack -mat_strumpack_gpu`; also offers BLR/HSS compression |
| SuperLU_DIST 9.3 | yes | yes ("GPU-resident solve"), whether PETSc reaches it is unconfirmed | yes | PETSc `superlu_dist`, `-mat_superlu_dist_gpuoffload` |
| MUMPS 5.9 | BLAS-3 offload (`ICNTL(51)`) | **no** | yes | already used; needs a MUMPS built with `-DUSE_GPU` |
| Trilinos Tacho | yes | yes | unclear | via Tpetra/Amesos2, would require leaving PETSc |
| deal.II itself | `Portable::MatrixFree` is for matrix-free operators (FE_Q); no GPU direct solver wrapper | | | |

Assessment:

- Published speed-ups for the single right hand side triangular solve are in the range of 2 to 4 (cuDSS release
  notes and blog, SuperLU_DIST SC23 paper), not tens. Triangular solves are sequential along the elimination tree,
  and the fronts of a subdomain with a few 100 000 unknowns are too small to fill a GPU. Factorization gains are
  larger but the factorization happens once.
- GPU memory limits the subdomain size: complex factors of 1e6 unknowns of a 3D Nédélec matrix need tens of GB.
  PETSc does not expose cuDSS' hybrid memory mode. The hierarchical splitting helps here because it keeps the
  lowest level blocks small.
- The sweep runs one subdomain at a time. One GPU shared by all processes of a sweep is therefore enough in
  principle, but the factors of all subdomains must then fit on it at once.

Recommended order of work:

1. **Done here:** MUMPS' default ordering instead of AMF (factorization 2.8 times faster) and FGMRES on the upper
   levels (35 % fewer subdomain solves in the 3-level test).
2. **Try on the CPU first:** block low rank compression in MUMPS for the subdomain factors
   (`-local_mat_mumps_icntl_35 2 -local_mat_mumps_cntl_7 1e-6`). The subdomain solves only have to be as accurate as
   the PML approximation they are part of, so a tolerance of 1e-4 to 1e-6 should not cost iterations, and both the
   memory and the solve time shrink. MUMPS 5.9 adds mixed precision factors (`ICNTL(47)`). This needs no new
   hardware and is one command line option now.
3. **Batch the right hand sides.** GPUs (and CPUs) are far more efficient with many right hand sides at once. The
   structure of the algorithm offers this on the lower levels: all y-sweeps (and x-sweeps) that belong to different
   z-blocks are independent. Today they are executed one after the other.
4. **cuDSS through PETSc** once it is in a release (3.27), on a machine with an NVIDIA GPU: change the solver type of
   `local_ksp` and use `aijcusparse` matrices and CUDA vectors for the local problem. The code change is confined to
   `LocalProblem`.
5. **Longer term, no factorization:** deal.II 9.8 has matrix-free evaluation for Nédélec elements. Recent work
   solves time-harmonic Maxwell with PML with complex shifted, matrix-free multigrid on GPUs at the scale of 1e10
   unknowns (arXiv:2608.22903). That is a different solver, but it is where GPU hardware fits naturally; the
   sweeping preconditioner could remain the coarse or reference solver.

## 6. Usability recommendations

Done: `--help`, readable errors for wrong parameter files and process counts, working defaults, independence of the
working directory, `results.txt`, results printed at the default log level, a single install script, `ctest`.

Recommended:

1. **One parameter file with sections and cross-checks**, and derive what can be derived: the sweeping level from the
   process counts, and the cells per direction from a target "cells per wavelength". Today a mismatch between level
   and process counts fails late. Add `--print-parameters` (ParameterHandler can write a documented template).
2. **Global cell counts instead of cells per process**, so that changing the partition does not change the problem.
3. **Remove the global objects** (`GlobalParams`, `Geometry`, ...). They make tests order-dependent (each test has to
   reset them) and prevent two problems in one process. Pass a parameter object and a geometry object explicitly.
4. **Make the taper region, the mode file and the wavelength parameters.** Add the propagation constant and the core
   size to the header of the mode file and check them against the case file at start-up; the swapped-convention
   error would have been caught by such a check (the divergence test of section 4).
5. **A mode solver.** Either a small script next to the code (the 2D vectorial eigenproblem with a 2D Nédélec/Q
   discretization and SLEPc) or at least a documented, scripted EIMS export. Right now a new cross-section or
   wavelength cannot be simulated.
6. **Checkpointing for long runs** and writing `results.txt` incrementally (residual history per level), since the
   large runs take days.
7. **Logging:** plain text without color codes when the output is not a terminal; one log file; a progress line with
   an estimate of the remaining subdomain solves.
8. **Clean the repository:** remove `compile_commands.json`, `.vs`, `.project`, `.settings`, `Skripte/Old*`, the
   committed LaTeX of the documentation; move the Slurm scripts into one parametrized script; delete dead code
   (`RectangularMode`, `ModeManager`, `Sector`, unused exact solutions).
9. **A Python post-processing script** that reads `results.txt` and the convergence histories and produces the plots
   (instead of calling gnuplot from C++).

## 7. A contribution to deal.II

deal.II already has step-81 (time-harmonic Maxwell with PML in 2D, serial, direct solver) and step-97 (Nédélec
elements, electromagnetics). What this code can add is the solver: nothing in the tutorials or the code gallery
shows a sweeping (or any other domain decomposition) preconditioner for an indefinite wave problem.

Suggested path:

1. **Code gallery first.** The requirements are lighter than for a tutorial (a self-contained program with a
   README, the author as maintainer). Title along the lines of "A sweeping preconditioner for time-harmonic Maxwell's
   equations".
2. **Scope of the program:** the setting of step-81 extended to a waveguide-like domain in 2D or a short 3D
   waveguide with `FE_NedelecSZ`; slabs in the propagation direction; each slab problem with an auxiliary PML and a
   direct solver; `SolverFGMRES`; a table of iteration counts against the number of slabs and against PML
   parameters. One level of sweeping. The hierarchical variant can be the "possibilities for extensions" section, or
   a second program.
3. **Keep it dependency-light.** Tutorials that need a complex PETSc reach few users (and deal.II's own complex
   PETSc build is currently broken, see section 1). A version with deal.II's own complex types
   (`SparseMatrix<std::complex<double>>`, `SparseDirectUMFPACK`, or `LinearAlgebra::distributed::Vector` with MPI)
   can be run by everyone. The slab solves can be distributed with MPI or simply with threads.
4. **What has to change compared with this code:** no global state, one file, the dof bookkeeping between slabs done
   with deal.II means (a `parallel::distributed` or `shared` triangulation with one subdomain id per slab and
   `DoFTools` index sets, instead of one triangulation per process with hand-made interface numbering), and an
   introduction that derives the block LDLᵀ factorization and the moving PML as chapter 3 of the dissertation does.
5. **Smaller contributions on the way:** the two bug reports of section 1; a pull request that removes the forced
   AMF ordering and the forced preconditioner reuse from `PETScWrappers::SparseDirectMUMPS` or makes them options
   (section 2.3, items 10 and 11); a test for `FE_NedelecSZ` with complex PETSc vectors.

Before submitting, read the deal.II contribution guidelines and ask on the mailing list whether the maintainers
prefer a code gallery entry or a tutorial; the license of the contribution has to follow the current deal.II
licensing terms.

## 8. Shape optimization: analysis of the code and the cases

### 8.1 What the code did

A gradient check was added (`perform gradient check = true` in a case file with `perform optimization = true`): it
compares the adjoint gradient at the initial shape with central finite differences of the loss. For a waveguide of
6 µm with a hump of 0.5 µm (4 processes, 16 x 16 x 48 cells, all field solves converged to 1e-8):

| Shape parameter | Finite difference | Original adjoint gradient | New, without PML term | New, with PML term |
| --- | --- | --- | --- | --- |
| 0 | 0.10792 | −0.0300 | 0.1044 | 0.10792 |
| 1 | 0.11802 | −0.0362 | 0.1148 | 0.11802 |
| 2 | 0.06700 | −0.0073 | 0.0650 | 0.06700 |

(Finite differences with a step of 0.005; the new gradient agrees with them to 5e-5 relative.)

The original gradient had the wrong sign and no fixed ratio to the true one. The causes, all in the path
`OptimizationRun::perform_step` → `set_rhs_for_adjoint_problem` → `solve_adjoint` → `compute_shape_gradient`:

1. **Adjoint source at the wrong place.** `InnerDomain::set_rhs_for_adjoint_problem` wrote into the distributed
   vector with the *local* dof indices of the process that owns the output face. These are valid global indices of
   the first process, so the source of the adjoint field sat in the first subdomain instead of at the output.
2. **Wrong adjoint source.** The entries were `−2 Re(κ · conj(u_i) ∫ m·φ_i)`: real numbers that contain the solution
   coefficient of the dof itself and the mode instead of its conjugate. The signal `S(u) = ∫ u·conj(m)` is linear in
   `u`, so the correct right hand side is the complex vector `G_i = ∫ conj(m)·φ_i` and does not depend on `u`.
3. **Conjugated adjoint in the gradient.** The gradient used `conj(λ)` and `conj(curl λ)`. The system matrix is
   complex symmetric, not hermitian; the derivative is the bilinear expression `λᵀ (∂A/∂p) u` without conjugation.
4. **No tensor derivative.** `get_Tensor_for_step` read the old value with `get_dof(j)` (index in *all* dofs, the
   first two of which are constraints) but wrote with `set_free_dof(j)`. It therefore computed
   `T(p_j = [value of dof j] + 0.01) − T(p)`, did not divide by the step, and "restored" the free parameter to the
   wrong value. After the first quadrature point the shape parameters were overwritten, so the tensor differences
   belonged to a different shape than the fields.
5. **No quadrature weights** and one point per cell; the factor `conj(s)/|s|` of the derivative of `|s|` was missing
   (only the real part was taken); the loss is `−|s|` while the source was built for `−|s|²`.
6. **The right hand side depends on the shape as well** (the tapered source contains `μ⁻¹`), which was ignored.

Together with the two errors of section 2 (interior not transformed, stale factorization) this means the
optimization in the state of the dissertation could not work, independent of resolution.

### 8.2 What it does now

`InnerDomain::assemble_adjoint_rhs_for_signal_strength` and `InnerDomain::compute_shape_gradient_integrals`
implement, for `J = −|s|`, `s = S/N`, `S = Gᵀu`, `A λ = G`:

`dJ/dp_j = Re( conj(s)/|s| · λᵀ (∂A/∂p_j u − ∂b/∂p_j) ) / N`

with the assembly quadrature, central differences of the transformation tensor (step 1e-5) and the derivative of the
tapered source. The cells of the PML domains contribute to `∂A/∂p` as well, because their material tensors contain
the space transformation (`PMLSurface::compute_shape_gradient_integrals`); without this term the gradient was 3 %
too small. The old functions (`compute_kappa`, `set_rhs_for_adjoint_problem`,
`compute_local_shape_gradient_data`, `get_Tensor_for_step`) are still in the code but no longer called; they can be
deleted.

The optimization now has stopping criteria, all set in the case file:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `optimization gradient tolerance` | 1e-2 | stop when the gradient norm is below this factor times its initial value |
| `optimization loss tolerance` | 1e-5 | stop when four consecutive evaluations did not improve the best loss by more than this factor times the loss |
| `optimization maximum evaluations` | 40 | forward + adjoint solves |
| `optimization maximum steps` | 15 | BFGS steps |

At the end the fields of the best shape are computed again, so that the outputs belong to the result, and
`optimization_history.txt` lists loss, gradient norm and shape parameters of every evaluation.

A BFGS run on the same case reduces the loss from −0.9072 to −0.9535 in two evaluations (gradient norm 0.173 →
0.008) and then creeps to −0.9547 until evaluation 13. After that the line search steps to shapes with slopes
around ±1, reaches −0.9655 and stops by itself after 23 evaluations because the loss stagnates. The final shape
was solved to tolerance (40 GMRES steps), but several evaluations of this last phase ran into the iteration limit,
and with slopes near ±1 on this coarse mesh the gain from 0.955 to 0.965 needs confirmation on a finer mesh
(see 8.3, item 1).

### 8.3 What is still wrong or missing

1. **Unconverged solves are used** (section 3, item 2), and there are no bounds on the slopes. The transformation
   itself stays regular for any slope, but the mesh resolves the sheared medium less and less: in the run above
   GMRES needed 14 steps per solve for the initial shape, about 33 around evaluation 5, and from evaluation 15 on
   several solves hit the limit of 60. A failed solve should end the evaluation with a large loss, and the slopes need bounds (or
   the mesh has to be fine enough for the largest admissible slope).
2. Check whether the lateral PML is still absorbing for a strongly sheared tensor; the PML and the shear do not
   commute.
3. **Loss functional.** `s` is a point-sample (midpoint) overlap including the longitudinal component, on the last
   cell layer before the PML. A power-normalized overlap with a proper quadrature would be the physical quantity;
   the dissertation states `1 − |·|²`, the code minimizes `−|·|`.
4. **Parametrization.** `ShapeFunction` mirrors the shape at the middle of the domain, so "vertical waveguide
   displacement" is the height of a hump and input and output are always at the same height; an S-bend between two
   different connector positions cannot be described. Its constructor uses `z_max/2` as the mirror plane, which is
   only correct if the domain starts at z = 0. The free parameters are slopes at the nodes; one slope is eliminated
   to hit the prescribed height, which makes the last sector carry the correction.

### 8.4 The case files

| Case | Setup | Assessment |
| --- | --- | --- |
| `optimization/base.prm` | 4 x 3.6 x 10 µm, hump 2 µm, taper, PML 1.5 µm / 8 layers / σ 20, run file `10-1-1` (10 processes in z, 20³ cells each) | Usable. Slope of the initial shape 0.4: resolution in z (h = 0.05) is fine, in x and y (0.2 and 0.18) it is marginal. `GMRES maximum steps = 20` is too low for bent shapes |
| `optimization/straight_geometry_taper.prm` | as above, hump 0 | With the old code the start was a random shape. Now the start is the straight waveguide, which is already optimal: only useful as a test that the gradient is zero there |
| `optimization/larger_geometry_dirichlet.prm` | 15 µm, hump 4 µm, variable width and height, Dirichlet input | Dirichlet coupling reflects the backward wave of a bent waveguide back into the domain (see the section on Dirichlet data in the dissertation), so it is the wrong coupling for optimization. The Jacobian for variable width/height was wrong (section 2.2, item 8) and is untested in a solve |
| `optimization/larger_geometry_taper.prm` | 8 x 7.2 x 50 µm, hump 10 µm, variable height, 640 processes (`8-8-10`: 10 x 10 x 60 cells per process, 3 levels) | Cells of 0.1 x 0.09 x 0.083 µm are fine, but `solver precision = 1e-3`, `GMRES maximum steps = 5` and inner tolerance 1e-2 are far too loose for gradients; with plain GMRES on the upper levels (section 2.3, item 9) the fields of this run were not reliable |

The experiment of the dissertation (6 µm, hump 1.5 µm, 12 x 12 x 48 cells, 4 processes) has no case file; the
closest is the one used for the gradient check here. Its transverse resolution (0.33 µm, three cells per wavelength
in the core) is too coarse to trust loss differences of a fraction of a percent.

Recommended next steps: treat failed solves and bound the slopes; rerun the
dissertation experiment at 16 to 20 cells per direction and process with tapered coupling; then decide about the
parametrization (free end height, control points instead of slopes).

## 9. Block low rank compression of the subdomain factors

New run parameter `local solver compression tolerance` (0 = off). It switches on BLR in MUMPS for the subdomain
factorizations (`ICNTL(35) = 2`, `CNTL(7)` = tolerance); `results.txt` now reports the factorization time, the time
per subdomain solve, the entries of the factors with and without compression and the memory.

Measured for the straight waveguide with tapered coupling, 2 processes, lowest order elements, `n` cells per process
and direction (unknowns per subdomain in brackets). MUMPS runs sequentially here (no OpenMP), on an M5 Max:

| n | Tolerance | Stored / full factor entries | Memory per process | Factorization | Time per solve | GMRES steps |
| --- | --- | --- | --- | --- | --- | --- |
| 32 (≈230k) | off | 1.00 | 3.6 GB | 15.7 s | 0.51 s | 4 |
| | 1e-6 | 0.77 | 3.2 GB | 20.0 s | 0.39 s | 4 |
| | 1e-4 | 0.63 | 2.8 GB | 15.8 s | 0.34 s | 4 |
| | 1e-3 | 0.55 | 2.6 GB | 13.4 s | 0.34 s | 5 |
| | 1e-2 | 0.47 | 2.4 GB | 12.1 s | 0.29 s | 16 |
| 40 (≈390k) | off | 1.00 | 7.0 GB | 42.3 s | 1.13 s | 4 |
| | 1e-6 | 0.68 | 5.8 GB | 43.4 s | 0.76 s | 4 |
| | 1e-4 | 0.55 | 5.0 GB | 31.1 s | 0.59 s | 4 |
| | 1e-3 | 0.47 | 4.6 GB | 28.1 s | 0.53 s | 7 |
| | 1e-2 | 0.39 | 4.2 GB | 22.2 s | 0.45 s | 22 |

The transmitted signal is identical to 8 digits in all rows, because the outer GMRES still solves the uncompressed
global system.

Reading:

- A tolerance of **1e-4** is the sweet spot here: no additional GMRES steps, the solve phase is 1.5 (n = 32) to 1.9
  (n = 40) times faster, the factorization is not slower, and the memory drops by 20 to 30 %. At 1e-3 the iteration
  count starts to grow and from 1e-2 on the compression costs more than it saves.
- The gain grows with the size of the subdomain, as expected for BLR (at n = 20 the factors only shrink to 0.77).
  It should be larger for the subdomain sizes of production runs and for `fem order = 1`; neither was measured.
- With tolerances of 1e-6 and tighter the factorization gets slower for these sizes: the compression work is not
  yet paid back.

Not done yet:

1. Levels 2 and 3. There the subdomain solver sits below inner iterative solvers, which tolerate more error; the
   tolerance could be tied to `relative convergence criterion`.
2. A MUMPS build with OpenMP and a threaded BLAS. BLR was designed for multithreaded fronts; PETSc was configured
   without OpenMP here.
3. `ICNTL(36) = 1` (the UCFS variant, more compression), `ICNTL(38)` (estimated compression rate for the memory
   allocation), mixed precision (`ICNTL(47)`, MUMPS 5.9) and adaptive precision BLR (`ICNTL(40)`).
4. The memory reported is the peak during the factorization, not the size of the factors that have to be kept; the
   relevant number for the limit on the subdomain size needs `INFOG` 36 to 39.
5. An automatic choice: start compressed and fall back to an exact factorization if GMRES needs more than a few
   steps more than expected.

## 10. PML against Hardy space infinite elements: accuracy and memory

Script: `examples/pml_vs_hsie.py` (data and figures in `examples/output/pml_vs_hsie/`).

### 10.1 State of the HSIE implementation

The comparison could not be done with the solver alone, because **the HSIE implementation does currently not
absorb**. Findings:

- With HSIE on several faces the dof numbering leaves indices that belong to no element (444 of 2375 on a 4³
  mesh); the matrix was singular. `LocalProblem::assemble` now turns such rows into `u_i = 0`.
- With that fixed, the Hertz dipole in free space (five HSIE faces) has a relative error of 2.5 to 10 for every
  `κ₀` and degree; PML gives 0.19 on the same mesh.
- To separate the method from the known edge and corner problem, a test with **one** absorbing face was added: the
  TE10 mode of a waveguide with perfectly conducting walls (`source type = 4`, `Lateral boundary = PEC`), which has
  the exact solution `E = (0, cos(πx/a), 0) exp(iβz)`. PML: error 0.069 at 12 cells per direction, converging with
  the mesh. HSIE: error 1.1 to 1.8, independent of the degree and of the sign of `κ₀`, i.e. worse than a fully
  reflecting wall.
- `HSIESurface::evaluate_a` lacks the factor `−2iκ₀` of the identity `∫ u v dξ = −2iκ₀ Σ aₖbₖ`. Adding it does not
  repair the method, so there is at least one more error (candidates: the orientation of the Nédélec functions of
  the 2D surface mesh relative to the 3D interior, the coupling of surface and exterior unknowns). This was not
  pursued further. The git history has no state in which the factor was present.

Two more errors surfaced on the way and are fixed: the exact solution of the Hertz dipole used the wavenumber `2π`
instead of that of the equation (`2π/λ`), so no solver could converge to it; and `κ₀` from the case file was read as
an integer and then overwritten by a constant.

### 10.2 Method of the comparison

- **Memory** comes from the 3D solver: number of unknowns, matrix entries and entries of the MUMPS factors for PML
  with 2 to 12 cell layers and for HSIE of degree 1 to 12. These depend only on the structure of the matrix, which
  is right for HSIE even though its values are not.
- **Accuracy of PML** comes from the 3D solver as well (best `σ_max` out of 5, 10, 20, 40 for each number of
  layers, quadratic profile, PML cells of the size of the interior cells).
- **Accuracy of both methods** comes from a reference model (`examples/absorbing_boundary_model.py`): for one
  transverse mode the test problem is `u″ + β²u = 0` on a line, and both truncations can be written down in a few
  lines. The HSIE part was validated by its exponential convergence in the degree. The model gives the error for
  the mode of the test problem (β = 0.92 k) and the worst case over angles of incidence from 0° to 75°
  (β = k cos θ), with `κ₀ = k`.

### 10.3 Results

3D solver, lowest order elements, error of PML against the exact solution (best `σ_max`) and factor memory:

| n (cells per direction) | 2 layers | 4 layers | 6 layers | 8 layers | 12 layers |
| --- | --- | --- | --- | --- | --- |
| 12 | 0.108 / 26 MB | 0.071 / 30 MB | 0.068 / 36 MB | 0.068 / 41 MB | 0.068 / 37 MB |
| 16 | 0.101 / 57 MB | 0.057 / 64 MB | 0.040 / 72 MB | 0.039 / 80 MB | 0.039 / 98 MB |
| 24 | 0.190 / 278 MB | 0.055 / 303 MB | 0.026 / 330 MB | 0.019 / 356 MB | 0.018 / 417 MB |

With lowest order elements the error stops improving after 6 to 8 layers: what remains (0.068, 0.039, 0.018) is
the discretization error of the interior. More absorber buys nothing there.

Factor memory at n = 24 (one absorbing face of 24 × 24 cells; 278 MB of it is the interior):

| PML layers | 2 | 3 | 4 | 6 | 8 | 12 |
| --- | --- | --- | --- | --- | --- | --- |
| MB | 278 | 291 | 303 | 330 | 356 | 417 |
| **HSIE degree** | **1** | **2** | **3** | **5** | **8** | **12** |
| MB | 292 | 308 | 320 | 373 | 446 | 573 |

One PML layer costs about 14 MB here, one HSIE degree 14 MB at low degrees and 32 MB at high degrees: all exterior
unknowns above one surface cell are coupled with each other, so the fronts grow with the degree, while PML layers
only couple to their neighbors. In unknowns the two are equal (59 600 for 8 layers, 62 050 for degree 8).

Reference model, error of the boundary method alone (fine interior mesh, second order elements), worst case over
0° to 75°:

| Target error | PML layers | HSIE degree | Memory PML | Memory HSIE |
| --- | --- | --- | --- | --- |
| 1e-1 | 3 | 1 | 291 MB | 292 MB |
| 1e-2 | 4 | 3 | 303 MB | 320 MB |
| 1e-3 | 8 | 5 | 356 MB | 373 MB |
| 1e-4 | 12 | 8 | 417 MB | 446 MB |

(Memory from the 3D measurements above; the PML layer counts assume second order elements in the PML.)

### 10.4 Reading

1. **Per unknown, HSIE is the better absorber.** Its error falls exponentially with the degree, by about a factor
   of 3 per degree in the worst case over angles, and it is exact up to the interior discretization for waves with
   `β ≈ κ₀` already at degree 0 or 1. PML needs about 1.5 times as many layers as HSIE needs degrees (3 against 1
   for the loosest target).
2. **Per megabyte, the two are close to equal for one face.** The denser coupling of HSIE costs what its faster
   convergence gains: at equal accuracy the factor memory differed by 0 to 7 % in favor of PML.
3. **The order of the elements matters for PML, not for HSIE.** With first order elements the PML itself is
   discretized poorly: the model needs 16 layers for 1e-2 and does not reach 1e-3, while HSIE keeps its rate. For
   the lowest order elements this code uses by default, HSIE of degree 3 (320 MB) replaces a PML of 16 layers
   (more than 417 MB), provided such accuracy is needed at all (see the first table: it is not, as long
   as the interior error is a few percent).
4. **PML has four parameters, HSIE one.** The best `σ_max` changed with the number of layers and with the mesh
   (5, 10, 20 and 40 all occur in the table above); a poor choice costs a factor of 2 to 10 in error. HSIE has
   `κ₀`, and `κ₀ = k` worked throughout.
5. **The large difference would be at edges and corners.** A PML of L layers around a block of n³ cells has
   (n + 2L)³ − n³ cells: for n = 20 and L = 8 that is 4.8 times the interior, and it is why up to 80 % of the
   unknowns in the sweeping runs are PML unknowns. Infinite elements only add unknowns on the faces: with the
   measured 20 unknowns per face cell for degree 5, all six faces of the same block would add about 1.8 times the
   unknowns of the interior. This saving is hypothetical as long as
   the treatment of edges and corners is unsolved (dissertation, section on sweeping with HSIE) - and it is the
   reason why repairing HSIE would be worth the effort for the subdomain problems.

Not covered: evanescent waves and grazing incidence beyond 75°, materials that vary along the boundary (the
waveguide core crossing the absorbing face), and `κ₀` other than `k`.

### 10.5 Second order elements in 3D (`--scale paper`)

With second order elements the interior error is small enough for the absorber to matter: the PML error against
the exact solution reaches 4.3e-3 (n = 8), 1.2e-3 (n = 12) and 5e-4 (n = 16) with 6 to 8 layers. The memory picture
shifts in favor of HSIE, because a PML layer of second order elements is more expensive while the infinite
elements grow less. Factor memory at n = 16 (926 MB of it interior), for the layers / degrees of the table in 10.3:

| Target error | PML | HSIE | HSIE relative to PML |
| --- | --- | --- | --- |
| 1e-2 | 4 layers: 1084 MB | degree 3: 1033 MB | −5 % |
| 1e-3 | 8 layers: 1380 MB | degree 5: 1134 MB | −18 % |
| 1e-4 | 12 layers: 1671 MB | degree 8: 1372 MB | −18 % |

Counted without the interior, the absorber itself costs 454 MB (PML, 8 layers) against 208 MB (HSIE, degree 5) for
1e-3. For lowest order elements at n = 32 the numbers stay within 0 to 9 % in favor of PML (for example 1049 MB
against 1060 MB at 1e-3), as in 10.3. So: parity for lowest order elements, roughly half the absorber memory with
HSIE for second order elements - on one face, and for an HSIE implementation that works.

## 11. Examples and web frontend

- `examples/`: one script per graph of the dissertation plus the comparison of section 10; see
  `examples/README.md`. The dissertation refers to them in the captions and in the section "Reproducing the
  Figures".
- `frontend/`: a browser application to configure a waveguide, start the solver and look at the result.
  `python3 frontend/server.py`, then open http://localhost:8750. The server uses only the Python standard library;
  the page has no external dependencies. It needs the field slices that the solver now writes
  (`field_slice_<plane><process>.txt`: the field at the cell centers on the planes x = 0, y = 0 and z = middle).
- New in the solver for these: `source type = 4` with `Lateral boundary = PEC`, `results.txt` for single process
  runs, `Kappa 0 Real / Imaginary` from the case file, a diagnostic for empty matrix rows at debug logging.
- Fixed on the way: a double delete at the end of `ConvergenceRun`.
