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
9. Not run to completion after the migration: the Hertz dipole convergence study and `fem order = 2`. (HSIE was
   repaired later, see section 10.1; second order elements are covered by the HSIE tests and the study of section 10.)
   A predefined hump shape was only run on a mesh far too coarse to be meaningful (it runs and GMRES converges).
   Memory is leaked at shutdown (children of `NonLocalProblem`, geometry levels).

## 4. Tests

Implemented (`ctest`, 17 targets, about 7 minutes in a release build):

- 39 unit tests. New are the tests in `tests/ModeAndGeometryTests.cpp`; each pins one of the errors above: the mode
  is divergence free in the core, it fulfills the Helmholtz equation for the configured β and ε, the core size of the
  parameters matches the profile, the analytic curl and the ramp derivative match finite differences, a straight
  waveguide is deterministic and has the identity as transformation, the Jacobian matches finite differences, and
  the neighbor ranks of a 2 x 3 x 2 layout.
- 15 integration tests (`tests/integration/run_integration_test.py`; `run_report` checks the run report against
  `results.txt` and the field outputs, `optimization_recovers_straight_waveguide` is the test of section 13; four
  more were added with sections 8, 9 and 10:
  compressed subdomain factorizations reproduce the exact ones, the adjoint gradient agrees with finite
  differences within 1 %, infinite elements absorb the mode of a metallic waveguide and agree with PML for the
  dielectric waveguide): transmission through a straight waveguide is
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
6. **HSIE**: done, see `hsie_single_face` and `hsie_matches_pml` (section 10.1).
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
| `optimization loss tolerance` | 1e-5 | stop when two consecutive BFGS steps did not improve the loss by more than this factor times the loss (until section 13: four consecutive evaluations) |
| `optimization maximum evaluations` | 40 | forward + adjoint solves |
| `optimization maximum steps` | 15 | BFGS steps |

At the end the fields of the best shape are computed again, so that the outputs belong to the result, and
`optimization_history.txt` lists loss, gradient norm and shape parameters of every evaluation.

A BFGS run on the same case reduces the loss from −0.9072 to −0.9535 in two evaluations (gradient norm 0.173 →
0.008) and then creeps to −0.9547 until evaluation 13. After that the line search steps to shapes with slopes
around ±1, reaches −0.9655 and stops by itself after 23 evaluations because the loss stagnates. The final shape
was solved to tolerance (40 GMRES steps), but several evaluations of this last phase ran into the iteration limit,
and with slopes near ±1 on this coarse mesh the gain from 0.955 to 0.965 needs confirmation on a finer mesh
(see 8.3, item 1). This run used the line search settings of deal.II that section 13 replaces; with the new
settings the same case reaches −0.9559 within the 14 evaluations of the example (before: −0.9547).

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

When the comparison was first made, the infinite elements of the solver did not absorb: on the test problem below
the error was 110 to 180 %, worse than a reflecting wall. The implementation has since been repaired. The errors,
all in the assembly of `HSIESurface` (none in the theory of the dissertation):

1. **Wrong volume factor.** `JacobianForCell::get_C_G_and_J` used the Frobenius norm of the Jacobian (`√3` for the
   identity) where the transformation rules need its determinant. This scaled the curl-curl term by `1/√3` and
   the mass term by `√3`.
2. **Missing factor of the Hardy space identity.** `∫ u v dξ = −2iκ₀ Σ aₖbₖ`; `evaluate_a` returned only the sum.
3. **Wrong operators.** The derivative in ray direction of a tangential function was computed by differentiating
   the polynomial on the unit disc (`applyDerivative`) instead of replacing `T₋` by `T₊`, i.e. `Ψⱼ` by `φⱼ`.
   In addition the matrices `D` and `D⁻¹` (multiplication by `1 + ξ`, needed for radial rays only) were applied
   to the curl of the tangential functions and to the ray functions; for axis parallel rays they do not belong
   there.
4. **Quadrature.** The shape function terms of the first quadrature point were reused for all points of a cell
   (vectors declared outside the loop and never cleared), and the rule had two points per direction regardless of
   the element order.
5. **Second order elements: wrong functions.** The orthogonal infinite cell functions were built from the cell
   functions of the Nédélec element instead of those of the nodal element, which duplicated the tangential
   functions and made the matrix singular.
6. **Second order elements: tangential continuity.** The functions of the two-dimensional Nédélec element on the
   surface mesh are not the traces of the interior functions they share their unknowns with: two of the four
   face functions differ by a factor of 2, and on the side faces they are linear combinations.
   `HSIESurface::compute_trace_coefficients` now computes, cell by cell, the trace of every interior function in
   the basis of the surface element and the infinite elements are built from these traces. It throws if a trace
   is not in the space of the surface element.

Earlier fixes on the way: unused dof indices are turned into `u_i = 0` rows; the exact solution of the Hertz
dipole used the wavenumber `2π` instead of `2π/λ`; `κ₀` was read as an integer and overwritten by a constant.

**What works now** (tests `hsie_single_face` and `hsie_matches_pml`):

| Case | PML | HSIE |
| --- | --- | --- |
| TE10 mode, one absorbing face, lowest order, n = 8 / 16 / 32 | 0.136 / 0.039 / 0.011 | 0.154 / 0.044 / 0.011 |
| TE10 mode, one absorbing face, second order, n = 4 / 8 / 16 | 0.045 / 4.3e-3 / 5.1e-4 | 0.047 / 4.6e-3 / 5.2e-4 |
| Dielectric waveguide, five absorbing faces, lowest order, 32³ cells | 0.295 (355 000 unknowns) | 0.299 (172 000) |
| Dielectric waveguide, five absorbing faces, second order, 12³ cells | 0.136 (560 000 unknowns) | 0.142 (83 000) |
| The same with tapered coupling, lowest order, 16³ cells | 0.796 | 0.797 |

(Relative L2 errors against the exact solution; in the waveguide rows the error is that of the coarse interior
mesh.) With `κ₀ = k` the result does not change beyond degree 1 to 2; with `κ₀ = 0.4 k` the error falls from 0.055
(degree 0) to the 0.044 of the mesh (degree 8), i.e. the convergence in the degree is as the theory predicts.
For the dielectric waveguide a complex `κ₀` is better, because the field in the cladding decays instead of
propagating: second order, 12³ cells, degree 2 / 5 gives 0.140 / 0.143 with `κ₀ = k`, 0.154 / 0.145 with the former
default `1 + i` and 0.1370 / 0.1370 with `κ₀ = k + 2i` (PML: 0.1359). `k + 2i` (`4.05 + 2i`) is now the default; the
single face test is unaffected by the imaginary part from degree 4 on.

**What does not work, and is not an implementation error:**

- **Fields that leave through the edges and corners.** The rays are parallel to the axes, so the regions beyond
  the edges and corners of the domain are not covered by infinite elements (dissertation, section "Sweeping
  Preconditioners with HSIE"). The code identifies the unknowns on the infinite faces of two surfaces over their
  common edge. For the Hertz dipole in free space, which radiates through all edges, the error is about 100 %
  (PML: 19 to 26 %), with this identification as well as without it. For a guided mode the field at the edges is
  negligible and the identification is harmless; it gave slightly better results than leaving these faces free
  (0.142 against 0.150 to 0.160 in the second order waveguide case), so it was kept.
- **Sweeping.** With two subdomains in z the sweeping preconditioner still reaches the PML result, but it used the
  full 30 GMRES steps where PML needs 5; with four subdomains it did not converge within 30 steps. HSIE is therefore
  a boundary condition for single process (direct) solves and for the outer boundary, as the dissertation
  concluded. The frontend restricts it to one process.

### 10.2 Method of the comparison

- **Memory** comes from the 3D solver: number of unknowns, matrix entries and entries of the MUMPS factors for PML
  with 2 to 12 cell layers and for HSIE of degree 1 to 12.
- **Accuracy of PML** comes from the 3D solver as well (best `σ_max` out of 5, 10, 20, 40 for each number of
  layers, quadratic profile, PML cells of the size of the interior cells).
- **Accuracy of HSIE** in the 3D solver is available since the repair (section 10.6). Sections 10.3 to 10.5 were
  written before it and take the degree HSIE needs from the reference model.
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
   unknowns of the interior. For the outer boundary of a waveguide problem this saving is real (section 10.6);
   for the subdomain problems of the sweeping preconditioner it stays hypothetical as long as the treatment of
   edges and corners is unsolved (dissertation, section on sweeping with HSIE).

Not covered: evanescent waves and grazing incidence beyond 75°, materials that vary along the boundary (the
waveguide core crossing the absorbing face), and `κ₀` other than `k`.

### 10.5 Second order elements in 3D (`--scale paper`)

With second order elements the interior error is small enough for the absorber to matter: the PML error against
the exact solution reaches 4.3e-3 (n = 8), 1.2e-3 (n = 12) and 5e-4 (n = 16) with 6 to 8 layers. The memory picture
shifts in favor of HSIE, because a PML layer of second order elements is more expensive while the infinite
elements grow less. Factor memory at n = 16 (926 MB of it interior), for the layers / degrees of the table in 10.3:

| Target error | PML | HSIE | HSIE relative to PML |
| --- | --- | --- | --- |
| 1e-2 | 4 layers: 1084 MB | degree 3: 1017 MB | −6 % |
| 1e-3 | 8 layers: 1380 MB | degree 5: 1100 MB | −20 % |
| 1e-4 | 12 layers: 1671 MB | degree 8: 1245 MB | −25 % |

(HSIE numbers from the repaired implementation; the earlier one had too many unknowns per cell for second order
elements.) For lowest order elements at n = 32 the numbers stay within 0 to 9 % in favor of PML (for example
1049 MB against 1060 MB at 1e-3), as in 10.3. So: parity for lowest order elements and a fifth to a quarter less
memory with HSIE for second order elements, on one face, if the degree has to cover a wide range of angles.

### 10.6 Measured with the repaired infinite elements

With the repaired implementation both methods can be compared in the 3D solver directly. On the test problem (one
mode with `β = 0.92 k`, `κ₀ = k`) the infinite elements reach the error of the mesh already at degree 1; higher
degrees change nothing. Cheapest configuration of each method whose error is within 15 % of the smallest error
reached on the mesh (`pml_vs_hsie_solver_tradeoff.txt`):

| Elements | n | Error of the mesh | PML | HSIE | HSIE / PML memory |
| --- | --- | --- | --- | --- | --- |
| lowest order | 12 | 6.8e-2 | 4 layers, 30 MB | degree 1, 26 MB | 0.86 |
| lowest order | 16 | 3.9e-2 | 6 layers, 72 MB | degree 1, 60 MB | 0.83 |
| lowest order | 24 | 1.8e-2 | 8 layers, 356 MB | degree 1, 292 MB | 0.82 |
| lowest order | 32 | 1.1e-2 | 12 layers, 1181 MB | degree 1, 894 MB | 0.76 |
| second order | 8 | 4.3e-3 | 6 layers, 102 MB | degree 1, 61 MB | 0.60 |
| second order | 12 | 1.2e-3 | 6 layers, 433 MB | degree 1, 302 MB | 0.70 |
| second order | 16 | 5.1e-4 | 6 layers, 1219 MB | degree 1, 915 MB | 0.75 |

Reading: for a single guided mode whose propagation constant is known (so that `κ₀` can be matched), HSIE needs
17 to 40 % less factor memory than a PML that is just thick enough, and it has no parameters to tune. The finer
the mesh, the more layers the PML needs to keep up with the interior accuracy, while the degree stays at 1. The
error of HSIE itself is 5 to 10 % above that of the best PML on the lowest order meshes (0.0200 against 0.0181 at
n = 24), which is a difference of the interior discretization, not of the absorber: it does not change with the
degree. This is the favorable case for HSIE; for fields with a wide range of angles the degrees of section 10.3
apply and the advantage shrinks to the numbers given there. With five absorbing faces (dielectric waveguide,
section 10.1) the difference is much larger, because PML also fills the edges and corners: 49 600 against 181 900
unknowns and 596 MB against 2283 MB of factors (second order, 8 × 8 × 12 cells) for the same error.

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

## 12. Performance: where the time goes and what was changed

Two example runs were profiled with a sampling profiler (`sample` on macOS, one process of each run).

| Run | Subdomain solves (MUMPS) | Waiting for other processes | Factorization | Assembly, setup, output |
| --- | --- | --- | --- | --- |
| Level 1, 4 subdomains of 20³ cells, 25 s | about 15 % | about 30 % | 31 % | about 22 % |
| Three levels, 16 processes (`hierarchical_sweeping.py`), 51 s | 6 % | 90 % | under 1 % | about 3 % |

- **The solve phase is serial.** A sweep solves the subdomains one after the other. In the three level run every
  process computes 1/16 of the time: one of 16 processes works at any moment. The wall time of the solution is the
  number of subdomain solves times the time per solve (3870 × 0.0115 s = 44 s of 51 s).
- **The code of the solver itself is about 1 % of the run.** The copies between the levels, the loops over single
  vector entries in the sweeps and the global matrix products of every sweep step are hardly visible. Compiler flags
  or small optimizations of this code gain nothing measurable.
- Every sweep step multiplies with the global matrix, so one sweep over n subdomains costs n global products
  (n² subdomain products). The wall time of one product is small against a subdomain solve, but it is wasted CPU
  time and n collective operations; a product restricted to the two subdomains involved would avoid it.

**Changed: flexible GMRES on level 1.** Right preconditioned GMRES applies the preconditioner once more at the end
of every solve to compute the solution. Flexible GMRES (already used on the higher levels) stores the preconditioned
vectors instead and gives the same iterates.

| Run | Subdomain solves before | after | Wall time before | after |
| --- | --- | --- | --- | --- |
| Three levels, 16 processes | 3870 | 1935 | 51 s | 26 s |
| Level 1, 8 subdomains | 90 | 75 | 44 s | 43 s |

The results are identical to all printed digits. The three level run halves because its level 1 solves need one
step, so the additional sweep was half of the work; with more subdomains per level the saving is 1 / (steps + 1).
On the level 1 run the wall time did not improve measurably on this machine although 17 % of the solves are saved
(the reported time per solve went up from 0.29 s to 0.34 s; the reason was not found).

**A parameter with a similar effect: the number of PML layers.** With 10 layers around subdomains of 20 cells the
absorbing layers hold about three quarters of the unknowns of a subdomain (the run report now lists this for
every level). 8 subdomains, thickness of the PML unchanged (0.5 µm):

| PML layers | Memory per process | Wall time | GMRES steps | Signal |
| --- | --- | --- | --- | --- |
| 10 | 1599 MB | 43 s | 5 | 0.987 |
| 7 | 913 MB | 26 s | 7 | 0.983 |
| 5 | 606 MB | 26 s | 13 | 0.966 |

The examples use 7 layers at laptop scale since then (`--scale paper` keeps the 10 layers of the dissertation).
This was one configuration and not a study.

**Tested without gain:** a looser or tighter tolerance of the inner solves (`relative convergence criterion` 0.1 or
0.001 instead of 0.01: 1962 and 2142 subdomain solves against 1935), and a threaded BLAS for the subdomain solves
(39 s with one thread, 43 s with the default of Accelerate).

**Not tested:** during a sweep all processes but one are idle. A MUMPS built with OpenMP could give the active
subdomain solve several threads on a cluster node. This needs a new build of PETSc and MUMPS and an MPI library
that does not wait actively. Together with the block low rank compression of section 9 this is the largest
remaining lever for the time per solve.

## 13. Run report, and an optimization with a known optimum

### 13.1 Run report

Every run now writes `run_report.txt` (for reading) and `run_report.json` (the same content plus the residual
histories). `python3 examples/run_report.py <output folder>` adds figures and writes `report.html`, one file
without external dependencies. The report has these parts:

1. **Run, geometry, mesh:** type of run, solver, tolerances, processes, absorbing boundary, input signal; domain,
   core, materials, shape parameters; cells, cell sizes, cells per wavelength.
2. **Unknowns:** for every level of the hierarchy the number of problems, the processes per problem and the
   unknowns split into interior and absorbing boundary; the unknowns of the absorbing boundary by side.
3. **Solution process:** for every level the number of solves, the GMRES steps per solve and the mean reduction of
   the residual per step; subdomain solves, factorization time, time per solve, memory and size of the factors.
   Figure: the residual histories of every level.
4. **Result:** signal transmission, difference to the straight mode. Figures: the field in the plane x = middle in
   the computational coordinates (straight waveguide) and in physical coordinates (displaced by f(z)), a top view
   and a cross-section, and the signal s(z) along the waveguide with the energy per cross-section.
5. **Optimization** (optimization runs): stopping reason, all evaluations with signal, gradient and shape
   parameters. Figures: all evaluated shapes, loss and gradient norm.

The code is `Code/OutputGenerators/RunReport.{h,cpp}` (sections with entries, tables and notes; one data model for
both files); `NonLocalProblem::fill_run_report` and `OptimizationRun::run` fill it.

Two things the first reports showed:

- The signal along z has to be evaluated on the cell faces z = const. At the cell centers the lowest order elements
  average the tangential field of two faces, which lowers the amplitude of a wave by cos(βh/2): 0.75 instead of
  0.84 in the three level example with 5 cells per wavelength. `signal_along_z` uses the faces and ends at the
  value of `results.txt` (0.8428 against 0.8427). The field slices still contain cell center values.
- In the three level example the absorbing boundaries hold 74 % of the unknowns of the global problem and up to
  92 % of a subdomain problem (table "Problems of the hierarchy"). This is the background of the PML numbers in
  section 12.

### 13.2 Optimization with a known optimum (`examples/straight_recovery.py`)

Input and output of a 6 µm waveguide are at the same height, so the straight waveguide is optimal. The optimization
starts from two bumps of 0.36 µm (new case parameter `Initial shape parameters = 0.3, 0.3, 0`). 4 processes,
16 × 16 × 12 cells each. The straight waveguide has the signal 0.9657978 on this mesh, and the shape gradient
there is 1e-14: the optimum of the discrete problem is where it should be.

**First result: the shape did not converge well.** The optimization stopped after 12 evaluations with "the loss
did not improve in four consecutive evaluations". The signal was 0.9657793 (1.9e-5 below the optimum), but the
largest displacement was still 0.013 µm, only a factor 27 below the start.

**Analysis.**

1. *The loss is flat around the optimum.* All evaluations fit a quadratic model, loss − optimum = ½ xᵀHx with the
   optimum at x = 0 (the model reproduces the measured gaps within 3 %). H has the eigenvalues 0.005, 0.025 and
   0.41, a condition number of 81. The steepest direction was eliminated after 2 evaluations, the second after 7.
   In the flattest direction the parameters moved from −0.118 to −0.083 only. A displacement of 0.013 µm costs
   2e-5 in the signal, so a criterion on the loss cannot see it.
2. *The line search of deal.II starts with tiny steps.* `SolverBFGS` takes the first trial step of a line search
   as min(1, 2 Δf / |g·p|) with the decrease Δf of the previous step. Close to the optimum Δf is tiny, the trial
   step as well, and the line search then doubles it evaluation by evaluation: the steps of evaluations 8 to 11 in
   the flat direction were 0.0006, 0.0008, 0.0018 and 0.0037. In addition the initial inverse Hessian is the
   identity, so the directions have the wrong length until the history contains every direction.
3. *The stagnation rule counted evaluations.* It stopped the run after four evaluations without an improvement of
   1e-5, in the middle of this line search.

With the stagnation rule switched off the old settings do converge, but they need about twice as many evaluations
(table below; that run was continued to 29 evaluations and a gradient of 4e-6 of its initial norm).

**Changes** (`OptimizationRun::run`): the line search starts with the step 1 (the natural step of a quasi-Newton
direction), the initial inverse Hessian is scaled with (s, y) / (y, y), and stagnation is judged per BFGS step
(two consecutive steps without improvement) instead of per evaluation.

| | Old settings, no stagnation stop | New settings |
| --- | --- | --- |
| Evaluations until max \|f\| < 0.01 µm | 14 | 8 |
| Evaluations until max \|f\| < 0.001 µm | 22 | 13 |
| Evaluations until the signal is within 1e-6 of the optimum | 17 | 13 |
| Evaluations until the gradient is below 1e-4 of its initial norm | 28 | 15 |
| Largest displacement at that evaluation | 2e-5 µm | 1e-4 µm |

The new run ends with max |f| = 9.6e-5 µm (a factor 3700 below the start), the signal 9e-10 below the optimum and
the gradient norm 8.6e-6. The last two evaluations reduce the distance by a factor 4 and 8: the superlinear
phase of BFGS is reached. `straight_recovery_convergence.png` shows this.

**What remains.**

- With the default `optimization loss tolerance = 1e-5` an optimization still stops when the signal is good, not
  when the shape has converged. That is the intended meaning of the tolerance, but it has to be 1e-9 or smaller
  when the shape itself is the result (as in this example).
- Between evaluation 2 and 10 the convergence is linear and slow. The cause is the parametrization by slopes: the
  three parameters act on the loss with very different strength (condition number 81). A parametrization by
  displacements at the nodes, or a scaling of the parameters, would make the first steps more effective.
- One mesh, one starting shape and three parameters were tested. The hump example (`shape_optimization.py`) is not
  converged after its 14 evaluations with either setting.

## 14. Documentation level and library replacements

### 14.1 Documentation

Doxygen with `EXTRACT_ALL = NO` and `WARN_IF_UNDOCUMENTED = YES` reported 789 undocumented classes, members and
functions in 52 files (131 in `Core/Types.h`, 91 in `Helpers/Parameters.h`, 77 in `Helpers/staticfunctions.h`).
All of them are documented now and the same audit reports no warning. In addition: the placeholder authors of the
file headers, `@param` names that did not match the signatures, LaTeX commands outside of formulas, and comments
that Doxygen did not see (`/* */` and `//` blocks). The comments were written from the implementation and the uses
of every item; where an item is unused or does something else than its name says, the comment says so.

Reading every declaration turned up places where the code looks wrong. None of them was changed, and none was
verified with a test. The ones that can affect results of the main code path come first.

| Where | Observation |
| --- | --- |
| `SpaceTransformation.cpp:56` | `operator()` ignores `apply_math_to_phys`; `switch_application_mode(false)` has no effect |
| `InnerDomain.cpp`, `output_results` | with `apply_transformation == true` the tensor is the identity, otherwise the transformation tensor: inverted relative to the name |
| `DirichletSurface.cpp:22`, `EmptySurface.cpp:21` | `surface_extremal_coordinate` is indexed with the level instead of the boundary id |
| `FEDomain.h`, `global_dof_indices` | never filled, but `DirichletSurface` and `EmptySurface` build their constraints from it |
| `GeometryManager.cpp:323,330` | level 1 neighbor in y uses the number of blocks in y as offset (the global version uses x) |
| `PMLTransformedExactSolution.cpp:92` | the lower z branch uses `in_p[0]` instead of `in_p[2]`; `curl()` always returns zero |
| `staticfunctions.cpp`, `hmax_for_cell_center` | compares the x coordinate with the height of the core |
| `staticfunctions.cpp`, `Triangulation_Shit_To_Local_Geometry` | writes `q[3]` of a 3D point |
| `staticfunctions.cpp`, `get_orientation` | uses `vertex_1[1]` instead of `vertex_1[i]` (the result is never read) |
| `staticfunctions.cpp`, `Transform_4_to_5`, `Transform_5_to_4` | only negate x |
| `Sector.cpp:16` | `boundary(in_left && in_right)`, the comment says "either" |
| `PointSourceField.cpp` | the Hertz dipole zeroes the field near the origin but does not return; both `vector_curl` functions are not the curl of the field |
| `ExactSolutionRamped.cpp:79` | the derivative of the C0 ramp is nonzero outside the ramp (the class is not instantiated) |
| `SquareMeshGenerator.cpp:106` | the refinement loop depends only on the last cell visited |
| `Parameters.cpp`, `check_validity` | always returns true (already open issue 8) |
| `ParameterReader.cpp:13` | "GMRES maximum steps" defaults to 30 in the reader and to 100 in the struct; "kappa angle" is declared but never read; `GMRES restart after` is read but not used |
| `JacobianForCell.cpp:27` | both branches are identical, the Jacobian is always the identity |
| `RectangularMode` | the eigensolver is commented out, `n_dofs_total` and `surfaces` are never set; the class is still instantiated for `source type = 0` |

Dead code that the comments now mark as unused: about 20 structs and enums in `Types.h` and `Enums.h`, 22 functions
of `staticfunctions.h`, `GradientTable`, `DofIndexData`, `LevelDofIndexData`, `LaguerreFunction`,
`PMLMeshTransformation`, `ExactSolutionRamped`, `ExactSolutionConjugate`, `ModeManager` (only loaded), ten members
of `Parameters`, and 15 functions that are declared but not defined. Removing them would shorten the code by an
estimated 1500 lines; this was not done because it is a change of its own.

### 14.2 Custom code replaced by library functions

Replaced (results unchanged, all tests pass):

| Custom code | Replacement |
| --- | --- |
| `crossproduct`, `dotproduct` for tensors | `dealii::cross_product_3d`, `operator*` |
| `conjugate`, `norm_squared` | `std::conj`, `std::norm` |
| `multiply_in_place` | `dealii::Vector::operator*=` |
| `Distance2D`, `Distance3D`, `file_exists`, `add_vector_of_indices`, `compute_center_of_triangulation`, `deal_vector_to_position` (all unused) | removed; `Point::distance`, `std::filesystem::exists`, `IndexSet::add_indices`, `BoundingBox::center` exist for these |

Candidates that are reasonable but were not done, in the order of benefit against risk:

1. **Copies between the levels and inside the sweeps.** `set_u_from_vec_object`, `perform_downward_sweep` and the
   functions around `S_inv` read and write single vector entries through the PETSc wrappers. `VecGetArray` on the
   local part, or one `VecScatter` per pair of levels (`PETScWrappers` exposes both), is the idiomatic form. No
   measurable gain today (section 12), but less code.
2. **Output of residuals and convergence tables.** `ResidualOutputGenerator` and `ConvergenceOutputGenerator` write
   gnuplot scripts and call gnuplot. The run report contains the same data; `dealii::ConvergenceTable` computes
   and prints convergence rates for `ConvergenceRun`.
3. **Signal and error evaluation.** `compute_signal_strength` and `compute_mode_strength` call
   `VectorTools::point_value` for one point per face. `FEFaceValues` with a Gauss quadrature on the output faces is
   faster, needs no point search and gives a proper surface integral (open point 3 of section 8.3).
4. **Logging.** `print_info` with its own levels duplicates `dealii::ConditionalOStream` and `LogStream`.
5. **String and parameter helpers.** `split` (10 uses) against `Utilities::split_string_list`, which also trims;
   `ParameterOverride` against `ParameterHandler::set`.
6. **The partition.** One triangulation per process with hand-made interface numbering (`NeighborSurface`,
   `finish_dof_index_initialization`, the sorting of interface dofs by position) could be replaced by
   `parallel::distributed::Triangulation` with `DoFTools` index sets. This is the largest block of custom code and
   the largest risk: the PML domains are separate meshes that are glued by the same numbering, and the level
   structure would need subdomain ids per level. Not recommended before the experiments on a cluster are done.

No library equivalent exists for the Hardy space infinite elements, the PML with its own meshes, the sweeping
preconditioner and the shape functions.

## 15. Cost of the predefined shapes on a commercial cloud, and how to partition

### 15.1 The problem

All 36 predefined shapes (`Modes/test.csv`, "Use predefined shape") are 100 µm long with a hump between 5.5 and
16 µm and a largest slope of 0.47. They use the same domain and mesh, so one estimate covers all of them. The run
of the dissertation: 48 × 48 × 1560 cells, lowest order, 19.7 million unknowns, three levels with 3 × 3 × 39 = 351
processes, 934 GB, 39.5 hours (13 900 core hours), 216 960 subdomain solves of 0.66 s.

### 15.2 Fewer and larger subdomains

The same problem solved with different partitions (hump of 0.5 µm, 8 µm long, this machine):

| Partition | Processes | Wall time | Core seconds | Subdomain solves | Memory per process | Sum of memory |
| --- | --- | --- | --- | --- | --- | --- |
| 16 × 16 × 40 cells: three levels, 2 × 2 × 4 | 16 | 25.5 s | 409 | 1935 | 181 MB | 2.9 GB |
| two levels, 1 × 2 × 4 | 8 | 18.6 s | 149 | 645 | 274 MB | 2.2 GB |
| one level, 8 slabs | 8 | 6.6 s | 53 | 135 | 292 MB | 2.3 GB |
| one level, 4 slabs | 4 | 5.8 s | 23 | 56 | 421 MB | 1.7 GB |
| one level, 2 slabs | 2 | 7.0 s | 14 | 21 | 695 MB | 1.4 GB |
| 32 × 32 × 80 cells: three levels, 2 × 2 × 4 | 16 | 243 s | 3881 | 3417 | 695 MB | 11.1 GB |
| two levels, 1 × 2 × 4 | 8 | 89 s | 712 | 558 | 1208 MB | 9.7 GB |
| one level, 8 slabs | 8 | 32 s | 255 | 135 | 1324 MB | 10.6 GB |
| one level, 4 slabs | 4 | 33 s | 133 | 56 | 2231 MB | 8.9 GB |
| one level, 2 slabs | 2 | 46 s | 93 | 21 | 4365 MB | 8.7 GB |

All partitions of a mesh give the same signal (0.8427 and 0.8783).

- **Every level of the hierarchy multiplies the number of subdomain solves** (by the GMRES steps of that level
  times the sweep length) and adds absorbing layers around every subdomain. Three levels need 7 times the wall
  time and 29 times the core seconds of one level with four slabs on the finer mesh, and more memory in total.
- **The hierarchy lowers the memory per process, not the total.** It is needed only if a slab of the full
  cross-section does not fit on one node.
- **Thicker slabs cost almost nothing as long as they are thin compared with the cross-section.** Time and memory
  of a slab grow linearly with its thickness (table below), so the time of one sweep is nearly independent of
  the number of slabs, and fewer slabs mean fewer processes and fewer GMRES steps (7, 8, 9 for 2, 4, 8 slabs).
  With 2 slabs the wall time goes up again because the factorization (20 s) is no longer small.

So the proposal to give the subdomains more volume when memory is available is right, and it goes further than
the choice of the slab thickness: for this cross-section the hierarchy should not be used at all.

**Fewer cores than processes.** A level 1 sweep with 54 processes on the 18 cores of this machine: the time per
subdomain solve is the same as with 16 processes on 16 cores (0.0247 s against 0.0255 s), with and without
`--mca mpi_yield_when_idle 1`. Only the factorization, which all processes do at the same time, takes three times
as long (3.5 s against 1.2 s). The sweep itself needs one core. This was tested up to a factor 3.

### 15.3 Slabs of the real cross-section

One level, 2 processes, cross-section 48 × 48 cells (6 × 5.4 µm), cell size in z as in the dissertation, 7 PML
layers, straight waveguide:

| Slab thickness | Unknowns of a slab | Memory | Time per solve | Factorization |
| --- | --- | --- | --- | --- |
| 10 cells | 290 000 | 3.4 GB | 0.42 s | 14 s |
| 20 cells | 408 000 | 5.8 GB | 0.80 s | 28 s |
| 20 cells, compression 1e-4 | 408 000 | 4.3 GB | 0.49 s | 24 s |
| 30 cells, compression 1e-4 | 526 000 | 6.0 GB | 0.70 s | 37 s |

Linear in the thickness t: 1.1 GB + 0.24 GB · t and 0.035 s + 0.038 s · t without compression, 0.7 GB + 0.18 GB · t
and 0.085 s + 0.020 s · t with block low rank compression (section 9).

### 15.4 Estimate for one predefined shape

One level, 39 slabs of 40 cells (extrapolated from the table; larger slabs did not fit on this machine):

| | Without compression | With compression 1e-4 |
| --- | --- | --- |
| Memory of the factors per slab | 10.3 GB | 7.6 GB |
| Memory in total, with 15 % for matrices and meshes | 460 GB | 340 GB |
| Time of one GMRES step (77 slab solves) on this machine | 120 s | 69 s |
| Factorization (all slabs at once) | 56 s | 49 s |

The number of GMRES steps is the main unknown. The run of the dissertation needed 10 applications of the
preconditioner on the top level with 39 slabs. The corrected code has not solved a 100 µm shape. The estimate uses
10 to 20 steps, and a core of a cloud server at half the speed of a core of this machine (an assumption).

| Configuration | Instance (on demand) | Wall time | Cost per shape |
| --- | --- | --- | --- |
| One level, 39 slabs, compression | 1 × 512 GB, 32 cores (AWS r7i.16xlarge, 4.23 $/h) | 0.5 to 0.9 h | 2 to 4 $ |
| One level, 39 slabs, no compression | the same | 0.7 to 1.4 h | 3 to 6 $ |
| As in the dissertation: three levels, 351 processes, 934 GB, 39.5 h | 2 × 768 GB, 192 cores (AWS hpc7a.96xlarge, 7.2 to 9.1 $/h each) | 39.5 h | 570 to 720 $ |

Prices are from price comparison sites in November 2026 and have to be checked against the provider. Other
providers are in the same range per GB (Google n2-highmem-128 with 864 GB: 7.71 $/h, 4.39 $/h as spot instance;
Azure HB176rs v4 with 768 GB: 7.2 $/h). Not included: start of the instance, storage, data transfer, failed runs.

What limits the accuracy of the estimate: the number of GMRES steps (factor 2), the speed of the cloud cores
(assumed), the extrapolation of the slab memory from 20 and 30 to 40 cells, 7 instead of 10 PML layers, and that
no run of this size was made with the current code. The first run on a cluster (JURECA test project) will replace
these numbers by measurements.

### 15.5 What determines the cost

- **Memory, not cores.** Cloud prices follow the memory (about 0.008 $ per GB and hour in the instances above).
  An instance with 512 GB and 16 instead of 32 cores costs about 20 % less (AWS x2idn.8xlarge, price scaled from
  the 32xlarge size), so
  running fewer cores saves little, although the sweep would allow it. The product of memory and time is what
  has to go down: one level instead of three, compression, fewer PML layers.
- **Idle processes cost nothing extra** once the instance is chosen by its memory. Using them would shorten the
  run: threads for the active subdomain solve (section 12), not tested.
- **Out-of-core factors, not tested.** Only one slab is solved at a time, so the factors of the other slabs could
  be on a local SSD (MUMPS `ICNTL(22)`). Memory would drop to a few slabs, the time per solve would grow by reading
  7 to 10 GB per solve. Whether this is cheaper depends on the price of a small instance with a fast disk.
- **Accuracy is not part of this estimate.** The mesh of the dissertation has 8 cells per wavelength across the
  waveguide with lowest order elements; the straight waveguide loses 3 % on comparable meshes (section 13). A
  service that sells results needs second order elements or a finer mesh, which changes memory and time.
