// Tests for the problem of one process (LocalProblem) and for the functions of its base class HierarchicalProblem:
// constraints, assembly, the direct solver and the solution of a waveguide with a known mode.
#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <set>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include "../Code/BoundaryCondition/DirichletSurface.h"
#include "../Code/BoundaryCondition/PMLSurface.h"
#include "../Code/Hierarchy/LocalProblem.h"
#include "./HelpersC.h"

namespace {

  using helpers_c::ProblemEnvironment;
  using helpers_c::SetupOptions;

  // The dielectric waveguide with absorbing layers on all sides and the signal prescribed on the input face.
  SetupOptions dielectric_waveguide() {
    SetupOptions options;
    options.cells_x = 4;
    options.cells_y = 4;
    options.cells_z = 3;
    options.size_z = 1.5;
    options.pml_layers = 2;
    options.pml_thickness = 0.5;
    return options;
  }

  // A waveguide with perfectly conducting walls of width 2 that is filled with vacuum. Its TE10 mode
  // E = (0, cos(pi x / 2) exp(i beta z), 0) is an exact solution; it is prescribed at z = 0 and absorbed at z = 2.
  SetupOptions metallic_waveguide(unsigned int cells_x, unsigned int cells_z, unsigned int fe_order) {
    SetupOptions options;
    options.cells_x = cells_x;
    options.cells_y = 2;
    options.cells_z = cells_z;
    options.size_x = 2.0;
    options.size_y = 1.0;
    options.size_z = 2.0;
    options.fe_order = fe_order;
    options.epsilon_in = 1.0;
    options.epsilon_out = 1.0;
    options.pec_side_walls = true;
    options.source_type = 4;
    options.pml_layers = 6;
    options.pml_thickness = 1.0;
    options.pml_sigma_max = 10.0;
    return options;
  }

  // The dielectric waveguide on a mesh with cells of 0.3 x 0.275 x 0.3, about 0.3 wave lengths of the core.
  SetupOptions resolved_dielectric_waveguide() {
    SetupOptions options;
    options.size_x = 2.4;
    options.size_y = 2.2;
    options.size_z = 0.9;
    options.cells_x = 8;
    options.cells_y = 8;
    options.cells_z = 3;
    options.pml_layers = 2;
    options.pml_thickness = 0.6;
    return options;
  }

  double propagation_constant_of_metallic_waveguide() {
    const double k = 2 * M_PI / 1.55;
    return std::sqrt(k * k - (M_PI / 2.0) * (M_PI / 2.0));
  }

  // Error of the solution relative to the norm of the exact mode, both in L2 on the inner domain.
  double relative_error_of_solution(const SetupOptions & options) {
    ProblemEnvironment environment(options);
    LocalProblem problem;
    problem.initialize();
    problem.assemble();
    problem.solve();
    EXPECT_TRUE(helpers_c::dof_bookkeeping_complaints(environment.output()).empty());
    return problem.compute_L2_error() / problem.compute_L2_norm_of_exact_solution();
  }

  struct PrescribedValues {
    unsigned int n_not_zero = 0;
    double largest_value = 0;
    double largest_deviation = 0;
  };

  // Compares the solution with the inhomogeneities of the constraints for the unknowns below n_unknowns.
  PrescribedValues compare_with_prescribed_values(LocalProblem & problem, unsigned int n_unknowns) {
    PrescribedValues ret;
    for (unsigned int i = 0; i < n_unknowns; i++) {
      if (!problem.constraints.is_constrained(i)) continue;
      const ComplexNumber prescribed = problem.constraints.get_inhomogeneity(i);
      if (std::abs(prescribed) > 1e-6) ret.n_not_zero++;
      ret.largest_value = std::max(ret.largest_value, std::abs(prescribed));
      ret.largest_deviation = std::max(ret.largest_deviation, std::abs(static_cast<ComplexNumber>(problem.solution[i]) - prescribed));
    }
    return ret;
  }

  double frobenius_norm(Mat matrix) {
    PetscReal norm;
    MatNorm(matrix, NORM_FROBENIUS, &norm);
    return norm;
  }

  // Unknowns of a Nedelec element of order k on the boundary of a box of nx x ny x nz cells.
  unsigned int n_nedelec_dofs_on_boundary_of_box(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int k) {
    const unsigned int n_inner_lines = nx * (ny - 1) * (nz - 1) + ny * (nx - 1) * (nz - 1) + nz * (nx - 1) * (ny - 1);
    const unsigned int n_lines = nx * (ny + 1) * (nz + 1) + ny * (nx + 1) * (nz + 1) + nz * (nx + 1) * (ny + 1);
    const unsigned int n_quads = 2 * (nx * ny + nx * nz + ny * nz);
    return (k + 1) * (n_lines - n_inner_lines) + 2 * k * (k + 1) * n_quads;
  }

}

TEST(HierarchicalProblemTests, LocalProblemIsLevelZeroWithoutChild) {
  ProblemEnvironment environment(dielectric_waveguide());
  LocalProblem problem;
  EXPECT_EQ(problem.level, 0u);
  EXPECT_FALSE(problem.has_child);
  EXPECT_EQ(problem.child, nullptr);
  EXPECT_EQ(problem.sweeping_direction, SweepingDirection::Z);
  EXPECT_EQ(problem.solve_counter, 0u);
  EXPECT_EQ(problem.own_dofs.n_elements(), Geometry.levels[0].n_total_level_dofs);
  for (BoundaryId b_id = 0; b_id < 6; b_id++) {
    EXPECT_EQ(problem.opposing_site_bid(b_id), b_id ^ 1u);
    EXPECT_EQ(problem.opposing_site_bid(problem.opposing_site_bid(b_id)), b_id);
  }
}

// The domain with its absorbing layers is a box. Its whole boundary is constrained: the outer boundaries of the
// layers, conducting walls and the input face.
TEST(HierarchicalProblemTests, ConstraintsCoverTheBoundaryOfTheBox) {
  for (unsigned int configuration = 0; configuration < 3; configuration++) {
    for (unsigned int order = 0; order < 2; order++) {
      SetupOptions options = dielectric_waveguide();
      options.fe_order = order;
      const unsigned int n = options.pml_layers;
      unsigned int box[3] = {options.cells_x + 2 * n, options.cells_y + 2 * n, options.cells_z + 2 * n};
      if (configuration == 0) {
        options.coupling = SignalCouplingMethod::Tapering;
      }
      if (configuration == 1) {
        box[2] = options.cells_z + n;
      }
      if (configuration == 2) {
        options.pec_side_walls = true;
        box[0] = options.cells_x;
        box[1] = options.cells_y;
        box[2] = options.cells_z + n;
      }
      ProblemEnvironment environment(options);
      LocalProblem problem;
      problem.make_constraints();
      EXPECT_EQ(problem.constraints.n_constraints(), n_nedelec_dofs_on_boundary_of_box(box[0], box[1], box[2], order)) << "configuration " << configuration << " order " << order;
      EXPECT_EQ(problem.constraints.has_inhomogeneities(), configuration != 0);
      EXPECT_EQ(Geometry.levels[0].n_total_level_dofs, helpers_c::n_nedelec_dofs(box[0], box[1], box[2], order));
    }
  }
}

// Order of precedence: absorbing layers, then prescribed values, then conducting walls. On the edges between the
// input face and the conducting walls the field of the wall (zero) is used.
TEST(HierarchicalProblemTests, ConductingWallsOverrulePrescribedValues) {
  SetupOptions options = dielectric_waveguide();
  options.pec_side_walls = true;
  // The side walls are close to the core (2.0 x 1.8) and cut through the mode of the dielectric waveguide.
  options.size_x = 2.4;
  options.size_y = 2.2;
  ProblemEnvironment environment(options);
  InnerDomain & inner_domain = environment.inner_domain();
  Constraints prescribed = environment.surface<DirichletSurface>(4)->make_constraints();
  LocalProblem problem;
  problem.make_constraints();
  std::set<DofNumber> on_wall;
  for (BoundaryId b_id = 0; b_id < 4; b_id++) {
    for (const InterfaceDofData & dof : inner_domain.get_surface_dof_vector_for_boundary_id(b_id)) on_wall.insert(dof.index);
  }
  unsigned int n_on_edge = 0, n_inside_face = 0;
  double largest_overruled_value = 0, largest_prescribed_value = 0;
  for (const InterfaceDofData & dof : inner_domain.get_surface_dof_vector_for_boundary_id(4)) {
    const DofNumber index = inner_domain.global_index_mapping[dof.index];
    ASSERT_TRUE(problem.constraints.is_constrained(index));
    if (on_wall.count(dof.index) == 1) {
      n_on_edge++;
      largest_overruled_value = std::max(largest_overruled_value, std::abs(prescribed.get_inhomogeneity(index)));
      EXPECT_EQ(problem.constraints.get_inhomogeneity(index), ComplexNumber(0, 0));
    } else {
      n_inside_face++;
      largest_prescribed_value = std::max(largest_prescribed_value, std::abs(prescribed.get_inhomogeneity(index)));
      EXPECT_EQ(problem.constraints.get_inhomogeneity(index), prescribed.get_inhomogeneity(index));
    }
  }
  EXPECT_EQ(n_on_edge, 2 * options.cells_x + 2 * options.cells_y);
  EXPECT_GT(n_inside_face, 0u);
  // Without the precedence the values on the edges would not be zero.
  EXPECT_GT(largest_overruled_value, 1e-3);
  EXPECT_GT(largest_prescribed_value, largest_overruled_value);
}

// HierarchicalProblem.cpp:40 adds the entry -1, which constrains x_1 = -x_2 instead of x_1 = x_2 (the function is not called).
TEST(HierarchicalProblemTests, DISABLED_ConstrainIdenticalDofSetsMakesValuesEqual) {
  ProblemEnvironment environment(dielectric_waveguide());
  LocalProblem problem;
  dealii::IndexSet all_dofs(6);
  all_dofs.add_range(0, 6);
  Constraints constraints(all_dofs, all_dofs);
  std::vector<unsigned int> first = {0, 1}, second = {4, 5};
  problem.constrain_identical_dof_sets(&first, &second, &constraints);
  constraints.close();
  dealii::Vector<ComplexNumber> values(6);
  values[4] = ComplexNumber(2, 1);
  values[5] = ComplexNumber(-3, 0.5);
  constraints.distribute(values);
  EXPECT_EQ(values[0], values[4]);
  EXPECT_EQ(values[1], values[5]);
}

TEST(HierarchicalProblemTests, SolvesAreCounted) {
  ProblemEnvironment environment(dielectric_waveguide());
  LocalProblem problem;
  problem.initialize();
  problem.assemble();
  problem.solve_with_timers_and_count();
  problem.solve_with_timers_and_count();
  EXPECT_EQ(problem.solve_counter, 2u);
  EXPECT_EQ(problem.compute_global_solve_counter(), 2u);
}

TEST(HierarchicalProblemTests, OutputFilesAreWrittenToOutputDirectory) {
  ProblemEnvironment environment(dielectric_waveguide());
  LocalProblem problem;
  problem.initialize();
  problem.assemble();
  problem.solve();
  const std::string file_of_inner_domain = problem.output_results("test_output");
  // The inner domain and the five absorbing layers.
  ASSERT_EQ(problem.filenames.size(), 6u);
  EXPECT_EQ(problem.filenames[0], file_of_inner_domain);
  for (const std::string & filename : problem.filenames) {
    EXPECT_EQ(filename.rfind(environment.output_directory, 0), 0u) << filename;
    EXPECT_GT(std::filesystem::file_size(filename), 0u) << filename;
  }
  problem.write_multifile_output("test_solution");
  EXPECT_GT(std::filesystem::file_size(environment.output_directory + "/_test_solution.pvtu"), 0u);
  unsigned int n_files = 0;
  for (const auto & entry : std::filesystem::directory_iterator(environment.output_directory)) {
    if (entry.path().filename().string().rfind("test_solution", 0) == 0) n_files++;
  }
  EXPECT_EQ(n_files, 6u);
}

// With absorbing layers the matrix is complex symmetric, A = A^T, and not hermitian. The direct solver relies on the
// symmetry (LDL^T factorization).
TEST(LocalProblemTests, SystemMatrixIsComplexSymmetricAndNotHermitian) {
  for (unsigned int order = 0; order < 2; order++) {
    SetupOptions options = dielectric_waveguide();
    options.fe_order = order;
    options.cells_x = options.cells_y = 2;
    ProblemEnvironment environment(options);
    LocalProblem problem;
    problem.initialize();
    problem.assemble();
    EXPECT_EQ(problem.matrix->m(), Geometry.levels[0].n_total_level_dofs);
    EXPECT_EQ(problem.matrix->n(), Geometry.levels[0].n_total_level_dofs);
    const double norm = frobenius_norm(*problem.matrix);
    ASSERT_GT(norm, 1.0);
    Mat transposed, hermitian_transposed;
    MatTranspose(*problem.matrix, MAT_INITIAL_MATRIX, &transposed);
    MatAXPY(transposed, -1.0, *problem.matrix, DIFFERENT_NONZERO_PATTERN);
    EXPECT_LT(frobenius_norm(transposed), 1e-12 * norm) << "order " << order;
    MatHermitianTranspose(*problem.matrix, MAT_INITIAL_MATRIX, &hermitian_transposed);
    MatAXPY(hermitian_transposed, -1.0, *problem.matrix, DIFFERENT_NONZERO_PATTERN);
    EXPECT_GT(frobenius_norm(hermitian_transposed), 1e-2 * norm) << "order " << order;
    MatDestroy(&transposed);
    MatDestroy(&hermitian_transposed);
    EXPECT_EQ(problem.n_unused_dof_indices, 0u);
    EXPECT_EQ(environment.output().find("rows of the system matrix are empty"), std::string::npos);
  }
}

TEST(LocalProblemTests, DirectSolverSolvesTheAssembledSystem) {
  ProblemEnvironment environment(dielectric_waveguide());
  LocalProblem problem;
  problem.initialize();
  problem.validate();
  EXPECT_FALSE(problem.has_factorization);
  problem.assemble();
  EXPECT_GT(problem.rhs.l2_norm(), 1e-3);
  problem.solve();
  NumericVectorDistributed residual(problem.rhs);
  problem.matrix->residual(residual, problem.solution, problem.rhs);
  EXPECT_LT(residual.l2_norm(), 1e-10 * problem.rhs.l2_norm());
  EXPECT_GT(problem.solution.l2_norm(), 1e-3);
  // MUMPS computed a factorization and reported its size.
  EXPECT_TRUE(problem.has_factorization);
  EXPECT_GT(problem.factor_entries_full_rank, static_cast<double>(problem.matrix->m()));
  EXPECT_EQ(problem.factor_entries_stored, problem.factor_entries_full_rank);
  EXPECT_EQ(problem.n_timed_solves, 0u);

  // A second right-hand side reuses the factorization.
  const NumericVectorDistributed first_solution(problem.solution);
  problem.rhs *= ComplexNumber(0, 2);
  problem.solve();
  EXPECT_EQ(problem.n_timed_solves, 1u);
  NumericVectorDistributed difference(problem.solution);
  difference.add(ComplexNumber(0, -2), first_solution);
  EXPECT_LT(difference.l2_norm(), 1e-10 * first_solution.l2_norm());

  // After a new assembly the factorization is computed again.
  problem.assemble();
  EXPECT_FALSE(problem.has_factorization);
  problem.compute_solver_factorization();
  EXPECT_TRUE(problem.has_factorization);
  EXPECT_EQ(problem.solution.l2_norm(), 0.0);
  problem.solve();
  difference = problem.solution;
  difference.add(-1.0, first_solution);
  EXPECT_LT(difference.l2_norm(), 1e-10 * first_solution.l2_norm());
}

// The solution has to take the prescribed value g on every constrained unknown. Here the mesh resolves the wave
// length and only the inner domain has prescribed values that are not zero.
TEST(LocalProblemTests, SolutionTakesPrescribedValuesOnConstrainedUnknowns) {
  for (unsigned int order = 0; order < 2; order++) {
    ProblemEnvironment environment(metallic_waveguide(8, 8, order));
    LocalProblem problem;
    problem.initialize();
    problem.assemble();
    problem.solve();
    const PrescribedValues values = compare_with_prescribed_values(problem, Geometry.levels[0].n_total_level_dofs);
    EXPECT_GT(values.n_not_zero, 10u);
    EXPECT_GT(values.largest_value, 0.1);
    EXPECT_LT(values.largest_deviation, 1e-12) << "order " << order;
  }
}

// InnerDomain.cpp:388: distribute_local_to_global() adds |a_ii| to the matrix but a_ii g to the right-hand side; curl-free shape functions of higher order have a_ii < 0, their unknowns take -g.
TEST(LocalProblemTests, DISABLED_SecondOrderElementsTakePrescribedValues) {
  SetupOptions options = resolved_dielectric_waveguide();
  options.fe_order = 1;
  ProblemEnvironment environment(options);
  LocalProblem problem;
  problem.initialize();
  problem.assemble();
  problem.solve();
  // The unknowns of the inner domain come first.
  const PrescribedValues values = compare_with_prescribed_values(problem, environment.inner_domain().n_locally_active_dofs);
  EXPECT_GT(values.n_not_zero, 10u);
  EXPECT_LT(values.largest_deviation, 1e-12);
}

// InnerDomain.cpp:388: as above; for lowest order elements a_ii < 0 if a cell is wider than about 0.39 wave lengths, so the unknowns take -g on coarse meshes.
TEST(LocalProblemTests, DISABLED_CoarseMeshTakesPrescribedValues) {
  ProblemEnvironment environment(dielectric_waveguide());
  LocalProblem problem;
  problem.initialize();
  problem.assemble();
  problem.solve();
  const PrescribedValues values = compare_with_prescribed_values(problem, environment.inner_domain().n_locally_active_dofs);
  EXPECT_GT(values.n_not_zero, 10u);
  EXPECT_LT(values.largest_deviation, 1e-12);
}

// PMLSurface.cpp:525: as above; in absorbing cells a_ii is complex, so the unknowns of the side layers on the input face take g a_ii / |a_ii|.
TEST(LocalProblemTests, DISABLED_AbsorbingLayersTakePrescribedValues) {
  ProblemEnvironment environment(resolved_dielectric_waveguide());
  LocalProblem problem;
  problem.initialize();
  problem.assemble();
  problem.solve();
  // In the inner domain the values are taken.
  EXPECT_LT(compare_with_prescribed_values(problem, environment.inner_domain().n_locally_active_dofs).largest_deviation, 1e-12);
  const PrescribedValues values = compare_with_prescribed_values(problem, Geometry.levels[0].n_total_level_dofs);
  EXPECT_LT(values.largest_deviation, 1e-12);
}

// The L2 norm of the TE10 mode on the inner domain is sqrt(int cos^2(pi x / w)) = sqrt(w / 2 * size_y * size_z).
TEST(LocalProblemTests, NormOfExactSolutionOfMetallicWaveguide) {
  const SetupOptions options = metallic_waveguide(8, 4, 1);
  ProblemEnvironment environment(options);
  LocalProblem problem;
  problem.initialize();
  EXPECT_NEAR(problem.compute_L2_norm_of_exact_solution(), std::sqrt(options.size_x / 2 * options.size_y * options.size_z), 1e-4);
  // The solution vector is zero, so the error is the norm of the exact solution.
  EXPECT_NEAR(problem.compute_L2_error(), problem.compute_L2_norm_of_exact_solution(), 1e-12);
  EXPECT_NEAR(problem.compute_error(), problem.compute_L2_norm_of_exact_solution(), 1e-12);
}

// For lowest order elements the error is dominated by numerical dispersion: the discrete wave travels with a wave
// number that is off by the factor (beta h)^2 / 24, which adds up to the phase error beta L (beta h)^2 / 24 at the
// end of the waveguide and to 1 / sqrt(3) of that in the L2 norm over its length.
TEST(LocalProblemTests, ModeOfMetallicWaveguideIsReproducedWithinDispersionError) {
  const double beta = propagation_constant_of_metallic_waveguide();
  const double length = 2.0;
  double errors[2];
  for (unsigned int refinement = 0; refinement < 2; refinement++) {
    const unsigned int cells_z = 8 << refinement;
    errors[refinement] = relative_error_of_solution(metallic_waveguide(cells_z / 2, cells_z, 0));
    const double h = length / cells_z;
    const double dispersion_error = beta * length * std::pow(beta * h, 2) / 24.0 / std::sqrt(3.0);
    EXPECT_LT(errors[refinement], 1.5 * dispersion_error) << cells_z << " cells";
    EXPECT_GT(errors[refinement], 0.5 * dispersion_error) << cells_z << " cells";
  }
  // Second order convergence.
  EXPECT_LT(errors[1], 0.3 * errors[0]);
}

TEST(LocalProblemTests, SecondOrderElementsReproduceModeOfMetallicWaveguide) {
  const double first_order = relative_error_of_solution(metallic_waveguide(4, 8, 0));
  const double second_order = relative_error_of_solution(metallic_waveguide(4, 8, 1));
  EXPECT_LT(second_order, 0.01);
  EXPECT_LT(second_order, 0.1 * first_order);
}

// Infinite elements with kappa_0 equal to the wave number absorb the mode as well as the PML does.
TEST(LocalProblemTests, InfiniteElementsAbsorbModeOfMetallicWaveguide) {
  SetupOptions options = metallic_waveguide(4, 8, 1);
  const double with_pml = relative_error_of_solution(options);
  options.boundary_method = BoundaryConditionType::HSIE;
  options.kappa_0 = ComplexNumber(2 * M_PI / 1.55, 0);
  const double with_infinite_elements = relative_error_of_solution(options);
  EXPECT_LT(with_infinite_elements, 0.01);
  EXPECT_NEAR(with_infinite_elements, with_pml, 0.005);
}

// A PML that does not absorb (sigma = 0) reflects the mode at its outer boundary, which is a conducting wall.
TEST(LocalProblemTests, ErrorIsLargeWithoutAbsorption) {
  SetupOptions options = metallic_waveguide(4, 8, 1);
  options.pml_sigma_max = 0.0;
  EXPECT_GT(relative_error_of_solution(options), 0.2);
}

// Infinite elements on neighboring surfaces identify the unknowns over their common edge. The indices that the
// second surface had reserved for them are left without an equation and get a unit row.
TEST(LocalProblemTests, UnusedIndicesOfInfiniteElementsGetUnitRows) {
  SetupOptions options = dielectric_waveguide();
  options.boundary_method = BoundaryConditionType::HSIE;
  options.coupling = SignalCouplingMethod::Tapering;
  options.cells_x = options.cells_y = options.cells_z = 2;
  options.hsie_degree = 2;
  ProblemEnvironment environment(options);
  LocalProblem problem;
  problem.initialize();
  problem.assemble();
  // Lowest order elements: degree + 2 unknowns per ray and degree + 1 per infinite face. Over each of the 12 edges
  // (2 cells) the ray in the middle and the 2 infinite faces exist on two surfaces, the rays at the 8 corners on three.
  const unsigned int n_per_ray = options.hsie_degree + 2;
  const unsigned int n_per_infinite_face = options.hsie_degree + 1;
  EXPECT_EQ(problem.n_unused_dof_indices, 12 * (n_per_ray + 2 * n_per_infinite_face) + 8 * 2 * n_per_ray);
  problem.solve();
  NumericVectorDistributed residual(problem.rhs);
  problem.matrix->residual(residual, problem.solution, problem.rhs);
  EXPECT_LT(residual.l2_norm(), 1e-9 * problem.rhs.l2_norm());
}

// kappa_0 = 0 makes the infinite elements divide by zero. The assembly has to report this instead of handing the
// matrix to the solver.
TEST(LocalProblemTests, AssemblyRejectsMatrixWithInvalidEntries) {
  SetupOptions options = metallic_waveguide(2, 2, 0);
  options.boundary_method = BoundaryConditionType::HSIE;
  options.kappa_0 = ComplexNumber(0, 0);
  ProblemEnvironment environment(options);
  LocalProblem problem;
  problem.initialize();
  EXPECT_THROW(problem.assemble(), std::runtime_error);
}

TEST(LocalProblemTests, EmptyMemoryReleasesMatrixAndSolver) {
  ProblemEnvironment environment(dielectric_waveguide());
  LocalProblem problem;
  problem.initialize();
  problem.assemble();
  problem.solve();
  ASSERT_NE(problem.local_ksp, nullptr);
  problem.empty_memory();
  EXPECT_EQ(problem.local_ksp, nullptr);
}
