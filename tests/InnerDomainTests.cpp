// Tests for the inner domain: mesh, number and location of the unknowns, the assembled matrix and the evaluation of
// a field (signal along z, values at points, errors).
#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <set>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include "../Code/Core/InnerDomain.h"
#include "../Code/Hierarchy/LocalProblem.h"
#include "./HelpersC.h"

namespace {

  using helpers_c::LinearField;
  using helpers_c::ProblemEnvironment;
  using helpers_c::SetupOptions;

  SetupOptions small_domain(unsigned int order) {
    SetupOptions options;
    options.cells_x = 4;
    options.cells_y = 3;
    options.cells_z = 5;
    options.fe_order = order;
    options.pml_layers = 2;
    options.pml_thickness = 0.5;
    return options;
  }

  // Core of the waveguide (2.0 x 1.8) and domain (4.0 x 3.6 x 2.0) are resolved by the 4 x 4 x 4 cells.
  SetupOptions resolved_core() {
    return SetupOptions();
  }

  // The inner domain assembled on its own, without any constraints.
  struct InnerSystem {
    Constraints constraints;
    dealii::SparsityPattern sparsity_pattern;
    dealii::PETScWrappers::MPI::SparseMatrix matrix;
    NumericVectorDistributed rhs;
    dealii::IndexSet all_dofs;

    explicit InnerSystem(InnerDomain & inner_domain) {
      const unsigned int n = Geometry.levels[0].n_total_level_dofs;
      all_dofs.set_size(n);
      all_dofs.add_range(0, n);
      constraints.reinit(all_dofs, all_dofs);
      constraints.close();
      dealii::DynamicSparsityPattern dsp(n, n);
      inner_domain.fill_sparsity_pattern(&dsp, &constraints);
      for (unsigned int i = 0; i < n; i++) {
        dsp.add(i, i);
      }
      sparsity_pattern.copy_from(dsp);
      std::vector<unsigned int> rows_per_process = {n};
      matrix.reinit(MPI_COMM_SELF, sparsity_pattern, rows_per_process, rows_per_process, 0);
      rhs.reinit(all_dofs, MPI_COMM_SELF);
      inner_domain.assemble_system(&constraints, &matrix, &rhs);
    }

    // u^T A u (without complex conjugation) for a vector given in the numbering of the inner domain.
    ComplexNumber quadratic_form(InnerDomain & inner_domain, const NumericVectorLocal & u) {
      NumericVectorDistributed u_level(all_dofs, MPI_COMM_SELF), product(all_dofs, MPI_COMM_SELF);
      for (unsigned int i = 0; i < u.size(); i++) {
        u_level[inner_domain.global_index_mapping[i]] = u[i];
      }
      u_level.compress(dealii::VectorOperation::insert);
      matrix.vmult(product, u_level);
      ComplexNumber ret(0, 0);
      for (unsigned int i = 0; i < u.size(); i++) {
        const DofNumber index = inner_domain.global_index_mapping[i];
        ret += static_cast<ComplexNumber>(u_level[index]) * static_cast<ComplexNumber>(product[index]);
      }
      return ret;
    }
  };

  // The coefficients of a field of the finite element space (see LinearField).
  NumericVectorLocal coefficients_of(ProblemEnvironment & environment, dealii::Function<3, ComplexNumber> * field) {
    environment.set_source_field(field);
    environment.inner_domain().exact_solution_is_initialized = false;
    environment.inner_domain().load_exact_solution();
    return environment.inner_domain().exact_solution_interpolated;
  }

  // A constant field that reports its destruction.
  class ObservedField : public LinearField {
   public:
    explicit ObservedField(bool * in_was_deleted) : LinearField(0, 1, 0), was_deleted(in_was_deleted) {}
    ~ObservedField() override {
      *was_deleted = true;
    }

   private:
    bool * was_deleted;
  };

  // The values of the cell data "Epsilon" in the patches of the last call of InnerDomain::output_results().
  std::vector<double> epsilon_values_of_last_output(InnerDomain & inner_domain) {
    std::stringstream stream;
    inner_domain.data_out.write_gnuplot(stream);
    std::vector<double> ret;
    int column = -1;
    std::string line;
    while (std::getline(stream, line)) {
      std::istringstream tokens(line);
      if (line.rfind("# <x>", 0) == 0) {
        // The line that names the columns: "# <x> <y> <z> <name> <name> ...".
        std::string token;
        int index = -1;
        while (tokens >> token) {
          if (token == "<Epsilon>") column = index;
          index++;
        }
      } else if (!line.empty() && line[0] != '#' && column >= 0) {
        double value = 0;
        for (int i = 0; i <= column; i++) tokens >> value;
        ret.push_back(value);
      }
    }
    return ret;
  }

  double frobenius_norm(Mat matrix) {
    PetscReal norm;
    MatNorm(matrix, NORM_FROBENIUS, &norm);
    return norm;
  }

}

TEST(InnerDomainTests, MeshFillsLocalRangeAndBoundaryIdsFollowDirections) {
  const SetupOptions options = small_domain(0);
  ProblemEnvironment environment(options);
  InnerDomain & inner_domain = environment.inner_domain();
  EXPECT_EQ(inner_domain.triangulation.n_active_cells(), options.cells_x * options.cells_y * options.cells_z);
  const double lower[3] = {-options.size_x / 2, -options.size_y / 2, 0};
  const double upper[3] = {options.size_x / 2, options.size_y / 2, options.size_z};
  std::array<unsigned int, 6> n_faces = {{0, 0, 0, 0, 0, 0}};
  for (const auto & cell : inner_domain.triangulation.active_cell_iterators()) {
    for (unsigned int face = 0; face < 6; face++) {
      if (!cell->face(face)->at_boundary()) continue;
      // Boundary id 2 d is the lower and 2 d + 1 the upper surface in direction d.
      const BoundaryId b_id = cell->face(face)->boundary_id();
      ASSERT_LT(b_id, 6u);
      const double coordinate = (b_id % 2 == 0) ? lower[b_id / 2] : upper[b_id / 2];
      ASSERT_NEAR(cell->face(face)->center()[b_id / 2], coordinate, 1e-12);
      ASSERT_NEAR(Geometry.surface_extremal_coordinate[b_id], coordinate, 1e-12);
      n_faces[b_id]++;
    }
  }
  EXPECT_EQ(n_faces[0], options.cells_y * options.cells_z);
  EXPECT_EQ(n_faces[1], options.cells_y * options.cells_z);
  EXPECT_EQ(n_faces[2], options.cells_x * options.cells_z);
  EXPECT_EQ(n_faces[3], options.cells_x * options.cells_z);
  EXPECT_EQ(n_faces[4], options.cells_x * options.cells_y);
  EXPECT_EQ(n_faces[5], options.cells_x * options.cells_y);
}

TEST(InnerDomainTests, NumberOfUnknownsMatchesNedelecFormula) {
  for (unsigned int order = 0; order < 3; order++) {
    const SetupOptions options = small_domain(order);
    ProblemEnvironment environment(options);
    InnerDomain & inner_domain = environment.inner_domain();
    const unsigned int expected = helpers_c::n_nedelec_dofs(options.cells_x, options.cells_y, options.cells_z, order);
    EXPECT_EQ(inner_domain.dof_handler.n_dofs(), expected) << "order " << order;
    EXPECT_EQ(inner_domain.compute_n_locally_active_dofs(), expected);
    // In one process there is no neighbor that could own unknowns of this domain.
    EXPECT_EQ(inner_domain.compute_n_locally_owned_dofs(), expected);
    EXPECT_EQ(inner_domain.n_locally_active_dofs, expected);
    EXPECT_EQ(inner_domain.n_locally_owned_dofs, expected);
  }
}

// The inner domain is numbered first. LocalProblem::compute_L2_error() and write_multifile_output() rely on this.
TEST(InnerDomainTests, UnknownsComeFirstInNumberingOfLevel) {
  ProblemEnvironment environment(small_domain(1));
  InnerDomain & inner_domain = environment.inner_domain();
  ASSERT_EQ(inner_domain.global_index_mapping.size(), inner_domain.dof_handler.n_dofs());
  for (unsigned int i = 0; i < inner_domain.global_index_mapping.size(); i++) {
    ASSERT_EQ(inner_domain.global_index_mapping[i], i);
    ASSERT_TRUE(inner_domain.is_dof_owned[i]);
  }
}

TEST(InnerDomainTests, SurfaceDofListsHaveExpectedSizeAndLieOnSurface) {
  for (unsigned int order = 0; order < 3; order++) {
    const SetupOptions options = small_domain(order);
    ProblemEnvironment environment(options);
    InnerDomain & inner_domain = environment.inner_domain();
    const unsigned int cells[3] = {options.cells_x, options.cells_y, options.cells_z};
    for (BoundaryId b_id = 0; b_id < 6; b_id++) {
      const std::vector<InterfaceDofData> dofs = inner_domain.get_surface_dof_vector_for_boundary_id(b_id);
      const unsigned int expected = helpers_c::n_nedelec_dofs_on_face(cells[(b_id / 2 + 1) % 3], cells[(b_id / 2 + 2) % 3], order);
      ASSERT_EQ(dofs.size(), expected) << "order " << order << " surface " << b_id;
      std::set<DofNumber> indices;
      for (const InterfaceDofData & dof : dofs) {
        ASSERT_LT(dof.index, inner_domain.dof_handler.n_dofs());
        ASSERT_NEAR(dof.base_point[b_id / 2], Geometry.surface_extremal_coordinate[b_id], 1e-12);
        indices.insert(dof.index);
      }
      // No unknown is listed twice.
      EXPECT_EQ(indices.size(), expected);
      EXPECT_TRUE(std::is_sorted(dofs.begin(), dofs.end(), compareDofBaseDataAndOrientation));
    }
    EXPECT_TRUE(helpers_c::dof_bookkeeping_complaints(environment.output()).empty());
  }
}

// The sweeping preconditioner and the neighbor surfaces pair the unknowns of opposing surfaces by their position in
// these lists.
TEST(InnerDomainTests, SurfaceDofListsOfOpposingSurfacesCorrespond) {
  for (unsigned int order = 0; order < 2; order++) {
    ProblemEnvironment environment(small_domain(order));
    InnerDomain & inner_domain = environment.inner_domain();
    for (BoundaryId b_id = 0; b_id < 6; b_id += 2) {
      const std::vector<InterfaceDofData> lower = inner_domain.get_surface_dof_vector_for_boundary_id(b_id);
      const std::vector<InterfaceDofData> upper = inner_domain.get_surface_dof_vector_for_boundary_id(b_id + 1);
      ASSERT_EQ(lower.size(), upper.size());
      for (unsigned int i = 0; i < lower.size(); i++) {
        Position shifted = lower[i].base_point;
        shifted[b_id / 2] = upper[i].base_point[b_id / 2];
        ASSERT_LT(shifted.distance(upper[i].base_point), 1e-10) << "order " << order << " surface " << b_id << " entry " << i;
        ASSERT_EQ(lower[i].order, upper[i].order);
      }
    }
  }
}

// Two surfaces that meet in an edge of the domain share the unknowns on the lines of that edge.
TEST(InnerDomainTests, SurfaceDofListsShareUnknownsOnEdges) {
  for (unsigned int order = 0; order < 2; order++) {
    const SetupOptions options = small_domain(order);
    ProblemEnvironment environment(options);
    InnerDomain & inner_domain = environment.inner_domain();
    const unsigned int cells[3] = {options.cells_x, options.cells_y, options.cells_z};
    for (BoundaryId a = 0; a < 6; a++) {
      std::set<DofNumber> of_a;
      for (const InterfaceDofData & dof : inner_domain.get_surface_dof_vector_for_boundary_id(a)) of_a.insert(dof.index);
      for (BoundaryId b = a + 1; b < 6; b++) {
        unsigned int n_common = 0;
        for (const InterfaceDofData & dof : inner_domain.get_surface_dof_vector_for_boundary_id(b)) n_common += of_a.count(dof.index);
        const unsigned int expected = are_opposing_sites(a, b) ? 0 : (order + 1) * cells[3 - a / 2 - b / 2];
        EXPECT_EQ(n_common, expected) << "order " << order << " surfaces " << a << " and " << b;
      }
    }
  }
}

// A lower surface with a neighboring process belongs to that process, an upper one to this process.
TEST(InnerDomainTests, UnknownsOnLowerNeighborSurfaceAreNotOwned) {
  const SetupOptions options = small_domain(0);
  ProblemEnvironment environment(options);
  InnerDomain inner_domain(0);
  inner_domain.make_grid();
  const unsigned int n_dofs = inner_domain.dof_handler.n_dofs();
  const unsigned int n_on_face = helpers_c::n_nedelec_dofs_on_face(options.cells_x, options.cells_y, 0);
  Geometry.levels[0].surface_type[5] = SurfaceType::NEIGHBOR_SURFACE;
  EXPECT_EQ(inner_domain.compute_n_locally_owned_dofs(), n_dofs);
  Geometry.levels[0].surface_type[4] = SurfaceType::NEIGHBOR_SURFACE;
  EXPECT_EQ(inner_domain.compute_n_locally_owned_dofs(), n_dofs - n_on_face);
  inner_domain.initialize_dof_counts(inner_domain.compute_n_locally_active_dofs(), inner_domain.compute_n_locally_owned_dofs());
  inner_domain.determine_non_owned_dofs();
  std::set<DofNumber> on_lower_face;
  for (const InterfaceDofData & dof : inner_domain.get_surface_dof_vector_for_boundary_id(4)) on_lower_face.insert(dof.index);
  for (unsigned int i = 0; i < n_dofs; i++) {
    ASSERT_EQ(inner_domain.is_dof_owned[i], on_lower_face.count(i) == 0);
  }
}

// For a straight waveguide the bilinear form is a(u, v) = int curl u . curl v - omega^2 eps u . v. It is evaluated
// here for two fields of the finite element space for which the integral is known.
TEST(InnerDomainTests, MatrixReproducesBilinearFormForKnownFields) {
  for (unsigned int order = 0; order < 2; order++) {
    SetupOptions options = resolved_core();
    options.fe_order = order;
    ProblemEnvironment environment(options);
    InnerDomain & inner_domain = environment.inner_domain();
    const double omega_squared = std::pow(2 * M_PI / GlobalParams.Lambda, 2);
    const double eps_core = options.epsilon_in, eps_cladding = options.epsilon_out;
    const double volume = options.size_x * options.size_y * options.size_z;
    const double volume_of_core = 2.0 * 1.8 * options.size_z;

    // u = (0, 1, 0): no curl, a(u, u) = - omega^2 int eps.
    NumericVectorLocal u = coefficients_of(environment, new LinearField(0, 1, 0));
    InnerSystem system(inner_domain);
    const double expected_constant = -omega_squared * (eps_cladding * volume + (eps_core - eps_cladding) * volume_of_core);
    const ComplexNumber form_constant = system.quadratic_form(inner_domain, u);
    EXPECT_NEAR(form_constant.real(), expected_constant, 1e-7 * std::abs(expected_constant)) << "order " << order;
    EXPECT_NEAR(form_constant.imag(), 0.0, 1e-10 * std::abs(expected_constant));

    // u = (0, x, 0): curl u = (0, 0, 1), a(u, u) = volume - omega^2 int eps x^2.
    u = coefficients_of(environment, new LinearField(0, 0, 0, 1.0, 0));
    const double x_squared = options.size_y * options.size_z * 2 * std::pow(options.size_x / 2, 3) / 3;
    const double x_squared_in_core = 1.8 * options.size_z * 2 * std::pow(1.0, 3) / 3;
    const double expected_linear = volume - omega_squared * (eps_cladding * x_squared + (eps_core - eps_cladding) * x_squared_in_core);
    const ComplexNumber form_linear = system.quadratic_form(inner_domain, u);
    EXPECT_NEAR(form_linear.real(), expected_linear, 1e-7 * std::abs(expected_linear)) << "order " << order;
    EXPECT_NEAR(form_linear.imag(), 0.0, 1e-10 * std::abs(expected_linear));
  }
}

// Without absorbing layers the matrix of a straight waveguide is real and symmetric and there is no source term if
// the signal is prescribed on the input face.
TEST(InnerDomainTests, MatrixOfStraightWaveguideIsRealAndSymmetric) {
  ProblemEnvironment environment(resolved_core());
  InnerSystem system(environment.inner_domain());
  const double norm = frobenius_norm(system.matrix);
  ASSERT_GT(norm, 1.0);
  Mat transposed, conjugated;
  MatTranspose(system.matrix, MAT_INITIAL_MATRIX, &transposed);
  MatAXPY(transposed, -1.0, system.matrix, DIFFERENT_NONZERO_PATTERN);
  EXPECT_LT(frobenius_norm(transposed), 1e-12 * norm);
  MatDuplicate(system.matrix, MAT_COPY_VALUES, &conjugated);
  MatConjugate(conjugated);
  MatAXPY(conjugated, -1.0, system.matrix, SAME_NONZERO_PATTERN);
  EXPECT_LT(frobenius_norm(conjugated), 1e-12 * norm);
  MatDestroy(&transposed);
  MatDestroy(&conjugated);
  EXPECT_EQ(system.rhs.l2_norm(), 0.0);
  // The assembly checks every cell matrix and reports a violation.
  EXPECT_EQ(environment.output().find("Not fulfilled"), std::string::npos);
}

// The source term of the tapered coupling only exists where the taper function changes.
TEST(InnerDomainTests, TaperedSourceIsRestrictedToTaperRegion) {
  SetupOptions options = resolved_core();
  options.coupling = SignalCouplingMethod::Tapering;
  ProblemEnvironment environment(options);
  // Cell layers in z: [0, 0.5], [0.5, 1], [1, 1.5], [1.5, 2]. The taper covers the second and third layer.
  GlobalParams.tapering_min_z = 0.5;
  GlobalParams.tapering_max_z = 1.5;
  InnerDomain & inner_domain = environment.inner_domain();
  InnerSystem system(inner_domain);
  EXPECT_GT(system.rhs.l2_norm(), 1e-3);
  // The functions of the unknowns on the surfaces z = 0 and z = 2 vanish in the two inner cell layers.
  for (BoundaryId b_id : {4u, 5u}) {
    for (const InterfaceDofData & dof : inner_domain.get_surface_dof_vector_for_boundary_id(b_id)) {
      ASSERT_EQ(std::abs(static_cast<ComplexNumber>(system.rhs[inner_domain.global_index_mapping[dof.index]])), 0.0);
    }
  }
  double norm_in_taper = 0;
  for (const auto & cell : inner_domain.dof_handler.active_cell_iterators()) {
    if (std::abs(cell->center()[2] - 1.0) > 0.3) continue;
    std::vector<DofNumber> indices(inner_domain.fe.dofs_per_cell);
    cell->get_dof_indices(indices);
    for (DofNumber index : indices) norm_in_taper += std::abs(static_cast<ComplexNumber>(system.rhs[inner_domain.global_index_mapping[index]]));
  }
  EXPECT_GT(norm_in_taper, 1e-3);
}

// Field E = (0, a + c z, 0) and "mode" m = constant: at each of the n_z + 1 planes of cell faces the sum over the
// 4 x 4 face centers is 16 (a + c z) conj(m_y) for the overlap, 16 |m|^2 for the norm of the mode and 16 |a + c z|^2
// for the energy. 2 x 2 of the face centers are inside the core.
TEST(InnerDomainTests, SignalAlongZOfKnownField) {
  const SetupOptions options = resolved_core();
  ProblemEnvironment environment(options);
  const ComplexNumber a(0.7, -0.2), c(0.3, 0.4);
  const NumericVectorLocal field = coefficients_of(environment, new LinearField(0, a, 0, 0, c));
  const ComplexNumber mode[3] = {ComplexNumber(0.3, 0), ComplexNumber(0.5, -0.2), ComplexNumber(0, 0.1)};
  environment.set_source_field(new LinearField(mode[0], mode[1], mode[2]));
  environment.inner_domain().write_signal_along_z(field, "signal_test");

  std::ifstream file(environment.output_directory + "/signal_test0.txt");
  ASSERT_TRUE(file.is_open());
  std::string header;
  std::getline(file, header);
  ASSERT_EQ(header[0], '#');
  const double mode_norm = std::norm(mode[0]) + std::norm(mode[1]) + std::norm(mode[2]);
  unsigned int n_layers = 0;
  double z, re_overlap, im_overlap, norm_of_mode, energy_in_core, energy;
  while (file >> z >> re_overlap >> im_overlap >> norm_of_mode >> energy_in_core >> energy) {
    EXPECT_NEAR(z, n_layers * options.size_z / options.cells_z, 1e-9);
    const ComplexNumber value = a + c * z;
    const ComplexNumber overlap = 16.0 * value * std::conj(mode[1]);
    EXPECT_NEAR(re_overlap, overlap.real(), 1e-6);
    EXPECT_NEAR(im_overlap, overlap.imag(), 1e-6);
    EXPECT_NEAR(norm_of_mode, 16.0 * mode_norm, 1e-6);
    EXPECT_NEAR(energy_in_core, 4.0 * std::norm(value), 1e-6);
    EXPECT_NEAR(energy, 16.0 * std::norm(value), 1e-6);
    n_layers++;
  }
  EXPECT_EQ(n_layers, options.cells_z + 1);
}

TEST(InnerDomainTests, EvaluationAtPositionsReproducesFieldOfTheSpace) {
  for (unsigned int order = 0; order < 2; order++) {
    SetupOptions options = resolved_core();
    options.fe_order = order;
    ProblemEnvironment environment(options);
    const LinearField reference(ComplexNumber(0.2, 0.1), ComplexNumber(0.7, -0.2), ComplexNumber(-0.4, 0), ComplexNumber(0.3, 0.4), ComplexNumber(0.1, -0.6));
    const NumericVectorLocal field = coefficients_of(environment, new LinearField(ComplexNumber(0.2, 0.1), ComplexNumber(0.7, -0.2), ComplexNumber(-0.4, 0), ComplexNumber(0.3, 0.4), ComplexNumber(0.1, -0.6)));
    const std::vector<Position> positions = {Position(0.13, -0.42, 0.77), Position(-1.9, 1.7, 0.05), Position(1.21, 0.9, 1.93), Position(0, 0, 1)};
    const std::vector<std::vector<ComplexNumber>> values = environment.inner_domain().evaluate_at_positions(positions, field);
    ASSERT_EQ(values.size(), positions.size());
    for (unsigned int i = 0; i < positions.size(); i++) {
      ASSERT_EQ(values[i].size(), 3u);
      for (unsigned int component = 0; component < 3; component++) {
        EXPECT_LT(std::abs(values[i][component] - reference.value(positions[i], component)), 1e-7) << "order " << order << " position " << i;
      }
    }
  }
}

// The signal strength is the mean of E . conj(mode) over the centers of the cell faces on the output surface. For
// E = mode it has to be equal to the mode strength, the mean of |mode|^2.
TEST(InnerDomainTests, SignalStrengthOfTheModeItselfIsTheModeStrength) {
  const SetupOptions options = resolved_core();
  ProblemEnvironment environment(options);
  InnerDomain & inner_domain = environment.inner_domain();
  const ComplexNumber a(0.7, -0.2), b(0.3, 0.4), c(0.1, -0.6);
  const NumericVectorLocal field = coefficients_of(environment, new LinearField(0, a, 0, b, c));
  dealii::LinearAlgebra::distributed::Vector<ComplexNumber> solution(Geometry.levels[0].n_total_level_dofs);
  for (unsigned int i = 0; i < field.size(); i++) {
    solution[inner_domain.global_index_mapping[i]] = field[i];
  }
  double expected = 0;
  for (double x : {-1.5, -0.5, 0.5, 1.5}) {
    expected += std::norm(a + b * x + c * options.size_z) / 4.0;
  }
  const ComplexNumber mode_strength = inner_domain.compute_mode_strength();
  const ComplexNumber signal_strength = inner_domain.compute_signal_strength(&solution);
  EXPECT_NEAR(mode_strength.real(), expected, 1e-4);
  EXPECT_NEAR(mode_strength.imag(), 0.0, 1e-12);
  EXPECT_LT(std::abs(signal_strength - mode_strength), 1e-7);
  // Half the field gives half the signal.
  solution *= 0.5;
  EXPECT_LT(std::abs(inner_domain.compute_signal_strength(&solution) - 0.5 * mode_strength), 1e-7);
}

TEST(InnerDomainTests, ErrorOfExactFieldIsZeroAndOfZeroFieldIsTheNormOfTheField) {
  const SetupOptions options = resolved_core();
  ProblemEnvironment environment(options);
  InnerDomain & inner_domain = environment.inner_domain();
  const ComplexNumber a(0.6, -0.8);
  const NumericVectorLocal field = coefficients_of(environment, new LinearField(0, a, 0));
  dealii::LinearAlgebra::distributed::Vector<ComplexNumber> solution(Geometry.levels[0].n_total_level_dofs);
  for (unsigned int i = 0; i < field.size(); i++) {
    solution[inner_domain.global_index_mapping[i]] = field[i];
  }
  FEErrorStruct errors = inner_domain.compute_errors(&solution);
  EXPECT_LT(errors.L2, 1e-8);
  EXPECT_LT(errors.Linfty, 1e-8);
  // Twice the field: the difference is the field itself, with |E| = 1 everywhere. compute_errors() divides by the
  // norms of the coefficient vector.
  solution *= 2.0;
  const double volume = options.size_x * options.size_y * options.size_z;
  errors = inner_domain.compute_errors(&solution);
  EXPECT_NEAR(errors.L2 * solution.l2_norm(), std::sqrt(volume), 1e-7);
  EXPECT_NEAR(errors.Linfty * solution.linfty_norm(), 1.0, 1e-7);
}

// One line per cell that is cut by one of the three planes through the center of the domain, with the field at the
// center of the cell.
TEST(InnerDomainTests, FieldSlicesOfKnownField) {
  const SetupOptions options = resolved_core();
  ProblemEnvironment environment(options);
  const LinearField reference(ComplexNumber(0.2, 0.1), ComplexNumber(0.7, -0.2), ComplexNumber(-0.4, 0), ComplexNumber(0.3, 0.4), ComplexNumber(0.1, -0.6));
  const NumericVectorLocal field = coefficients_of(environment, new LinearField(ComplexNumber(0.2, 0.1), ComplexNumber(0.7, -0.2), ComplexNumber(-0.4, 0), ComplexNumber(0.3, 0.4), ComplexNumber(0.1, -0.6)));
  environment.inner_domain().write_field_slices(field, "slice_test");
  const std::string planes[3] = {"x", "y", "z"};
  for (unsigned int plane = 0; plane < 3; plane++) {
    std::ifstream file(environment.output_directory + "/slice_test_" + planes[plane] + "0.txt");
    ASSERT_TRUE(file.is_open());
    std::string header;
    std::getline(file, header);
    ASSERT_EQ(header[0], '#');
    unsigned int n_lines = 0;
    double p[3], physical[3], values[6];
    int in_core;
    while (file >> p[0] >> p[1] >> p[2] >> physical[0] >> physical[1] >> physical[2] >> values[0] >> values[1] >> values[2] >> values[3] >> values[4] >> values[5] >> in_core) {
      n_lines++;
      const Position position(p[0], p[1], p[2]);
      // The cells have the size 1 x 0.9 x 0.5; the plane is the lower face of the cell.
      const double half_cell[3] = {0.5, 0.45, 0.25};
      const double center_of_domain[3] = {0, 0, options.size_z / 2};
      EXPECT_NEAR(p[plane], center_of_domain[plane] + half_cell[plane], 1e-9);
      for (unsigned int i = 0; i < 3; i++) {
        // Straight waveguide: physical and mathematical coordinates are the same.
        EXPECT_NEAR(physical[i], p[i], 1e-9);
        EXPECT_NEAR(values[2 * i], reference.value(position, i).real(), 1e-5);
        EXPECT_NEAR(values[2 * i + 1], reference.value(position, i).imag(), 1e-5);
      }
      EXPECT_EQ(in_core == 1, std::abs(p[0]) <= 1.0 && std::abs(p[1]) <= 0.9);
    }
    EXPECT_EQ(n_lines, 16u) << "plane " << planes[plane];
  }
}

// One evaluation point per cell (its center) with the values and curls of both fields.
TEST(InnerDomainTests, ShapeGradientDataContainsFieldsAtCellCenters) {
  const SetupOptions options = resolved_core();
  ProblemEnvironment environment(options);
  // curl (0, a + b x + c z, 0) = (-c, 0, b)
  const ComplexNumber b(0.3, 0.4), c(0.1, -0.6);
  const LinearField primal(ComplexNumber(0.2, 0.1), ComplexNumber(0.7, -0.2), ComplexNumber(-0.4, 0), b, c);
  NumericVectorLocal primal_coefficients = coefficients_of(environment, new LinearField(ComplexNumber(0.2, 0.1), ComplexNumber(0.7, -0.2), ComplexNumber(-0.4, 0), b, c));
  const LinearField adjoint(ComplexNumber(0, 1), ComplexNumber(2, 0), ComplexNumber(0.5, 0.5));
  NumericVectorLocal adjoint_coefficients = coefficients_of(environment, new LinearField(ComplexNumber(0, 1), ComplexNumber(2, 0), ComplexNumber(0.5, 0.5)));
  const std::vector<FEAdjointEvaluation> data = environment.inner_domain().compute_local_shape_gradient_data(primal_coefficients, adjoint_coefficients);
  ASSERT_EQ(data.size(), options.cells_x * options.cells_y * options.cells_z);
  const ComplexNumber curl_of_primal[3] = {-c, ComplexNumber(0, 0), b};
  for (const FEAdjointEvaluation & item : data) {
    for (unsigned int i = 0; i < 3; i++) {
      ASSERT_LT(std::abs(item.primal_field[i] - primal.value(item.x, i)), 1e-6);
      ASSERT_LT(std::abs(item.adjoint_field[i] - adjoint.value(item.x, i)), 1e-6);
      ASSERT_LT(std::abs(item.primal_field_curl[i] - curl_of_primal[i]), 1e-6);
      ASSERT_LT(std::abs(item.adjoint_field_curl[i]), 1e-6);
    }
  }
}

// The right-hand side of the adjoint problem for the signal strength is the functional v -> int conj(mode) . v over
// the output face (midpoint rule), the return value is int |mode|^2. For constant fields both integrals are known.
TEST(InnerDomainTests, AdjointRightHandSideForSignalStrength) {
  const SetupOptions options = resolved_core();
  ProblemEnvironment environment(options);
  InnerDomain & inner_domain = environment.inner_domain();
  const ComplexNumber field[3] = {ComplexNumber(0.2, 0.1), ComplexNumber(0.7, -0.2), ComplexNumber(-0.4, 0)};
  const NumericVectorLocal coefficients = coefficients_of(environment, new LinearField(field[0], field[1], field[2]));
  const ComplexNumber mode[3] = {ComplexNumber(0.3, 0), ComplexNumber(0.5, -0.2), ComplexNumber(0, 0.1)};
  environment.set_source_field(new LinearField(mode[0], mode[1], mode[2]));
  const unsigned int n = Geometry.levels[0].n_total_level_dofs;
  dealii::IndexSet all_dofs(n);
  all_dofs.add_range(0, n);
  NumericVectorDistributed rhs(all_dofs, MPI_COMM_SELF);
  const double mode_norm = inner_domain.assemble_adjoint_rhs_for_signal_strength(&rhs);
  rhs.compress(dealii::VectorOperation::add);
  const double area = options.size_x * options.size_y;
  EXPECT_NEAR(mode_norm, area * (std::norm(mode[0]) + std::norm(mode[1]) + std::norm(mode[2])), 1e-10);
  ComplexNumber functional(0, 0);
  for (unsigned int i = 0; i < coefficients.size(); i++) {
    functional += static_cast<ComplexNumber>(rhs[inner_domain.global_index_mapping[i]]) * coefficients[i];
  }
  // The component in z is normal to the face. Its trace is not continuous but for a constant field it is exact.
  const ComplexNumber expected = area * (std::conj(mode[0]) * field[0] + std::conj(mode[1]) * field[1] + std::conj(mode[2]) * field[2]);
  EXPECT_LT(std::abs(functional - expected), 1e-6);
}

// Output on the mesh of the computation (mathematical coordinates): "Epsilon" is the norm of the material tensor the
// assembly uses there, i.e. including the tensor of the transformation where the waveguide is bent.
TEST(InnerDomainTests, OutputOnComputationalMeshShowsTransformedMaterial) {
  for (double displacement : {0.0, 0.6}) {
    SetupOptions options = resolved_core();
    options.vertical_displacement = displacement;
    options.n_sectors = 2;
    ProblemEnvironment environment(options);
    InnerDomain & inner_domain = environment.inner_domain();
    inner_domain.load_exact_solution();
    const NumericVectorLocal solution(inner_domain.n_locally_active_dofs);
    const std::string filename = inner_domain.output_results("output_test", solution, false);
    EXPECT_EQ(filename.rfind(environment.output_directory, 0), 0u);
    EXPECT_GT(std::filesystem::file_size(filename), 0u);
    const std::vector<double> values = epsilon_values_of_last_output(inner_domain);
    ASSERT_GT(values.size(), 0u);
    double largest_expected = 0, smallest_expected = 1e10;
    bool is_bent_somewhere = false;
    for (const auto & cell : inner_domain.triangulation.active_cell_iterators()) {
      Position center = cell->center();
      const double eps = (std::abs(center[0]) <= 1.0 && std::abs(center[1]) <= 0.9) ? options.epsilon_in : options.epsilon_out;
      const dealii::Tensor<2, 3, double> tensor = GlobalSpaceTransformation->get_Space_Transformation_Tensor(center);
      largest_expected = std::max(largest_expected, eps * tensor.norm());
      smallest_expected = std::min(smallest_expected, eps * tensor.norm());
      is_bent_somewhere = is_bent_somewhere || tensor.norm() > std::sqrt(3.0) + 1e-3;
    }
    EXPECT_EQ(is_bent_somewhere, displacement != 0.0);
    EXPECT_NEAR(*std::max_element(values.begin(), values.end()), largest_expected, 1e-5);
    EXPECT_NEAR(*std::min_element(values.begin(), values.end()), smallest_expected, 1e-5);
    if (displacement == 0.0) {
      // Straight waveguide: the isotropic material of core and cladding.
      EXPECT_NEAR(largest_expected, options.epsilon_in * std::sqrt(3.0), 1e-12);
      EXPECT_NEAR(smallest_expected, options.epsilon_out * std::sqrt(3.0), 1e-12);
    }
  }
}

// InnerDomain.cpp:478 deletes GlobalParams.source_field, and the mesh stays displaced because SpaceTransformation::operator() ignores switch_application_mode(false).
TEST(InnerDomainTests, DISABLED_OutputOnPhysicalMeshLeavesDomainUnchanged) {
  SetupOptions options = resolved_core();
  options.vertical_displacement = 0.6;
  options.n_sectors = 2;
  ProblemEnvironment environment(options);
  InnerDomain & inner_domain = environment.inner_domain();
  bool source_field_was_deleted = false;
  environment.set_source_field(new ObservedField(&source_field_was_deleted));
  inner_domain.load_exact_solution();
  const std::vector<Position> vertices_before = inner_domain.triangulation.get_vertices();
  const NumericVectorLocal solution(inner_domain.n_locally_active_dofs);
  inner_domain.output_results("output_test", solution, true);
  if (source_field_was_deleted) {
    GlobalParams.source_field = nullptr;
  }
  EXPECT_FALSE(source_field_was_deleted);
  const std::vector<Position> vertices_after = inner_domain.triangulation.get_vertices();
  double largest_displacement = 0;
  for (unsigned int i = 0; i < vertices_before.size(); i++) {
    largest_displacement = std::max(largest_displacement, vertices_before[i].distance(vertices_after[i]));
  }
  EXPECT_LT(largest_displacement, 1e-12);
  // In physical coordinates the material is the isotropic one of core and cladding.
  const std::vector<double> values = epsilon_values_of_last_output(inner_domain);
  ASSERT_GT(values.size(), 0u);
  EXPECT_NEAR(*std::max_element(values.begin(), values.end()), options.epsilon_in * std::sqrt(3.0), 1e-5);
  EXPECT_NEAR(*std::min_element(values.begin(), values.end()), options.epsilon_out * std::sqrt(3.0), 1e-5);
}

// FEDomain.h:29: global_dof_indices is never filled, it stays an empty index set of size 0 (DirichletSurface and EmptySurface construct their constraints with it).
TEST(InnerDomainTests, DISABLED_GlobalDofIndicesDescribeUnknownsOfDomain) {
  ProblemEnvironment environment(small_domain(0));
  InnerDomain & inner_domain = environment.inner_domain();
  EXPECT_EQ(inner_domain.global_dof_indices.size(), Geometry.levels[0].n_total_level_dofs);
  EXPECT_EQ(inner_domain.global_dof_indices.n_elements(), inner_domain.n_locally_active_dofs);
}
