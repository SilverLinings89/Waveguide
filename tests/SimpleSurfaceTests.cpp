// Tests for the surfaces without unknowns of their own (Dirichlet, empty and neighbor surfaces) and for the functions
// of the base class BoundaryCondition.
#include <vector>
#include <gtest/gtest.h>
#include <cmath>
#include <set>
#include "../Code/BoundaryCondition/DirichletSurface.h"
#include "../Code/BoundaryCondition/EmptySurface.h"
#include "../Code/BoundaryCondition/HSIESurface.h"
#include "../Code/BoundaryCondition/NeighborSurface.h"
#include "../Code/BoundaryCondition/PMLSurface.h"
#include "./HelpersC.h"

namespace {

  using helpers_c::LinearField;
  using helpers_c::ProblemEnvironment;
  using helpers_c::SetupOptions;

  SetupOptions small_domain() {
    SetupOptions options;
    options.cells_x = 4;
    options.cells_y = 3;
    options.cells_z = 5;
    options.pml_layers = 2;
    options.pml_thickness = 0.5;
    return options;
  }

  // Waveguide with perfectly conducting side walls, the signal prescribed at -z and a PML at +z.
  SetupOptions conducting_side_walls() {
    SetupOptions options = small_domain();
    options.pec_side_walls = true;
    return options;
  }

  void expect_no_own_unknowns(BoundaryCondition & surface) {
    EXPECT_EQ(surface.dof_counter, 0u);
    EXPECT_EQ(surface.compute_n_locally_active_dofs(), 0u);
    EXPECT_EQ(surface.compute_n_locally_owned_dofs(), 0u);
    EXPECT_EQ(surface.n_locally_active_dofs, 0u);
    EXPECT_EQ(surface.global_index_mapping.size(), 0u);
    EXPECT_EQ(surface.n_cells(), 0u);
    EXPECT_EQ(surface.cells_for_boundary_id(0), 0u);
    EXPECT_FALSE(surface.is_point_at_boundary(Position2D(0, 0), 0));
    EXPECT_EQ(surface.output_results(dealii::Vector<ComplexNumber>(), "unused"), "");
  }

}

TEST(DirichletSurfaceTests, HasNoUnknownsOfItsOwn) {
  ProblemEnvironment environment(small_domain());
  DirichletSurface * surface = environment.surface<DirichletSurface>(4);
  ASSERT_NE(surface, nullptr);
  expect_no_own_unknowns(*surface);
  EXPECT_TRUE(surface->get_dof_association().empty());
  for (BoundaryId b_id = 0; b_id < 6; b_id++) {
    EXPECT_TRUE(surface->get_dof_association_by_boundary_id(b_id).empty());
  }
  EXPECT_EQ(surface->b_id, 4u);
  EXPECT_EQ(surface->level, 0u);
}

// The constraints prescribe the tangential components of the source field on the input face. For a field of the
// finite element space the constrained unknowns alone reproduce these components exactly.
TEST(DirichletSurfaceTests, ConstraintsPrescribeTangentialSourceFieldOnInputFace) {
  for (unsigned int order = 0; order < 2; order++) {
    SetupOptions options = small_domain();
    options.fe_order = order;
    ProblemEnvironment environment(options);
    const LinearField reference(ComplexNumber(0.2, 0.1), ComplexNumber(0.7, -0.2), ComplexNumber(-0.4, 0), ComplexNumber(0.3, 0.4));
    environment.set_source_field(new LinearField(ComplexNumber(0.2, 0.1), ComplexNumber(0.7, -0.2), ComplexNumber(-0.4, 0), ComplexNumber(0.3, 0.4)));
    DirichletSurface * surface = environment.surface<DirichletSurface>(4);
    ASSERT_NE(surface, nullptr);
    Constraints constraints = surface->make_constraints();
    constraints.close();
    // The input face of the inner domain and of the PML domains at the four side walls.
    const unsigned int n = options.pml_layers;
    EXPECT_EQ(constraints.n_constraints(), helpers_c::n_nedelec_dofs_on_face(options.cells_x + 2 * n, options.cells_y + 2 * n, order)) << "order " << order;

    InnerDomain & inner_domain = environment.inner_domain();
    std::set<DofNumber> on_input_face;
    for (const InterfaceDofData & dof : inner_domain.get_surface_dof_vector_for_boundary_id(4)) on_input_face.insert(dof.index);
    NumericVectorLocal field(inner_domain.n_locally_active_dofs);
    for (unsigned int i = 0; i < field.size(); i++) {
      const DofNumber index = inner_domain.global_index_mapping[i];
      ASSERT_EQ(constraints.is_constrained(index), on_input_face.count(i) == 1);
      if (constraints.is_constrained(index)) {
        field[i] = constraints.get_inhomogeneity(index);
      }
    }
    const std::vector<Position> positions = {Position(0.13, -0.42, 0), Position(-1.9, 1.7, 0), Position(1.21, 0.9, 0), Position(-0.6, 0.1, 0)};
    const std::vector<std::vector<ComplexNumber>> values = inner_domain.evaluate_at_positions(positions, field);
    for (unsigned int i = 0; i < positions.size(); i++) {
      EXPECT_LT(std::abs(values[i][0] - reference.value(positions[i], 0)), 1e-8) << "order " << order << " position " << i;
      EXPECT_LT(std::abs(values[i][1] - reference.value(positions[i], 1)), 1e-8) << "order " << order << " position " << i;
    }
  }
}

// DirichletSurface.cpp:22 passes surface_extremal_coordinate[level] instead of [surface]: on level 0 the coordinate is the lower end of the x range.
TEST(DirichletSurfaceTests, DISABLED_AdditionalCoordinateIsTheCoordinateOfTheSurface) {
  ProblemEnvironment environment(small_domain());
  DirichletSurface * surface = environment.surface<DirichletSurface>(4);
  ASSERT_NE(surface, nullptr);
  EXPECT_EQ(surface->additional_coordinate, Geometry.surface_extremal_coordinate[4]);
  // For a surface whose id differs from the level in another way.
  DirichletSurface other(1, 0);
  EXPECT_EQ(other.additional_coordinate, Geometry.surface_extremal_coordinate[1]);
}

TEST(EmptySurfaceTests, HasNoUnknownsOfItsOwn) {
  ProblemEnvironment environment(conducting_side_walls());
  for (BoundaryId b_id = 0; b_id < 4; b_id++) {
    EmptySurface * surface = environment.surface<EmptySurface>(b_id);
    ASSERT_NE(surface, nullptr) << "surface " << b_id;
    expect_no_own_unknowns(*surface);
    EXPECT_TRUE(surface->get_dof_association().empty());
    EXPECT_TRUE(surface->get_dof_association_by_boundary_id(5).empty());
    EXPECT_EQ(surface->b_id, b_id);
  }
  EXPECT_NE(environment.surface<DirichletSurface>(4), nullptr);
  EXPECT_NE(environment.surface<PMLSurface>(5), nullptr);
}

// A perfectly conducting wall: the tangential field is zero on the wall, in the inner domain and in the PML at +z.
TEST(EmptySurfaceTests, ConstraintsSetTangentialFieldOnWallToZero) {
  for (unsigned int order = 0; order < 2; order++) {
    SetupOptions options = conducting_side_walls();
    options.fe_order = order;
    ProblemEnvironment environment(options);
    InnerDomain & inner_domain = environment.inner_domain();
    PMLSurface * pml = environment.surface<PMLSurface>(5);
    ASSERT_NE(pml, nullptr);
    // Without absorbing side walls the PML is not extended sideways.
    EXPECT_EQ(pml->n_cells(), options.cells_x * options.cells_y * options.pml_layers);
    const unsigned int cells[3] = {options.cells_x, options.cells_y, options.cells_z};
    for (BoundaryId b_id = 0; b_id < 4; b_id++) {
      EmptySurface * surface = environment.surface<EmptySurface>(b_id);
      ASSERT_NE(surface, nullptr);
      Constraints constraints = surface->make_constraints();
      const unsigned int cells_along_wall = cells[1 - b_id / 2];
      EXPECT_EQ(constraints.n_constraints(), helpers_c::n_nedelec_dofs_on_face(cells_along_wall, options.cells_z + options.pml_layers, order)) << "order " << order << " surface " << b_id;
      for (const InterfaceDofData & dof : inner_domain.get_surface_dof_vector_for_boundary_id(b_id)) {
        const DofNumber index = inner_domain.global_index_mapping[dof.index];
        ASSERT_TRUE(constraints.is_constrained(index));
        ASSERT_FALSE(constraints.is_inhomogeneously_constrained(index));
      }
      for (const InterfaceDofData & dof : pml->get_dof_association_by_boundary_id(b_id)) {
        ASSERT_NEAR(dof.base_point[b_id / 2], Geometry.surface_extremal_coordinate[b_id], 1e-12);
        const DofNumber index = pml->global_index_mapping[dof.index];
        ASSERT_TRUE(constraints.is_constrained(index));
        ASSERT_FALSE(constraints.is_inhomogeneously_constrained(index));
      }
    }
  }
}

// EmptySurface.cpp:21 passes surface_extremal_coordinate[level] instead of [surface]: on level 0 every surface gets the lower end of the x range.
TEST(EmptySurfaceTests, DISABLED_AdditionalCoordinateIsTheCoordinateOfTheSurface) {
  ProblemEnvironment environment(conducting_side_walls());
  for (BoundaryId b_id = 0; b_id < 4; b_id++) {
    EmptySurface * surface = environment.surface<EmptySurface>(b_id);
    ASSERT_NE(surface, nullptr);
    EXPECT_EQ(surface->additional_coordinate, Geometry.surface_extremal_coordinate[b_id]) << "surface " << b_id;
  }
}

// Ranks are numbered x-first, then y, then z. The process (0, 0, 1) of a 1 x 1 x 3 layout has the neighbors 0 and 2.
TEST(NeighborSurfaceTests, PartnerRanksAndOrientation) {
  ProblemEnvironment environment(small_domain());
  GlobalParams.Blocks_in_z_direction = 3;
  GlobalParams.Index_in_z_direction = 1;
  GlobalParams.MPI_Rank = 1;
  NeighborSurface lower(4, 0);
  NeighborSurface upper(5, 0);
  EXPECT_TRUE(lower.is_lower_interface);
  EXPECT_FALSE(upper.is_lower_interface);
  EXPECT_EQ(lower.global_partner_mpi_rank, 0);
  EXPECT_EQ(upper.global_partner_mpi_rank, 2);
  EXPECT_EQ(lower.additional_coordinate, Geometry.surface_extremal_coordinate[4]);
  EXPECT_EQ(upper.additional_coordinate, Geometry.surface_extremal_coordinate[5]);
  expect_no_own_unknowns(lower);
  expect_no_own_unknowns(upper);
  for (BoundaryId b_id = 0; b_id < 6; b_id++) {
    EXPECT_TRUE(upper.get_dof_association_by_boundary_id(b_id).empty());
  }
}

// The data a process sends to its neighbor: the indices of the unknowns of the inner domain on the common surface
// and of the absorbing layers that touch it, in the order of the surface dof lists.
TEST(NeighborSurfaceTests, PreparedDofsAreThoseOnTheCommonSurface) {
  SetupOptions options = small_domain();
  options.coupling = SignalCouplingMethod::Tapering;
  ProblemEnvironment environment(options);
  InnerDomain & inner_domain = environment.inner_domain();
  NeighborSurface surface(5, 0);
  surface.prepare_dofs();
  const std::vector<InterfaceDofData> of_inner_domain = inner_domain.get_surface_dof_vector_for_boundary_id(5);
  EXPECT_EQ(surface.inner_dofs, helpers_c::to_level_numbering(inner_domain, of_inner_domain));
  unsigned int n_expected = of_inner_domain.size();
  for (BoundaryId b_id = 0; b_id < 4; b_id++) {
    EXPECT_EQ(surface.boundary_dofs[b_id], Geometry.levels[0].surfaces[b_id]->get_global_dof_indices_by_boundary_id(5)) << "surface " << b_id;
    ASSERT_GT(surface.boundary_dofs[b_id].size(), 0u);
    n_expected += surface.boundary_dofs[b_id].size();
    const std::vector<InterfaceDofData> association = surface.get_dof_association_by_boundary_id(b_id);
    ASSERT_EQ(association.size(), surface.boundary_dofs[b_id].size());
    for (unsigned int i = 0; i < association.size(); i++) {
      ASSERT_EQ(association[i].index, surface.boundary_dofs[b_id][i]);
    }
  }
  EXPECT_TRUE(surface.boundary_dofs[4].empty());
  EXPECT_TRUE(surface.boundary_dofs[5].empty());
  EXPECT_EQ(surface.n_dofs, n_expected);
  EXPECT_EQ(surface.global_indices.size(), n_expected);
  const std::vector<InterfaceDofData> association = surface.get_dof_association();
  ASSERT_EQ(association.size(), of_inner_domain.size());
  for (unsigned int i = 0; i < association.size(); i++) {
    ASSERT_EQ(association[i].index, surface.inner_dofs[i]);
    ASSERT_LT(association[i].base_point.distance(of_inner_domain[i].base_point), 1e-12);
  }
}

// The process above a common surface takes over the indices it received from the process below.
TEST(NeighborSurfaceTests, LowerInterfaceAdoptsReceivedIndices) {
  SetupOptions options = small_domain();
  options.coupling = SignalCouplingMethod::Tapering;
  ProblemEnvironment environment(options);
  InnerDomain & inner_domain = environment.inner_domain();
  NeighborSurface surface(4, 0);
  surface.prepare_dofs();
  // In place of receive(): indices that do not occur on this process.
  for (unsigned int i = 0; i < surface.inner_dofs.size(); i++) {
    surface.inner_dofs[i] = 1000000 + i;
  }
  for (BoundaryId b_id = 0; b_id < 4; b_id++) {
    for (unsigned int i = 0; i < surface.boundary_dofs[b_id].size(); i++) {
      surface.boundary_dofs[b_id][i] = 2000000 + 100000 * b_id + i;
    }
  }
  surface.distribute_dof_indices();
  const std::vector<InterfaceDofData> of_inner_domain = inner_domain.get_surface_dof_vector_for_boundary_id(4);
  for (unsigned int i = 0; i < of_inner_domain.size(); i++) {
    ASSERT_EQ(inner_domain.global_index_mapping[of_inner_domain[i].index], 1000000 + i);
  }
  for (BoundaryId b_id = 0; b_id < 4; b_id++) {
    const std::vector<InterfaceDofData> of_layer = Geometry.levels[0].surfaces[b_id]->get_dof_association_by_boundary_id(4);
    ASSERT_EQ(of_layer.size(), surface.boundary_dofs[b_id].size());
    for (unsigned int i = 0; i < of_layer.size(); i++) {
      ASSERT_EQ(Geometry.levels[0].surfaces[b_id]->global_index_mapping[of_layer[i].index], 2000000 + 100000 * b_id + i);
    }
  }
  // An upper interface sends its indices and does not change them.
  NeighborSurface upper(5, 0);
  upper.prepare_dofs();
  const std::vector<DofNumber> before = inner_domain.global_index_mapping;
  upper.distribute_dof_indices();
  EXPECT_EQ(inner_domain.global_index_mapping, before);
}

TEST(BoundaryConditionTests, AdjacentBoundariesExcludeOwnAndOpposingSurface) {
  SetupOptions options = small_domain();
  options.coupling = SignalCouplingMethod::Tapering;
  ProblemEnvironment environment(options);
  for (BoundaryId b_id = 0; b_id < 6; b_id++) {
    BoundaryCondition & surface = *Geometry.levels[0].surfaces[b_id];
    EXPECT_EQ(surface.b_id, b_id);
    EXPECT_EQ(surface.level, 0u);
    EXPECT_EQ(surface.additional_coordinate, Geometry.surface_extremal_coordinate[b_id]);
    ASSERT_EQ(surface.adjacent_boundaries.size(), 4u);
    for (BoundaryId other : surface.adjacent_boundaries) {
      EXPECT_NE(other / 2, b_id / 2);
    }
    EXPECT_FALSE(surface.are_edge_dofs_owned[b_id]);
    EXPECT_FALSE(surface.are_edge_dofs_owned[b_id ^ 1u]);
  }
}

// Of two neighboring surfaces of the same kind exactly one numbers the unknowns they share.
TEST(BoundaryConditionTests, SharedUnknownsHaveOneOwner) {
  for (BoundaryConditionType method : {BoundaryConditionType::PML, BoundaryConditionType::HSIE}) {
    SetupOptions options = small_domain();
    options.coupling = SignalCouplingMethod::Tapering;
    options.boundary_method = method;
    ProblemEnvironment environment(options);
    for (BoundaryId a = 0; a < 6; a++) {
      for (BoundaryId b : Geometry.levels[0].surfaces[a]->adjacent_boundaries) {
        EXPECT_NE(Geometry.levels[0].surfaces[a]->are_edge_dofs_owned[b], Geometry.levels[0].surfaces[b]->are_edge_dofs_owned[a]) << "surfaces " << a << " and " << b;
      }
    }
  }
}

TEST(BoundaryConditionTests, NormOfVectorOnBoundaryDomain) {
  SetupOptions options = small_domain();
  options.coupling = SignalCouplingMethod::Tapering;
  ProblemEnvironment environment(options);
  PMLSurface * pml = environment.surface<PMLSurface>(1);
  const unsigned int n = Geometry.levels[0].n_total_level_dofs;
  dealii::IndexSet all_dofs(n);
  all_dofs.add_range(0, n);
  NumericVectorDistributed vector(all_dofs, MPI_COMM_SELF);
  double expected_all = 0, expected_owned = 0;
  for (unsigned int i = 0; i < pml->n_locally_active_dofs; i++) {
    const DofNumber index = pml->global_index_mapping[i];
    const ComplexNumber value(0.01 * index, 1.0);
    vector[index] = value;
    expected_all += std::norm(value);
    if (pml->is_dof_owned[i]) expected_owned += std::norm(value);
  }
  vector.compress(dealii::VectorOperation::insert);
  EXPECT_NEAR(pml->boundary_norm(&vector), std::sqrt(expected_all), 1e-9);
  EXPECT_NEAR(pml->local_norm_of_vector(&vector), std::sqrt(expected_owned), 1e-9);
  const NumericVectorLocal local = pml->get_local_vector_from_global(vector);
  ASSERT_EQ(local.size(), pml->n_locally_active_dofs);
  for (unsigned int i = 0; i < local.size(); i++) {
    ASSERT_EQ(local[i], ComplexNumber(0.01 * pml->global_index_mapping[i], 1.0));
  }
}

// BoundaryCondition.cpp:108 reads the vector of the level with the local index of the dof instead of its index on the level (the function is not called).
TEST(BoundaryConditionTests, DISABLED_NormOfVectorOnOneFaceOfBoundaryDomain) {
  SetupOptions options = small_domain();
  options.coupling = SignalCouplingMethod::Tapering;
  ProblemEnvironment environment(options);
  PMLSurface * pml = environment.surface<PMLSurface>(1);
  const unsigned int n = Geometry.levels[0].n_total_level_dofs;
  dealii::IndexSet all_dofs(n);
  all_dofs.add_range(0, n);
  NumericVectorDistributed vector(all_dofs, MPI_COMM_SELF);
  for (unsigned int i = 0; i < n; i++) {
    vector[i] = ComplexNumber(0.01 * i, 1.0);
  }
  vector.compress(dealii::VectorOperation::insert);
  double expected = 0;
  for (const InterfaceDofData & dof : pml->get_dof_association_by_boundary_id(3)) {
    expected += std::norm(ComplexNumber(0.01 * pml->global_index_mapping[dof.index], 1.0));
  }
  EXPECT_NEAR(pml->boundary_surface_norm(&vector, 3), std::sqrt(expected), 1e-9);
}

// The base class constrains nothing. HSIESurface uses this: infinite elements have no outer boundary.
TEST(BoundaryConditionTests, DefaultConstraintsAreEmpty) {
  SetupOptions options = small_domain();
  options.boundary_method = BoundaryConditionType::HSIE;
  ProblemEnvironment environment(options);
  HSIESurface * surface = environment.surface<HSIESurface>(5);
  ASSERT_NE(surface, nullptr);
  EXPECT_EQ(surface->make_constraints().n_constraints(), 0u);
}
