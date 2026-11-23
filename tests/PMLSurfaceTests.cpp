// Tests for the PML domains: the meshes of the six layers, their unknowns and the coupling of these unknowns to the
// inner domain and to the neighboring layers, the material tensor and the constraints on the outer boundary.
#include <gtest/gtest.h>
#include <cmath>
#include <set>
#include <deal.II/base/bounding_box.h>
#include <deal.II/grid/grid_tools.h>
#include "../Code/BoundaryCondition/DirichletSurface.h"
#include "../Code/BoundaryCondition/PMLSurface.h"
#include "./HelpersC.h"

namespace {

  using helpers_c::ProblemEnvironment;
  using helpers_c::SetupOptions;

  // Different cell counts and sizes in all directions, so that mixed up directions are noticed.
  SetupOptions pml_on_all_sides() {
    SetupOptions options;
    options.cells_x = 3;
    options.cells_y = 4;
    options.cells_z = 5;
    options.size_x = 3.0;
    options.size_y = 3.6;
    options.size_z = 2.5;
    options.pml_layers = 2;
    options.pml_thickness = 0.8;
    options.coupling = SignalCouplingMethod::Tapering;
    return options;
  }

  // Cells per direction of the layer on a surface. The layers in y also cover the edges towards the layers in x, the
  // layers in z cover all edges and corners they touch. Together with the inner domain they fill a box.
  std::array<unsigned int, 3> expected_cells(const SetupOptions & options, BoundaryId b_id) {
    const unsigned int n = options.pml_layers;
    switch (b_id / 2) {
      case 0:
        return {{n, options.cells_y, options.cells_z}};
      case 1:
        return {{options.cells_x + 2 * n, n, options.cells_z}};
      default:
        return {{options.cells_x + 2 * n, options.cells_y + 2 * n, n}};
    }
  }

  // Lower and upper corner of the layer on a surface.
  std::pair<Position, Position> expected_extent(const SetupOptions & options, BoundaryId b_id) {
    const double t = options.pml_thickness;
    Position lower(-options.size_x / 2, -options.size_y / 2, 0);
    Position upper(options.size_x / 2, options.size_y / 2, options.size_z);
    const unsigned int direction = b_id / 2;
    for (unsigned int i = 0; i < direction; i++) {
      lower[i] -= t;
      upper[i] += t;
    }
    if (b_id % 2 == 0) {
      upper[direction] = lower[direction];
      lower[direction] -= t;
    } else {
      lower[direction] = upper[direction];
      upper[direction] += t;
    }
    return {lower, upper};
  }

  // Relative permittivity of the straight waveguide: the core has the cross-section 2.0 x 1.8.
  double expected_epsilon(const SetupOptions & options, const Position & p) {
    return (std::abs(p[0]) <= 1.0 && std::abs(p[1]) <= 0.9) ? options.epsilon_in : options.epsilon_out;
  }

  ComplexNumber stretching(double fraction) {
    return ComplexNumber(1.0, GlobalParams.PML_Sigma_Max * std::pow(fraction, GlobalParams.PML_skaling_order));
  }

  void expect_diagonal(const dealii::Tensor<2, 3, ComplexNumber> & tensor, ComplexNumber d0, ComplexNumber d1, ComplexNumber d2) {
    const ComplexNumber expected[3] = {d0, d1, d2};
    for (unsigned int i = 0; i < 3; i++) {
      for (unsigned int j = 0; j < 3; j++) {
        const ComplexNumber reference = (i == j) ? expected[i] : ComplexNumber(0, 0);
        EXPECT_NEAR(tensor[i][j].real(), reference.real(), 1e-12) << "entry " << i << "," << j;
        EXPECT_NEAR(tensor[i][j].imag(), reference.imag(), 1e-12) << "entry " << i << "," << j;
      }
    }
  }

}

TEST(PMLSurfaceTests, CellCountsOfAllSixLayers) {
  const SetupOptions options = pml_on_all_sides();
  ProblemEnvironment environment(options);
  unsigned int n_cells_total = options.cells_x * options.cells_y * options.cells_z;
  for (BoundaryId b_id = 0; b_id < 6; b_id++) {
    PMLSurface * pml = environment.surface<PMLSurface>(b_id);
    ASSERT_NE(pml, nullptr) << "surface " << b_id;
    const std::array<unsigned int, 3> cells = expected_cells(options, b_id);
    EXPECT_EQ(pml->n_cells(), cells[0] * cells[1] * cells[2]) << "surface " << b_id;
    n_cells_total += pml->n_cells();
  }
  const unsigned int n = options.pml_layers;
  EXPECT_EQ(n_cells_total, (options.cells_x + 2 * n) * (options.cells_y + 2 * n) * (options.cells_z + 2 * n));
}

TEST(PMLSurfaceTests, ExtentsOfAllSixLayers) {
  const SetupOptions options = pml_on_all_sides();
  ProblemEnvironment environment(options);
  for (BoundaryId b_id = 0; b_id < 6; b_id++) {
    PMLSurface * pml = environment.surface<PMLSurface>(b_id);
    ASSERT_NE(pml, nullptr);
    const std::pair<Position, Position> expected = expected_extent(options, b_id);
    const dealii::BoundingBox<3> box = dealii::GridTools::compute_bounding_box(pml->triangulation);
    double volume = 1;
    for (unsigned int i = 0; i < 3; i++) {
      EXPECT_NEAR(box.get_boundary_points().first[i], expected.first[i], 1e-12) << "surface " << b_id << " direction " << i;
      EXPECT_NEAR(box.get_boundary_points().second[i], expected.second[i], 1e-12) << "surface " << b_id << " direction " << i;
      volume *= expected.second[i] - expected.first[i];
    }
    // The mesh fills its bounding box without holes or overlapping parts.
    EXPECT_NEAR(dealii::GridTools::volume(pml->triangulation), volume, 1e-10) << "surface " << b_id;
  }
}

// All cells of a layer have the same thickness: PML thickness / number of layers.
TEST(PMLSurfaceTests, LayersHaveUniformThickness) {
  const SetupOptions options = pml_on_all_sides();
  ProblemEnvironment environment(options);
  for (BoundaryId b_id = 0; b_id < 6; b_id++) {
    PMLSurface * pml = environment.surface<PMLSurface>(b_id);
    ASSERT_NE(pml, nullptr);
    EXPECT_NEAR(pml->non_pml_layer_thickness, options.pml_thickness / options.pml_layers, 1e-14);
    for (const auto & cell : pml->triangulation.active_cell_iterators()) {
      ASSERT_NEAR(cell->extent_in_direction(b_id / 2), options.pml_thickness / options.pml_layers, 1e-12) << "surface " << b_id;
    }
  }
}

TEST(PMLSurfaceTests, NumberOfUnknownsMatchesNedelecFormula) {
  for (unsigned int order = 0; order < 3; order++) {
    SetupOptions options = pml_on_all_sides();
    options.fe_order = order;
    ProblemEnvironment environment(options);
    for (BoundaryId b_id = 0; b_id < 6; b_id++) {
      PMLSurface * pml = environment.surface<PMLSurface>(b_id);
      ASSERT_NE(pml, nullptr);
      const std::array<unsigned int, 3> cells = expected_cells(options, b_id);
      const unsigned int expected = helpers_c::n_nedelec_dofs(cells[0], cells[1], cells[2], order);
      EXPECT_EQ(pml->dof_counter, expected) << "order " << order << " surface " << b_id;
      EXPECT_EQ(pml->compute_n_locally_active_dofs(), expected);
      EXPECT_EQ(pml->n_locally_active_dofs, expected);
      EXPECT_EQ(pml->global_index_mapping.size(), expected);
    }
  }
}

// The inner domain and the six layers form a conforming mesh of a box. Every unknown on a common face has to be
// counted once, so the level has as many unknowns as a Nedelec element on the mesh of the whole box.
TEST(PMLSurfaceTests, UnknownsOfLevelAreThoseOfTheSurroundingBox) {
  for (unsigned int order = 0; order < 2; order++) {
    SetupOptions options = pml_on_all_sides();
    options.fe_order = order;
    ProblemEnvironment environment(options);
    const unsigned int n = options.pml_layers;
    const unsigned int expected = helpers_c::n_nedelec_dofs(options.cells_x + 2 * n, options.cells_y + 2 * n, options.cells_z + 2 * n, order);
    EXPECT_EQ(Geometry.levels[0].n_total_level_dofs, expected) << "order " << order;
    EXPECT_EQ(Geometry.levels[0].n_local_dofs, expected);
    // Every index of the level belongs to at least one domain and every unknown of a domain has an index.
    std::vector<bool> is_used(expected, false);
    unsigned int n_owned = environment.inner_domain().n_locally_owned_dofs;
    for (DofNumber index : environment.inner_domain().global_index_mapping) {
      ASSERT_LT(index, expected);
      is_used[index] = true;
    }
    for (BoundaryId b_id = 0; b_id < 6; b_id++) {
      n_owned += Geometry.levels[0].surfaces[b_id]->n_locally_owned_dofs;
      for (DofNumber index : Geometry.levels[0].surfaces[b_id]->global_index_mapping) {
        ASSERT_LT(index, expected) << "surface " << b_id;
        is_used[index] = true;
      }
    }
    EXPECT_EQ(n_owned, expected);
    EXPECT_EQ(std::count(is_used.begin(), is_used.end(), false), 0);
    EXPECT_TRUE(helpers_c::dof_bookkeeping_complaints(environment.output()).empty());
  }
}

// With the signal prescribed on the input face there is no layer at -z.
TEST(PMLSurfaceTests, NoLayerOnDirichletInputFace) {
  SetupOptions options = pml_on_all_sides();
  options.coupling = SignalCouplingMethod::Dirichlet;
  ProblemEnvironment environment(options);
  EXPECT_EQ(environment.surface<PMLSurface>(4), nullptr);
  EXPECT_NE(environment.surface<DirichletSurface>(4), nullptr);
  const unsigned int n = options.pml_layers;
  for (BoundaryId b_id : {0u, 1u, 2u, 3u, 5u}) {
    PMLSurface * pml = environment.surface<PMLSurface>(b_id);
    ASSERT_NE(pml, nullptr);
    const std::array<unsigned int, 3> cells = expected_cells(options, b_id);
    EXPECT_EQ(pml->n_cells(), cells[0] * cells[1] * cells[2]) << "surface " << b_id;
  }
  EXPECT_EQ(Geometry.levels[0].n_total_level_dofs, helpers_c::n_nedelec_dofs(options.cells_x + 2 * n, options.cells_y + 2 * n, options.cells_z + n, 0));
}

// The unknowns on the face towards the inner domain are the unknowns of the inner domain on that surface.
TEST(PMLSurfaceTests, InterfaceToInnerDomainSharesUnknowns) {
  for (unsigned int order = 0; order < 2; order++) {
    SetupOptions options = pml_on_all_sides();
    options.fe_order = order;
    ProblemEnvironment environment(options);
    const unsigned int cells[3] = {options.cells_x, options.cells_y, options.cells_z};
    for (BoundaryId b_id = 0; b_id < 6; b_id++) {
      PMLSurface * pml = environment.surface<PMLSurface>(b_id);
      ASSERT_NE(pml, nullptr);
      const std::vector<InterfaceDofData> of_layer = pml->get_dof_association();
      const std::vector<InterfaceDofData> of_inner_domain = environment.inner_domain().get_surface_dof_vector_for_boundary_id(b_id);
      const unsigned int expected = helpers_c::n_nedelec_dofs_on_face(cells[(b_id / 2 + 1) % 3], cells[(b_id / 2 + 2) % 3], order);
      ASSERT_EQ(of_layer.size(), expected) << "order " << order << " surface " << b_id;
      ASSERT_EQ(of_inner_domain.size(), expected);
      for (unsigned int i = 0; i < expected; i++) {
        ASSERT_LT(of_layer[i].base_point.distance(of_inner_domain[i].base_point), 1e-10);
        ASSERT_NEAR(of_layer[i].base_point[b_id / 2], Geometry.surface_extremal_coordinate[b_id], 1e-12);
        ASSERT_FALSE(pml->is_dof_owned[of_layer[i].index]);
      }
      EXPECT_EQ(helpers_c::to_level_numbering(*pml, of_layer), helpers_c::to_level_numbering(environment.inner_domain(), of_inner_domain));
    }
  }
}

// Two neighboring layers meet in a face. Both sides have to list the same unknowns in the same order, and exactly
// one of the two layers numbers them.
TEST(PMLSurfaceTests, NeighboringLayersShareUnknowns) {
  for (unsigned int order = 0; order < 2; order++) {
    SetupOptions options = pml_on_all_sides();
    options.fe_order = order;
    ProblemEnvironment environment(options);
    const unsigned int cells[3] = {options.cells_x, options.cells_y, options.cells_z};
    for (BoundaryId a = 0; a < 6; a++) {
      for (BoundaryId b = a + 1; b < 6; b++) {
        if (are_opposing_sites(a, b)) continue;
        PMLSurface * first = environment.surface<PMLSurface>(a);
        PMLSurface * second = environment.surface<PMLSurface>(b);
        const std::vector<InterfaceDofData> of_first = first->get_dof_association_by_boundary_id(b);
        const std::vector<InterfaceDofData> of_second = second->get_dof_association_by_boundary_id(a);
        // The common face spans the layers in one direction. In the third direction it spans the inner domain and,
        // between the layers in y and z, also the layers in x.
        const unsigned int third_direction = 3 - a / 2 - b / 2;
        const unsigned int cells_in_third_direction = cells[third_direction] + ((third_direction == 0) ? 2 * options.pml_layers : 0);
        const unsigned int expected = helpers_c::n_nedelec_dofs_on_face(options.pml_layers, cells_in_third_direction, order);
        ASSERT_EQ(of_first.size(), expected) << "order " << order << " surfaces " << a << " and " << b;
        ASSERT_EQ(of_second.size(), expected) << "order " << order << " surfaces " << a << " and " << b;
        for (unsigned int i = 0; i < expected; i++) {
          ASSERT_LT(of_first[i].base_point.distance(of_second[i].base_point), 1e-10);
          ASSERT_FALSE(first->is_dof_owned[of_first[i].index] && second->is_dof_owned[of_second[i].index]);
        }
        EXPECT_EQ(first->get_global_dof_indices_by_boundary_id(b), second->get_global_dof_indices_by_boundary_id(a));
        EXPECT_NE(first->are_edge_dofs_owned[b], second->are_edge_dofs_owned[a]);
        EXPECT_EQ(first->cells_for_boundary_id(b), options.pml_layers * cells_in_third_direction);
        EXPECT_EQ(second->cells_for_boundary_id(a), options.pml_layers * cells_in_third_direction);
      }
    }
  }
}

// The tangential field vanishes on the outer boundary of every layer.
TEST(PMLSurfaceTests, ConstraintsAreHomogeneousAndCoverTheOuterBoundary) {
  const SetupOptions options = pml_on_all_sides();
  ProblemEnvironment environment(options);
  for (BoundaryId b_id = 0; b_id < 6; b_id++) {
    PMLSurface * pml = environment.surface<PMLSurface>(b_id);
    ASSERT_NE(pml, nullptr);
    Constraints constraints = pml->make_constraints();
    const std::vector<InterfaceDofData> outer_dofs = pml->get_dof_association_by_boundary_id(b_id);
    ASSERT_GT(outer_dofs.size(), 0u);
    EXPECT_EQ(constraints.n_constraints(), outer_dofs.size());
    const std::pair<Position, Position> extent = expected_extent(options, b_id);
    bool reaches_far_face = false;
    for (const InterfaceDofData & dof : outer_dofs) {
      const DofNumber index = pml->global_index_mapping[dof.index];
      ASSERT_TRUE(constraints.is_constrained(index));
      ASSERT_FALSE(constraints.is_inhomogeneously_constrained(index));
      // The dof lies on the boundary of the box that the layer fills.
      bool on_boundary = false;
      for (unsigned int i = 0; i < 3; i++) {
        on_boundary = on_boundary || std::abs(dof.base_point[i] - extent.first[i]) < 1e-10 || std::abs(dof.base_point[i] - extent.second[i]) < 1e-10;
      }
      ASSERT_TRUE(on_boundary);
      const double far_coordinate = (b_id % 2 == 0) ? extent.first[b_id / 2] : extent.second[b_id / 2];
      reaches_far_face = reaches_far_face || std::abs(dof.base_point[b_id / 2] - far_coordinate) < 1e-10;
    }
    EXPECT_TRUE(reaches_far_face);
    // The far face is covered completely.
    const std::array<unsigned int, 3> cells = expected_cells(options, b_id);
    EXPECT_GE(outer_dofs.size(), helpers_c::n_nedelec_dofs_on_face(cells[(b_id / 2 + 1) % 3], cells[(b_id / 2 + 2) % 3], 0));
  }
}

// Inside the range of the inner domain and in the cell layer next to it nothing is absorbed.
TEST(PMLSurfaceTests, TensorIsIdentityWithoutAbsorption) {
  const SetupOptions options = pml_on_all_sides();
  ProblemEnvironment environment(options);
  const double first_layer = options.pml_thickness / options.pml_layers;
  for (BoundaryId b_id = 0; b_id < 6; b_id++) {
    PMLSurface * pml = environment.surface<PMLSurface>(b_id);
    Position p(0.1, -0.2, 1.1);
    const double sign = (b_id % 2 == 0) ? -1.0 : 1.0;
    p[b_id / 2] = Geometry.surface_extremal_coordinate[b_id] + sign * 0.9 * first_layer;
    const std::array<double, 3> fractions = pml->fraction_of_pml_direction(p);
    for (unsigned int i = 0; i < 3; i++) {
      EXPECT_EQ(fractions[i], 0.0);
    }
    expect_diagonal(pml->get_pml_tensor(p), 1.0, 1.0, 1.0);
    expect_diagonal(pml->get_pml_tensor_mu(p), 1.0, 1.0, 1.0);
    const double eps = expected_epsilon(options, p);
    expect_diagonal(pml->get_pml_tensor_epsilon(p), eps, eps, eps);
  }
}

// With the stretching s = 1 + i sigma in the direction n of the layer, the tensor is 1/s in the entry (n, n) and s in
// the other two diagonal entries. The time dependence is exp(-i omega t), so sigma > 0 is the absorbing sign: the
// wave exp(i k x) becomes exp(i k x) exp(-k int sigma) in the stretched coordinate.
TEST(PMLSurfaceTests, TensorAbsorbsInDirectionOfTheLayer) {
  const SetupOptions options = pml_on_all_sides();
  ProblemEnvironment environment(options);
  const double first_layer = options.pml_thickness / options.pml_layers;
  const double absorbing_thickness = options.pml_thickness - first_layer;
  for (BoundaryId b_id = 0; b_id < 6; b_id++) {
    PMLSurface * pml = environment.surface<PMLSurface>(b_id);
    const unsigned int direction = b_id / 2;
    const double sign = (b_id % 2 == 0) ? -1.0 : 1.0;
    for (double fraction : {0.25, 0.5, 1.0}) {
      Position p(0.1, -0.2, 1.1);
      p[direction] = Geometry.surface_extremal_coordinate[b_id] + sign * (first_layer + fraction * absorbing_thickness);
      const std::array<double, 3> fractions = pml->fraction_of_pml_direction(p);
      for (unsigned int i = 0; i < 3; i++) {
        EXPECT_NEAR(fractions[i], (i == direction) ? fraction : 0.0, 1e-12) << "surface " << b_id;
      }
      const ComplexNumber s = stretching(fraction);
      ComplexNumber diagonal[3] = {s, s, s};
      diagonal[direction] = 1.0 / s;
      const dealii::Tensor<2, 3, ComplexNumber> tensor = pml->get_pml_tensor(p);
      expect_diagonal(tensor, diagonal[0], diagonal[1], diagonal[2]);
      EXPECT_GT(tensor[(direction + 1) % 3][(direction + 1) % 3].imag(), 0.0);
      EXPECT_LT(tensor[direction][direction].imag(), 0.0);
      expect_diagonal(pml->get_pml_tensor_mu(p), diagonal[0], diagonal[1], diagonal[2]);
      const double eps = expected_epsilon(options, p);
      expect_diagonal(pml->get_pml_tensor_epsilon(p), eps * diagonal[0], eps * diagonal[1], eps * diagonal[2]);
    }
  }
}

TEST(PMLSurfaceTests, EpsilonTensorUsesCladdingOutsideOfCore) {
  const SetupOptions options = pml_on_all_sides();
  ProblemEnvironment environment(options);
  PMLSurface * pml = environment.surface<PMLSurface>(5);
  // In the layer at +z, in the cell layer without absorption.
  const double z = options.size_z + 0.1;
  const double eps_core = options.epsilon_in;
  const double eps_cladding = options.epsilon_out;
  expect_diagonal(pml->get_pml_tensor_epsilon(Position(0.9, 0.8, z)), eps_core, eps_core, eps_core);
  expect_diagonal(pml->get_pml_tensor_epsilon(Position(1.1, 0.0, z)), eps_cladding, eps_cladding, eps_cladding);
  expect_diagonal(pml->get_pml_tensor_epsilon(Position(0.0, 1.0, z)), eps_cladding, eps_cladding, eps_cladding);
}

// sigma = sigma_max * (relative depth)^order: zero at the start, sigma_max at the outer boundary for every order, and
// for a larger order smaller in between.
TEST(PMLSurfaceTests, SigmaProfileFollowsScalingOrderAndReachesSigmaMax) {
  SetupOptions options = pml_on_all_sides();
  options.pml_sigma_max = 7.5;
  ProblemEnvironment environment(options);
  PMLSurface * pml = environment.surface<PMLSurface>(1);
  const double first_layer = options.pml_thickness / options.pml_layers;
  const double absorbing_thickness = options.pml_thickness - first_layer;
  const double x_start = options.size_x / 2 + first_layer;
  auto sigma_at = [&](double fraction) {
    // The entry (1, 1) of the tensor is s_x = 1 + i sigma.
    const dealii::Tensor<2, 3, ComplexNumber> tensor = pml->get_pml_tensor(Position(x_start + fraction * absorbing_thickness, 0, 1));
    EXPECT_NEAR(tensor[1][1].real(), 1.0, 1e-12);
    return tensor[1][1].imag();
  };
  double sigma_of_previous_order = 0;
  for (unsigned int order = 1; order <= 4; order++) {
    GlobalParams.PML_skaling_order = order;
    EXPECT_NEAR(sigma_at(0.0), 0.0, 1e-12);
    EXPECT_NEAR(sigma_at(1.0), options.pml_sigma_max, 1e-10);
    double previous = 0;
    for (double fraction = 0.1; fraction < 1.05; fraction += 0.1) {
      const double sigma = sigma_at(fraction);
      EXPECT_NEAR(sigma, options.pml_sigma_max * std::pow(fraction, order), 1e-10);
      EXPECT_GT(sigma, previous);
      previous = sigma;
    }
    if (order > 1) {
      EXPECT_LT(sigma_at(0.5), sigma_of_previous_order);
    }
    sigma_of_previous_order = sigma_at(0.5);
  }
}

// In the edge and corner regions the stretchings of the directions are combined:
// diag(s_y s_z / s_x, s_x s_z / s_y, s_x s_y / s_z).
TEST(PMLSurfaceTests, TensorCombinesDirectionsInEdgesAndCorners) {
  const SetupOptions options = pml_on_all_sides();
  ProblemEnvironment environment(options);
  const double first_layer = options.pml_thickness / options.pml_layers;
  const double absorbing_thickness = options.pml_thickness - first_layer;
  PMLSurface * pml = environment.surface<PMLSurface>(5);
  // Edge between +y and +z.
  Position on_edge(0.3, options.size_y / 2 + first_layer + 0.4 * absorbing_thickness, options.size_z + first_layer + 0.7 * absorbing_thickness);
  ComplexNumber sy = stretching(0.4), sz = stretching(0.7);
  expect_diagonal(pml->get_pml_tensor(on_edge), sy * sz, sz / sy, sy / sz);
  // Corner at -x, +y, +z.
  Position in_corner(-options.size_x / 2 - first_layer - 0.9 * absorbing_thickness, on_edge[1], on_edge[2]);
  const ComplexNumber sx = stretching(0.9);
  expect_diagonal(pml->get_pml_tensor(in_corner), sy * sz / sx, sx * sz / sy, sx * sy / sz);
  // The neighboring layers describe the same material where they are evaluated at the same point.
  for (BoundaryId b_id = 0; b_id < 6; b_id++) {
    const dealii::Tensor<2, 3, ComplexNumber> other = environment.surface<PMLSurface>(b_id)->get_pml_tensor(in_corner);
    expect_diagonal(other, sy * sz / sx, sx * sz / sy, sx * sy / sz);
  }
}

// Mirror images with respect to the center of the domain have the same material.
TEST(PMLSurfaceTests, TensorIsSymmetricWithRespectToTheDomain) {
  const SetupOptions options = pml_on_all_sides();
  ProblemEnvironment environment(options);
  PMLSurface * lower = environment.surface<PMLSurface>(4);
  PMLSurface * upper = environment.surface<PMLSurface>(5);
  for (double depth = 0.05; depth < options.pml_thickness; depth += 0.1) {
    const dealii::Tensor<2, 3, ComplexNumber> at_lower = lower->get_pml_tensor(Position(-1.7, 2.0, -depth));
    const dealii::Tensor<2, 3, ComplexNumber> at_upper = upper->get_pml_tensor(Position(1.7, -2.0, options.size_z + depth));
    for (unsigned int i = 0; i < 3; i++) {
      EXPECT_NEAR(std::abs(at_lower[i][i] - at_upper[i][i]), 0.0, 1e-12);
    }
  }
}

TEST(PMLSurfaceTests, PositionChecksOfMeshBoundary) {
  const SetupOptions options = pml_on_all_sides();
  ProblemEnvironment environment(options);
  PMLSurface * pml = environment.surface<PMLSurface>(1);
  const double x_inner = options.size_x / 2;
  const double x_outer = x_inner + options.pml_thickness;
  EXPECT_TRUE(pml->is_position_at_boundary(Position(x_inner, 0, 1), 0));
  EXPECT_TRUE(pml->is_position_at_boundary(Position(x_outer, 0, 1), 1));
  EXPECT_FALSE(pml->is_position_at_boundary(Position(x_inner + 0.1, 0, 1), 0));
  EXPECT_TRUE(pml->is_position_at_boundary(Position(x_inner + 0.1, -options.size_y / 2, 1), 2));
  EXPECT_TRUE(pml->is_position_at_boundary(Position(x_inner + 0.1, options.size_y / 2, 1), 3));
  EXPECT_TRUE(pml->is_position_at_boundary(Position(x_inner + 0.1, 0, 0), 4));
  EXPECT_TRUE(pml->is_position_at_boundary(Position(x_inner + 0.1, 0, options.size_z), 5));
  EXPECT_FALSE(pml->is_position_at_boundary(Position(x_inner + 0.1, 0, 1), 5));
}

// The layers in x are never extended, the layers in y are extended towards x, the layers in z in all directions.
TEST(PMLSurfaceTests, OnlyLayersOfHigherDirectionsCoverEdges) {
  const SetupOptions options = pml_on_all_sides();
  ProblemEnvironment environment(options);
  for (BoundaryId b_id = 0; b_id < 6; b_id++) {
    PMLSurface * pml = environment.surface<PMLSurface>(b_id);
    for (BoundaryId other = 0; other < 6; other++) {
      if (other / 2 == b_id / 2) continue;
      EXPECT_EQ(pml->extend_mesh_in_direction(other), other / 2 < b_id / 2) << "surface " << b_id << " towards " << other;
    }
  }
}

TEST(PMLSurfaceTests, OutputIsWrittenToOutputDirectory) {
  const SetupOptions options = pml_on_all_sides();
  ProblemEnvironment environment(options);
  PMLSurface * pml = environment.surface<PMLSurface>(3);
  dealii::Vector<ComplexNumber> values(pml->dof_counter);
  const std::string filename = pml->output_results(values, "pml_test");
  EXPECT_EQ(filename.rfind(environment.output_directory, 0), 0u);
  EXPECT_GT(std::filesystem::file_size(filename), 0u);
}
