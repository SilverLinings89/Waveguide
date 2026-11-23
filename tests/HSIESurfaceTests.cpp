// Tests for the Hardy space infinite elements on a surface: the number and the kinds of unknowns, their association
// with the unknowns of the inner domain and of neighboring surfaces, and the terms of the bilinear form.
#include <gtest/gtest.h>
#include <cmath>
#include <map>
#include <set>
#include "../Code/BoundaryCondition/HSIEPolynomial.h"
#include "../Code/BoundaryCondition/HSIESurface.h"
#include "./HelpersC.h"

namespace {

  using helpers_c::ProblemEnvironment;
  using helpers_c::SetupOptions;

  SetupOptions infinite_elements_on_all_sides(unsigned int fe_order, unsigned int hsie_degree) {
    SetupOptions options;
    options.cells_x = 2;
    options.cells_y = 3;
    options.cells_z = 4;
    options.fe_order = fe_order;
    options.hsie_degree = hsie_degree;
    options.boundary_method = BoundaryConditionType::HSIE;
    options.coupling = SignalCouplingMethod::Tapering;
    return options;
  }

  // The unknowns of an infinite element over a surface mesh, for the Nedelec element of order k (k + 1 unknowns per
  // edge, 2 k (k + 1) per cell), the nodal element of degree k + 1 (1 per vertex, k per edge, k^2 per cell) and the
  // Hardy space degree d: functions with the Hardy space index -1 carry the boundary value, the indices 0 to d
  // belong to the infinite direction. Tangential functions (Nedelec) of the infinite direction have no index -1.
  struct ExpectedDofs {
    unsigned int k;
    unsigned int d;
    unsigned int ray() const { return d + 2; }
    unsigned int edge() const { return k + 1; }
    unsigned int infinite_face_a() const { return (k + 1) * (d + 1); }
    unsigned int infinite_face_b() const { return k * (d + 2); }
    unsigned int surface() const { return 2 * k * (k + 1); }
    unsigned int segment_a() const { return 2 * k * (k + 1) * (d + 1); }
    unsigned int segment_b() const { return k * k * (d + 2); }
    unsigned int per_vertex() const { return ray(); }
    unsigned int per_edge() const { return edge() + infinite_face_a() + infinite_face_b(); }
    unsigned int per_cell() const { return surface() + segment_a() + segment_b(); }
  };

  // Cells of the surface mesh in its two directions.
  std::pair<unsigned int, unsigned int> cells_of_surface(const SetupOptions & options, BoundaryId b_id) {
    const unsigned int cells[3] = {options.cells_x, options.cells_y, options.cells_z};
    return {cells[(b_id / 2 + 1) % 3], cells[(b_id / 2 + 2) % 3]};
  }

  std::map<DofType, unsigned int> count_types(const DofDataVector & dofs) {
    std::map<DofType, unsigned int> ret;
    for (const DofData & dof : dofs) {
      ret[dof.type]++;
    }
    return ret;
  }

}

TEST(HSIESurfaceTests, DofsPerVertexEdgeAndCellFollowElementDefinition) {
  for (unsigned int fe_order = 0; fe_order < 3; fe_order++) {
    for (unsigned int degree : {0u, 3u, 6u}) {
      SetupOptions options = infinite_elements_on_all_sides(fe_order, degree);
      options.cells_x = options.cells_y = options.cells_z = 1;
      ProblemEnvironment environment(options);
      HSIESurface * surface = environment.surface<HSIESurface>(5);
      ASSERT_NE(surface, nullptr);
      const ExpectedDofs expected = {fe_order, degree};
      EXPECT_EQ(surface->compute_dofs_per_vertex(), expected.per_vertex()) << "order " << fe_order << " degree " << degree;
      EXPECT_EQ(surface->compute_dofs_per_edge(false), expected.per_edge()) << "order " << fe_order << " degree " << degree;
      EXPECT_EQ(surface->compute_dofs_per_edge(true), expected.per_edge() - expected.edge());
      EXPECT_EQ(surface->compute_dofs_per_face(false), expected.per_cell()) << "order " << fe_order << " degree " << degree;
      EXPECT_EQ(surface->compute_dofs_per_face(true), expected.per_cell() - expected.surface());
    }
  }
}

TEST(HSIESurfaceTests, NumberAndKindsOfUnknownsOnEverySurface) {
  for (unsigned int fe_order = 0; fe_order < 3; fe_order++) {
    const unsigned int degree = 3;
    const SetupOptions options = infinite_elements_on_all_sides(fe_order, degree);
    ProblemEnvironment environment(options);
    const ExpectedDofs expected = {fe_order, degree};
    for (BoundaryId b_id = 0; b_id < 6; b_id++) {
      HSIESurface * surface = environment.surface<HSIESurface>(b_id);
      ASSERT_NE(surface, nullptr);
      const std::pair<unsigned int, unsigned int> cells = cells_of_surface(options, b_id);
      const unsigned int n_vertices = (cells.first + 1) * (cells.second + 1);
      const unsigned int n_edges = cells.first * (cells.second + 1) + cells.second * (cells.first + 1);
      const unsigned int n_cells = cells.first * cells.second;
      EXPECT_EQ(surface->vertex_dof_data.size(), n_vertices * expected.per_vertex()) << "order " << fe_order << " surface " << b_id;
      EXPECT_EQ(surface->edge_dof_data.size(), n_edges * expected.per_edge()) << "order " << fe_order << " surface " << b_id;
      EXPECT_EQ(surface->face_dof_data.size(), n_cells * expected.per_cell()) << "order " << fe_order << " surface " << b_id;
      const unsigned int n_dofs = n_vertices * expected.per_vertex() + n_edges * expected.per_edge() + n_cells * expected.per_cell();
      EXPECT_EQ(surface->dof_counter, n_dofs);
      EXPECT_EQ(surface->compute_n_locally_active_dofs(), n_dofs);
      // Only the unknowns shared with the inner domain belong to another domain.
      const unsigned int n_shared_with_inner_domain = n_edges * expected.edge() + n_cells * expected.surface();
      EXPECT_EQ(n_shared_with_inner_domain, helpers_c::n_nedelec_dofs_on_face(cells.first, cells.second, fe_order));
      EXPECT_EQ(surface->compute_n_locally_owned_dofs(), n_dofs - n_shared_with_inner_domain);

      std::map<DofType, unsigned int> vertex_types = count_types(surface->vertex_dof_data);
      std::map<DofType, unsigned int> edge_types = count_types(surface->edge_dof_data);
      std::map<DofType, unsigned int> cell_types = count_types(surface->face_dof_data);
      EXPECT_EQ(vertex_types[DofType::RAY], n_vertices * expected.ray());
      EXPECT_EQ(edge_types[DofType::EDGE], n_edges * expected.edge());
      EXPECT_EQ(edge_types[DofType::IFFa], n_edges * expected.infinite_face_a());
      EXPECT_EQ(edge_types[DofType::IFFb], n_edges * expected.infinite_face_b());
      EXPECT_EQ(cell_types[DofType::SURFACE], n_cells * expected.surface());
      EXPECT_EQ(cell_types[DofType::SEGMENTa], n_cells * expected.segment_a());
      EXPECT_EQ(cell_types[DofType::SEGMENTb], n_cells * expected.segment_b());

      // Every unknown has its own index, a Hardy space index in the admissible range and the right kind of surface
      // function.
      std::set<unsigned int> indices;
      for (const DofDataVector * dofs : {&surface->vertex_dof_data, &surface->edge_dof_data, &surface->face_dof_data}) {
        for (const DofData & dof : *dofs) {
          indices.insert(dof.global_index);
          ASSERT_TRUE(dof.got_base_dof_index);
          const bool points_in_infinite_direction = (dof.type == DofType::RAY || dof.type == DofType::IFFb || dof.type == DofType::SEGMENTb);
          ASSERT_EQ(dof.nodal_basis, points_in_infinite_direction);
          if (dof.type == DofType::EDGE || dof.type == DofType::SURFACE) {
            ASSERT_EQ(dof.hsie_order, -1);
          } else {
            ASSERT_GE(dof.hsie_order, points_in_infinite_direction ? -1 : 0);
            ASSERT_LE(dof.hsie_order, static_cast<int>(degree));
          }
        }
      }
      EXPECT_EQ(indices.size(), n_dofs);
      EXPECT_EQ(*indices.rbegin(), n_dofs - 1);
    }
  }
}

TEST(HSIESurfaceTests, EveryCellSeesTheUnknownsOfItsVerticesEdgesAndInterior) {
  for (unsigned int fe_order = 0; fe_order < 2; fe_order++) {
    const unsigned int degree = 2;
    ProblemEnvironment environment(infinite_elements_on_all_sides(fe_order, degree));
    const ExpectedDofs expected = {fe_order, degree};
    for (BoundaryId b_id = 0; b_id < 6; b_id++) {
      HSIESurface * surface = environment.surface<HSIESurface>(b_id);
      ASSERT_NE(surface, nullptr);
      EXPECT_TRUE(surface->check_number_of_dofs_for_cell_integrity()) << "surface " << b_id;
      EXPECT_TRUE(surface->check_dof_assignment_integrity()) << "surface " << b_id;
      // The function only uses the cell, its edges and its vertices, so any dof handler of the surface mesh will do.
      dealii::FE_Q<2> fe(1);
      DofHandler2D dof_handler(Geometry.surface_meshes[b_id]);
      dof_handler.distribute_dofs(fe);
      const DofDataVector cell_dofs = surface->get_dof_data_for_cell(dof_handler.begin_active(), dof_handler.begin_active());
      EXPECT_EQ(cell_dofs.size(), 4 * expected.per_vertex() + 4 * expected.per_edge() + expected.per_cell());
    }
    EXPECT_TRUE(helpers_c::dof_bookkeeping_complaints(environment.output()).empty());
  }
}

// The unknowns of the surface functions are the unknowns of the inner domain on that surface.
TEST(HSIESurfaceTests, SurfaceDofAssociationMatchesInnerDomain) {
  for (unsigned int fe_order = 0; fe_order < 3; fe_order++) {
    const SetupOptions options = infinite_elements_on_all_sides(fe_order, 2);
    ProblemEnvironment environment(options);
    InnerDomain & inner_domain = environment.inner_domain();
    for (BoundaryId b_id = 0; b_id < 6; b_id++) {
      HSIESurface * surface = environment.surface<HSIESurface>(b_id);
      ASSERT_NE(surface, nullptr);
      const std::vector<InterfaceDofData> of_surface = surface->get_dof_association();
      const std::vector<InterfaceDofData> of_inner_domain = inner_domain.get_surface_dof_vector_for_boundary_id(b_id);
      const std::pair<unsigned int, unsigned int> cells = cells_of_surface(options, b_id);
      ASSERT_EQ(of_surface.size(), helpers_c::n_nedelec_dofs_on_face(cells.first, cells.second, fe_order)) << "order " << fe_order << " surface " << b_id;
      ASSERT_EQ(of_inner_domain.size(), of_surface.size());
      std::set<unsigned int> indices_of_surface_functions;
      for (const DofData & dof : surface->edge_dof_data) {
        if (dof.type == DofType::EDGE) indices_of_surface_functions.insert(dof.global_index);
      }
      for (const DofData & dof : surface->face_dof_data) {
        if (dof.type == DofType::SURFACE) indices_of_surface_functions.insert(dof.global_index);
      }
      for (unsigned int i = 0; i < of_surface.size(); i++) {
        ASSERT_LT(of_surface[i].base_point.distance(of_inner_domain[i].base_point), 1e-10) << "order " << fe_order << " surface " << b_id << " entry " << i;
        ASSERT_NEAR(of_surface[i].base_point[b_id / 2], Geometry.surface_extremal_coordinate[b_id], 1e-12);
        ASSERT_EQ(indices_of_surface_functions.count(of_surface[i].index), 1u);
        ASSERT_FALSE(surface->is_dof_owned[of_surface[i].index]);
      }
      EXPECT_EQ(helpers_c::to_level_numbering(*surface, of_surface), helpers_c::to_level_numbering(inner_domain, of_inner_domain));
      // The opposing boundary id denotes the same list.
      EXPECT_EQ(surface->get_dof_association_by_boundary_id(b_id ^ 1u).size(), of_surface.size());
    }
  }
}

// Two surfaces that meet in an edge of the domain identify the unknowns of the rays and of the infinite faces over
// that edge (see HSIESurface::finish_dof_index_initialization()). Both have to list them in the same order.
TEST(HSIESurfaceTests, NeighboringSurfacesListTheSameUnknownsOverCommonEdge) {
  for (unsigned int fe_order = 0; fe_order < 2; fe_order++) {
    const unsigned int degree = 2;
    const SetupOptions options = infinite_elements_on_all_sides(fe_order, degree);
    ProblemEnvironment environment(options);
    const ExpectedDofs expected = {fe_order, degree};
    const unsigned int cells[3] = {options.cells_x, options.cells_y, options.cells_z};
    for (BoundaryId a = 0; a < 6; a++) {
      for (BoundaryId b = a + 1; b < 6; b++) {
        if (are_opposing_sites(a, b)) continue;
        HSIESurface * first = environment.surface<HSIESurface>(a);
        HSIESurface * second = environment.surface<HSIESurface>(b);
        const unsigned int cells_along_edge = cells[3 - a / 2 - b / 2];
        EXPECT_EQ(first->get_vertices_for_boundary_id(b).size(), cells_along_edge + 1);
        EXPECT_EQ(first->get_lines_for_boundary_id(b).size(), cells_along_edge);
        EXPECT_EQ(second->get_vertices_for_boundary_id(a).size(), cells_along_edge + 1);
        EXPECT_EQ(second->get_lines_for_boundary_id(a).size(), cells_along_edge);
        const std::vector<InterfaceDofData> of_first = first->get_dof_association_by_boundary_id(b);
        const std::vector<InterfaceDofData> of_second = second->get_dof_association_by_boundary_id(a);
        const unsigned int n_expected = (cells_along_edge + 1) * expected.per_vertex() + cells_along_edge * expected.per_edge();
        ASSERT_EQ(of_first.size(), n_expected) << "order " << fe_order << " surfaces " << a << " and " << b;
        ASSERT_EQ(of_second.size(), n_expected) << "order " << fe_order << " surfaces " << a << " and " << b;
        for (unsigned int i = 0; i < n_expected; i++) {
          ASSERT_LT(of_first[i].base_point.distance(of_second[i].base_point), 1e-10) << "surfaces " << a << " and " << b << " entry " << i;
          ASSERT_NEAR(of_first[i].base_point[a / 2], Geometry.surface_extremal_coordinate[a], 1e-12);
          ASSERT_NEAR(of_first[i].base_point[b / 2], Geometry.surface_extremal_coordinate[b], 1e-12);
        }
        EXPECT_EQ(first->get_global_dof_indices_by_boundary_id(b), second->get_global_dof_indices_by_boundary_id(a)) << "surfaces " << a << " and " << b;
      }
    }
  }
}

// The surface mesh is a flat copy of a face of the inner domain. Its boundary ids name the neighboring surfaces, and
// undo_transform() maps its points back onto the face.
TEST(HSIESurfaceTests, SurfaceMeshIsMappedOntoFaceOfInnerDomain) {
  const SetupOptions options = infinite_elements_on_all_sides(0, 2);
  ProblemEnvironment environment(options);
  const double lower[3] = {-options.size_x / 2, -options.size_y / 2, 0};
  const double upper[3] = {options.size_x / 2, options.size_y / 2, options.size_z};
  for (BoundaryId b_id = 0; b_id < 6; b_id++) {
    HSIESurface * surface = environment.surface<HSIESurface>(b_id);
    ASSERT_NE(surface, nullptr);
    const std::pair<unsigned int, unsigned int> cells = cells_of_surface(options, b_id);
    EXPECT_EQ(Geometry.surface_meshes[b_id].n_active_cells(), cells.first * cells.second);
    std::vector<BoundaryId> boundary_ids = surface->get_boundary_ids();
    std::sort(boundary_ids.begin(), boundary_ids.end());
    EXPECT_EQ(boundary_ids, surface->adjacent_boundaries) << "surface " << b_id;
    for (const auto & cell : Geometry.surface_meshes[b_id].active_cell_iterators()) {
      const Position center = surface->undo_transform(cell->center());
      ASSERT_NEAR(center[b_id / 2], Geometry.surface_extremal_coordinate[b_id], 1e-12);
      for (unsigned int i = 0; i < 3; i++) {
        ASSERT_GT(center[i], lower[i] - 1e-12);
        ASSERT_LT(center[i], upper[i] + 1e-12);
      }
      for (unsigned int face = 0; face < 4; face++) {
        if (!cell->face(face)->at_boundary()) continue;
        // An edge with the boundary id s lies on the surface s of the inner domain.
        const BoundaryId other = cell->face(face)->boundary_id();
        const Position on_edge = surface->undo_transform(cell->face(face)->center());
        ASSERT_NEAR(on_edge[other / 2], Geometry.surface_extremal_coordinate[other], 1e-12) << "surface " << b_id << " towards " << other;
        ASSERT_TRUE(surface->is_point_at_boundary(cell->face(face)->center(), other));
      }
      ASSERT_FALSE(surface->is_point_at_boundary(cell->center(), surface->adjacent_boundaries[0]));
    }
  }
}

// int_0^infinity u v dxi = -2 i kappa_0 sum_k a_k b_k for the coefficients a and b of the Hardy space representations;
// evaluate_a() sums this over the components with the weights G_ij.
TEST(HSIESurfaceTests, BilinearFormOfHardySpaceRepresentations) {
  SetupOptions options = infinite_elements_on_all_sides(0, 2);
  options.kappa_0 = ComplexNumber(4.0, 0.0);
  ProblemEnvironment environment(options);
  HSIESurface * surface = environment.surface<HSIESurface>(5);
  ASSERT_NE(surface, nullptr);
  const ComplexNumber k0 = options.kappa_0;
  const HSIEPolynomial zero({ComplexNumber(0, 0)}, k0);
  std::vector<HSIEPolynomial> u = {HSIEPolynomial({ComplexNumber(1, 0), ComplexNumber(2, 0)}, k0), zero, HSIEPolynomial({ComplexNumber(0, 1)}, k0)};
  std::vector<HSIEPolynomial> v = {HSIEPolynomial({ComplexNumber(3, 0), ComplexNumber(4, 0), ComplexNumber(5, 0)}, k0), HSIEPolynomial({ComplexNumber(7, 0)}, k0), HSIEPolynomial({ComplexNumber(2, 0), ComplexNumber(9, 0)}, k0)};
  dealii::Tensor<2, 3, double> identity;
  for (unsigned int i = 0; i < 3; i++) identity[i][i] = 1.0;
  // (1 * 3 + 2 * 4) + 0 + (i * 2) = 11 + 2 i
  const ComplexNumber expected = ComplexNumber(0, -2) * k0 * ComplexNumber(11, 2);
  EXPECT_LT(std::abs(surface->evaluate_a(u, v, identity) - expected), 1e-12);
  EXPECT_LT(std::abs(surface->evaluate_a(v, u, identity) - expected), 1e-12);
  // An off-diagonal weight couples different components: G_01 u_0 v_1 = 0.5 * (1 * 7).
  dealii::Tensor<2, 3, double> weights = identity;
  weights[0][1] = 0.5;
  EXPECT_LT(std::abs(surface->evaluate_a(u, v, weights) - (expected + ComplexNumber(0, -2) * k0 * 3.5)), 1e-12);
}

// Functions with tangential surface part have no component in the infinite direction, functions with nodal surface
// part only have that component. The infinite direction is the normal of the surface.
TEST(HSIESurfaceTests, TermsOfShapeFunctionsPointInTheRightDirections) {
  ProblemEnvironment environment(infinite_elements_on_all_sides(0, 2));
  auto is_zero = [](HSIEPolynomial & polynomial) {
    double sum = 0;
    for (ComplexNumber coefficient : polynomial.a) sum += std::abs(coefficient);
    return sum == 0.0;
  };
  for (BoundaryId b_id = 0; b_id < 6; b_id++) {
    HSIESurface * surface = environment.surface<HSIESurface>(b_id);
    ASSERT_NE(surface, nullptr);
    const unsigned int normal = b_id / 2;
    std::vector<HSIEPolynomial> tangential = surface->build_non_curl_term_nedelec(1, 0.3, -0.7);
    std::vector<HSIEPolynomial> orthogonal = surface->build_non_curl_term_q(1, 0.3);
    ASSERT_EQ(tangential.size(), 3u);
    ASSERT_EQ(orthogonal.size(), 3u);
    for (unsigned int i = 0; i < 3; i++) {
      EXPECT_EQ(is_zero(tangential[i]), i == normal) << "surface " << b_id << " component " << i;
      EXPECT_EQ(is_zero(orthogonal[i]), i != normal) << "surface " << b_id << " component " << i;
    }
    // The curl of a function in the infinite direction, grad(f) x n, is tangential.
    dealii::Tensor<1, 2> gradient;
    gradient[0] = 0.4;
    gradient[1] = -0.9;
    std::vector<HSIEPolynomial> curl_of_orthogonal = surface->build_curl_term_q(1, gradient);
    for (unsigned int i = 0; i < 3; i++) {
      EXPECT_EQ(is_zero(curl_of_orthogonal[i]), i == normal) << "surface " << b_id << " component " << i;
    }
  }
}
