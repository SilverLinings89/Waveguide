#include "../Code/HSIEPreconditioner/LaguerreFunction.h"
#include "../third_party/googletest/googletest/include/gtest/gtest.h"
#include "../Code/HSIEPreconditioner/HSIESurface.h"
#include "../Code/HSIEPreconditioner/HSIEPolynomial.h"
#include "../Code/HSIEPreconditioner/HSIESurface.cpp"
#include "../Code/Core/Types.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/tensor.h>

Parameters GlobalParams;
GeometryManager Geometry;
MPICommunicator GlobalMPI;
ModeManager GlobalModeManager;
std::string solutionpath = "";
std::ofstream log_stream;
std::string constraints_filename = "constraints.log";
std::string assemble_filename = "assemble.log";
std::string precondition_filename = "precondition.log";
std::string solver_filename = "solver.log";
std::string total_filename = "total.log";
int StepsR = 10;
int StepsPhi = 10;
int alert_counter = 0;
std::string input_file_name;
SpaceTransformation *the_st;

static HSIEPolynomial random_poly(unsigned int Order, std::complex<double> k0) {
  std::vector<std::complex<double>> a;
  for (unsigned int i = 0; i < Order; i++) {
    a.emplace_back(rand() % 10, rand() % 10);
  }
  return HSIEPolynomial(a, k0);
}

static unsigned int dofs_per_edge(unsigned int hsie_Order,
    unsigned int inner_order) {
  switch (hsie_Order) {
  case 5:
    if (inner_order == 0)
      return 7;
    if (inner_order == 1)
      return 21;
    if (inner_order == 2)
      return 35;
    break;
  case 10:
    if (inner_order == 0)
      return 12;
    if (inner_order == 1)
      return 36;
    if (inner_order == 2)
      return 60;
    break;
  default:
    return 0;
  }
  return 0;
}

static unsigned int dofs_per_vertex(unsigned int hsie_Order) {
  return hsie_Order + 2;
}

static unsigned int dofs_per_face(unsigned int hsie_Order,
    unsigned int inner_order) {
  if (inner_order == 0) {
    return 0;
  }
  if (inner_order == 1) {
    return (hsie_Order + 2) * 3 * 4;
  }
  if (inner_order == 2) {
    return (hsie_Order + 2) * 3 * 12;
  }
  return 0;
}

class TestData {
public:
  unsigned int Inner_Element_Order;
  unsigned int Cells_Per_Direction;

  TestData(unsigned int in_inner_element, unsigned int in_cells) {
    Inner_Element_Order = in_inner_element;
    Cells_Per_Direction = in_cells;
  }

  TestData(const TestData &in_td) {
    Inner_Element_Order = in_td.Inner_Element_Order;
    Cells_Per_Direction = in_td.Cells_Per_Direction;
  }
};

class TestOrderFixture: public ::testing::TestWithParam<
    std::tuple<unsigned int, unsigned int>> {
public:
  unsigned int Cells_Per_Direction;
  unsigned int InnerOrder;
  dealii::Triangulation<3> tria;
  dealii::Triangulation<2, 3> temp_triangulation;
  dealii::Triangulation<2> surf_tria;
  std::complex<double> k0;
protected:
  void SetUp() override {
    std::tuple<unsigned int, unsigned int> Params = GetParam();
    Cells_Per_Direction = std::get<1>(Params);
    InnerOrder = std::get<0>(Params);
    k0 = { 0.0, -1.0 };
    std::vector<unsigned int> repetitions;
    repetitions.push_back(Cells_Per_Direction);
    repetitions.push_back(Cells_Per_Direction);
    repetitions.push_back(Cells_Per_Direction);
    Position left(-1, -1, -1);
    Position right(1, 1, 1);
    dealii::GridGenerator::subdivided_hyper_rectangle(tria, repetitions, left,
        right, true);
    std::set<unsigned int> b_ids;
    b_ids.insert(4);
    dealii::GridTools::transform(Transform_4_to_5, tria);
    dealii::GridGenerator::extract_boundary_mesh(tria, temp_triangulation,
        b_ids);
    dealii::GridGenerator::flatten_triangulation(temp_triangulation, surf_tria);
  }
};

class FullCubeFixture: public ::testing::Test {
public:
  unsigned int InnerOrder;
  dealii::Triangulation<3> full_tria;
  dealii::Triangulation<2, 3> temp_triangulation;
  dealii::Triangulation<2> surf_tria;
  std::complex<double> k0;
  std::array<std::shared_ptr<HSIESurface>,6> surfaces;
protected:
  void SetUp() override {
    InnerOrder = 5;
    k0 = { 0.0, -1.0 };
    std::vector<unsigned int> repetitions;
    repetitions.push_back(3);
    repetitions.push_back(4);
    repetitions.push_back(5);
    Position left(-1, -1, -1);
    Position right(1, 1, 1);
    dealii::GridGenerator::subdivided_hyper_rectangle(full_tria, repetitions, left, right, true);

    for(unsigned int side = 0; side < 6; side++) {
      dealii::Triangulation<2, 3> temp_triangulation;
      const unsigned int component = side / 2;
      double additional_coorindate = 0;
      bool found = false;
      for (auto it : full_tria.active_cell_iterators()) {
        if (it->at_boundary(side)) {
          for (auto i = 0; i < 6 && !found; i++) {
            if (it->face(i)->boundary_id() == side) {
              found = true;
              additional_coorindate = it->face(i)->center()[component];
            }
          }
        }
        if (found) {
          break;
        }
      }
      dealii::Triangulation<2> surf_tria;
      Mesh tria;
      tria.copy_triangulation(full_tria);
      std::set<unsigned int> b_ids;
      b_ids.insert(side);
      switch (side) {
        case 0:
          dealii::GridTools::transform(Transform_0_to_5, tria);
          break;
        case 1:
          dealii::GridTools::transform(Transform_1_to_5, tria);
          break;
        case 2:
          dealii::GridTools::transform(Transform_2_to_5, tria);
          break;
        case 3:
          dealii::GridTools::transform(Transform_3_to_5, tria);
          break;
        case 4:
          dealii::GridTools::transform(Transform_4_to_5, tria);
          break;
        default:
          break;
      }
      dealii::GridGenerator::extract_boundary_mesh(tria, temp_triangulation, b_ids);
      dealii::GridGenerator::flatten_triangulation(temp_triangulation, surf_tria);
      surfaces[side] = std::shared_ptr<HSIESurface>(new HSIESurface(GlobalParams.HSIE_polynomial_degree, std::ref(surf_tria), side,
              GlobalParams.Nedelec_element_order, GlobalParams.kappa_0, additional_coorindate));
      surfaces[side]->initialize();
    }
  }
};

class TestDirectionFixture: public ::testing::TestWithParam<
    std::tuple<unsigned int, unsigned int, unsigned int>> {
public:
  unsigned int Cells_Per_Direction;
  unsigned int InnerOrder;
  unsigned int boundary_id;
  dealii::Triangulation<3> tria;
  dealii::Triangulation<2, 3> temp_triangulation;
  dealii::Triangulation<2> surf_tria;
  std::complex<double> k0;
  std::set<unsigned int> b_ids;
protected:
  void SetUp() override {
    std::tuple<unsigned int, unsigned int, unsigned int> Params = GetParam();
    Cells_Per_Direction = std::get<1>(Params);
    InnerOrder = std::get<0>(Params);
    boundary_id = std::get<2>(Params);
    k0 = { 0.0, -1.0 };
    std::vector<unsigned int> repetitions;
    repetitions.push_back(Cells_Per_Direction);
    repetitions.push_back(Cells_Per_Direction);
    repetitions.push_back(Cells_Per_Direction);
    Position left(-1, -1, -1);
    Position right(1, 1, 1);
    dealii::GridGenerator::subdivided_hyper_rectangle(tria, repetitions, left,
        right, true);
    const unsigned int dest_cells = Cells_Per_Direction * Cells_Per_Direction
        * Cells_Per_Direction;
    ASSERT_EQ(tria.n_active_cells(), dest_cells);
    b_ids.insert(boundary_id);
    switch (boundary_id) {
      case 0:
        dealii::GridTools::transform(Transform_0_to_5, tria);
        break;
      case 1:
        dealii::GridTools::transform(Transform_1_to_5, tria);
        break;
      case 2:
        dealii::GridTools::transform(Transform_2_to_5, tria);
        break;
      case 3:
        dealii::GridTools::transform(Transform_3_to_5, tria);
        break;
      case 4:
        dealii::GridTools::transform(Transform_4_to_5, tria);
        break;
      }
  }
};

TEST_P(TestDirectionFixture, TestCellRequirements) {
  dealii::GridGenerator::extract_boundary_mesh(tria, temp_triangulation, b_ids);
  const unsigned int dest_surf_cells = Cells_Per_Direction
      * Cells_Per_Direction;
  ASSERT_EQ(temp_triangulation.n_active_cells(), dest_surf_cells);
  dealii::GridGenerator::flatten_triangulation(temp_triangulation, surf_tria);
  ASSERT_EQ(surf_tria.n_active_cells(), dest_surf_cells);
}

TEST_P(TestDirectionFixture, DofNumberingTest1) {
  dealii::GridGenerator::extract_boundary_mesh(tria, temp_triangulation, b_ids);
  dealii::GridGenerator::flatten_triangulation(temp_triangulation, surf_tria);
  const unsigned int component = 0;
  auto temp_it = temp_triangulation.begin();
  double additional_coorindate = temp_it->center()[component];
  HSIESurface surf(5, surf_tria, boundary_id, InnerOrder, k0,
      additional_coorindate);
  surf.initialize();
  std::vector<types::boundary_id> boundary_ids_of_flattened_mesh =
      surf.get_boundary_ids();
  ASSERT_EQ(boundary_ids_of_flattened_mesh.size(), 4);
  auto it = std::find(boundary_ids_of_flattened_mesh.begin(),
      boundary_ids_of_flattened_mesh.end(), boundary_id);
  ASSERT_EQ(boundary_ids_of_flattened_mesh.end(), it);
}

TEST_F(FullCubeFixture, ExtremeCoordinates) {
  for(unsigned int i = 0; i < 6; i++) {
    surfaces[i]->compute_extreme_vertex_coordinates();
    for(unsigned int coord = 0; coord < 6; coord++) {
      ASSERT_TRUE(surfaces[i]->boundary_vertex_coordinates[coord] == -1 || surfaces[i]->boundary_vertex_coordinates[coord] == 1);
    }
  }
}

TEST_F(FullCubeFixture, BoundaryVertexTest1) {
  for(unsigned int i = 0; i < 6; i++) {
    for(unsigned int j = 0; j < 6; j++) {
      if(!are_opposing_sites(i,j)) {
        ASSERT_EQ(surfaces[i]->get_vertices_for_boundary_id(j).size(), surfaces[j]->get_vertices_for_boundary_id(i).size());
      }
    }  
  }
}

TEST_F(FullCubeFixture, BoundaryVertexTest2) {
  for(unsigned int i = 0; i < 6; i++) {
    for(unsigned int j = 0; j < 6; j++) {
      if(! are_opposing_sites(i,j)) {
        std::vector<unsigned int> first_indices = surfaces[i]->get_vertices_for_boundary_id(j);
        std::vector<Position> positions_first = surfaces[i]->vertex_positions_for_ids(first_indices);
        std::vector<unsigned int> second_indices = surfaces[j]->get_vertices_for_boundary_id(i);
        std::vector<Position> positions_second = surfaces[j]->vertex_positions_for_ids(second_indices);
        ASSERT_TRUE(positions_second.size() > 0);
        ASSERT_TRUE(positions_second.size() == positions_first.size());
      }
    }  
  }
}

TEST_F(FullCubeFixture, BoundaryCouplingTest1) {
  for(unsigned int i = 0; i < 6; i++) {
    for(unsigned int j = 0; j < 6; j++) {
      if(! are_opposing_sites(i,j)) {
        std::vector<DofIndexAndOrientationAndPosition> first_dofs = surfaces[i]->get_dof_association_by_boundary_id(j);
        std::vector<DofIndexAndOrientationAndPosition> second_dofs = surfaces[j]->get_dof_association_by_boundary_id(i);
        ASSERT_TRUE(first_dofs.size() ==  second_dofs.size());
        ASSERT_TRUE(first_dofs.size() > 0 );
        for(unsigned int dof = 0; dof < first_dofs.size(); dof++) {
          ASSERT_EQ(first_dofs[dof].position, second_dofs[dof].position);
        }
      }
    }  
  }
}

TEST_F(FullCubeFixture, BoundaryCouplingTest2) {
  for(unsigned int i = 0; i < 6; i++) {
    for(unsigned int j = 0; j < 6; j++) {
      if(! are_opposing_sites(i,j)) {
        std::vector<DofIndexAndOrientationAndPosition> first_dofs = surfaces[i]->get_dof_association_by_boundary_id(j);
        ASSERT_TRUE(first_dofs.size() > 0 );
        for(unsigned int dof = 0; dof < first_dofs.size(); dof++) {
          ASSERT_TRUE(first_dofs[dof].position[0] >= -1);
          ASSERT_TRUE(first_dofs[dof].position[0] <= 1);
          ASSERT_TRUE(first_dofs[dof].position[1] >= -1);
          ASSERT_TRUE(first_dofs[dof].position[1] <= 1);
          ASSERT_TRUE(first_dofs[dof].position[2] >= -1);
          ASSERT_TRUE(first_dofs[dof].position[2] <= 1);
        }
      }
    }  
  }
}

TEST_F(FullCubeFixture, BoundaryCouplingTest3) {
  for(unsigned int i = 0; i < 6; i++) {
    for(unsigned int j = 0; j < 6; j++) {
      if(! are_opposing_sites(i,j)) {
        std::vector<unsigned int> first_dofs = surfaces[i]->get_vertices_for_boundary_id(j);
        std::vector<Position> positions_first = surfaces[i]->vertex_positions_for_ids(first_dofs);
        ASSERT_TRUE(positions_first.size() > 0 );
        for(unsigned int dof = 0; dof < first_dofs.size(); dof++) {
          ASSERT_TRUE(positions_first[dof][0] >= -1);
          ASSERT_TRUE(positions_first[dof][0] <= 1);
          ASSERT_TRUE(positions_first[dof][1] >= -1);
          ASSERT_TRUE(positions_first[dof][1] <= 1);
          ASSERT_TRUE(positions_first[dof][2] >= -1);
          ASSERT_TRUE(positions_first[dof][2] <= 1);
        }
      }
    }  
  }
}

TEST_F(FullCubeFixture, BoundaryLineTest1) {
  for(unsigned int i = 0; i < 6; i++) {
    for(unsigned int j = 0; j < 6; j++) {
      if(! are_opposing_sites(i,j)) {
        ASSERT_EQ(surfaces[i]->get_lines_for_boundary_id(j).size(), surfaces[j]->get_lines_for_boundary_id(i).size());
      }
    }
  }
}

TEST_F(FullCubeFixture, OpposingSideTests) {
  for(unsigned int i = 0; i < 6; i++) {
    for(unsigned int j = 0; j < 6; j++) {
      if(are_opposing_sites(i,j)) {
        ASSERT_EQ(surfaces[i]->get_lines_for_boundary_id(j).size(), 0);
        ASSERT_EQ(surfaces[i]->get_vertices_for_boundary_id(j).size(), 0);
        ASSERT_EQ(surfaces[j]->get_lines_for_boundary_id(i).size(), 0);
        ASSERT_EQ(surfaces[j]->get_vertices_for_boundary_id(i).size(), 0);
        std::vector<DofIndexAndOrientationAndPosition> first_dofs = surfaces[i]->get_dof_association_by_boundary_id(j);
        std::vector<DofIndexAndOrientationAndPosition> second_dofs = surfaces[j]->get_dof_association_by_boundary_id(i);
        ASSERT_EQ(first_dofs.size(), 0);
        ASSERT_EQ(second_dofs.size(), 0);
      }
    }
  }
}

TEST_P(TestOrderFixture, AssemblationTestOrder5) {
  const unsigned int component = 0;
  auto temp_it = temp_triangulation.begin();
  double additional_coorindate = temp_it->center()[component];
  HSIESurface surf(5, surf_tria, 0, InnerOrder, k0, additional_coorindate);
  surf.initialize();
  ASSERT_EQ(surf.compute_dofs_per_vertex(), dofs_per_vertex(5));
  ASSERT_EQ(surf.compute_dofs_per_edge(false), dofs_per_edge(5, InnerOrder));
  ASSERT_EQ(surf.compute_dofs_per_face(false), dofs_per_face(5, InnerOrder));
  ASSERT_EQ( (Cells_Per_Direction + 1) * (Cells_Per_Direction + 1) * surf.compute_dofs_per_vertex(), surf.vertex_dof_data.size());
  ASSERT_EQ( 2 * Cells_Per_Direction * (Cells_Per_Direction + 1) * surf.compute_dofs_per_edge(false), surf.edge_dof_data.size());
  ASSERT_EQ( Cells_Per_Direction * Cells_Per_Direction * surf.compute_dofs_per_face(false), surf.face_dof_data.size());
  ASSERT_TRUE(surf.check_number_of_dofs_for_cell_integrity());
  ASSERT_TRUE(surf.check_dof_assignment_integrity());
}

TEST_P(TestOrderFixture, AssemblationTestOrder10) {
  const unsigned int component = 0;
  auto temp_it = temp_triangulation.begin();
  double additional_coorindate = temp_it->center()[component];
  HSIESurface surf(10, surf_tria, 0, InnerOrder, k0, additional_coorindate);
  surf.initialize();
  ASSERT_EQ(surf.compute_dofs_per_vertex(), dofs_per_vertex(10));
  ASSERT_EQ(surf.compute_dofs_per_edge(false), dofs_per_edge(10, InnerOrder));
  ASSERT_EQ(surf.compute_dofs_per_face(false), dofs_per_face(10, InnerOrder));
  ASSERT_EQ((Cells_Per_Direction + 1) * (Cells_Per_Direction + 1)
          * surf.compute_dofs_per_vertex(), surf.vertex_dof_data.size());
  ASSERT_EQ(2 * Cells_Per_Direction * (Cells_Per_Direction + 1)
          * surf.compute_dofs_per_edge(false), surf.edge_dof_data.size());
  ASSERT_EQ(Cells_Per_Direction * Cells_Per_Direction * surf.compute_dofs_per_face(false), surf.face_dof_data.size());
  ASSERT_TRUE(surf.check_number_of_dofs_for_cell_integrity());
  ASSERT_TRUE(surf.check_dof_assignment_integrity());
}

INSTANTIATE_TEST_SUITE_P(HSIESurfaceTests, TestOrderFixture, ::testing::Combine( ::testing::Values(0,1,2), ::testing::Values(5,9)));
// INSTANTIATE_TEST_SUITE_P(HSIESurfaceTests, TestOrderFixture, ::testing::Combine(::testing::Values(0), ::testing::Values(5, 9)));

INSTANTIATE_TEST_SUITE_P(MespPreparationTests, TestDirectionFixture,
    ::testing::Combine(::testing::Values(0), ::testing::Values(5, 9),
        ::testing::Values(0, 1, 2, 3, 4, 5)));

TEST(HSIEPolynomialTests, TestOperatorTplus) {
  std::vector<std::complex<double>> in_a;
  in_a.emplace_back(0.0, 0.0);
  in_a.emplace_back(0.0, 0.0);
  in_a.emplace_back(1.0, 0.0);
  in_a.emplace_back(0.0, 1.0);
  in_a.emplace_back(0.0, 0.0);
  std::complex<double> k0(0.0, 1.0);
  HSIEPolynomial poly(in_a, k0);
  poly.applyTplus(std::complex<double>(1, -1));
  ASSERT_EQ(poly.a[0], std::complex<double>(0.5, -0.5));
  ASSERT_EQ(poly.a[1], std::complex<double>(0, 0));
  ASSERT_EQ(poly.a[2], std::complex<double>(0.5, 0));
  ASSERT_EQ(poly.a[3], std::complex<double>(0.5, 0.5));
  ASSERT_EQ(poly.a[4], std::complex<double>(0, 0.5));
}

TEST(HSIEPolynomialTests, ProductOfDandIShouldBeIdentity) {
  std::vector<std::complex<double>> in_a;
  in_a.emplace_back(0.0, 0.0);
  in_a.emplace_back(0.0, 0.0);
  in_a.emplace_back(1.0, 0.0);
  in_a.emplace_back(0.0, 1.0);
  in_a.emplace_back(0.0, 0.0);
  std::complex<double> k0(0.0, 1.0);
  HSIEPolynomial poly(in_a, k0);
  poly.applyD();
  dealii::FullMatrix<std::complex<double>> product(HSIEPolynomial::D.size(0),
      HSIEPolynomial::D.size(1));
  HSIEPolynomial::D.mmult(product, HSIEPolynomial::I, false);
  ASSERT_NEAR(product.frobenius_norm(), std::sqrt(HSIEPolynomial::I.size(0)),
      0.0001);
}

TEST(HSIE_ORTHOGONALITY_TESTS, EvaluationOfA) {
  std::complex<double> k0(0, -1);
  std::vector<std::complex<double>> zeroes;
  for (unsigned k = 0; k < 10; k++) {
    zeroes.emplace_back(0, 0);
  }
  for (unsigned int i = 0; i < 10; i++) {
    for (unsigned int j = 0; j < 10; j++) {
      std::vector<HSIEPolynomial> u;
      u.emplace_back(i, k0);
      u.emplace_back(zeroes, k0);
      u.emplace_back(zeroes, k0);
      std::vector<HSIEPolynomial> v;
      v.emplace_back(j, k0);
      v.emplace_back(zeroes, k0);
      v.emplace_back(zeroes, k0);
      dealii::Tensor<2, 3, double> G;
      G[0][0] = 1;
      G[1][1] = 1;
      G[2][2] = 1;
      std::complex<double> res = HSIESurface::evaluate_a(u, v, G);
      if (j != i) {
        ASSERT_NEAR(0, res.real(), 0.001);
        ASSERT_NEAR(0, res.imag(), 0.001);
      } else {
        bool eitherOr = std::abs(res.real()) > 0.0001
            || std::abs(res.imag()) > 0.0001;
        ASSERT_TRUE(eitherOr);
      }
    }
  }
}

TEST(HSIE_ORTHOGONALITY_TESTS, RandomPolynomialProductTest) {
  std::complex<double> k0(0, -1);
  std::vector<std::complex<double>> zeroes;
  for (unsigned k = 0; k < 10; k++) {
    zeroes.emplace_back(0, 0);
  }

  std::vector<HSIEPolynomial> u;
  u.push_back(random_poly(10, k0));
  u.push_back(random_poly(10, k0));
  u.push_back(random_poly(10, k0));
  std::vector<HSIEPolynomial> v;
  v.push_back(random_poly(10, k0));
  v.push_back(random_poly(10, k0));
  v.push_back(random_poly(10, k0));
  dealii::Tensor<2, 3, double> G;
  G[0][0] = 1;
  G[1][1] = 1;
  G[2][2] = 1;
      std::complex<double> res = HSIESurface::evaluate_a(u, v, G);
  std::complex<double> expected_result(0, 0);
  for (unsigned j = 0; j < 3; j++) {
    for (unsigned int i = 0; i < 10; i++) {
      expected_result += u[j].a[i] * v[j].a[i];
    }
  }

  ASSERT_NEAR(expected_result.real(), res.real(),
      expected_result.real() / 100.0);
  ASSERT_NEAR(expected_result.imag(), res.imag(),
      expected_result.imag() / 100.0);
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_1) {
  ASSERT_EQ(LaguerreFunction::evaluate(0, 13, 2), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_2) {
  ASSERT_EQ(LaguerreFunction::evaluate(0, 1, 4), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_3) {
  for (int i = 0; i < 10; i++) {
    ASSERT_EQ(LaguerreFunction::evaluate(1, i, 2), -2 + i + 1);
  }
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_4) {
  for (int i = 0; i < 10; i++) {
    ASSERT_EQ(LaguerreFunction::evaluate(2, i, 2),
        2 - 2 * i - 4 + (i + 2) * (i + 1) / 2);
  }
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_5) {
  for (int i = 0; i < 10; i++) {
    for (unsigned int j = 0; j < 10; j++) {
      ASSERT_EQ(LaguerreFunction::evaluate(i, j, 0),
          LaguerreFunction::binomial_coefficient(i + j, i));
    }
  }
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_1) {
  ASSERT_EQ(LaguerreFunction::binomial_coefficient(5, 2), 10);
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_2) {
  ASSERT_EQ(LaguerreFunction::binomial_coefficient(0, 0), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_3) {
  ASSERT_EQ(LaguerreFunction::binomial_coefficient(1, 0), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_4) {
  ASSERT_EQ(LaguerreFunction::binomial_coefficient(1, 1), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_5) {
  ASSERT_EQ(LaguerreFunction::binomial_coefficient(15, 5), 3003);
}

TEST(LAGUERRE_FUNCTION_TESTS, FACTORIAL_1) {
  ASSERT_EQ(LaguerreFunction::factorial(1), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, FACTORIAL_2) {
  ASSERT_EQ(LaguerreFunction::factorial(2), 2);
}

TEST(LAGUERRE_FUNCTION_TESTS, FACTORIAL_3) {
  ASSERT_EQ(LaguerreFunction::factorial(4), 24);
}

TEST(LAGUERRE_FUNCTION_TESTS, FACTORIAL_5) {
  ASSERT_EQ(LaguerreFunction::factorial(5), 120);
}

TEST(STATIC_FUNCTION_TEST, COMPARISON_TESTS_1) {
  Position p1 = {0,0,1};
  Position p2 = {0,0,0};
  Position p3 = {0,1,1};
  Position p4 = {0,1,0};
  Position p5 = {1,0,1};
  Position p6 = {1,0,0};
  Position p7 = {1,1,1};
  Position p8 = {1,1,0};
  std::pair<DofNumber, Position> pair_1 = {1,p1};
  std::pair<DofNumber, Position> pair_2 = {2,p2};
  std::pair<DofNumber, Position> pair_3 = {3,p3};
  std::pair<DofNumber, Position> pair_4 = {4,p4};
  std::pair<DofNumber, Position> pair_5 = {5,p5};
  std::pair<DofNumber, Position> pair_6 = {6,p6};
  std::pair<DofNumber, Position> pair_7 = {7,p7};
  std::pair<DofNumber, Position> pair_8 = {8,p8};
  std::vector<std::pair<DofNumber, Position>> dofs;
  dofs.push_back(pair_1);
  dofs.push_back(pair_3);
  dofs.push_back(pair_5);
  dofs.push_back(pair_6);
  dofs.push_back(pair_7);
  dofs.push_back(pair_2);
  dofs.push_back(pair_4);
  dofs.push_back(pair_8);
  std::sort(dofs.begin(), dofs.end(), compareDofBaseData);
  std::string ord = "";
  for(unsigned int i = 0; i < 8; i++) ord += std::to_string(dofs[i].first);
  ASSERT_EQ(ord, "26481537");
}