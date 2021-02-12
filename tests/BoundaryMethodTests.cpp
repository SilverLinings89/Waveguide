#include "../Code/BoundaryCondition/LaguerreFunction.h"
#include "../third_party/googletest/googletest/include/gtest/gtest.h"
#include "../Code/BoundaryCondition/HSIESurface.h"
#include "../Code/BoundaryCondition/HSIEPolynomial.h"
#include "../Code/BoundaryCondition/BoundaryCondition.h"
#include "../Code/Core/Types.h"
#include "../Code/Helpers/staticfunctions.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/tensor.h>
#include <deal.II/grid/grid_tools.h>
#include "./Fixtures.h"
#include "./TestHelperFunctions.h"
#include "./TestData.h"

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
  std::vector<types::boundary_id> boundary_ids_of_flattened_mesh = surf.get_boundary_ids();
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

TEST_F(PMLCubeFixture, PMLCellCountTests) {
  const unsigned int default_layers = GlobalParams.PML_N_Layers - 1;
  ASSERT_EQ(surfaces[0]->triangulation.n_active_cells(), 4 * 5 * default_layers);
  ASSERT_EQ(surfaces[1]->triangulation.n_active_cells(), 4 * 5 * default_layers);
  ASSERT_EQ(surfaces[2]->triangulation.n_active_cells(), 3 * 5 * default_layers);
  ASSERT_EQ(surfaces[3]->triangulation.n_active_cells(), 3 * 5 * default_layers);
  ASSERT_EQ(surfaces[4]->triangulation.n_active_cells(), 3 * 4 * default_layers);
  ASSERT_EQ(surfaces[5]->triangulation.n_active_cells(), 3 * 4 * default_layers);
}

TEST_F(PMLCubeFixture, PMLSurfaceStructureCounts) {
  ASSERT_EQ(surfaces[0]->edge_ids_by_boundary_id[0].size(), 5*5+4*6);
  ASSERT_EQ(surfaces[0]->edge_ids_by_boundary_id[1].size(), 5*5+4*6);

  ASSERT_EQ(surfaces[0]->edge_ids_by_boundary_id[2].size(), 6*7+5*8);
  ASSERT_EQ(surfaces[0]->edge_ids_by_boundary_id[3].size(), 6*7+5*8);

  ASSERT_EQ(surfaces[0]->edge_ids_by_boundary_id[4].size(), 5*7+4*8);
  ASSERT_EQ(surfaces[0]->edge_ids_by_boundary_id[5].size(), 5*7+4*8);
}

TEST_F(PMLCubeFixture, PMLDomainMeasurements) {
  ASSERT_EQ(surfaces[0]->x_range.first, -2.0);
  ASSERT_EQ(surfaces[0]->x_range.second, -1.0);
  ASSERT_EQ(surfaces[0]->y_range.first, -1.0);
  ASSERT_EQ(surfaces[0]->y_range.second, 1.0);
  ASSERT_EQ(surfaces[0]->z_range.first, -1.0);
  ASSERT_EQ(surfaces[0]->z_range.second, 1.0);

  ASSERT_EQ(surfaces[1]->x_range.first, 1.0);
  ASSERT_EQ(surfaces[1]->x_range.second, 2.0);
  ASSERT_EQ(surfaces[1]->y_range.first, -1.0);
  ASSERT_EQ(surfaces[1]->y_range.second, 1.0);
  ASSERT_EQ(surfaces[1]->z_range.first, -1.0);
  ASSERT_EQ(surfaces[1]->z_range.second, 1.0);

  ASSERT_EQ(surfaces[2]->x_range.first, -1.0);
  ASSERT_EQ(surfaces[2]->x_range.second, 1.0);
  ASSERT_EQ(surfaces[2]->y_range.first, -2.0);
  ASSERT_EQ(surfaces[2]->y_range.second, -1.0);
  ASSERT_EQ(surfaces[2]->z_range.first, -1.0);
  ASSERT_EQ(surfaces[2]->z_range.second, 1.0);

  ASSERT_EQ(surfaces[3]->x_range.first, -1.0);
  ASSERT_EQ(surfaces[3]->x_range.second, 1.0);
  ASSERT_EQ(surfaces[3]->y_range.first, 1.0);
  ASSERT_EQ(surfaces[3]->y_range.second, 2.0);
  ASSERT_EQ(surfaces[3]->z_range.first, -1.0);
  ASSERT_EQ(surfaces[3]->z_range.second, 1.0);

  ASSERT_EQ(surfaces[4]->x_range.first, -1.0);
  ASSERT_EQ(surfaces[4]->x_range.second, 1.0);
  ASSERT_EQ(surfaces[4]->y_range.first, -1.0);
  ASSERT_EQ(surfaces[4]->y_range.second, 1.0);
  ASSERT_EQ(surfaces[4]->z_range.first, -2.0);
  ASSERT_EQ(surfaces[4]->z_range.second, -1.0);

  ASSERT_EQ(surfaces[5]->x_range.first, -1.0);
  ASSERT_EQ(surfaces[5]->x_range.second, 1.0);
  ASSERT_EQ(surfaces[5]->y_range.first, -1.0);
  ASSERT_EQ(surfaces[5]->y_range.second, 1.0);
  ASSERT_EQ(surfaces[5]->z_range.first, 1.0);
  ASSERT_EQ(surfaces[5]->z_range.second, 2.0);
}

TEST_F(PMLCubeFixture, PMLSurfaceDofCountTests) {
  const unsigned int default_layers = GlobalParams.PML_N_Layers - 1;
  std::array<unsigned int, 3> cells_by_direction;
  
  cells_by_direction[0] = 3;
  cells_by_direction[1] = 4;
  cells_by_direction[2] = 5;

  for(unsigned int i = 0; i < 6; i++) {
    for(unsigned int j = 0; j < 6; j++) {
      std::cout << "i: " << i << " j: " << j << std::endl; 
      if(i / 2 == j / 2) {
        if(i/2 == 0) {
          ASSERT_EQ(surfaces[i]->get_dof_count_by_boundary_id(j), (cells_by_direction[1] + 1) * cells_by_direction[2] + (cells_by_direction[2] + 1) * cells_by_direction[1]);
        }
        if(i/2 == 1) {
          ASSERT_EQ(surfaces[i]->get_dof_count_by_boundary_id(j), (cells_by_direction[2] + 1) * cells_by_direction[0] + (cells_by_direction[0] + 1) * cells_by_direction[2]);
        }
        if(i/2 == 2) {
          ASSERT_EQ(surfaces[i]->get_dof_count_by_boundary_id(j), (cells_by_direction[0] + 1) * cells_by_direction[1] + (cells_by_direction[1] + 1) * cells_by_direction[0]);
        }
      } else {
        unsigned int directional_component = third_number_in_zero_one_two(i/2, j/2);
        ASSERT_EQ(surfaces[i]->get_dof_count_by_boundary_id(j), cells_by_direction[directional_component] * GlobalParams.PML_N_Layers + (cells_by_direction[directional_component]+1) * (GlobalParams.PML_N_Layers-1));
      }
    }
  }
}

TEST_F(PMLCubeFixture, BoundaryToBoundaryAssociationTests) {
  for(unsigned int i = 0; i < 6; i++) {
    for(unsigned int j = 0; j < 6; j++) {
      if(i/2 != j/2) {
        unsigned int dof_count_i = surfaces[i]->get_dof_count_by_boundary_id(j);
        unsigned int dof_count_j = surfaces[j]->get_dof_count_by_boundary_id(i);
        std::cout << "Pair " << i << " - " << j << std::endl;
        ASSERT_EQ(dof_count_i, dof_count_j);
      }
    }
  }
}

TEST_F(PMLCubeFixture, PMLLayerSideDofCounts) {
  for(unsigned int i = 0; i < 6; i++) {
    
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
        std::vector<InterfaceDofData> first_dofs = surfaces[i]->get_dof_association_by_boundary_id(j);
        std::vector<InterfaceDofData> second_dofs = surfaces[j]->get_dof_association_by_boundary_id(i);
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
        std::vector<InterfaceDofData> first_dofs = surfaces[i]->get_dof_association_by_boundary_id(j);
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
        std::vector<InterfaceDofData> first_dofs = surfaces[i]->get_dof_association_by_boundary_id(j);
        std::vector<InterfaceDofData> second_dofs = surfaces[j]->get_dof_association_by_boundary_id(i);
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

TEST_F(FullCubeFixture, EvaluationOfA) {
  ComplexNumber k0(0, -1);
  std::vector<ComplexNumber> zeroes;
  
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
      ComplexNumber res = surfaces[0]->evaluate_a(u, v, G);
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

TEST_F(FullCubeFixture, RandomPolynomialProductTest) {
  ComplexNumber k0(0, -1);
  std::vector<ComplexNumber> zeroes;
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
  ComplexNumber res = surfaces[0]->evaluate_a(u, v, G);
  ComplexNumber expected_result(0, 0);
  for (unsigned j = 0; j < 3; j++) {
    for (unsigned int i = 0; i < 10; i++) {
      expected_result += u[j].a[i] * v[j].a[i];
    }
  }

  ASSERT_NEAR(expected_result.real(), res.real(), expected_result.real() / 100.0);
  ASSERT_NEAR(expected_result.imag(), res.imag(), expected_result.imag() / 100.0);
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