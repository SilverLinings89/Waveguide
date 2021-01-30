#pragma once

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
#include "./TestHelperFunctions.h"

class TestOrderFixture: public ::testing::TestWithParam<
    std::tuple<unsigned int, unsigned int>> {
public:
  unsigned int Cells_Per_Direction;
  unsigned int InnerOrder;
  dealii::Triangulation<3> tria;
  dealii::Triangulation<2, 3> temp_triangulation;
  dealii::Triangulation<2> surf_tria;
  ComplexNumber k0;
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
  ComplexNumber k0;
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

class TestDirectionFixture: public ::testing::TestWithParam<std::tuple<unsigned int, unsigned int, unsigned int>> {
public:
  unsigned int Cells_Per_Direction;
  unsigned int InnerOrder;
  unsigned int boundary_id;
  dealii::Triangulation<3> tria;
  dealii::Triangulation<2, 3> temp_triangulation;
  dealii::Triangulation<2> surf_tria;
  ComplexNumber k0;
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
    dealii::GridGenerator::subdivided_hyper_rectangle(tria, repetitions, left, right, true);
    const unsigned int dest_cells = Cells_Per_Direction * Cells_Per_Direction * Cells_Per_Direction;
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