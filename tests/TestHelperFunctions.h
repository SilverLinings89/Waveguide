#pragma once

#include "../Code/Core/Types.h"
#include "../Code/Helpers/Parameters.h"
#include "../Code/GlobalObjects/GeometryManager.h"
#include "../Code/Hierarchy/MPICommunicator.h"
#include "../Code/GlobalObjects/ModeManager.h"
#include "../Code/SpaceTransformations/SpaceTransformation.h"
#include "../Code/BoundaryCondition/HSIEPolynomial.h"
#include <fstream>

Parameters GlobalParams;
GeometryManager Geometry;
MPICommunicator GlobalMPI;
ModeManager GlobalModeManager;
OutputManager GlobalOutputManager;
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

static void setup_GlobalParams_for_tests(unsigned int n_cells_x, unsigned int n_cells_y, unsigned int n_cells_z, BoundaryConditionType b_type, double size_x, double size_y, double size_z) {
  GlobalParams.Blocks_in_x_direction = 1;
  GlobalParams.Blocks_in_y_direction = 1;
  GlobalParams.Blocks_in_z_direction = 1;
  GlobalParams.Index_in_x_direction = 0;
  GlobalParams.Index_in_y_direction = 0;
  GlobalParams.Index_in_z_direction = 0;
  GlobalParams.Cells_in_x = n_cells_x;
  GlobalParams.Cells_in_y = n_cells_y;
  GlobalParams.Cells_in_z = n_cells_z;
  GlobalParams.Geometry_Size_X = size_x;
  GlobalParams.Geometry_Size_Y = size_y;
  GlobalParams.Geometry_Size_Z = size_z;
  GlobalParams.HSIE_SWEEPING_LEVEL = 1;
  GlobalParams.MPI_Rank = 0;
  GlobalParams.Nedelec_element_order = 0;
  GlobalParams.BoundaryCondition = b_type;
}

static unsigned int third_number_in_zero_one_two(unsigned int first, unsigned int second) {
  for(unsigned int i = 0; i < 3; i++) {
    if(first != i && second != i) {
      return i;
    }
  }
  return 3;
}

static HSIEPolynomial random_poly(unsigned int Order, ComplexNumber k0) {
  std::vector<ComplexNumber> a;
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