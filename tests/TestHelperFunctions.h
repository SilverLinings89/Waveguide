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
TimerManager GlobalTimerManager;
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
