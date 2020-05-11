#ifndef MainCppFlag
#define MainCppFlag

#include <sys/types.h>
#include <sys/stat.h>
#include <deal.II/base/parameter_handler.h>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "../Code/Core/NumericProblem.h"
#include "../Code/Helpers/Parameters.h"
#include "../Code/Helpers/ParameterReader.h"
#include "../Code/Helpers/staticfunctions.h"
#include "../Code/Helpers/ModeManager.h"
#include "../Code/Core/Sector.h"
#include "../Code/Helpers/PointVal.h"
#include "../Code/Helpers/ExactSolution.h"
#include "../Code/Core/SolutionWeight.h"
#include "../Code/OutputGenerators/Console/GradientTable.h"
#include "../Code/SpaceTransformations/SpaceTransformation.h"
#include "../Code/SpaceTransformations/InhomogenousTransformationRectangular.h"
#include "../Code/SpaceTransformations/HomogenousTransformationRectangular.h"
#include "../Code/SpaceTransformations/DualProblemTransformationWrapper.h"
#include "../Code/MeshGenerators/SquareMeshGenerator.h"
#include "../Code/Helpers/ShapeDescription.h"
#include "../Code/Core/Simulation.h"

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
Parameters GlobalParams;
GeometryManager Geometry;
MPICommunicator GlobalMPI;
ModeManager GlobalModeManager;

int main(int argc, char *argv[]) {

  if (argc > 1) {
    input_file_name = argv[1];
  } else {
    input_file_name = "../Parameters/Parameters.xml";
  }

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  PrepareStreams();

  Simulation simulation;

  simulation.prepare();

  simulation.run();

  return 0;

}

#endif
