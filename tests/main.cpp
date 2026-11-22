// Entry point of the test executable. Initializes MPI (the code under test uses PETSc data
// structures) and defines the global objects that the library expects the application to provide.
#include <gtest/gtest.h>
#include <deal.II/base/mpi.h>
#include <fstream>
#include <string>
#include "../Code/GlobalObjects/GlobalObjects.h"

Parameters GlobalParams;
GeometryManager Geometry;
MPICommunicator GlobalMPI;
ModeManager GlobalModeManager;
OutputManager GlobalOutputManager;
TimerManager GlobalTimerManager;
SpaceTransformation * GlobalSpaceTransformation;

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

int main(int argc, char ** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  return RUN_ALL_TESTS();
}
