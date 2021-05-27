#include <sys/types.h>
#include <sys/stat.h>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "../Code/Core/InnerDomain.h"
#include "../Code/Helpers/Parameters.h"
#include "../Code/Helpers/ParameterReader.h"
#include "../Code/Helpers/staticfunctions.h"
#include "../Code/GlobalObjects/ModeManager.h"
#include "../Code/Core/Sector.h"
#include "../Code/Helpers/PointVal.h"
#include "../Code/GlobalObjects/OutputManager.h"
#include "../Code/GlobalObjects/TimerManager.h"
#include "../Code/Solutions/ExactSolution.h"
#include "../Code/Core/SolutionWeight.h"
#include "../Code/OutputGenerators/Console/GradientTable.h"
#include "../Code/SpaceTransformations/SpaceTransformation.h"
#include "../Code/SpaceTransformations/InhomogenousTransformationRectangular.h"
#include "../Code/SpaceTransformations/HomogenousTransformationRectangular.h"
#include "../Code/SpaceTransformations/DualProblemTransformationWrapper.h"
#include "../Code/MeshGenerators/SquareMeshGenerator.h"
#include "../Code/Helpers/ShapeDescription.h"
#include "../Code/Runners/Simulation.h"
#include "../Code/Runners/ParameterSweep.h"
#include "../Code/Runners/SingleCoreRun.h"
#include "../Code/Runners/SweepingRun.h"

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
SpaceTransformation *the_st;
Parameters GlobalParams;
GeometryManager Geometry;
MPICommunicator GlobalMPI;
ModeManager GlobalModeManager;
OutputManager GlobalOutputManager;
TimerManager GlobalTimerManager;

int main(int argc, char *argv[]) {
  std::string run_file = "../Parameters/Run/base.prm";
  std::string case_file = "../Parameters/Case/base.prm";
  std::vector<std::string> all_args;

  if (argc > 1) {
    all_args.assign(argv, argv + argc);
  }
  if (argc >= 3) {
    if(all_args[1] == "--case") {
      case_file = all_args[2];
    }
    if(all_args[1] == "--run") {
      run_file = all_args[2];
    }
  }

  if (argc >= 5) {
    if(all_args[3] == "--case") {
      case_file = all_args[4];
    }
    if(all_args[3] == "--run") {
      run_file = all_args[4];
    }
  }

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  print_info("Main", "Prepare Streams", true);
  initialize_global_variables(run_file, case_file);
  Simulation * simulation;
  if (GlobalParams.Enable_Parameter_Run) {
    simulation = new ParameterSweep();
  } else {
    if (GlobalParams.NumberProcesses == 1) {
      simulation = new SingleCoreRun();
    } else {
      simulation = new SweepingRun();
    }
  }
  
  simulation->create_output_directory();
  simulation->prepare_transformed_geometry();

  print_info("Main", "Prepare Simulation", true);
  simulation->prepare();

  print_info("Main", "Run Simulation", true);
  simulation->run();

  return 0;
}
