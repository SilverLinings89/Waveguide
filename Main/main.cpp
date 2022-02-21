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
#include "../Code/OutputGenerators/Console/GradientTable.h"
#include "../Code/SpaceTransformations/SpaceTransformation.h"
#include "../Code/MeshGenerators/SquareMeshGenerator.h"
#include "../Code/Helpers/ShapeDescription.h"
#include "../Code/Runners/Simulation.h"
#include "../Code/Runners/ConvergenceRun.h"
#include "../Code/Runners/ParameterSweep.h"
#include "../Code/Runners/SingleCoreRun.h"
#include "../Code/Runners/SweepingRun.h"
#include "../Code/Runners/OptimizationRun.h"

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
Parameters GlobalParams;
GeometryManager Geometry;
MPICommunicator GlobalMPI;
ModeManager GlobalModeManager;
OutputManager GlobalOutputManager;
TimerManager GlobalTimerManager;
SpaceTransformation * GlobalSpaceTransformation;

int main(int argc, char *argv[]) {
  std::string run_file = "../Parameters/Run/base.prm";
  std::string case_file = "../Parameters/Case/base.prm";
  std::vector<std::string> all_args;
  int argc_stripped;
  argc_stripped = argc;
  if (argc > 1) {
    all_args.assign(argv, argv + argc);
  }
  bool arg1 = false;
  bool arg2 = false;
  bool arg3 = false;
  std::string override_data = "";
  if (argc >= 3) {
    if(all_args[1] == "--case") {
      arg1 = true;
      case_file = all_args[2];
      argc_stripped -= 2;
    }
    if(all_args[1] == "--run") {
      arg1 = true;
      run_file = all_args[2];
      argc_stripped -= 2;
    }
    if(all_args[1] == "--override") {
      arg1 = true;
      override_data = all_args[2];
      argc_stripped -= 2;
    }
  }

  if (argc >= 5) {
    if(all_args[3] == "--case") {
      arg2 = true;
      case_file = all_args[4];
      argc_stripped -= 2;
    }
    if(all_args[3] == "--run") {
      arg2 = true;
      run_file = all_args[4];
      argc_stripped -=2;
    }
    if(all_args[3] == "--override") {
      arg2 = true;
      override_data = all_args[4];
      argc_stripped -= 2;
    }
  }

  if (argc >= 7) {
    if(all_args[5] == "--case") {
      arg3 = true;
      case_file = all_args[6];
      argc_stripped -= 2;
    }
    if(all_args[5] == "--run") {
      arg3 = true;
      run_file = all_args[6];
      argc_stripped -= 2;
    }
    if(all_args[5] == "--override") {
      arg3 = true;
      override_data = all_args[6];
      argc_stripped -= 2;
    }
  }

  char** argv_stripped = new char*[argc_stripped];
  unsigned int counter = 0;
  for( int i = 0; i < argc; i++) {
    bool is_processed_argument = (i == 1 || i == 2) && arg1;
    is_processed_argument = is_processed_argument || ((i == 3 || i == 4) && arg2);
    is_processed_argument = is_processed_argument || ( (i == 5 || i == 6) && arg3);
    if(! is_processed_argument) {
      argv_stripped[counter] = argv[i];
      counter++;
    }
  }
  unsigned int rank_temp = 0;
  {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc_stripped, argv_stripped, 1);
    rank_temp = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    initialize_global_variables(run_file, case_file, override_data);
    Simulation * simulation;
    if(GlobalParams.Perform_Optimization) {
      simulation = new OptimizationRun();
    } else {
      if(GlobalParams.Perform_Convergence_Test) {
        simulation = new ConvergenceRun();
      } else {
        if (GlobalParams.Enable_Parameter_Run) {
          simulation = new ParameterSweep();
        } else {
          if (GlobalParams.NumberProcesses == 1) {
            simulation = new SingleCoreRun();
          } else {
            simulation = new SweepingRun();
          }
        }
      }
    }
    simulation->create_output_directory();
    simulation->prepare_transformed_geometry();

    print_info("Main", "Prepare Simulation");
    simulation->prepare();

    print_info("Main", "Run Simulation");
    simulation->run();
    
    print_info("Main", "Shutting down");
    delete simulation;
    GlobalMPI.destroy_comms();
    delete GlobalSpaceTransformation;
    print_info("Main", "End");
  }
  return 0;
}
