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

void print_usage() {
  std::cout << "Usage: mpirun -np <N> main [--case <case file>] [--run <run file>] [--override \"key=value;key=value\"] [PETSc options]\n"
            << "\n"
            << "  --case      Parameter file that describes WHAT is computed (geometry, materials, signal input, boundary method).\n"
            << "              Default: Parameters/Case/base.prm\n"
            << "  --run       Parameter file that describes HOW it is computed (process layout, cells per process, solver).\n"
            << "              Default: Parameters/Run/base.prm\n"
            << "  --override  Overrides single parameters, e.g. \"n_pml_cells=8;pml_sigma_max=10\". Supported keys: n_pml_cells,\n"
            << "              pml_sigma_max, pml_order, solver_type, geometry_size_z, system_length, processes_in_z,\n"
            << "              predefined_case_number.\n"
            << "  --help      Prints this message.\n"
            << "\n"
            << "N has to be equal to (processes in x) * (processes in y) * (processes in z) of the run file.\n"
            << "Environment variables: WAVEGUIDE_ROOT (repository root, used to locate Modes/ and Parameters/) and\n"
            << "WAVEGUIDE_OUTPUT_DIR (folder for the outputs, default: <repository>/Solutions).\n"
            << "Examples of case and run files can be found in the folders Parameters/Case and Parameters/Run." << std::endl;
}

int main(int argc, char *argv[]) {
  std::string run_file = repository_path("Parameters/Run/base.prm");
  std::string case_file = repository_path("Parameters/Case/base.prm");
  std::string override_data = "";
  // All arguments that are not consumed here are passed on to MPI / PETSc.
  std::vector<char *> remaining_arguments;
  remaining_arguments.push_back(argv[0]);
  for(int i = 1; i < argc; i++) {
    const std::string argument = argv[i];
    if(argument == "--help" || argument == "-h") {
      print_usage();
      return 0;
    }
    const bool has_value = (i + 1 < argc);
    if(argument == "--case" && has_value) {
      case_file = argv[++i];
    } else if(argument == "--run" && has_value) {
      run_file = argv[++i];
    } else if(argument == "--override" && has_value) {
      override_data = argv[++i];
    } else {
      remaining_arguments.push_back(argv[i]);
    }
  }
  for(const std::string & file : {case_file, run_file}) {
    if(!std::ifstream(file).good()) {
      std::cerr << "The parameter file '" << file << "' could not be opened.\n" << std::endl;
      print_usage();
      return 1;
    }
  }
  int argc_stripped = remaining_arguments.size();
  char** argv_stripped = remaining_arguments.data();

  {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc_stripped, argv_stripped, 1);
    try {
      initialize_global_variables(run_file, case_file, override_data);
    } catch(std::exception & e) {
      // Typically an unknown or malformed entry in one of the parameter files.
      if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
        std::cerr << "The simulation could not be set up with the case file '" << case_file << "' and the run file '" << run_file << "':\n" << e.what() << std::endl;
      }
      return 1;
    }
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
    GlobalTimerManager.finalize();
    GlobalMPI.destroy_comms();
    print_info("Main", "End");
  }
  return 0;
}
