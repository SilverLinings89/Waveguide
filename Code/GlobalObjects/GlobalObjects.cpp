#include "../Helpers/Parameters.h"
#include "GeometryManager.h"
#include "../Hierarchy/MPICommunicator.h"
#include "../Helpers/staticfunctions.h"
#include "ModeManager.h"
#include "GlobalObjects.h"

void initialize_global_variables(const std::string run_file, const std::string case_file) {
  // Read parameters into Parameter Object
  GlobalParams = GetParameters(run_file, case_file);
  
  // Build MPI Communicator
  GlobalMPI.initialize();
  
  // Build Global Geometry
  Geometry.initialize();

  // Build Mode Manager
  GlobalModeManager.load();

  GlobalOutputManager.initialize();

  GlobalTimerManager.initialize();
  print_info("GlobalPreparation", "Complete");
}
