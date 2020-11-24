#include "../Helpers/Parameters.h"
#include "../Helpers/GeometryManager.h"
#include "../Hierarchy/MPICommunicator.h"
#include "../Helpers/staticfunctions.h"
#include "../Helpers/ModeManager.h"
#include "GlobalObjects.h"

void initialize_global_variables(const std::string run_file, const std::string case_file) {
  // Read parameters into Parameter Object
  GlobalParams = GetParameters(run_file, case_file);

  // Build Global Geometry
  Geometry.initialize();

  // Build MPI Communicator
  GlobalMPI.initialize();

  // Build Mode Manager
  GlobalModeManager.load();
}
