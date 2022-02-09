#include "../Helpers/Parameters.h"
#include "GeometryManager.h"
#include "../SpaceTransformations/InhomogenousTransformationRectangular.h"
#include "../SpaceTransformations/BendTransformation.h"
#include "../SpaceTransformations/AngleWaveguideTransformation.h"
#include "../Hierarchy/MPICommunicator.h"
#include "../Helpers/staticfunctions.h"
#include "ModeManager.h"
#include "GlobalObjects.h"
#include "../Helpers/ParameterOverride.h"

void initialize_global_variables(const std::string run_file, const std::string case_file, std::string override_data) {
  // Read parameters into Parameter Object
  
  ParameterOverride po;
  if(override_data.size() != 0) {
    bool success = po.read(override_data);
    if(!success) {
      std::cout << "The override data was incorrect. Usage: Seperate overrides by \";\" and pass key-value-pairs like \"pml_order=4;pml_sigma_max=10\". Also, spaces are not allowed. Use \"_\" instead. Also, remember to wrap the list in \"...\" in case you have multiple statements because the \";\" will otherwise be interpreted by the shell." <<std::endl;
      exit(0);
    }
  }
  
  GlobalParams = GetParameters(run_file, case_file, po);

  // Build MPI Communicator
  GlobalMPI.initialize();
  
  // Build Global Geometry
  Geometry.initialize();

  // Build Mode Manager
  GlobalModeManager.load();

  GlobalOutputManager.initialize();

  GlobalTimerManager.initialize();

  if(GlobalParams.transformation_type == TransformationType::InhomogenousWavegeuideTransformationType) {
    GlobalSpaceTransformation = new InhomogenousTransformationRectangular();
  }
  if(GlobalParams.transformation_type == TransformationType::BendTransformationType) {
    GlobalSpaceTransformation = new BendTransformation();
  }
  if(GlobalParams.transformation_type == TransformationType::AngleWaveguideTransformationType) {
    GlobalSpaceTransformation = new AngleWaveguideTransformation();
  }

  GlobalSpaceTransformation->estimate_and_initialize();
  
  print_info("GlobalPreparation", "Complete");
}
