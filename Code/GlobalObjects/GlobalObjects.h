#pragma once

#include "../Helpers/Parameters.h"
#include "../GlobalObjects/GeometryManager.h"
#include "../Hierarchy/MPICommunicator.h"
#include "../GlobalObjects/ModeManager.h"
#include "../GlobalObjects/OutputManager.h"

extern Parameters GlobalParams;
extern GeometryManager Geometry;
extern MPICommunicator GlobalMPI;
extern ModeManager GlobalModeManager;
extern OutputManager GlobalOutputManager;

void initialize_global_variables(const std::string run_file,const std::string case_file);
