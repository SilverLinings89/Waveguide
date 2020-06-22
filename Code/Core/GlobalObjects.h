#pragma once

#include "../Helpers/Parameters.h"
#include "../Helpers/GeometryManager.h"
#include "../Hierarchy/MPICommunicator.h"
#include "../Helpers/ModeManager.h"

extern Parameters GlobalParams;
extern GeometryManager Geometry;
extern MPICommunicator GlobalMPI;
extern ModeManager GlobalModeManager;
void initialize_global_variables();
