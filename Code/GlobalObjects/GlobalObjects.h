#pragma once

#include "../Helpers/Parameters.h"
#include "GeometryManager.h"
#include "../Hierarchy/MPICommunicator.h"
#include "ModeManager.h"
#include "OutputManager.h"
#include "TimerManager.h"
#include "../SpaceTransformations/SpaceTransformation.h"

extern Parameters GlobalParams; // 
extern GeometryManager Geometry;
extern MPICommunicator GlobalMPI;
extern ModeManager GlobalModeManager;
extern OutputManager GlobalOutputManager;
extern TimerManager GlobalTimerManager;
extern SpaceTransformation * GlobalSpaceTransformation;

void initialize_global_variables(const std::string run_file,const std::string case_file);
