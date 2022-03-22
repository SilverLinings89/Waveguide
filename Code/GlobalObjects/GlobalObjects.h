#pragma once

/**
 * @file GlobalObjects.h
 * @author your name (you@domain.com)
 * @brief Contains the declaration of some global objects that contain the parameter values as well as some values derived from them, like the geometry and information about other processes.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "../Helpers/Parameters.h"
#include "GeometryManager.h"
#include "../Hierarchy/MPICommunicator.h"
#include "ModeManager.h"
#include "OutputManager.h"
#include "TimerManager.h"
#include "../SpaceTransformations/SpaceTransformation.h"

extern Parameters GlobalParams; 
extern GeometryManager Geometry;
extern MPICommunicator GlobalMPI;
extern ModeManager GlobalModeManager;
extern OutputManager GlobalOutputManager;
extern TimerManager GlobalTimerManager;
extern SpaceTransformation * GlobalSpaceTransformation;

void initialize_global_variables(const std::string run_file,const std::string case_file, std::string override_data = "");
