/*
 * GlobalObjects.cpp
 *
 *  Created on: May 11, 2020
 *      Author: kraft
 */

#ifndef CODE_CORE_GlobalObjects_
#define CODE_CORE_GlobalObjects_

#include "../Helpers/Parameters.h"
#include "../Helpers/GeometryManager.h"
#include "../Hierarchy/MPICommunicator.h"
#include "../Helpers/staticfunctions.h"
#include "../Helpers/ModeManager.h"
#include "GlobalObjects.h"

void initialize_global_variables() {
  // Read parameters into Parameter Object
  GlobalParams = GetParameters();

  // Build Global Geometry
  Geometry.initialize();

  // Build MPI Communicator
  GlobalMPI.initialize();

  // Build Mode Manager
  GlobalModeManager.load();
}

#endif
