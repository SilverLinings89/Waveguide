/*
 * GlobalObjects.h
 *
 *  Created on: May 11, 2020
 *      Author: kraft
 */

#ifndef CODE_CORE_GLOBALOBJECTS_H_
#define CODE_CORE_GLOBALOBJECTS_H_

#include "../Helpers/Parameters.h"
#include "../Helpers/GeometryManager.h"
#include "../Hierarchy/MPICommunicator.h"
#include "../Helpers/ModeManager.h"

extern Parameters GlobalParams;
extern GeometryManager Geometry;
extern MPICommunicator GlobalMPI;
extern ModeManager GlobalModeManager;
void initialize_global_variables();

#endif /* CODE_CORE_GLOBALOBJECTS_H_ */
