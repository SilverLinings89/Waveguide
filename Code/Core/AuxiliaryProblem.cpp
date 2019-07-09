/*
 * AuxiliaryProblem.cpp
 *
 *  Created on: Jun 24, 2019
 *      Author: kraft
 */

#include "AuxiliaryProblem.h"

AuxiliaryProblem::AuxiliaryProblem(MPI_Comm inMpiComm, SquareMeshGenerator *inMg, SpaceTransformation *inSt)
        : NumericProblem(inMpiComm, inMg, inSt) {
    // TODO Auto-generated constructor stub
}

AuxiliaryProblem::~AuxiliaryProblem() {
    // TODO Auto-generated destructor stub
}
