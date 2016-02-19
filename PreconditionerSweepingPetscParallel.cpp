/*
 * PreconditionerSweepingPetscParallel.cpp
 *
 *  Created on: 19.02.2016
 *      Author: kraft
 */

#include "PreconditionerSweepingPetscParallel.h"

PreconditionerSweepingPetscParallel::PreconditionerSweepingPetscParallel(unsigned int in_sub_lowest, unsigned int in_lowest, unsigned int in_highest, dealii::PETScWrappers::SparseMatrix * in_matrix) :
sub_lowest(in_sub_lowest),
lowest(in_lowest),
highest(in_highest) {

	A = in_matrix;


}

void PreconditionerSweepingPetscParallel::vmult( dealii::PETScWrappers::MPI::Vector & out_vec ,const dealii::PETScWrappers::MPI::Vector & in_vec ) const {

}


