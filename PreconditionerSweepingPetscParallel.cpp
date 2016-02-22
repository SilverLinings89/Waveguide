/*
 * PreconditionerSweepingPetscParallel.cpp
 *
 *  Created on: 19.02.2016
 *      Author: kraft
 */

#include "PreconditionerSweepingPetscParallel.h"
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/solver_control.h>


PreconditionerSweepingPetscParallel::PreconditionerSweepingPetscParallel(dealii::PETScWrappers::MPI::SparseMatrix * in_matrix) :
sub_lowest(GlobalParams.sub_block_lowest),
lowest(GlobalParams.block_lowest),
highest(GlobalParams.block_highest),
cn(),
solver(cn)
{
	A = in_matrix;
}

void PreconditionerSweepingPetscParallel::vmult( dealii::PETScWrappers::MPI::Vector & out_vec ,const dealii::PETScWrappers::MPI::Vector & in_vec ) {
	dealii::PETScWrappers::MPI::Vector temp(MPI_COMM_SELF, highest - sub_lowest, highest - sub_lowest);

	for( unsigned int i = 0; i<= highest - lowest; i++) {
		temp( i + lowest - sub_lowest ) = in_vec( lowest + i);
	}
	dealii::PETScWrappers::MPI::Vector out(MPI_COMM_SELF, highest - sub_lowest, highest - sub_lowest);

	solver.solve( *A ,out,temp);

	for(unsigned int i = 0; i < highest - lowest; i++) {
		out_vec( lowest + i ) = out(i + lowest- sub_lowest);
	}
}


