/*
 * PreconditionerSweepingPetscParallel.h
 *
 *  Created on: 19.02.2016
 *      Author: kraft
 */

#ifndef PRECONDITIONERSWEEPINGPETSCPARALLEL_H_
#define PRECONDITIONERSWEEPINGPETSCPARALLEL_H_

#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>

class PreconditionerSweepingPetscParallel : public dealii::PETScWrappers::PreconditionerBase {
public:
	PreconditionerSweepingPetscParallel (unsigned int in_sub_lowest, unsigned int in_lowest, unsigned int in_highest, dealii::PETScWrappers::SparseMatrix * in_matrix);

	unsigned int sub_lowest, lowest, highest;
	dealii::PETScWrappers::SparseMatrix A;

	dealii::PETScWrappers::SparseDirectMUMPS solver;

	void vmult(dealii::PETScWrappers::MPI::Vector & in_vec, const dealii::PETScWrappers::MPI::Vector & out_vec) const ;
};



#endif /* PRECONDITIONERSWEEPINGPETSCPARALLEL_H_ */
