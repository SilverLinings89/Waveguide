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
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/solver_control.h>

class PreconditionerSweepingPetscParallel : public dealii::PETScWrappers::PreconditionerBase {
public:
	PreconditionerSweepingPetscParallel (dealii::PETScWrappers::MPI::SparseMatrix * in_matrix);

	unsigned int 								sub_lowest, lowest, highest;
	SolverControl								cn;
	dealii::PETScWrappers::SparseDirectMUMPS 	solver;

	dealii::PETScWrappers::MPI::SparseMatrix *	A;


	void vmult(dealii::PETScWrappers::MPI::Vector & in_vec, const dealii::PETScWrappers::MPI::Vector & out_vec) ;

};



#endif /* PRECONDITIONERSWEEPINGPETSCPARALLEL_H_ */
