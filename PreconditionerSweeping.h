/*
 * PreconditionerSweepingPetscParallel.h
 *
 *  Created on: 19.02.2016
 *      Author: kraft
 */

#ifndef PRECONDITIONERSWEEPING_H_
#define PRECONDITIONERSWEEPING_H_

using namespace dealii;
#include <deal.II/base/config.h>
#include <deal.II/lac/exceptions.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>


class PreconditionerSweeping : public TrilinosWrappers::PreconditionBase
  {
	using TrilinosWrappers::PreconditionBase::vmult;

  public:
	PreconditionerSweeping (const TrilinosWrappers::SparseMatrix  &S, int in_own, int in_others);

	virtual void vmult (TrilinosWrappers::MPI::Vector       &dst,      const TrilinosWrappers::MPI::Vector &src) const;

  private:
	int own, others;
      const SmartPointer<const TrilinosWrappers::SparseMatrix> preconditioner_matrix;
      TrilinosWrappers::MPI::Vector itmp, otmp;
      TrilinosWrappers::MPI::BlockVector input, output;
  };

#endif /* PRECONDITIONERSWEEPING_H_ */
