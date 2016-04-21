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
#include <petscpc.h>


class PreconditionerSweeping : public TrilinosWrappers::PreconditionBase
  {
  public:
	PreconditionerSweeping (const TrilinosWrappers::SparseMatrix  &S);

	void vmult (TrilinosWrappers::MPI::Vector       &dst,      const TrilinosWrappers::MPI::Vector &src) const;

  private:
      const SmartPointer<const TrilinosWrappers::SparseMatrix> preconditioner_matrix;
  };

#endif /* PRECONDITIONERSWEEPING_H_ */
