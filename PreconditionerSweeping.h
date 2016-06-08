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
#include <deal.II/lac/vector.h>


class PreconditionerSweeping : TrilinosWrappers::PreconditionBase
  {

  public:
	PreconditionerSweeping ( dealii::SparseDirectUMFPACK * S, int in_own, int in_others);

	virtual void vmult (TrilinosWrappers::MPI::Vector       &dst,      const TrilinosWrappers::MPI::Vector &src) const;

	// template <typename T> inline void vmult (T &src, const T &dst) const ;
  private:
	SolverControl solver_control;

	int own, others;
      // const SmartPointer<const TrilinosWrappers::SparseMatrix> preconditioner_matrix;
      TrilinosWrappers::MPI::Vector itmp, otmp;
      //dealii::Vector<double> inputb, outputb ;
      //TrilinosWrappers::MPI::BlockVector input, output;
      std::vector<unsigned int> sizes;
      dealii::SparseDirectUMFPACK * solver;
  };

#endif /* PRECONDITIONERSWEEPING_H_ */
