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

static SolverControl s(10,1.e-10, false, false);
dealii::TrilinosWrappers::SolverDirect * solver;

class PreconditionerSweeping : TrilinosWrappers::PreconditionBase
  {

  public:
	PreconditionerSweeping ( int in_own, int in_others, int bandwidth, IndexSet locallyowned);

    ~PreconditionerSweeping ();
        
	virtual void vmult (TrilinosWrappers::MPI::Vector       &dst,      const TrilinosWrappers::MPI::Vector &src) const;

	TrilinosWrappers::SparseMatrix matrix;

  private:
	int * indices;
	int own, others;
	  // const SmartPointer<const TrilinosWrappers::SparseMatrix> preconditioner_matrix;
      TrilinosWrappers::MPI::Vector itmp, otmp;
      //dealii::Vector<double> inputb, outputb ;
      //TrilinosWrappers::MPI::BlockVector input, output;
      std::vector<unsigned int> sizes;
  };

#endif /* PRECONDITIONERSWEEPING_H_ */
