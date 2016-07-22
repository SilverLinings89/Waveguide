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
	PreconditionerSweeping ( int in_own, int in_others, int bandwidth, IndexSet locallyowned, int in_upper);

    ~PreconditionerSweeping ();

    void Hinv(dealii::Vector<double> src, dealii::Vector<double> dst) const ;
        
    void LowerProduct(dealii::Vector<double> src, dealii::Vector<double> dst) const ;

    void UpperProduct(dealii::Vector<double> src, dealii::Vector<double> dst) const ;

	virtual void vmult (TrilinosWrappers::MPI::Vector       &dst,      const TrilinosWrappers::MPI::Vector &src) const;

	TrilinosWrappers::SparseMatrix matrix;

  private:
	int * indices;
	int own, others, upper;
	TrilinosWrappers::MPI::Vector itmp, otmp;
  };

#endif /* PRECONDITIONERSWEEPING_H_ */
