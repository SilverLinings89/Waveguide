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
	PreconditionerSweeping ( int in_own, int in_others, int bandwidth, IndexSet sweepable, IndexSet locally_owned,  int in_upper, ConstraintMatrix * in_cm);

    ~PreconditionerSweeping ();

    void Hinv(const dealii::Vector<double> &src, dealii::Vector<double> &dst) const ;
        
    void LowerProduct(const dealii::Vector<double> &src, dealii::Vector<double> &dst) const ;

    void UpperProduct(const dealii::Vector<double> &src, dealii::Vector<double> &dst) const ;

	virtual void vmult (TrilinosWrappers::MPI::Vector       &dst,      const TrilinosWrappers::MPI::Vector &src) const;

	TrilinosWrappers::SparseMatrix matrix, prec_matrix_lower, prec_matrix_upper;

	void Prepare(TrilinosWrappers::MPI::Vector &src);

  private:
	int * indices;
	int own, others, upper;
	TrilinosWrappers::MPI::Vector itmp, otmp;
	ConstraintMatrix * cm;
	Vector<double> boundary;
	unsigned int sweepable;
	IndexSet locally_owned_dofs;
  };

#endif /* PRECONDITIONERSWEEPING_H_ */
