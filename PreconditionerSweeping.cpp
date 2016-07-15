#include <cmath>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector.templates.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_vector_base.h>
#include <deal.II/lac/solver.h>

#include "PreconditionerSweeping.h"

using namespace dealii;

PreconditionerSweeping::~PreconditionerSweeping (){
	// delete solver;
}
PreconditionerSweeping::PreconditionerSweeping (  int in_own, int in_others, int bandwidth):

		matrix(in_own+in_others, in_own+in_others, bandwidth)
{
		own = in_own;
		others = in_others;
		sizes.push_back(others);
		sizes.push_back(own);
   }


void PreconditionerSweeping::vmult (TrilinosWrappers::MPI::Vector       &dst,
			const TrilinosWrappers::MPI::Vector &src)const
{

	TrilinosWrappers::Vector inputb(own + others);
	for(int i = 0; i < others; i++) {
		inputb[i] = 0;
	}

	IndexSet owneddofs = src.locally_owned_elements();
	for(int i = 0; i < own; i++) {
		inputb[i + others] = src(owneddofs.nth_index_in_set(i));
	}

	TrilinosWrappers::Vector outputb(own + others);

	// const TrilinosWrappers::MPI::Vector inp(input);

	//TrilinosWrappers::PreconditionBlockwiseDirect::vmult(outputb, inputb);
	solver.solve( matrix , outputb, inputb);

	for(int i = 0; i < own; i++) {
		dst[owneddofs.nth_index_in_set(i)] = outputb[others + i];
	}

	// dealii::Vector<double> outputb(own + others);


	// std::cout << GlobalParams.MPI_Rank << "Non-prec L2: " << src.l2_norm() << ", Prec L2: "<< dst.l2_norm() << std::endl;

}
