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
	delete solver;
}
PreconditionerSweeping::PreconditionerSweeping (  int in_own, int in_others, int bandwidth,  IndexSet locally_owned):

		matrix(in_own+in_others, in_own+in_others, bandwidth)
{
		own = in_own;
		others = in_others;
		sizes.push_back(others);
		sizes.push_back(own);
		IndexSet elements (own+others);
		elements.add_range(0,own+elements);
		solver = new TrilinosWrappers::SolverDirect(s, TrilinosWrappers::SolverDirect::AdditionalData(false, GlobalParams.PRM_S_Preconditioner));
		itmp(elements , MPI_COMM_SELF);
		indices = new int[locally_owned.n_elements()];
		for(int i = 0; i < own; i++){
			indices[i] = locally_owned.nth_index_in_set(i);
		}
   }


void PreconditionerSweeping::vmult (TrilinosWrappers::MPI::Vector       &dst,
			const TrilinosWrappers::MPI::Vector &src)const
{


	for(int i = 0; i < own; i++) {
		itmp[i + others] = src(indices[i]);
	}

	solver->solve( matrix , otmp, itmp);

	for(int i = 0; i < own; i++) {
		dst[indices[i]] = otmp[others + i];
	}

}
