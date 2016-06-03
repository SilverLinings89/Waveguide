#include <cmath>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector.templates.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_vector_base.h>

#include "PreconditionerSweeping.h"

using namespace dealii;


PreconditionerSweeping::PreconditionerSweeping ( TrilinosWrappers::SolverDirect * S, TrilinosWrappers::SparseMatrix & M, int in_own, int in_others)
      :
      preconditioner_matrix     (&M)

    {
		solver = S;
		own = in_own;
		others = in_others;
		//itmp = TrilinosWrappers::MPI::Vector(complete_index_set(own + others));
		//otmp = TrilinosWrappers::MPI::Vector(complete_index_set(own + others));
		//input.block(0).reinit(complete_index_set(others), MPI_COMM_SELF);
		//input.block(1).reinit(complete_index_set(own), MPI_COMM_SELF);
		//output.block(0).reinit(complete_index_set(others), MPI_COMM_SELF);
		//output.block(1).reinit(complete_index_set(own), MPI_COMM_SELF);
		sizes.push_back(others);
		sizes.push_back(own);


		//inputb.reinit(own + others, false);
		//outputb.reinit(own + others, false);
		//TrilinosWrappers::PreconditionBlockwiseDirect::initialize(S, TrilinosWrappers::PreconditionBlockwiseDirect::AdditionalData());

    }


void PreconditionerSweeping::vmult (TrilinosWrappers::MPI::Vector       &dst,
			const TrilinosWrappers::MPI::Vector &src) const
{

	if(GlobalParams.MPI_Rank == 0) {
		dealii::Vector<double> inputb(own + others);
		for(unsigned int i = 0; i < others; i++) {
			inputb[i] = 0;
		}

		IndexSet owneddofs = src.locally_owned_elements();
		for(unsigned int i = 0; i < own; i++) {
			inputb[i + others] = src(owneddofs.nth_index_in_set(i));
		}

		dealii::Vector<double> outputb(own + others);

		// const TrilinosWrappers::MPI::Vector inp(input);

		//TrilinosWrappers::PreconditionBlockwiseDirect::vmult(outputb, inputb);
		solver->solve(*preconditioner_matrix, outputb, inputb);

		for(int i = 0; i < own; i++) {
			dst[owneddofs.nth_index_in_set(i)] = outputb[others + i];
		}

		// dealii::Vector<double> outputb(own + others);


		// std::cout << GlobalParams.MPI_Rank << "Non-prec L2: " << src.l2_norm() << ", Prec L2: "<< dst.l2_norm() << std::endl;
	} else {
		dst = src;
	}
}
