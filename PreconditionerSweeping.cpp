#include <cmath>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector.templates.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_vector_base.h>

#include "PreconditionerSweeping.h"

using namespace dealii;


PreconditionerSweeping::PreconditionerSweeping (const TrilinosWrappers::SparseMatrix  &S, int in_own, int in_others)
      :
      preconditioner_matrix     (&S)
	  //input(2),
	  //output(2)
	  //inputb(in_own + in_others)
    {
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

	}

template<typename T>
inline void PreconditionerSweeping::vmult (T &dst, const T &src) const {
	std::cout << typeid(T).name() << std::endl;
}

/**
inline void PreconditionerSweeping::vmult (TrilinosWrappers::MPI::Vector       &dst,
			const TrilinosWrappers::MPI::Vector &src) const
{

	dealii::Vector<double> inputb(own + others);
	for(unsigned int i = 0; i < others; i++) {
		inputb[i] = 0;
	}

	for(unsigned int i = 0; i < own; i++) {
		inputb[i + others] = src(i);
	}

	dealii::Vector<double> outputb(own + others);

	// const TrilinosWrappers::MPI::Vector inp(input);
	SolverControl solver_control(5000, 1e-6 * src.l2_norm());
	TrilinosWrappers::SolverDirect solver(solver_control, TrilinosWrappers::SolverDirect::AdditionalData(true, "Amesos_Umfpack"));
	solver.solve(*preconditioner_matrix, outputb, inputb);

	for(int i = 0; i < own; i++) {
		dst[i] = outputb[others + i];
	}

	std::cout << GlobalParams.MPI_Rank << "Non-prec L2: " << src.l2_norm() << ", Prec L2: "<< dst.l2_norm() << std::endl;

}

**/
