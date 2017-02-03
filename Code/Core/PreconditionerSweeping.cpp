

#include <cmath>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector.templates.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_vector_base.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>

#include "PreconditionerSweeping.h"

using namespace dealii;

PreconditionerSweeping::~PreconditionerSweeping (){
  delete temp;
  delete solver;
}
PreconditionerSweeping::PreconditionerSweeping (  int in_own, int in_others, int in_bandwidth, IndexSet in_locally_owned_dofs)
:
    sparsity_pattern(in_own+in_others, in_own+in_others, in_bandwidth)
	{
		locally_owned_dofs = in_locally_owned_dofs;
		own = in_own;
		others = in_others;
		IndexSet elements (own+others);
		elements.add_range(0,own+others);
		indices = new int[in_locally_owned_dofs.n_elements()];
		sweepable = in_locally_owned_dofs.n_elements();
		for(unsigned int i = 0; i < sweepable; i++){
			indices[i] = in_locally_owned_dofs.nth_index_in_set(i);
		}

   }


void PreconditionerSweeping::Prepare ( TrilinosWrappers::MPI::BlockVector & inp) {
	boundary.reinit(own, false);
	for(int i = 0; i<own; i++) {
		boundary[i] = inp[locally_owned_dofs.nth_index_in_set(i)];
	}
	return;
}

void PreconditionerSweeping::vmult (TrilinosWrappers::MPI::BlockVector       &dst,
			const TrilinosWrappers::MPI::BlockVector &src)const
{
	// line 1
	dealii::Vector<double> input(own);
	for (unsigned int i = 0; i < sweepable; i++) {
		input[i] = src[indices[i]];
	}

	// double norm_in = input.l2_norm();

	if ((int)GlobalParams.MPI_Rank+1 == GlobalParams.NumberProcesses) {

		// dealii::Vector<double> outputb(own);

		// solver.solve( input);
		solver->solve( input);
		// double *  trans1 = new double[own];

		// for(int i = 0; i < own; i++) {
			// input[i] = outputb[i];

		// }
		MPI_Send(&input[0], own, MPI_DOUBLE, GlobalParams.MPI_Rank-1, 0, MPI_COMM_WORLD);

	} else {

		double * trans2 = new double[others];
		MPI_Recv(trans2, others, MPI_DOUBLE, GlobalParams.MPI_Rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
		dealii::Vector<double> temp2 (others);
		for (int i = 0; i < others; i++) {
			temp2[i] = trans2[i];
		}

		// std::cout << "A" << GlobalParams.MPI_Rank << " " << temp2.l2_norm() <<  std::endl;

		dealii::Vector<double> temp3 (own);

		LowerProduct(temp2, temp3);

		// std::cout << "B" << GlobalParams.MPI_Rank << " " << temp3.l2_norm() << std::endl;


		//Line 4
		input -= temp3;
		
		// std::cout << "C" << GlobalParams.MPI_Rank << " " << input.l2_norm() << std::endl;


		if(GlobalParams.MPI_Rank != 0) {
			dealii::Vector<double> temp4 (own);

			Hinv(input, temp4);

			// std::cout << "D" << GlobalParams.MPI_Rank << " " << temp4.l2_norm() << std::endl;

			MPI_Send(&temp4[0], own, MPI_DOUBLE, GlobalParams.MPI_Rank - 1, 0, MPI_COMM_WORLD);
		}
	}
        
    MPI_Barrier(MPI_COMM_WORLD);
        
    /**
    if (GlobalParams.MPI_Rank == 0) {
        std::cout << "S1 done ..." << std::endl;
    }
       **/
    // Line 8
                
    if((int)GlobalParams.MPI_Rank +1 != GlobalParams.NumberProcesses) {
        dealii::Vector<double> temp (own);
        for(int i =0; i < own; i++) {
            temp[i] = input[i];
        }
        Hinv(temp, input);
    }
        
    MPI_Barrier(MPI_COMM_WORLD);
    // Line 11
    if ( GlobalParams.MPI_Rank == 0) {
            dealii::Vector<double> back_sweep (others);
            UpperProduct(input, back_sweep);
            MPI_Send(&back_sweep[0], others, MPI_DOUBLE, GlobalParams.MPI_Rank + 1, 0, MPI_COMM_WORLD);
    } else {
            double * trans4 = new double [own];
            MPI_Recv(trans4, own, MPI_DOUBLE, GlobalParams.MPI_Rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            dealii::Vector<double> back_sweep (own);
            dealii::Vector<double> temp_calc (own);
            for (int i = 0; i < own; i++) {
                    temp_calc[i] = trans4[i];
            }
			// std::cout << "E" << GlobalParams.MPI_Rank << " " << temp_calc.l2_norm() << std::endl;
            Hinv(temp_calc, back_sweep);
            // std::cout << "F" << GlobalParams.MPI_Rank << " " << back_sweep.l2_norm() << std::endl;
            input -= back_sweep;
                
            if((int)GlobalParams.MPI_Rank +1< GlobalParams.NumberProcesses) {
                    dealii::Vector<double> back_sweep2 (others);
                    UpperProduct(input, back_sweep2);
                    // std::cout << "G" << GlobalParams.MPI_Rank << " " << back_sweep2.l2_norm() << std::endl;
                    MPI_Send(&back_sweep2[0], others, MPI_DOUBLE, GlobalParams.MPI_Rank + 1, 0, MPI_COMM_WORLD);
            }
    }
        
    if (GlobalParams.MPI_Rank == 0) {
        // std::cout << "S2 done ..." << std::endl;

        for(int i = 0 ; i < own; i++) {
        	if(boundary[i] != 0.0){
        		input[i] = boundary[i];
        	}
        }
    }
    
	for(int i = 0; i < own; i++ ){
		dst[i + locally_owned_dofs.nth_index_in_set(0)] = input[i];
    }

}

void PreconditionerSweeping::Hinv(const dealii::Vector<double> & src, dealii::Vector<double> & dst) const {
	IndexSet is (own+others);
	is.add_range(0, own+others-1);

	// TrilinosWrappers::MPI::Vector inputb(is, MPI_COMM_SELF<dealii::TrilinosWrappers::BlockSparseMatrix>);
	dealii::Vector<double> inputb(own + others);
	for(int i = 0; i < own; i++) {
		inputb[i ] = src(i);
	}

	dealii::TrilinosWrappers::Vector  outputb(own + others);
	// TrilinosWrappers::MPI::Vector outputb(is, MPI_COMM_SELF);

	solver->solve(  inputb );

	for(int i = 0; i < own; i++) {
		dst[i] = inputb[ i];
	}
}

void PreconditionerSweeping::LowerProduct(const dealii::Vector<double> & src, dealii::Vector<double> & dst) const {

	if((int)GlobalParams.MPI_Rank +1 == GlobalParams.NumberProcesses) {
		std::cout << "ERROR!" <<std::endl;
	}

	prec_matrix_lower->vmult(dst, src);

}

void PreconditionerSweeping::init(SolverControl solver_control) {
  //  solver->initialize(*matrix, dealii::SparseDirectUMFPACK::AdditionalData());
  solver = new SparseDirectUMFPACK();
  IndexSet local (matrix->m());
  local.add_range(0, matrix->m());

  TrilinosWrappers::SparseMatrix::iterator it = matrix->begin();
  TrilinosWrappers::SparseMatrix::iterator end = matrix->end();
  for(; it != end; it++){
    sparsity_pattern.add(it->row(), it->column());
  }


  sparsity_pattern.compress();

  std::ofstream out ("sparsity_pattern"+ std::to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) + ".plot");
  sparsity_pattern.print_gnuplot (out);

  temp = new dealii::SparseMatrix<double>(sparsity_pattern);

  temp->copy_from(* matrix);

  solver->factorize(*temp);
  // solver->factorize(matrix);
}

void PreconditionerSweeping::UpperProduct(const dealii::Vector<double> & src, dealii::Vector<double> & dst) const {


	if((int)GlobalParams.MPI_Rank+1 == GlobalParams.NumberProcesses) {
		std::cout << "ERROR!" <<std::endl;
	}

	prec_matrix_lower->Tvmult(dst, src);

}


