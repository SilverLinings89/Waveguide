

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

PreconditionerSweeping::PreconditionerSweeping (  MPI_Comm in_mpi_comm, int in_own, int in_others, int in_above, int in_bandwidth, IndexSet in_locally_owned_dofs, IndexSet * in_fixed_dofs, int in_rank)
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
		fixed_dofs = in_fixed_dofs;
		rank = in_rank;
		bandwidth = in_bandwidth;
		mpi_comm = in_mpi_comm;
		above = in_above;
}


void PreconditionerSweeping::Prepare ( TrilinosWrappers::MPI::BlockVector & inp) {
	boundary.reinit(own, false);
	for(int i = 0; i<own; i++) {
		boundary[i] = inp[i];
	}
	return;
}

void PreconditionerSweeping::vmult (TrilinosWrappers::MPI::BlockVector       &dst,
			const TrilinosWrappers::MPI::BlockVector &src)const
{
  dealii::Vector<double> input(own);
	for (unsigned int i = 0; i < sweepable; i++) {
		input[i] = src[indices[i]];
	}

	if ((int)rank+1 == GlobalParams.NumberProcesses) {
	  solver->solve( input);
		MPI_Send(&input[0], own, MPI_DOUBLE, rank-1, 0, mpi_comm);
	} else {
	  double * trans2 = new double[others];
		MPI_Recv(trans2, others, MPI_DOUBLE, rank+1, 0, mpi_comm, MPI_STATUS_IGNORE);
		dealii::Vector<double> temp2 (others);
		for (int i = 0; i < others; i++) {
			temp2[i] = trans2[i];
		}
		dealii::Vector<double> temp3 (own);
		UpperProduct(temp2, temp3);
		input -= temp3;
		if(rank != 0) {
			dealii::Vector<double> temp4 (own);
			Hinv(input, temp4);
			MPI_Send(&temp4[0], own, MPI_DOUBLE, rank - 1, 0, mpi_comm);
		}

	}
        
  MPI_Barrier(mpi_comm);
  if((int)rank +1 != GlobalParams.NumberProcesses) {
      dealii::Vector<double> temp (own);
      for(int i =0; i < own; i++) {
          temp[i] = input[i];
      }
      Hinv(temp, input);
  }

  MPI_Barrier(mpi_comm);
  if ( rank == 0) {
      MPI_Send(&input[0], own, MPI_DOUBLE, rank + 1, 0, mpi_comm);
  } else {
      double * trans4 = new double [above];
      MPI_Recv(trans4, above, MPI_DOUBLE, rank-1, 0, mpi_comm, MPI_STATUS_IGNORE);
      dealii::Vector<double> back_sweep (own);
      dealii::Vector<double> back_sweep2 (own);
      dealii::Vector<double> temp_calc (above);
      for (int i = 0; i < own; i++) {
              temp_calc[i] = trans4[i];
      }
      LowerProduct(temp_calc, back_sweep);
      Hinv(back_sweep, back_sweep2);
      input -= back_sweep2;

      if((int)rank +1< GlobalParams.NumberProcesses) {

          MPI_Send(&input[0], own, MPI_DOUBLE, rank + 1, 0, mpi_comm);

      }
  }

  for(int i = 0; i < own; i++ ){
    if(! fixed_dofs->is_element(indices[i])){
      dst[indices[i]] = input[i];
    }
  }


}

void PreconditionerSweeping::Hinv(const dealii::Vector<double> & src, dealii::Vector<double> & dst) const {
	IndexSet is (own+others);
	is.add_range(0, own+others-1);

	dealii::Vector<double> inputb(own + others);
	for(int i = 0; i < own; i++) {
		inputb[i ] = src(i);
	}

	// dealii::TrilinosWrappers::Vector  outputb(own + others);

	solver->solve(  inputb );

	for(int i = 0; i < own; i++) {
		dst[i] = inputb[ i];
	}
}


void PreconditionerSweeping::init(SolverControl , TrilinosWrappers::SparseMatrix * in_prec_upper, TrilinosWrappers::SparseMatrix * in_prec_lower) {

  deallog.push("Init Preconditioner");
  deallog << "Prepare Objects" <<std::endl;
  solver = new SparseDirectUMFPACK();
  IndexSet local (matrix->m());
  local.add_range(0, matrix->m());

  // Main Matrix Preparation

  sparsity_pattern.reinit(own+ others, own+others, bandwidth);

  TrilinosWrappers::SparseMatrix::iterator it = matrix->begin();
  TrilinosWrappers::SparseMatrix::iterator end = matrix->end();
  for(; it != end; it++){
    sparsity_pattern.add(it->row(), it->column());
  }


  sparsity_pattern.compress();

  temp = new dealii::SparseMatrix<double>(sparsity_pattern);
  deallog << "Copy Matrix" <<std::endl;
  temp->copy_from(* matrix);
  deallog << "Factorize Matrix" <<std::endl;
  solver->factorize(*temp);

  // Prec Matrix lower Preparation
  deallog << "Prepare Lower Block" <<std::endl;
  off_diag_block_lower.reinit(own, above, bandwidth);

  if(above != 0){
    it = in_prec_lower->begin();
    end = in_prec_lower->end();
    for(; it != end; it++){
      off_diag_block_lower.add(it->row(), it->column());
    }
  } else {
    off_diag_block_lower.reinit(own,own, 120);
    it = in_prec_lower->begin();
    end = in_prec_lower->end();
    for(; it != end; it++){
      off_diag_block_lower.add(it->row(), it->column());
    }
  }
  off_diag_block_lower.compress();

  prec_matrix_lower = new dealii::SparseMatrix<double>(off_diag_block_lower);
  prec_matrix_lower->copy_from(*in_prec_lower);

  // Prec Matrix upper Preparation
  deallog << "Prepare Upper Block" <<std::endl;
  off_diag_block_upper.reinit(own, others, bandwidth);

  if(others != 0){
    it = in_prec_upper->begin();
    end = in_prec_upper->end();
    for(; it != end; it++){
      off_diag_block_upper.add(it->row(), it->column());
    }
  } else {
    off_diag_block_upper.reinit(own,own, 120);
    it = in_prec_upper->begin();
    end = in_prec_upper->end();
    for(; it != end; it++){
      off_diag_block_upper.add(it->row(), it->column());
    }
  }
  off_diag_block_upper.compress();

  prec_matrix_upper = new dealii::SparseMatrix<double>(off_diag_block_upper);
  prec_matrix_upper->copy_from(*in_prec_upper);
  deallog.pop();
}

void PreconditionerSweeping::UpperProduct(const dealii::Vector<double> & src, dealii::Vector<double> & dst) const {


	if((int)rank+1 == GlobalParams.NumberProcesses) {
		std::cout << "ERROR!" <<std::endl;
	}

	prec_matrix_upper->vmult(dst, src);

}

void PreconditionerSweeping::LowerProduct(const dealii::Vector<double> & src, dealii::Vector<double> & dst) const {

  if((int)rank  == 0) {
    std::cout << "ERROR!" <<std::endl;
  }

  prec_matrix_lower->vmult(dst, src);

}


