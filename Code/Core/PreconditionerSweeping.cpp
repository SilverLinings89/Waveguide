// Copyright 2018 Pascal Kraft
#ifndef CODE_CORE_PRECONDITIONERSWEEPING_CPP_
#define CODE_CORE_PRECONDITIONERSWEEPING_CPP_

#include <deal.II/base/utilities.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/vector.h>
#include <cmath>

#include <deal.II/lac/solver.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include "PreconditionerSweeping.h"

using namespace dealii;

dealii::SolverControl s(10, 1.e-10, false, false);
dealii::SparseDirectUMFPACK *solver = 0;

dealii::SparsityPattern off_diag_block_lower, off_diag_block_upper;

PreconditionerSweeping::~PreconditionerSweeping() { delete solver; }

PreconditionerSweeping::PreconditionerSweeping(MPI_Comm in_mpi_comm, int in_own,
                                               int in_others, int in_above,
                                               int in_bandwidth,
                                               IndexSet in_locally_owned_dofs,
                                               IndexSet *in_fixed_dofs,
                                               int in_rank, bool in_fast) {
  locally_owned_dofs = in_locally_owned_dofs;
  own = in_own;
  others = in_others;
  IndexSet elements(own + others);
  elements.add_range(0, own + others);
  indices = new int[in_locally_owned_dofs.n_elements()];
  sweepable = in_locally_owned_dofs.n_elements();
  for (unsigned int i = 0; i < sweepable; i++) {
    indices[i] = in_locally_owned_dofs.nth_index_in_set(i);
  }
  fixed_dofs = in_fixed_dofs;
  rank = in_rank;
  bandwidth = in_bandwidth;
  mpi_comm = in_mpi_comm;
  above = in_above;
  prec_matrix_lower = 0;
  prec_matrix_upper = 0;
  matrix = 0;
  fast = in_fast;
}

void PreconditionerSweeping::Prepare(TrilinosWrappers::MPI::BlockVector &inp) {
  boundary.reinit(own, false);
  for (int i = 0; i < own; i++) {
    boundary[i] = inp[i];
  }
  return;
}

void PreconditionerSweeping::vmult(
    TrilinosWrappers::MPI::BlockVector &dst,
    const TrilinosWrappers::MPI::BlockVector &src) const {
  if (fast) {
    this->vmult_fast(dst, src);
  } else {
    this->vmult_slow(dst, src);
  }
}

void PreconditionerSweeping::vmult_fast(
    TrilinosWrappers::MPI::BlockVector &dst,
    const TrilinosWrappers::MPI::BlockVector &src) const {
  dealii::Vector<double> recv_buffer_above(above);
  dealii::Vector<double> recv_buffer_below(others);
  dealii::Vector<double> temp_own(own);
  dealii::Vector<double> temp_own_2(own);
  dealii::Vector<double> input(own);
  for (unsigned int i = 0; i < sweepable; i++) {
    input[i] = src[indices[i]];
  }

  if (rank == GlobalParams.NumberProcesses - 1) {
    solver->solve(input);
  } else {
    if (rank > 0) {
      Hinv(input, temp_own);
    }
  }
  if (rank == GlobalParams.NumberProcesses - 1) {
    MPI_Send(&input[0], own, MPI_DOUBLE, rank - 1, 0, mpi_comm);
  } else {
    MPI_Recv(&recv_buffer_below[0], others, MPI_DOUBLE, rank + 1, 0, mpi_comm,
             MPI_STATUS_IGNORE);
    if (rank > 0)
      MPI_Send(&temp_own[0], own, MPI_DOUBLE, rank - 1, 0, mpi_comm);
  }

  if (rank < GlobalParams.NumberProcesses - 1) {
    UpperProduct(recv_buffer_below, temp_own);
    input -= temp_own;
  }

  if (rank < GlobalParams.NumberProcesses - 1) {
    for (int i = 0; i < own; i++) {
      temp_own[i] = input[i];
    }
    Hinv(temp_own, input);
  }

  /**
   * fast version starting here.
    if ( rank == 0) {
      MPI_Send(&input[0], own, MPI_DOUBLE, rank + 1, 0, mpi_comm);
    } else {
      MPI_Recv(& recv_buffer_above[0], above, MPI_DOUBLE, rank-1, 0, mpi_comm,
   MPI_STATUS_IGNORE); if(rank < GlobalParams.NumberProcesses-1)
   MPI_Send(&input[0], own, MPI_DOUBLE, rank + 1, 0, mpi_comm);
    }

    if(rank > 0) {
      LowerProduct(recv_buffer_above, temp_own);
      if(rank == GlobalParams.NumberProcesses-1) {
        solver->solve(temp_own);
        input -= temp_own;
      } else {
        Hinv(temp_own, temp_own_2);
        input -= temp_own_2;
      }
    }

    for(int i = 0; i < own; i++ ){
      if(! fixed_dofs->is_element(indices[i])){
        dst[indices[i]] = input[i];
      }
    }
    **/
  if (rank == 0) {
    MPI_Send(&input[0], own, MPI_DOUBLE, rank + 1, 0, mpi_comm);
  } else {
    MPI_Recv(&recv_buffer_above[0], above, MPI_DOUBLE, rank - 1, 0, mpi_comm,
             MPI_STATUS_IGNORE);
    LowerProduct(recv_buffer_above, temp_own);
    Hinv(temp_own, temp_own_2);
    input -= temp_own_2;
    if ((int)rank + 1 < GlobalParams.NumberProcesses) {
      MPI_Send(&input[0], own, MPI_DOUBLE, rank + 1, 0, mpi_comm);
    }
  }

  for (int i = 0; i < own; i++) {
    if (!fixed_dofs->is_element(indices[i])) {
      dst[indices[i]] = input[i];
    }
  }
}

void PreconditionerSweeping::vmult_slow(
    TrilinosWrappers::MPI::BlockVector &dst,
    const TrilinosWrappers::MPI::BlockVector &src) const {
  std::cout << "L2 norm before vmult:" << src.l2_norm() << std::endl;
  dealii::Vector<double> recv_buffer_above(above);
  dealii::Vector<double> recv_buffer_below(others);
  dealii::Vector<double> temp_own(own);
  dealii::Vector<double> temp_own_2(own);
  dealii::Vector<double> input(own);

  for (unsigned int i = 0; i < sweepable; i++) {
    input[i] = src[indices[i]];
  }

  if ((int)rank + 1 == GlobalParams.NumberProcesses) {
    solver->solve(input);
    MPI_Send(&input[0], own, MPI_DOUBLE, rank - 1, 0, mpi_comm);
  } else {
    MPI_Recv(&recv_buffer_below[0], others, MPI_DOUBLE, rank + 1, 0, mpi_comm,
             MPI_STATUS_IGNORE);
    UpperProduct(recv_buffer_below, temp_own);
    input -= temp_own;
    if (rank != 0) {
      Hinv(input, temp_own);
      MPI_Send(&temp_own[0], own, MPI_DOUBLE, rank - 1, 0, mpi_comm);
    }
  }
  std::cout << "L2 after sweep 1:" << input.l2_norm() << std::endl;
  if ((int)rank + 1 != GlobalParams.NumberProcesses) {
    for (int i = 0; i < own; i++) {
      temp_own[i] = input[i];
    }
    Hinv(temp_own, input);
  }
  std::cout << "L2 after application of inverse:" << input.l2_norm()
            << std::endl;
  if (rank == 0) {
    MPI_Send(&input[0], own, MPI_DOUBLE, rank + 1, 0, mpi_comm);
  } else {
    MPI_Recv(&recv_buffer_above[0], above, MPI_DOUBLE, rank - 1, 0, mpi_comm,
             MPI_STATUS_IGNORE);
    LowerProduct(recv_buffer_above, temp_own);
    Hinv(temp_own, temp_own_2);
    input -= temp_own_2;
    if ((int)rank + 1 < GlobalParams.NumberProcesses) {
      MPI_Send(&input[0], own, MPI_DOUBLE, rank + 1, 0, mpi_comm);
    }
  }

  for (int i = 0; i < own; i++) {
    if (!fixed_dofs->is_element(indices[i])) {
      dst[indices[i]] = input[i];
    }
  }
  std::cout << "L2 norm after vmult:" << dst.l2_norm() << std::endl;
}

void PreconditionerSweeping::Hinv(const dealii::Vector<double> &src,
                                  dealii::Vector<double> &dst) const {
  dealii::Vector<double> inputb(own + others);
  for (int i = 0; i < own; i++) {
    inputb[i] = src(i);
  }

  solver->solve(inputb);

  for (int i = 0; i < own; i++) {
    dst[i] = inputb[i];
  }
}

void PreconditionerSweeping::init(
    SolverControl, TrilinosWrappers::SparseMatrix *in_prec_upper,
    TrilinosWrappers::SparseMatrix *in_prec_lower) {
  deallog.push("Init Preconditioner");
  deallog << "Prepare Objects" << std::endl;
  solver = new SparseDirectUMFPACK();
  IndexSet local(matrix->m());
  local.add_range(0, matrix->m());
  dealii::SparsityPattern sparsity_pattern;
  dealii::SparseMatrix<double> *temp;
  // Main Matrix Preparation

  sparsity_pattern.reinit(own + others, own + others, bandwidth);
  TrilinosWrappers::SparseMatrix::iterator it = matrix->begin();
  TrilinosWrappers::SparseMatrix::iterator end = matrix->end();
  for (; it != end; it++) {
    sparsity_pattern.add(it->row(), it->column());
  }
  sparsity_pattern.compress();

  temp = new dealii::SparseMatrix<double>(sparsity_pattern);
  deallog << "Copy Matrix" << std::endl;
  temp->copy_from(*matrix);
  deallog << "Factorize Matrix" << std::endl;
  solver->factorize(*temp);
  std::cout << this->rank << " is done." << std::endl;
  temp->clear();
  temp = 0;
  // Prec Matrix lower Preparation
  deallog << "Prepare Lower Block" << std::endl;
  off_diag_block_lower.reinit(own, above, bandwidth);

  if (above != 0) {
    it = in_prec_lower->begin();
    end = in_prec_lower->end();
    for (; it != end; it++) {
      off_diag_block_lower.add(it->row(), it->column());
    }
  } else {
    off_diag_block_lower.reinit(own, own, 120);
    it = in_prec_lower->begin();
    end = in_prec_lower->end();
    for (; it != end; it++) {
      off_diag_block_lower.add(it->row(), it->column());
    }
  }
  off_diag_block_lower.compress();

  prec_matrix_lower = new dealii::SparseMatrix<double>(off_diag_block_lower);
  prec_matrix_lower->copy_from(*in_prec_lower);

  // Prec Matrix upper Preparation
  deallog << "Prepare Upper Block" << std::endl;
  off_diag_block_upper.reinit(own, others, bandwidth);

  if (others != 0) {
    it = in_prec_upper->begin();
    end = in_prec_upper->end();
    for (; it != end; it++) {
      off_diag_block_upper.add(it->row(), it->column());
    }
  } else {
    off_diag_block_upper.reinit(own, own, 120);
    it = in_prec_upper->begin();
    end = in_prec_upper->end();
    for (; it != end; it++) {
      off_diag_block_upper.add(it->row(), it->column());
    }
  }
  off_diag_block_upper.compress();

  prec_matrix_upper = new dealii::SparseMatrix<double>(off_diag_block_upper);
  prec_matrix_upper->copy_from(*in_prec_upper);
  deallog.pop();
}

void PreconditionerSweeping::UpperProduct(const dealii::Vector<double> &src,
                                          dealii::Vector<double> &dst) const {
  if ((int)rank + 1 == GlobalParams.NumberProcesses) {
    std::cout << "ERROR!" << std::endl;
  }

  prec_matrix_upper->vmult(dst, src);
}

void PreconditionerSweeping::LowerProduct(const dealii::Vector<double> &src,
                                          dealii::Vector<double> &dst) const {
  if ((int)rank == 0) {
    std::cout << "ERROR!" << std::endl;
  }

  prec_matrix_lower->vmult(dst, src);
}

#endif
