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

PreconditionerSweeping::PreconditionerSweeping(
    MPI_Comm in_mpi_comm, int in_own, int in_others, int in_above,
    unsigned int interface, int in_bandwidth, IndexSet in_locally_owned_dofs,
    IndexSet *in_fixed_dofs, int in_rank) {
  interface_dof_count = interface;
  locally_owned_dofs = in_locally_owned_dofs;
  own = in_own;
  others = in_others;
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
}

void PreconditionerSweeping::Prepare(TrilinosWrappers::MPI::BlockVector &inp) {
  boundary.reinit(own, false);
  for (int i = 0; i < own; i++) {
    boundary[i] = inp[i];
  }
}

void PreconditionerSweeping::vmult(
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
  deallog << "N1: " << input.l2_norm() << std::endl;
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
  if ((int)rank + 1 != GlobalParams.NumberProcesses) {
    for (int i = 0; i < own; i++) {
      temp_own[i] = input[i];
    }
    Hinv(temp_own, input);
  }
  if (rank == 0) {
    MPI_Send(&input[0], own, MPI_DOUBLE, rank + 1, 0, mpi_comm);
  } else {
    MPI_Recv(&recv_buffer_above[0], above, MPI_DOUBLE, rank - 1, 0, mpi_comm,
             MPI_STATUS_IGNORE);
    double recv_norm = 0.0;
    for (unsigned int i = 0; i < above; i++) {
      recv_norm += std::abs(recv_buffer_above[i]);
    }
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

  double delta = 0;
  for (unsigned int i = 0; i < own; i++) {
    if (!fixed_dofs->is_element(indices[i])) {
      delta += std::abs(dst[indices[i]] - src[indices[i]]);
    }
  }
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

void PreconditionerSweeping::init(SolverControl,
                                  dealii::SparseMatrix<double> *in_prec_upper,
                                  dealii::SparseMatrix<double> *in_prec_lower) {
  deallog.push("Init Preconditioner");
  deallog << "Prepare Objects" << std::endl;
  solver = new SparseDirectUMFPACK();
  IndexSet local(matrix->m());
  deallog << "Found m = " << matrix->m() << std::endl;
  local.add_range(0, matrix->m());
  dealii::DynamicSparsityPattern sparsity_pattern;
  dealii::SparseMatrix<double> *temp;
  // Main Matrix Preparation

  sparsity_pattern.reinit(own + others, own + others);
  TrilinosWrappers::SparseMatrix::iterator it = matrix->begin();
  TrilinosWrappers::SparseMatrix::iterator end = matrix->end();
  int cnt = 0;
  for (; it != end; it++) {
    sparsity_pattern.add(it->row(), it->column());
    cnt++;
  }
  sparsity_pattern.compress();
  deallog << "Added " << cnt << " entries to sp." << std::endl;
  SparsityPattern tsp;
  tsp.copy_from(sparsity_pattern);
  temp = new dealii::SparseMatrix<double>(tsp);
  deallog << "Copy Matrix" << std::endl;
  temp->copy_from(*matrix);
  deallog << "Factorize Matrix" << std::endl;
  solver->factorize(*temp);
  temp->clear();
  temp = 0;

  prec_matrix_lower = in_prec_lower;
  prec_matrix_upper = in_prec_upper;
  std::ofstream matrix_data1(
      "temp/Process" + std::to_string(rank) + "-upper.dat", std::ofstream::out);
  prec_matrix_upper->print(matrix_data1);
  std::ofstream matrix_data2(
      "temp/Process" + std::to_string(rank) + "-lower.dat", std::ofstream::out);
  prec_matrix_lower->print(matrix_data2);
  deallog.pop();
}

void PreconditionerSweeping::init(
    SolverControl, TrilinosWrappers::SparseMatrix *in_prec_upper,
    TrilinosWrappers::SparseMatrix *in_prec_lower) {
  deallog.push("Init Preconditioner");
  deallog << "Prepare Objects" << std::endl;
  solver = new SparseDirectUMFPACK();
  IndexSet local(matrix->m());
  deallog << "Found m = " << matrix->m() << std::endl;
  local.add_range(0, matrix->m());
  dealii::SparsityPattern sparsity_pattern;
  dealii::SparseMatrix<double> *temp;
  // Main Matrix Preparation

  sparsity_pattern.reinit(own + others, own + others, bandwidth);
  TrilinosWrappers::SparseMatrix::iterator it = matrix->begin();
  TrilinosWrappers::SparseMatrix::iterator end = matrix->end();
  int cnt = 0;
  for (; it != end; it++) {
    sparsity_pattern.add(it->row(), it->column());
    cnt++;
  }
  sparsity_pattern.compress();
  deallog << "Added " << cnt << " entries to sp." << std::endl;
  temp = new dealii::SparseMatrix<double>(sparsity_pattern);
  deallog << "Copy Matrix" << std::endl;
  temp->copy_from(*matrix);
  deallog << "Factorize Matrix" << std::endl;
  solver->factorize(*temp);
  temp->clear();
  temp = 0;

  // Prec Matrix lower Preparation
  deallog << "Prepare Lower Block" << std::endl;
  if (above != 0) {
    off_diag_block_lower.reinit(own, above, bandwidth);
  } else {
    off_diag_block_lower.reinit(own, own, bandwidth);
  }

  it = in_prec_lower->begin();
  end = in_prec_lower->end();
  for (; it != end; it++) {
    off_diag_block_lower.add(it->row(), it->column());
  }
  off_diag_block_lower.compress();

  prec_matrix_lower = new dealii::SparseMatrix<double>(off_diag_block_lower);
  it = in_prec_lower->begin();
  end = in_prec_lower->end();
  for (; it != end; it++) {
    prec_matrix_lower->set(it->row(), it->column(),
                           it->value().operator double());
  }
  prec_matrix_lower->compress(dealii::VectorOperation::unknown);
  deallog << "Norm of lower: " << prec_matrix_lower->l1_norm() << std::endl;

  // Prec Matrix upper Preparation
  deallog << "Prepare Upper Block" << std::endl;
  if (others != 0) {
    off_diag_block_upper.reinit(own, others, bandwidth);
    it = in_prec_upper->begin();
    end = in_prec_upper->end();
    for (; it != end; it++) {
      off_diag_block_upper.add(it->row(), it->column());
    }
  } else {
    off_diag_block_upper.reinit(own, own, bandwidth);
    it = in_prec_upper->begin();
    end = in_prec_upper->end();
    for (; it != end; it++) {
      off_diag_block_upper.add(it->row(), it->column());
    }
  }
  off_diag_block_upper.compress();

  prec_matrix_upper = new dealii::SparseMatrix<double>(off_diag_block_upper);
  it = in_prec_upper->begin();
  end = in_prec_upper->end();
  for (; it != end; it++) {
    prec_matrix_upper->set(it->row(), it->column(),
                           it->value().operator double());
  }
  prec_matrix_upper->compress(dealii::VectorOperation::unknown);
  deallog << "Norm of upper: " << prec_matrix_upper->l1_norm() << std::endl;
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
