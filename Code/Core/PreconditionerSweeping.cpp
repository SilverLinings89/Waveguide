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
  std::cout << "I am " << GlobalParams.MPI_Rank << " and I am at 1."
            << std::endl;
  deallog << "N1: " << input.l2_norm() << std::endl;
  if ((int)rank + 1 == GlobalParams.NumberProcesses) {
    solver->solve(input);
    deallog << "N2a: " << input.l2_norm() << std::endl;
    MPI_Send(&input[0], own, MPI_DOUBLE, rank - 1, 0, mpi_comm);
    std::cout << "I am " << GlobalParams.MPI_Rank << " and I am at 2a."
              << std::endl;
  } else {
    MPI_Recv(&recv_buffer_below[0], others, MPI_DOUBLE, rank + 1, 0, mpi_comm,
             MPI_STATUS_IGNORE);
    std::cout << "I am " << GlobalParams.MPI_Rank << " and I am at 2b."
              << std::endl;
    UpperProduct(recv_buffer_below, temp_own);
    std::cout << "I am " << GlobalParams.MPI_Rank << " and I am at 2c."
              << std::endl;
    input -= temp_own;
    deallog << "N2b: " << input.l2_norm() << std::endl;
    if (rank != 0) {
      Hinv(input, temp_own);
      std::cout << "I am " << GlobalParams.MPI_Rank << " and I am at 2d."
                << std::endl;
      MPI_Send(&temp_own[0], own, MPI_DOUBLE, rank - 1, 0, mpi_comm);
      std::cout << "I am " << GlobalParams.MPI_Rank << " and I am at 2e."
                << std::endl;
    }
  }
  std::cout << "I am " << GlobalParams.MPI_Rank << " and I am at 3."
            << std::endl;
  if ((int)rank + 1 != GlobalParams.NumberProcesses) {
    for (int i = 0; i < own; i++) {
      temp_own[i] = input[i];
    }
    Hinv(temp_own, input);
  }
  std::cout << "I am " << GlobalParams.MPI_Rank << " and I am at 4."
            << std::endl;
  deallog << "N3: " << input.l2_norm() << std::endl;
  if (rank == 0) {
    MPI_Send(&input[0], own, MPI_DOUBLE, rank + 1, 0, mpi_comm);
    std::cout << "I am " << GlobalParams.MPI_Rank << " and I am at 5a."
              << std::endl;
  } else {
    MPI_Recv(&recv_buffer_above[0], above, MPI_DOUBLE, rank - 1, 0, mpi_comm,
             MPI_STATUS_IGNORE);
    std::cout << "I am " << GlobalParams.MPI_Rank << " and I am at 5b."
              << std::endl;
    double recv_norm = 0.0;
    for (unsigned int i = 0; i < above; i++) {
      recv_norm += std::abs(recv_buffer_above[i]);
    }
    deallog << "Recv_Norm: " << recv_norm << std::endl;
    LowerProduct(recv_buffer_above, temp_own);
    std::cout << "I am " << GlobalParams.MPI_Rank << " and I am at 5c."
              << std::endl;
    deallog << "After L Product: " << temp_own.l2_norm() << std::endl;
    Hinv(temp_own, temp_own_2);
    std::cout << "I am " << GlobalParams.MPI_Rank << " and I am at 5d."
              << std::endl;
    deallog << "After Hinv: " << temp_own_2.l2_norm() << std::endl;
    input -= temp_own_2;
    deallog << "N4: " << input.l2_norm() << std::endl;
    if ((int)rank + 1 < GlobalParams.NumberProcesses) {
      MPI_Send(&input[0], own, MPI_DOUBLE, rank + 1, 0, mpi_comm);
    }
    std::cout << "I am " << GlobalParams.MPI_Rank << " and I am at 5e."
              << std::endl;
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
  std::cout << "Delta: " << delta << " from " << GlobalParams.MPI_Rank
            << std::endl;
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
  std::cout << this->rank << " is done." << std::endl;
  temp->clear();
  temp = 0;
  // Prec Matrix lower Preparation
  deallog << "Prepare Lower Block" << std::endl;
  off_diag_block_lower.reinit(own, above, bandwidth);
  dealii::DynamicSparsityPattern dsp(own, above);

  it = in_prec_lower->begin();
  end = in_prec_lower->end();
  for (; it != end; it++) {
    dsp.add(it->row(), it->column());
  }
  off_diag_block_lower.copy_from(dsp);

  prec_matrix_lower = new dealii::SparseMatrix<double>(off_diag_block_lower);
  it = in_prec_lower->begin();
  end = in_prec_lower->end();
  int cnt_lower = 0;
  double val_lower = 0.0;
  for (; it != end; it++) {
    prec_matrix_lower->set(it->row(), it->column(), it->value());
    cnt_lower++;
    val_lower += std::abs(it->value());
  }

  deallog << "Norm of the lower off_diag block: "
          << prec_matrix_lower->linfty_norm() << " copied " << cnt_lower
          << " entries with sum " << val_lower << std::endl;

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
  it = in_prec_upper->begin();
  end = in_prec_upper->end();
  for (; it != end; it++) {
    prec_matrix_upper->set(it->row(), it->column(), it->value());
  }
  // prec_matrix_upper->compress(VectorOperation::unknown);

  deallog.pop();
}

void PreconditionerSweeping::UpperProduct(const dealii::Vector<double> &src,
                                          dealii::Vector<double> &dst) const {
  if ((int)rank + 1 == GlobalParams.NumberProcesses) {
    std::cout << "ERROR!" << std::endl;
  }

  prec_matrix_upper->vmult(dst, src);
  deallog   << " computed upper product with result " << dst.l2_norm()
            << " input norm was " << src.l2_norm() << std::endl;

  deallog << "src smallest index: " << src.locally_owned_elements().nth_index_in_set(0)<< std::endl;
  deallog << "src largest index: " << src.locally_owned_elements().nth_index_in_set(src.locally_owned_elements().n_elements()-1)<< std::endl;
  deallog << "dst smallest index: " << dst.locally_owned_elements().nth_index_in_set(0)<< std::endl;
  deallog << "dst largest index: " << dst.locally_owned_elements().nth_index_in_set(dst.locally_owned_elements().n_elements()-1)<< std::endl;
  deallog << "Matrix first entry: Col: " << prec_matrix_upper->begin()->column() << " Row: " << prec_matrix_upper->begin()->row() << " Value " << prec_matrix_upper->begin()->value() << std::endl;

}

void PreconditionerSweeping::LowerProduct(const dealii::Vector<double> &src,
                                          dealii::Vector<double> &dst) const {
  if ((int)rank == 0) {
    std::cout << "ERROR!" << std::endl;
  }

  prec_matrix_lower->vmult(dst, src);
  std::cout << "Process " << GlobalParams.MPI_Rank
            << " computed lower product with result " << dst.l2_norm()
            << " input norm was " << src.l2_norm() << std::endl;
  deallog << "src smallest index: " << src.locally_owned_elements().nth_index_in_set(0)<< std::endl;
    deallog << "src largest index: " << src.locally_owned_elements().nth_index_in_set(src.locally_owned_elements().n_elements()-1)<< std::endl;
    deallog << "dst smallest index: " << dst.locally_owned_elements().nth_index_in_set(0)<< std::endl;
    deallog << "dst largest index: " << dst.locally_owned_elements().nth_index_in_set(dst.locally_owned_elements().n_elements()-1)<< std::endl;
    deallog << "Matrix first entry: Col: " << prec_matrix_upper->begin()->column() << " Row: " << prec_matrix_upper->begin()->row() << " Value " << prec_matrix_upper->begin()->value() << std::endl;
}

#endif
