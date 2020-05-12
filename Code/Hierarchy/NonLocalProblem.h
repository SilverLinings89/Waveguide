//
// Created by pascal on 03.02.20.
//

#ifndef WAVEGUIDEPROBLEM_NONLOCALPROBLEM_H
#define WAVEGUIDEPROBLEM_NONLOCALPROBLEM_H

#include <mpi.h>
#include "HierarchicalProblem.h"
#include "./LocalProblem.h"

class NonLocalProblem: public HierarchicalProblem {
  unsigned int compute_own_dofs();

  void initialize_MPI_communicator_for_level();

 public:
  NonLocalProblem(unsigned int, unsigned int);
  ~NonLocalProblem() override;

  unsigned int compute_lower_interface_dof_count() override;

  unsigned int compute_upper_interface_dof_count() override;

  void assemble() override;

  void solve() override;

  void initialize() override;

  void generate_sparsity_pattern() override;

  void initialize_index_sets() override;

  void apply_sweep(dealii::LinearAlgebra::distributed::Vector<double>) override;

  LocalProblem* get_local_problem() override;
};

#endif  // WAVEGUIDEPROBLEM_NONLOCALPROBLEM_H
