//
// Created by pascal on 03.02.20.
//

#ifndef WAVEGUIDEPROBLEM_NONLOCALPROBLEM_H
#define WAVEGUIDEPROBLEM_NONLOCALPROBLEM_H

#include <mpi.h>
#include "../Helpers/Parameters.h"
#include "HierarchicalProblem.h"
#include "./LocalProblem.h"

class NonLocalProblem : public HierarchicalProblem {
  unsigned int compute_own_dofs();
  MPI_Comm level_communicator;

  void initialize_MPI_communicator_for_level();

 public:
  NonLocalProblem(unsigned int, unsigned int, DOFManager*, MPI_Comm);

  void solve() override;

  void initialize() override;

  void generate_sparsity_pattern() override;

  void initialize_index_sets() override;

  IndexSet get_owned_dofs_for_level(unsigned int level) override;

  LocalProblem* get_local_problem() override;
};

#endif  // WAVEGUIDEPROBLEM_NONLOCALPROBLEM_H
