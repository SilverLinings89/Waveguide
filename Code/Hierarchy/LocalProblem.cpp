//
// Created by pascal on 03.02.20.
//

#include "LocalProblem.h"

LocalProblem::LocalProblem(unsigned int , unsigned int global_level,
    DOFManager *dof_manager) :
    HierarchicalProblem(0, global_level, dof_manager) {
}

void LocalProblem::solve() {}

void LocalProblem::initialize() {}

void LocalProblem::generate_sparsity_pattern() {
}

dealii::IndexSet LocalProblem::get_owned_dofs_for_level(unsigned int level) {
  unsigned int n_owned_dofs = compute_own_dofs();
  if (level >= 2) {
    n_owned_dofs += 1;  // TODO: implement this.
  }
  if (level >= 1) {
    n_owned_dofs += 1; // TODO: implement this.
  }
  return dealii::IndexSet(n_owned_dofs);
}

LocalProblem* LocalProblem::get_local_problem() {
  return this;
}
