//
// Created by pascal on 03.02.20.
//

#include "LocalProblem.h"
#include "../Helpers/Parameters.h"

LocalProblem::LocalProblem(unsigned int , unsigned int global_level,
                           DOFManager* dof_manager, Parameters* params)
    : HierarchicalProblem(0, global_level, dof_manager, params) {}

unsigned int LocalProblem::compute_own_dofs() {
  return dof_manager->compute_n_own_dofs();
}

void LocalProblem::compute_level_dofs_total() {}

void LocalProblem::solve() {}

void LocalProblem::initialize() {}

void LocalProblem::generateSparsityPattern() {}

IndexSet LocalProblem::get_owned_dofs_for_level(unsigned int level) {
  unsigned int n_owned_dofs = compute_own_dofs();
  if (level >= 2) {
    n_owned_dofs += 1;  // TODO: implement this.
  }
  if (level >= 1) {
    n_owned_dofs += 1; // TODO: implement this.
  }
  return IndexSet(n_owned_dofs);
}
