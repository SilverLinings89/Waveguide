//
// Created by pascal on 03.02.20.
//

#include "HierarchicalProblem.h"
#include "../Helpers/Parameters.h"

HierarchicalProblem::~HierarchicalProblem() {
  delete dof_manager;
}

HierarchicalProblem::HierarchicalProblem(unsigned int in_own_level,
    unsigned int in_global_level)
    : global_level(in_global_level), local_level(in_own_level) {
  is_dof_manager_set = false;
  dof_manager = nullptr;
  has_child = in_own_level > 0;
  child = nullptr;
}

void HierarchicalProblem::setup_dof_manager(DOFManager *in_dof_manager) {
  dof_manager = in_dof_manager;
  is_dof_manager_set = true;
}


