//
// Created by pascal on 03.02.20.
//

#include "HierarchicalProblem.h"
#include "../Helpers/Parameters.h"

HierarchicalProblem::~HierarchicalProblem() {
  delete dof_manager;
}

HierarchicalProblem::HierarchicalProblem(unsigned int in_own_level) :
    local_level(in_own_level) {
  is_dof_manager_set = false;
  dof_manager = nullptr;
  has_child = in_own_level > 0;
  child = nullptr;
  matrix = nullptr;
  n_own_dofs = 0;
  first_own_index = 0;
  for (unsigned int i = 0; i < 6; i++) {
    surface_first_dofs.push_back(0);
  }
  dofs_process_above = 0;
  dofs_process_below = 0;
  rank = 0;
  n_procs_in_sweep = 0;
}

void HierarchicalProblem::setup_dof_manager(DOFManager *in_dof_manager) {
  dof_manager = in_dof_manager;
  is_dof_manager_set = true;
}

void HierarchicalProblem::constrain_identical_dof_sets(
    std::vector<unsigned int> *set_one, std::vector<unsigned int> *set_two,
    dealii::AffineConstraints<double> *affine_constraints) {
  const unsigned int n_entries = set_one->size();
  if (n_entries != set_two->size()) {
    std::cout
        << "There was an error in constrain_identical_dof_sets. No changes made."
        << std::endl;
  }

  for (unsigned int index = 0; index < n_entries; index++) {
    affine_constraints->add_line(set_one->operator [](index));
    affine_constraints->add_entry(set_one->operator [](index),
        set_two->operator [](index), -1);
  }
}
