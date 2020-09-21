//
// Created by pascal on 03.02.20.
//

#include "HierarchicalProblem.h"
#include "../Helpers/Parameters.h"

HierarchicalProblem::~HierarchicalProblem() { }

HierarchicalProblem::HierarchicalProblem(unsigned int in_own_level) :
    local_level(in_own_level) {
  has_child = in_own_level > 0;
  child = nullptr;
  n_own_dofs = 0;
  first_own_index = 0;
  for (unsigned int i = 0; i < 6; i++) {
    surface_first_dofs.push_back(0);
  }
  dofs_process_above = 0;
  dofs_process_below = 0;
  rank = 0;
  n_procs_in_sweep = 0;
  for(unsigned int i = 0; i < 6; i++) {
    is_surface_locked.push_back(false);
  }
}

void HierarchicalProblem::constrain_identical_dof_sets(
    std::vector<unsigned int> *set_one, std::vector<unsigned int> *set_two,
    dealii::AffineConstraints<ComplexNumber> *affine_constraints) {
  const unsigned int n_entries = set_one->size();
  if (n_entries != set_two->size()) {
    std::cout
        << "There was an error in constrain_identical_dof_sets. No changes made."
        << std::endl;
  }

  for (unsigned int index = 0; index < n_entries; index++) {
    affine_constraints->add_line(set_one->operator [](index));
    affine_constraints->add_entry(set_one->operator [](index),
        set_two->operator [](index), ComplexNumber(-1, 0));
  }
}

void HierarchicalProblem::lock_boundary_dofs(BoundaryId in_bid) {
  if(in_bid < 6) {
    is_surface_locked[in_bid] = true;
    child->lock_boundary_dofs(in_bid);
  }
}

void HierarchicalProblem::unlock_boundary_dofs(BoundaryId in_bid) {
  if(in_bid < 6) {
    is_surface_locked[in_bid] = false;
    child->unlock_boundary_dofs(in_bid);
  }
}
