//
// Created by pascal on 03.02.20.
//

#include "HierarchicalProblem.h"
#include "../Helpers/Parameters.h"
#include "../Core/Types.h"
#include "../Helpers/staticfunctions.h"

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
    print_info("HierarchicalProblem::constrain_identical_dof_sets", "There was an error in constrain_identical_dof_sets. No changes made.", false, LoggingLevel::PRODUCTION_ALL);
  }

  for (unsigned int index = 0; index < n_entries; index++) {
    affine_constraints->add_line(set_one->operator [](index));
    affine_constraints->add_entry(set_one->operator [](index),
        set_two->operator [](index), ComplexNumber(-1, 0));
  }
}

auto HierarchicalProblem::opposing_site_bid(BoundaryId in_bid) -> BoundaryId {
  if((in_bid % 2) == 0) {
    return in_bid + 1;
  }
  else {
    return in_bid - 1;
  }
}
