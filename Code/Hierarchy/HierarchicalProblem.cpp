#include "HierarchicalProblem.h"
#include "../Helpers/Parameters.h"
#include "../Core/Types.h"
#include "../Helpers/staticfunctions.h"
#include "../Core/NumericProblem.h"

HierarchicalProblem::~HierarchicalProblem() { }

HierarchicalProblem::HierarchicalProblem(unsigned int in_own_level, SweepingDirection in_direction) :
  sweeping_direction(in_direction),
  level(in_own_level) {
  has_child = in_own_level > 0;
  child = nullptr;
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

void HierarchicalProblem::make_constraints() {
  print_info("HierarchicalProblem::make_constraints", "Start");

  IndexSet total_dofs_global(Geometry.levels[level].n_total_level_dofs);
  total_dofs_global.add_range(0,Geometry.levels[level].n_total_level_dofs);
  constraints.reinit(total_dofs_global);
  
  // Inner constraints
  Geometry.inner_domain->make_constraints(&constraints, Geometry.levels[level].inner_first_dof, own_dofs);

  // Surface constraints
  for(unsigned int i = 0; i < 6; i++) {
    Geometry.levels[level].surfaces[i]->make_surface_constraints(&constraints);
  }

  // Edge constraints
  Geometry.levels[level].surfaces[4]->make_edge_constraints(&constraints, 0);
  Geometry.levels[level].surfaces[4]->make_edge_constraints(&constraints, 1);
  Geometry.levels[level].surfaces[4]->make_edge_constraints(&constraints, 2);
  Geometry.levels[level].surfaces[4]->make_edge_constraints(&constraints, 3);
  Geometry.levels[level].surfaces[2]->make_edge_constraints(&constraints, 0);
  Geometry.levels[level].surfaces[2]->make_edge_constraints(&constraints, 1);
  Geometry.levels[level].surfaces[3]->make_edge_constraints(&constraints, 0);
  Geometry.levels[level].surfaces[3]->make_edge_constraints(&constraints, 1);
  Geometry.levels[level].surfaces[5]->make_edge_constraints(&constraints, 0);
  Geometry.levels[level].surfaces[5]->make_edge_constraints(&constraints, 1);
  Geometry.levels[level].surfaces[5]->make_edge_constraints(&constraints, 2);
  Geometry.levels[level].surfaces[5]->make_edge_constraints(&constraints, 3);

  constraints.close();
  
  print_info("HierarchicalProblem::make_constraints", "End");
}

void HierarchicalProblem::make_sparsity_pattern() {
  print_info("HierarchicalProblem::make_sparsity_patter", "Start on level "  + std::to_string(level));
  dealii::DynamicSparsityPattern dsp = {Geometry.levels[level].n_total_level_dofs, Geometry.levels[level].n_total_level_dofs};
  dealii::IndexSet is(Geometry.levels[level].n_total_level_dofs);
  is.add_range(0, Geometry.levels[level].n_total_level_dofs);
  
  Geometry.inner_domain->make_sparsity_pattern(&dsp, Geometry.levels[level].inner_first_dof, &constraints);
  for (unsigned int surface = 0; surface < 6; surface++) {
    Geometry.levels[level].surfaces[surface]->fill_sparsity_pattern(&dsp, &constraints);
  }
  
  sp.copy_from(dsp);
  sp.compress();
  print_info("HierarchicalProblem::make_sparsity_patter", "End on level "  + std::to_string(level));
}