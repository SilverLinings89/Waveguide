#include "HierarchicalProblem.h"
#include <string>
#include "../Helpers/Parameters.h"
#include "../Core/Types.h"
#include "../Helpers/staticfunctions.h"
#include "../Core/InnerDomain.h"

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

  for(unsigned int i = 0; i< 6; i++) {
    for(unsigned int j = i+1; j < 6; j++) {
      if(!are_opposing_sites(i, j)) {
        if((!Geometry.levels[level].surface_type[i] == SurfaceType::OPEN_SURFACE) && (!Geometry.levels[level].surface_type[j] == SurfaceType::OPEN_SURFACE)) {
          Geometry.levels[level].surfaces[i]->make_edge_constraints(&constraints, j);
        }
      }
    }
  }
  /**
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
  **/
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

void HierarchicalProblem::output_results() {
  print_info("Hierarchical::output_results()", "Start on level " + std::to_string(level));
  compute_final_rhs_mismatch();
  NumericVectorLocal in_solution(Geometry.inner_domain->n_dofs);
  for(unsigned int i = 0; i < Geometry.inner_domain->n_dofs; i++) {
    in_solution[i] = solution[Geometry.levels[level].inner_first_dof + i];
  }
  Geometry.inner_domain->output_results("solution_inner_domain_level" + std::to_string(level) , in_solution);

  if(GlobalParams.BoundaryCondition == BoundaryConditionType::PML) {
    for(unsigned int i = 0; i < 6; i++){
      if(Geometry.levels[level].is_surface_truncated[i]){
        dealii::Vector<ComplexNumber> ds (Geometry.levels[level].surfaces[i]->dof_counter);
        for(unsigned int index = 0; index < Geometry.levels[level].surfaces[i]->dof_counter; index++) {
          ds[index] = solution(index + Geometry.levels[level].surface_first_dof[i]);
        }
        Geometry.levels[level].surfaces[i]->output_results(ds, "PML_domain_level_" + std::to_string(level));
      }
    }
  }

  for(unsigned int i = 0; i < Geometry.inner_domain->n_dofs; i++) {
    in_solution[i] = final_rhs_mismatch[Geometry.levels[level].inner_first_dof + i];
  }
  Geometry.inner_domain->output_results("rhs_mismatch" + std::to_string(level) , in_solution);

  if(GlobalParams.BoundaryCondition == BoundaryConditionType::PML) {
    for(unsigned int i = 0; i < 6; i++){
      if(Geometry.levels[level].is_surface_truncated[i]){
        dealii::Vector<ComplexNumber> ds (Geometry.levels[level].surfaces[i]->dof_counter);
        for(unsigned int index = 0; index < Geometry.levels[level].surfaces[i]->dof_counter; index++) {
          ds[index] = final_rhs_mismatch(index + Geometry.levels[level].surface_first_dof[i]);
        }
        Geometry.levels[level].surfaces[i]->output_results(ds, "PML_domain_rhs_mismatch" + std::to_string(level));
      }
    }
  }

  if(level != 0) {
    child->output_results();
  }

  print_info("Hierarchical::output_results()", "End on level " + std::to_string(level));
}

void HierarchicalProblem::compute_final_rhs_mismatch() {
  matrix->vmult(final_rhs_mismatch, solution);
  final_rhs_mismatch -= rhs;
}

void HierarchicalProblem::execute_vmult() {
  matrix->vmult(temp_solution, solution);
  solution = temp_solution;
}