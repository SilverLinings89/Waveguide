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
    Constraints *affine_constraints) {
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
    bool has_inhomogeneity = (GlobalParams.Index_in_z_direction == 0 && i == 4) && (GlobalParams.Signal_coupling_method == SignalCouplingMethod::Jump);
    Geometry.levels[level].surfaces[i]->make_surface_constraints(&constraints, has_inhomogeneity);
  }

  if(GlobalParams.Signal_coupling_method == SignalCouplingMethod::Dirichlet) {
    if(GlobalParams.Index_in_z_direction == 0) {
      IndexSet owned_dofs(Geometry.inner_domain->dof_handler.n_dofs());
      owned_dofs.add_range(0, Geometry.inner_domain->dof_handler.n_dofs());
      AffineConstraints<ComplexNumber> constraints_local(owned_dofs);
      VectorTools::project_boundary_values_curl_conforming_l2(Geometry.inner_domain->dof_handler, 0, *GlobalParams.source_field , 4, constraints_local);
      constraints_local.shift(Geometry.levels[level].inner_first_dof);
      constraints.merge(constraints_local, Constraints::MergeConflictBehavior::right_object_wins, true);
    }
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

std::string HierarchicalProblem::output_results() {
  print_info("Hierarchical::output_results()", "Start on level " + std::to_string(level));
  std::string ret = "";
  compute_final_rhs_mismatch();
  NumericVectorLocal in_solution(Geometry.inner_domain->n_dofs);
  for(unsigned int i = 0; i < Geometry.inner_domain->n_dofs; i++) {
    in_solution[i] = solution[Geometry.levels[level].inner_first_dof + i];
  }
  std::string file_1 = Geometry.inner_domain->output_results("solution_inner_domain_level" + std::to_string(level) , in_solution);
  ret = file_1;
  filenames.clear();
  filenames.push_back(file_1);

  if(GlobalParams.BoundaryCondition == BoundaryConditionType::PML) {
    for(unsigned int i = 0; i < 6; i++){
      if(Geometry.levels[level].is_surface_truncated[i]){
        dealii::Vector<ComplexNumber> ds (Geometry.levels[level].surfaces[i]->dof_counter);
        for(unsigned int index = 0; index < Geometry.levels[level].surfaces[i]->dof_counter; index++) {
          ds[index] = solution(index + Geometry.levels[level].surface_first_dof[i]);
        }
        std::string file_2 = Geometry.levels[level].surfaces[i]->output_results(ds, "PML_domain_level_" + std::to_string(level));
        filenames.push_back(file_2);
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
  return ret;
}

void HierarchicalProblem::compute_final_rhs_mismatch() {
  matrix->vmult(final_rhs_mismatch, solution);
  final_rhs_mismatch -= rhs;
}

void HierarchicalProblem::execute_vmult() {
  // constraints.set_zero(solution);
  matrix->vmult(temp_solution, solution);
  solution = temp_solution;
  // constraints.distribute(solution);
}

void HierarchicalProblem::compute_rhs_representation_of_incoming_wave() {
  // reinit_rhs();
  /**
  NumericVectorLocal temp(Geometry.inner_domain->dof_handler.n_dofs());
  if(GlobalParams.Index_in_z_direction == 0) {
    // there can be an incoming signal here
    IndexSet owned_dofs(Geometry.inner_domain->dof_handler.n_dofs());
    owned_dofs.add_range(0, Geometry.inner_domain->dof_handler.n_dofs());
    AffineConstraints<ComplexNumber> constraints_local(owned_dofs);
    VectorTools::project_boundary_values_curl_conforming_l2(Geometry.inner_domain->dof_handler, 0, *GlobalParams.source_field , 4, constraints_local);
    for(unsigned int i =0; i < Geometry.inner_domain->dof_handler.n_dofs(); i++) {
      if(constraints_local.is_inhomogeneously_constrained(i)) {
        temp[i] = constraints_local.get_inhomogeneity(i);
      }
    }
    NumericVectorLocal vmult_result = Geometry.inner_domain->vmult(temp);
    for(unsigned int i = 0; i < Geometry.inner_domain->dof_handler.n_dofs(); i++) {
      if(constraints_local.is_inhomogeneously_constrained(i)) {
        vmult_result[i] = ComplexNumber(0,0);
      }
    }
    std::vector<unsigned int> indices;
    for(unsigned int i = 0; i < Geometry.inner_domain->dof_handler.n_dofs(); i++) {
      indices.push_back(i + Geometry.levels[level].inner_first_dof);
    }
    rhs.add(indices,vmult_result);
    rhs.compress(VectorOperation::add);
  } else {
    // there cannot be an incoming signal here
    rhs.compress(VectorOperation::add);
  }
  **/
}