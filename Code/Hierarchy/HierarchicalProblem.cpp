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


  // Edge constraints
  for(unsigned int i = 0; i< 6; i++) {
    for(unsigned int j = i+1; j < 6; j++) {
      if(!are_opposing_sites(i, j)) {
        if(is_absorbing_boundary(Geometry.levels[level].surface_type[i]) && is_absorbing_boundary(Geometry.levels[level].surface_type[j])) {
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

std::string HierarchicalProblem::output_results(std::string in_fname_part) {
  print_info("Hierarchical::output_results()", "Start on level " + std::to_string(level));
  std::string ret = "";
  NumericVectorLocal in_solution(Geometry.inner_domain->n_dofs);
  for(unsigned int i = 0; i < Geometry.inner_domain->n_dofs; i++) {
    in_solution[i] = solution[Geometry.levels[level].inner_first_dof + i];
  }
  std::string file_1 = Geometry.inner_domain->output_results(in_fname_part + std::to_string(level) , in_solution);
  ret = file_1;
  filenames.clear();
  filenames.push_back(file_1);

  if(GlobalParams.BoundaryCondition == BoundaryConditionType::PML) {
    for(unsigned int i = 0; i < 6; i++){
      if(Geometry.levels[level].surface_type[i] == SurfaceType::ABC_SURFACE){
        dealii::Vector<ComplexNumber> ds (Geometry.levels[level].surfaces[i]->dof_counter);
        for(unsigned int index = 0; index < Geometry.levels[level].surfaces[i]->dof_counter; index++) {
          ds[index] = solution(index + Geometry.levels[level].surface_first_dof[i]);
        }
        std::string file_2 = Geometry.levels[level].surfaces[i]->output_results(ds, "pml_domain" + std::to_string(level));
        filenames.push_back(file_2);
      }
    }
  }

  // End of core output

  for(unsigned int i = 0; i < Geometry.inner_domain->n_dofs; i++) {
    in_solution[i] = solution_error[Geometry.levels[level].inner_first_dof + i];
  }
  Geometry.inner_domain->output_results("error" + std::to_string(level) , in_solution);

  if(GlobalParams.BoundaryCondition == BoundaryConditionType::PML) {
    for(unsigned int i = 0; i < 6; i++){
      if(Geometry.levels[level].is_surface_truncated[i]){
        dealii::Vector<ComplexNumber> ds (Geometry.levels[level].surfaces[i]->dof_counter);
        for(unsigned int index = 0; index < Geometry.levels[level].surfaces[i]->dof_counter; index++) {
          ds[index] = solution_error(index + Geometry.levels[level].surface_first_dof[i]);
        }
        Geometry.levels[level].surfaces[i]->output_results(ds, "error_in_pml" + std::to_string(level));
      }
    }
  }

  if(level != 0) {
  //  child->output_results();
  }

  print_info("Hierarchical::output_results()", "End on level " + std::to_string(level));
  return ret;
}


void HierarchicalProblem::execute_vmult() {
  // constraints.set_zero(solution);
  NumericVectorDistributed temp_solution;
  temp_solution.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
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