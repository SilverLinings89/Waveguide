#include "HierarchicalProblem.h"
#include <string>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <sstream>
#include "../Helpers/Parameters.h"
#include "../Core/Types.h"
#include "../Helpers/staticfunctions.h"
#include "../Core/InnerDomain.h"
#include "../Core/Enums.h"
#include "../BoundaryCondition/NeighborSurface.h"

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
  for(unsigned int i = 0; i < 6; i++) {
    Constraints local_constraints = Geometry.levels[level].surfaces[i]->make_constraints();
    constraints.merge(local_constraints, Constraints::MergeConflictBehavior::left_object_wins,true);
  }
  constraints.close();
  
  print_info("HierarchicalProblem::make_constraints", "End");
}

void HierarchicalProblem::make_sparsity_pattern() {
  print_info("HierarchicalProblem::make_sparsity_pattern", "Start on level "  + std::to_string(level));
  dealii::DynamicSparsityPattern dsp = {Geometry.levels[level].n_total_level_dofs, Geometry.levels[level].n_total_level_dofs};
  
  Geometry.levels[level].inner_domain->fill_sparsity_pattern(&dsp, &constraints);
  for (unsigned int surface = 0; surface < 6; surface++) {
    Geometry.levels[level].surfaces[surface]->fill_sparsity_pattern(&dsp, &constraints);
  }
  
  sp.copy_from(dsp);
  sp.compress();
  print_info("HierarchicalProblem::make_sparsity_pattern", "End on level "  + std::to_string(level));
}

std::string HierarchicalProblem::output_results(std::string in_fname_part) {
  GlobalTimerManager.switch_context("Output Results", level);
  Timer timer;
  timer.start();
  print_info("Hierarchical::output_results()", "Start on level " + std::to_string(level));
  std::string ret = "";
  NumericVectorLocal in_solution(Geometry.levels[level].inner_domain->dof_handler.n_dofs());
  for(unsigned int i = 0; i < Geometry.levels[level].inner_domain->dof_handler.n_dofs(); i++) {
    in_solution[i] = solution[Geometry.levels[level].inner_domain->global_index_mapping[i]];
  }
  std::string file_1 = Geometry.levels[level].inner_domain->output_results(in_fname_part + std::to_string(level) , in_solution);
  ret = file_1;
  filenames.clear();
  filenames.push_back(file_1);

  if(GlobalParams.BoundaryCondition == BoundaryConditionType::PML) {
    for(unsigned int i = 0; i < 6; i++){
      if(Geometry.levels[level].surface_type[i] == SurfaceType::ABC_SURFACE){
        dealii::Vector<ComplexNumber> ds (Geometry.levels[level].surfaces[i]->dof_counter);
        for(unsigned int index = 0; index < Geometry.levels[level].surfaces[i]->dof_counter; index++) {
          ds[index] = solution[Geometry.levels[level].surfaces[i]->global_index_mapping[index]];
        }
        std::string file_2 = Geometry.levels[level].surfaces[i]->output_results(ds, "pml_domain" + std::to_string(level));
        filenames.push_back(file_2);
      }
    }
  }

  // End of core output
  if(level != 0) {
  //  child->output_results();
  }

  print_info("Hierarchical::output_results()", "End on level " + std::to_string(level));
  timer.stop();
  GlobalTimerManager.leave_context(level);
  return ret;
}
