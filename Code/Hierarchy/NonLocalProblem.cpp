//
// Created by pascal on 03.02.20.
//

#include "NonLocalProblem.h"
#include "../Helpers/GeometryManager.h"
#include "LocalProblem.h"
#include "../Core/NumericProblem.h"

NonLocalProblem::NonLocalProblem(unsigned int local_level,
    unsigned int global_level, DOFManager *dof_manager, MPI_Comm communicator) :
    HierarchicalProblem(local_level, global_level, dof_manager) {
    const unsigned int diff = global_level - local_level;
    level_communicator = MPI_COMM_WORLD;
    if(diff == 3) {
    MPI_Comm_split(communicator, GlobalParams.Index_in_x_direction,
        GlobalParams.MPI_Rank, &level_communicator);
    }
    if(diff == 2) {
    MPI_Comm_split(communicator, GlobalParams.Index_in_y_direction,
        GlobalParams.MPI_Rank, &level_communicator);
    }
    if(diff == 1) {
    MPI_Comm_split(communicator, GlobalParams.Index_in_z_direction,
        GlobalParams.MPI_Rank, &level_communicator);
    }
    if(local_level > 1) {
    child = new NonLocalProblem(local_level - 1, global_level, dof_manager,
        communicator);
    } else {
    child = new LocalProblem(local_level - 1, global_level, dof_manager);
    }
}

void NonLocalProblem::initialize_MPI_communicator_for_level() {

}

void NonLocalProblem::solve() {

}

void NonLocalProblem::initialize() {

}

void NonLocalProblem::generate_sparsity_pattern() {

}

void NonLocalProblem::initialize_index_sets() {

}

IndexSet NonLocalProblem::get_owned_dofs_for_level(unsigned int level) {
  return this->child->get_owned_dofs_for_level(level);
}

LocalProblem* NonLocalProblem::get_local_problem() {
  return child->get_local_problem();
}
