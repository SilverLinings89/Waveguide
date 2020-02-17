//
// Created by pascal on 03.02.20.
//

#include "NonLocalProblem.h"
#include "../Helpers/GeometryManager.h"
#include "LocalProblem.h"

NonLocalProblem::NonLocalProblem(unsigned int local_level, unsigned int global_level, DOFManager * dof_manager, MPI_Comm communicator, Parameters * params):
    HierarchicalProblem(local_level, global_level, dof_manager, params){
    const unsigned int diff = global_level - local_level;
    level_communicator = MPI_COMM_WORLD;
    if(diff == 3) {
        MPI_Comm_split(communicator, params->Index_in_x_direction, params->MPI_Rank, & level_communicator);
    }
    if(diff == 2) {
        MPI_Comm_split(communicator, params->Index_in_y_direction, params->MPI_Rank, & level_communicator);
    }
    if(diff == 1) {
        MPI_Comm_split(communicator, params->Index_in_z_direction, params->MPI_Rank, & level_communicator);
    }
    if(local_level > 1) {
        child = new NonLocalProblem(local_level - 1, global_level, dof_manager, communicator,  params);
    } else {
        child = new LocalProblem(local_level - 1, global_level, dof_manager, params);
    }
}

unsigned int NonLocalProblem::compute_own_dofs() {
    number_of_own_indices = child->compute_own_dofs();
    return number_of_own_indices;
}

void NonLocalProblem::initialize_MPI_communicator_for_level() {

}

void NonLocalProblem::compute_level_dofs_total() {

}

void NonLocalProblem::solve() {

}

void NonLocalProblem::initialize() {

}

void NonLocalProblem::generateSparsityPattern() {

}

void NonLocalProblem::initialize_index_sets() {
    
}