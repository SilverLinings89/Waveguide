//
// Created by pascal on 03.02.20.
//

#include "LocalProblem.h"
#include "../Helpers/Parameters.h"

LocalProblem::LocalProblem(unsigned int local_level, unsigned int global_level, DOFManager * dof_manager, Parameters * params):
    HierarchicalProblem(local_level, global_level, dof_manager, params){

}

unsigned int LocalProblem::compute_own_dofs() {
    dof_manager->compute_n_own_dofs();
}

void LocalProblem::compute_level_dofs_total() {

}

void LocalProblem::solve() {

}

void LocalProblem::initialize() {

}

void LocalProblem::generateSparsityPattern() {

}
