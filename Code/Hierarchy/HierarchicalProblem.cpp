//
// Created by pascal on 03.02.20.
//

#include "HierarchicalProblem.h"
#include "../Helpers/Parameters.h"

HierarchicalProblem::HierarchicalProblem(
        unsigned int in_own_level,
        unsigned int in_global_level,
        DOFManager * in_dof_manager,
        Parameters * in_params):
        global_level(in_global_level),
        local_level(in_own_level) {
    has_child = in_own_level > 0;
    dof_manager = in_dof_manager;
    parameters = in_params;
}