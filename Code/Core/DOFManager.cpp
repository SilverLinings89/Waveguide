//
// Created by kraft on 16.07.19.
//

#include "DOFManager.h"
#include "NumericProblem.h"

void DOFManager::compute_and_communicate_edge_dofs() {
    if(GlobalParams.Blocks_in_x_direction > 1 && GlobalParams.Blocks_in_y_direction > 1) {

    } else {
        if(GlobalParams.Blocks_in_y_direction > 1) {

        } else {
            
        }
    }
}

void DOFManager::MPI_build_global_index_set_vector() {

}

void DOFManager::init() {

}

unsigned int DOFManager::compute_n_own_dofs() {
    return 0;
}
