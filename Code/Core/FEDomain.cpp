#include "FEDomain.h"

FEDomain::FEDomain() {
    n_locally_active_dofs = 0;
    n_locally_owned_dofs = 0;
    first_own_dof = UINT_MAX;
}

void FEDomain::initialize_dof_counts(DofCount in_n_locally_active_dofs, DofCount in_n_locally_owned_dofs) {
    n_locally_owned_dofs = in_n_locally_owned_dofs;
    n_locally_active_dofs = in_n_locally_active_dofs;
    global_index_mapping.resize(n_locally_active_dofs);
    for(unsigned int i = 0; i < n_locally_active_dofs; i++) {
        global_index_mapping[i] = UINT_MAX;
    }
}

bool FEDomain::finish_initialization(DofNumber first_own_index) {
    unsigned int n_non_owned_dofs = 0;
    for(unsigned int i = 0; i < n_locally_active_dofs; i++) {
        if(global_index_mapping[i] != UINT_MAX) {
            n_non_owned_dofs++;
        }
    }
    if(n_non_owned_dofs + n_locally_owned_dofs != n_locally_active_dofs) {
        return false;
    }
    DofNumber running_index = first_own_index;
    for(unsigned int i = 0; i < n_locally_active_dofs; i++) {
        if(global_index_mapping[i] == UINT_MAX) {
            global_index_mapping[i] = running_index;
            running_index++;
        }
    }
    return true;
}

std::vector<DofNumber> FEDomain::transform_local_to_global_dofs(std::vector<DofNumber> in_dofs) {
    std::vector<DofNumber> global_dof_indices;
    for(unsigned int i = 0; i < in_dofs.size(); i++) {
        global_dof_indices.push_back(global_index_mapping[in_dofs[i]]);
    }
    return global_dof_indices;
}

void FEDomain::set_non_local_dof_indices(DofIndexVector local_indices, DofIndexVector global_indices) {
    for(unsigned int i = 0; i < local_indices.size(); i++) {
        global_index_mapping[local_indices[i]] = global_indices[i];
    }
}

void FEDomain::mark_local_dofs_as_non_local(DofIndexVector in_dofs) {
    for(unsigned int i = 0; i < in_dofs.size(); i++) {
        global_index_mapping[in_dofs[i]] = 0;
    }
}
