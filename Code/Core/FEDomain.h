#pragma once

#include <deal.II/base/index_set.h>
#include <climits>
#include "../Core/Types.h"

class FEDomain {
    public:
    DofCount n_locally_active_dofs;
    DofCount n_locally_owned_dofs;
    dealii::IndexSet global_dof_indices;
    DofIndexVector global_index_mapping;
    DofNumber first_own_dof;
    
    FEDomain();
    void initialize_dof_counts(DofCount n_locally_active_dofs, DofCount n_locally_owned_dofs);
    DofIndexVector transform_local_to_global_dofs(DofIndexVector local_indices);
    void mark_local_dofs_as_non_local(DofIndexVector);
    bool finish_initialization(DofNumber first_own_index);
    void set_non_local_dof_indices(DofIndexVector local_indices, DofIndexVector global_indices);
    virtual DofCount compute_n_locally_owned_dofs(std::array<bool, 6> is_locally_owned_surfac) = 0;
    virtual DofCount compute_n_locally_active_dofs() = 0;
};
