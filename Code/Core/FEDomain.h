#pragma once

/**
 * @file FEDomain.h
 * @author Pascal Kraft (kraft.pascal@gmail.com)
 * @brief A base class for all objects that have either locally owned or active dofs.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <deal.II/base/index_set.h>
#include <climits>
#include "../Core/Types.h"

/**
 * @brief This class is a base type for all objects that own their own dofs. For all such objects we have to manage the sets of locally active and owned dofs. This object provides an abstract interface for these tasks.
 * 
 */
class FEDomain {
    public:
    DofCount n_locally_active_dofs;
    DofCount n_locally_owned_dofs;
    dealii::IndexSet global_dof_indices;
    DofIndexVector global_index_mapping;
    std::vector<bool> is_dof_owned;
    bool is_ownership_ready;
    
    FEDomain();
    
    /**
     * @brief In derived objects, this function will check for all dofs if they are locally owned or not. It will store the result in the vector is_dof_owned.
     * Once this is done we can count how many new dofs this object introduces.
     * 
     */
    virtual void determine_non_owned_dofs() = 0;

    /**
     * @brief Function for internal use. This sets the number of locally owned and active dofs.
     * 
     * @param n_locally_active_dofs The number of dofs that have support on the domain represented by this object. This is usually non-zero.
     * @param n_locally_owned_dofs The number of dofs that are either only active on the domain represented by this object or alternatively dofs that that are shared but this object has been determined to be the owner.
     */
    void initialize_dof_counts(DofCount n_locally_active_dofs, DofCount n_locally_owned_dofs);

    /**
     * @brief Returns the global number for a local index. Local indices always range from zero to n_locally_active_dofs. Global indices depend on the sweeping level and many other factors.
     * 
     * @param local_index The local index to be transformed into global numbering
     * @return DofIndexVector 
     */
    DofIndexVector transform_local_to_global_dofs(DofIndexVector local_index);

    /**
     * @brief Takes an index set and marks all indices in the set as non locally owned.
     * @param indices The set containing the dofs that are non-locally-owned.
     */
    void mark_local_dofs_as_non_local(DofIndexVector indices);

    /**
     * @brief Once all ownerships have been decided, this function numbers the locally owned dofs starting at the number provided.
     * 
     * @param first_own_index The index the first locally owned dof should have.
     * @return true If all dofs now have a valid index.
     * @return false If there are still dofs that have no valid index
     */
    virtual bool finish_initialization(DofNumber first_own_index);

    /**
     * @brief For a given index vector in local and global numbering, this function stores the global indices. After this call, the global index of any of the provided local indices is what was provided.
     * The data usually comes from another boundary or process or the interior domain
     * @param local_indices Indices in local numbering.
     * @param global_indices Indices in global numbering.
     */
    void set_non_local_dof_indices(DofIndexVector local_indices, DofIndexVector global_indices);

    /**
     * @brief Counts the number of locally owned dofs.
     * 
     * @return DofCount The number of locally owned dofs.
     */
    virtual DofCount compute_n_locally_owned_dofs() = 0;

    /**
     * @brief Counts the number of locally active dofs.
     * 
     * @return DofCount The number of locally active dofs.
     */
    virtual DofCount compute_n_locally_active_dofs() = 0;

    /**
     * @brief After this is called, ownership of dofs cannot be changed.
     * 
     */
    void freeze_ownership();

    /**
     * @brief For a provided vector of a global problem, this function extracts the locally active vector and returns it.
     * 
     * @param in_vector The global solution vector.
     * @return NumericVectorLocal The excerpt of the global vector in local numbering.
     */
    NumericVectorLocal get_local_vector_from_global(const NumericVectorDistributed in_vector);

    /**
     * @brief Computes the L2 norm of the contributions to the provided vector by the local object.
     * 
     * @return double L2 norm of the local part.
     */
    double local_norm_of_vector(NumericVectorDistributed *);
};
