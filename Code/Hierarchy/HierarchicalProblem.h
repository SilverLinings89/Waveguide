//
// Created by pascal on 03.02.20.
//

#ifndef WAVEGUIDEPROBLEM_HIERARCHICALPROBLEM_H
#define WAVEGUIDEPROBLEM_HIERARCHICALPROBLEM_H
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include "../Core/DOFManager.h"
#include "../Helpers/Parameters.h"
#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

class HierarchicalProblem {
public:
    bool has_child;
    HierarchicalProblem * child;
    const unsigned int global_level, local_level;
    unsigned int number_of_own_indices;
    DOFManager * dof_manager;
    Parameters * parameters;
    IndexSet own_indices;
    IndexSet locally_owned_dofs, locally_relevant_dofs;
    unsigned int level_total_dofs;
    dealii::TrilinosWrappers::SparseMatrix matrix;

    HierarchicalProblem(unsigned int in_own_level, unsigned int in_global_level, DOFManager * in_dof_manager, Parameters * in_params);

    virtual void send_vector_to_upper();
    virtual void receive_vector_from_upper();
    virtual void send_vector_to_lower();
    virtual void receive_vector_from_lower();

    virtual unsigned int compute_lower_interface_dof_count();
    virtual unsigned int compute_upper_interface_dof_count();

    virtual void compute_level_dofs_total() = 0;
    virtual void solve() = 0;
    virtual void initialize() = 0;
    virtual void generateSparsityPattern() = 0;
    virtual unsigned int compute_own_dofs() = 0;

    virtual void solve();
    virtual void solve_inner();
    virtual void assemble();
    virtual void make_sparsity_pattern();
    virtual void initialize_index_sets();
    virtual void apply_sweep(LinearAlgebra::distributed::Vector<double>)
};


#endif //WAVEGUIDEPROBLEM_HIERARCHICALPROBLEM_H
