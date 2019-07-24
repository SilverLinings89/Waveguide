//
// Created by kraft on 16.07.19.
//

#ifndef WAVEGUIDEPROBLEM_DOFMANAGER_H
#define WAVEGUIDEPROBLEM_DOFMANAGER_H


#include <vector>
#include <deal.II/base/index_set.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>

class DOFManager {
    std::vector<dealii::IndexSet> own_dofs_per_process; // via MPI_Build...
    dealii::IndexSet local_dofs;    // via MPI_Build
    const unsigned int dofs_per_edge; //constructor
    const unsigned int dofs_per_face; //constructor
    const unsigned int dofs_per_cell; //constructor
    unsigned int n_global_dofs; // via MPI_Build
    unsigned int n_locally_owned_dofs; // via MPI_Build
    dealii::Triangulation<3,3> * triangulation;
    dealii::DoFHandler<3,3> * dof_handler;
    const dealii::FiniteElement<3,3> * fe;
    bool computed_n_global;

    DOFManager(unsigned int, unsigned int, unsigned int, dealii::DoFHandler<3,3> *, dealii::Triangulation<3,3> *, const dealii::FiniteElement<3,3> *);

    void init ();
    unsigned int compute_n_own_dofs();
    void MPI_build_global_index_set_vector();
    void compute_and_communicate_edge_dofs();
    void compute_and_communicate_face_dofs();
    void SortDofs();

    void compute_and_send_x_dofs();
    void compute_and_send_y_dofs();
    void compute_and_send_z_dofs();
    void receive_x_dofs();
    void receive_y_dofs();
    void receive_z_dofs();
    dealii::IndexSet get_dofs_for_boundary_id(dealii::types::boundary_id in_bid);
    dealii::IndexSet get_non_owned_dofs();
    void shift_own_to_final_dof_numbers();

    void update_interface_dofs_with_IndexSet(dealii::IndexSet in_new_indices, dealii::types::boundary_id in_bid);

    // TODO: There should be some functions checking for neighboring processes if the communicated IndexSets are the same.
};


#endif //WAVEGUIDEPROBLEM_DOFMANAGER_H
