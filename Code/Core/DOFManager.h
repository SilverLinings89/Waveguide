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
    std::vector<dealii::IndexSet> own_dofs_per_process;
    dealii::IndexSet local_dofs;
    const unsigned int dofs_per_cell;
    const unsigned int dofs_per_face;
    const unsigned int dofs_per_edge;
    unsigned int n_global_dofs;
    unsigned int n_locally_active_dofs;
    unsigned int n_locally_owned_dofs;
    dealii::Triangulation<3,3> * triangulation;
    dealii::DoFHandler<3,3> * dof_handler;

    DOFManager(unsigned int i_dofs_per_cell, unsigned int i_dofs_per_face, unsigned int i_dofs_per_edge);

    void init ();
    unsigned int compute_n_own_dofs();
    void MPI_build_global_index_set_vector();
    void compute_and_communicate_edge_dofs();



};


#endif //WAVEGUIDEPROBLEM_DOFMANAGER_H
