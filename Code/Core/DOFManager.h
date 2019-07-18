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
    const unsigned int dofs_per_cell; //constructor
    const unsigned int dofs_per_face; //constructor
    const unsigned int dofs_per_edge; //constructor
    unsigned int n_global_dofs; // via MPI_Build
    unsigned int n_locally_owned_dofs; // via MPI_Build
    dealii::Triangulation<3,3> * triangulation;
    dealii::DoFHandler<3,3> * dof_handler;
    const dealii::FiniteElement<3,3> * fe;

    DOFManager(unsigned int, unsigned int, unsigned int, dealii::DoFHandler<3,3> *, dealii::Triangulation<3,3> *, const dealii::FiniteElement<3,3> *);

    void init ();
    unsigned int compute_n_own_dofs();
    void MPI_build_global_index_set_vector();
    void compute_and_communicate_edge_dofs();
    void SortDofs();


};


#endif //WAVEGUIDEPROBLEM_DOFMANAGER_H
