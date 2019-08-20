//
// Created by kraft on 16.08.19.
//

#ifndef WAVEGUIDEPROBLEM_HSIESURFACE_H
#define WAVEGUIDEPROBLEM_HSIESURFACE_H

#include <deal.II/grid/tria.h>

struct DofCount {
    unsigned int owned, non_owned,  hsie, non_hsie, owned_hsie, total;
};

template<unsigned int ORDER>
class HSIESurface {
    unsigned int n_total_hsie_dofs;
    dealii::Triangulation<3,3> * main_triangulation;
    dealii::Triangulation<2,3> surface_triangulation;
    unsigned int n_dofs_shared_with_interior;
    unsigned int n_pure_hsie_dofs;
    const unsigned int b_id;
    unsigned int level;
    DofCount n_vertex_dofs, n_face_dofs, n_edge_dofs;

public:
    HSIESurface(dealii::Triangulation<3,3> * in_main_triangulation, unsigned int in_boundary_id, unsigned int in_level);

    void prepare_surface_triangulation();
    void compute_dof_numbers();
    void fill_matrix(dealii::SparseMatrix<double>* , dealii::IndexSet);
    unsigned int get_n_own_hsie_dofs();
    DofCount compute_n_edge_dofs();
    DofCount compute_n_vertex_dofs();
    DofCount compute_n_face_dofs();
    unsigned int compute_dofs_per_edge(bool only_edgge_dofs);
    unsigned int compute_dofs_per_face(bool only_face_dofs);
    unsigned int compute_dofs_per_vertex();
    void initialize();
    void initialize_dof_handlers();



};




#endif //WAVEGUIDEPROBLEM_HSIESURFACE_H
