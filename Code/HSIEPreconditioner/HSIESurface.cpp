//
// Created by kraft on 16.08.19.
//

#include <deal.II/grid/grid_generator.h>
#include "HSIESurface.h"

template<unsigned int ORDER>
void HSIESurface<ORDER>::prepare_surface_triangulation() {
    dealii::GridGenerator::extract_boundary_mesh(main_triangulation, surface_triangulation, b_id);
}

template<unsigned int ORDER>
HSIESurface<ORDER>::HSIESurface(dealii::Triangulation<3, 3> *in_main_triangulation, unsigned int in_boundary_id,
                                unsigned int in_level):
                                main_triangulation(in_main_triangulation),
                                b_id(in_boundary_id),
                                level(in_level) {

}

template<unsigned int ORDER>
void HSIESurface<ORDER>::compute_dof_numbers() {
    this->n_edge_dofs = compute_n_edge_dofs();
    this->n_face_dofs = compute_n_face_dofs();
    this->n_vertex_dofs = compute_n_vertex_dofs();
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::fill_matrix(dealii::SparseMatrix<double> *, dealii::IndexSet) {

}

template<unsigned int ORDER>
unsigned int HSIESurface<ORDER>::get_n_own_hsie_dofs() {
    return n_vertex_dofs.owned_hsie + n_edge_dofs.owned_hsie + n_face_dofs.owned_hsie;
}

template<unsigned int ORDER>
DofCount HSIESurface<ORDER>::compute_n_edge_dofs() {
    DofCount ret;
    return ret;
}

template<unsigned int ORDER>
DofCount HSIESurface<ORDER>::compute_n_vertex_dofs() {
    DofCount ret;
    return ret;
}

template<unsigned int ORDER>
DofCount HSIESurface<ORDER>::compute_n_face_dofs() {
    DofCount ret;
    return ret;
}

template<unsigned int ORDER>
unsigned int HSIESurface<ORDER>::compute_dofs_per_edge(bool only_edgge_dofs) {
    return 0;
}

template<unsigned int ORDER>
unsigned int HSIESurface<ORDER>::compute_dofs_per_face(bool only_face_dofs) {
    return 0;
}

template<unsigned int ORDER>
unsigned int HSIESurface<ORDER>::compute_dofs_per_vertex() {
    return 0;
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::initialize() {
    initialize_dof_handlers();
    compute_dof_numbers();
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::initialize_dof_handlers() {

}
