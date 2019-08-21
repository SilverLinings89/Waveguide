//
// Created by kraft on 16.08.19.
//

#ifndef WAVEGUIDEPROBLEM_HSIESURFACE_H
#define WAVEGUIDEPROBLEM_HSIESURFACE_H

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

struct DofCount {
    unsigned int owned = 0;
    unsigned int non_owned = 0;
    unsigned int hsie = 0;
    unsigned int non_hsie = 0;
    unsigned int owned_hsie = 0;
    unsigned int total = 0;
};

template<unsigned int ORDER>
class HSIESurface {
    unsigned int n_total_hsie_dofs;
    dealii::Triangulation<3,3> * main_triangulation;
    dealii::Triangulation<2,3> surface_triangulation;
    unsigned int n_dofs_shared_with_interior;
    unsigned int n_pure_hsie_dofs;
    const unsigned int b_id;
    const unsigned int Inner_Element_Order;
    unsigned int level;
    DofCount n_vertex_dofs, n_face_dofs, n_edge_dofs;
    dealii::DoFHandler<2,2> dof_h_nedelec;
    dealii::DoFHandler<2,2> dof_h_q;
    dealii::FESystem<2> fe_nedelec;
    dealii::FESystem<2> fe_q;

public:
    HSIESurface(dealii::Triangulation<3,3> * in_main_triangulation, unsigned int in_boundary_id, unsigned int in_level, unsigned int in_inner_order);

    void prepare_surface_triangulation();
    void compute_dof_numbers();
    void fill_matrix(dealii::SparseMatrix<double>* , dealii::IndexSet);
    unsigned int get_n_own_hsie_dofs();
    DofCount compute_n_edge_dofs();
    DofCount compute_n_vertex_dofs();
    DofCount compute_n_face_dofs();
    unsigned int compute_dofs_per_edge(bool only_hsie_dofs);
    unsigned int compute_dofs_per_face(bool only_hsie_dofs);
    unsigned int compute_dofs_per_vertex();
    void initialize();
    void initialize_dof_handlers_and_fe();
    void update_dof_counts_for_edge(dealii::DoFHandler<2>::active_cell_iterator cell, unsigned  int edge, DofCount & );
    void update_dof_counts_for_face(dealii::DoFHandler<2>::active_cell_iterator cell, DofCount & );
    void update_dof_counts_for_vertex(dealii::DoFHandler<2>::active_cell_iterator cell, unsigned  int edge, unsigned int vertex, DofCount & );
    bool is_edge_owned(dealii::DoFHandler<2>::active_cell_iterator cell, unsigned  int edge);
    bool is_face_owned(dealii::DoFHandler<2>::active_cell_iterator cell);
    bool is_vertex_owned(dealii::DoFHandler<2>::active_cell_iterator cell, unsigned  int edge, unsigned int vertex);

};




#endif //WAVEGUIDEPROBLEM_HSIESURFACE_H
