//
// Created by kraft on 16.08.19.
//

#include <deal.II/grid/grid_generator.h>
#include "HSIESurface.h"
#include "../Core/NumericProblem.h"
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/grid/tria_accessor.h>


template<unsigned int ORDER>
void HSIESurface<ORDER>::prepare_surface_triangulation() {
    std::set<unsigned int> b_ids;
    b_ids.insert(this->b_id);
    dealii::GridGenerator::extract_boundary_mesh(*main_triangulation, surface_triangulation, b_ids);
}

template<unsigned int ORDER>
HSIESurface<ORDER>::HSIESurface(dealii::Triangulation<3, 3> *in_main_triangulation, unsigned int in_boundary_id,
                                unsigned int in_level, unsigned int in_inner_order):
                                main_triangulation(in_main_triangulation),
                                b_id(in_boundary_id),
                                level(in_level),
                                Inner_Element_Order(in_inner_order),
                                fe_nedelec(dealii::FE_Nedelec<2>(Inner_Element_Order), 2),
                                fe_q(dealii::FE_Q<2>(GlobalParams.So_ElementOrder), 2)
                                {

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
    std::set<unsigned int> touched_edges;
    DoFHandler<2>::active_cell_iterator cell, endc;
    endc = dof_h_nedelec.end();
    DofCount ret;
    // for each cell
    for(cell = dof_h_nedelec.begin(); cell != endc; cell++) {
        // for each edge
        for(unsigned int edge = 0; edge < GeometryInfo<2>::lines_per_cell; edge++) {
            // if it wasn't handled before
            if(!(touched_edges.end() == touched_edges.find(cell->line_index(edge)))) {
                // handle it
                update_dof_counts_for_edge(cell, edge, &ret);

                // remember that it has been handled
                touched_edges.insert(cell->line_index(edge));
            }
        }
    }
    return ret;
}

template<unsigned int ORDER>
DofCount HSIESurface<ORDER>::compute_n_vertex_dofs() {
    std::set<unsigned int> touched_vertices;
    DoFHandler<2>::active_cell_iterator cell, endc;
    endc = dof_h_nedelec.end();
    DofCount ret;
    // for each cell
    for(cell = dof_h_nedelec.begin(); cell != endc; cell++) {
        // for each edge
        for(unsigned int edge = 0; edge < GeometryInfo<2>::lines_per_cell; edge++) {
            // First Vertex
            if(!(touched_vertices.end() == touched_vertices.find(cell->line(edge)->vertex_index(0)))) {
                // handle it
                update_dof_counts_for_vertex(cell, edge, 0, &ret);

                // remember that it has been handled
                touched_vertices.insert(cell->line_index(edge));
            }
            // if it wasn't handled before
            if(!(touched_vertices.end() == touched_vertices.find(cell->line(edge)->vertex_index(1)))) {
                // handle it
                update_dof_counts_for_vertex(cell, edge, 1, &ret);

                // remember that it has been handled
                touched_vertices.insert(cell->line_index(edge));
            }
        }
    }
    return ret;
}

template<unsigned int ORDER>
DofCount HSIESurface<ORDER>::compute_n_face_dofs() {
    std::set<std::string> touched_faces;
    DoFHandler<2>::active_cell_iterator cell, endc;
    endc = dof_h_nedelec.end();
    DofCount ret;
    // for each cell
    for(cell = dof_h_nedelec.begin(); cell != endc; cell++) {
        // if it wasn't handled before
        if(!(touched_faces.end() == touched_faces.find(cell->id().to_string()))) {
            // handle it
            update_dof_counts_for_face(cell, &ret);

            // remember that it has been handled
            touched_faces.insert(cell->id().to_string());
        }
    }
    return ret;
}

template<unsigned int ORDER>
unsigned int HSIESurface<ORDER>::compute_dofs_per_edge(bool only_hsie_dofs) {
    unsigned int ret = 0;
    const unsigned int INNER_REAL_DOFS_PER_LINE = fe_nedelec.dofs_per_line / 2;

    // Number of inner dofs 1.
    if(! only_hsie_dofs) {
      ret += INNER_REAL_DOFS_PER_LINE;
    }

    //Number of infinite face functions 4a and 4b.
    ret += INNER_REAL_DOFS_PER_LINE * (ORDER+1) + (INNER_REAL_DOFS_PER_LINE-1)*(ORDER+2);

    // everything double for real and imaginary part.
    ret *= 2;
    return ret;
}

template<unsigned int ORDER>
unsigned int HSIESurface<ORDER>::compute_dofs_per_face(bool only_hsie_dofs) {
    unsigned int ret = 0;
    const unsigned int INNER_REAL_NEDELEC_DOFS_PER_FACE = fe_nedelec.dofs_per_face / 2;
    const unsigned int INNER_REAL_Q_DOFS_PER_FACE = fe_q.dofs_per_face / 2;
    if(! only_hsie_dofs) {
        ret += INNER_REAL_NEDELEC_DOFS_PER_FACE;
    }

    // Number of elements of type 5a.
    ret += INNER_REAL_NEDELEC_DOFS_PER_FACE * (ORDER + 1);

    // Number of elements of type 5b.
    ret += INNER_REAL_Q_DOFS_PER_FACE* (ORDER + 2);

    return ret;
}

template<unsigned int ORDER>
unsigned int HSIESurface<ORDER>::compute_dofs_per_vertex() {
    // Number of elements of type 3
    unsigned int ret = ORDER + 2;

    // Real and imaginary part.
    ret *= 2;
    return ret;
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::initialize() {
    prepare_surface_triangulation();
    initialize_dof_handlers_and_fe();
    compute_dof_numbers();
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::initialize_dof_handlers_and_fe() {
    dof_h_nedelec = new DoFHandler(surface_triangulation);
    dof_h_q = new DoFHandler(surface_triangulation);
    dof_h_q.distribute_dofs(fe_q);
    dof_h_nedelec.distribute_dofs(fe_nedelec);
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::update_dof_counts_for_edge(const dealii::DoFHandler<2>::active_cell_iterator cell, unsigned  int edge, DofCount & in_dof_count) {
    bool edge_is_owned = is_edge_owned(cell, edge);
    const unsigned int dofs_per_edge_all = compute_dofs_per_edge(false);
    const unsigned int dofs_per_edge_hsie = compute_dofs_per_edge(true);
    in_dof_count.total +=  dofs_per_edge_all;
    in_dof_count.hsie += dofs_per_edge_hsie;
    in_dof_count.non_hsie += dofs_per_edge_all - dofs_per_edge_hsie;
    if(edge_is_owned) {
        in_dof_count.owned += dofs_per_edge_all;
        in_dof_count.owned_hsie += dofs_per_edge_hsie;
    }
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::update_dof_counts_for_face(const dealii::DoFHandler<2>::active_cell_iterator cell, DofCount & in_dof_count) {
    bool edge_is_owned = is_face_owned(cell);
    const unsigned int dofs_per_face_all = compute_dofs_per_face(false);
    const unsigned int dofs_per_face_hsie = compute_dofs_per_face(true);
    in_dof_count.total +=  dofs_per_face_all;
    in_dof_count.hsie += dofs_per_face_hsie;
    in_dof_count.non_hsie += dofs_per_face_all - dofs_per_face_hsie;
    if(edge_is_owned) {
        in_dof_count.owned += dofs_per_face_all;
        in_dof_count.owned_hsie += dofs_per_face_hsie;
    }
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::update_dof_counts_for_vertex(const dealii::DoFHandler<2>::active_cell_iterator cell, unsigned  int edge, unsigned int vertex, DofCount & in_dof_count) {
    bool edge_is_owned = is_vertex_owned(cell, edge, vertex);

    // These are only hsie dofs.
    const unsigned int dofs_per_vertex_all = compute_dofs_per_vertex();

    in_dof_count.total +=  dofs_per_vertex_all;
    in_dof_count.hsie += dofs_per_vertex_all;
    if(edge_is_owned) {
        in_dof_count.owned += dofs_per_vertex_all;
        in_dof_count.owned_hsie += dofs_per_vertex_all;
    }
}

template<unsigned int ORDER>
bool HSIESurface<ORDER>::is_edge_owned(dealii::DoFHandler<2>::active_cell_iterator cell, unsigned int edge) {
    // TODO Implement this. Not really easy.
    return false;
}

template<unsigned int ORDER>
bool HSIESurface<ORDER>::is_face_owned(dealii::DoFHandler<2>::active_cell_iterator cell) {
    // TODO Implement this. Not really easy.
    return false;
}

template<unsigned int ORDER>
bool HSIESurface<ORDER>::is_vertex_owned(dealii::DoFHandler<2>::active_cell_iterator cell, unsigned int edge,
                                         unsigned int vertex) {
    // TODO Implement this. Not really easy.
    return false;
}

// TODO This file should be easily testable ..... maybe I should do that :D
