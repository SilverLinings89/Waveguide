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

union DofBaseStructureID {
    std::string face_id;
    unsigned int non_face_id;

    DofBaseStructureID() {};
    ~DofBaseStructureID() {};
};

struct DofCount {
    unsigned int owned = 0;
    unsigned int non_owned = 0;
    unsigned int hsie = 0;
    unsigned int non_hsie = 0;
    unsigned int owned_hsie = 0;
    unsigned int total = 0;
};

// IFF = Infinite Face Function (a and b)
enum DofType {
    EDGE, SURFACE, RAY, IFFa, IFFb, SEGMENTa, SEGMENTb
};

struct DofData {
    DofType type;
    int hsie_order{};
    int inner_order{};
    bool is_real{};
    unsigned int global_index{};
    DofBaseStructureID base_structure_id;

    DofData() {
        base_structure_id.face_id = "";
    }

    DofData(std::string in_id) {
        base_structure_id.face_id = in_id;
    }

    DofData(unsigned int in_id) {
        base_structure_id.non_face_id = in_id;
    }
};

template<unsigned int ORDER>
class HSIESurface {
    dealii::Triangulation<3,3> * main_triangulation;
    dealii::Triangulation<2,3> surface_triangulation;
    const unsigned int b_id;
    const unsigned int Inner_Element_Order;
    unsigned int level;
    DofCount n_vertex_dofs, n_face_dofs, n_edge_dofs;
    dealii::DoFHandler<2,3> dof_h_nedelec;
    dealii::DoFHandler<2,3> dof_h_q;
    dealii::FESystem<2,3> fe_nedelec;
    dealii::FESystem<2,3> fe_q;
    std::vector<DofData> face_dof_data, edge_dof_data, vertex_dof_data;
    unsigned int dof_counter;
    dealii::Point<3> reference_point;
    std::map<dealii::Triangulation<2,3>::cell_iterator, dealii::Triangulation<3,3>::face_iterator > association;

public:
    HSIESurface(dealii::Triangulation<3,3> * in_main_triangulation, unsigned int in_boundary_id, unsigned int in_level, unsigned int in_inner_order);

    void prepare_surface_triangulation();
    void compute_dof_numbers();
    void fill_matrix(dealii::SparseMatrix<double>* , dealii::IndexSet);
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
    void register_new_vertex_dofs(dealii::DoFHandler<2>::active_cell_iterator cell, unsigned  int edge, unsigned int vertex);
    void register_new_edge_dofs(dealii::DoFHandler<2>::active_cell_iterator cell, unsigned  int edge);
    void register_new_surface_dofs(dealii::DoFHandler<2>::active_cell_iterator cell);
    unsigned int register_dof();
    void register_single_dof(std::string & in_id, int in_hsie_order, int in_inner_order, bool in_is_real, DofType in_dof_type, std::vector<DofData> &);
    void register_single_dof(unsigned int in_id, int in_hsie_order, int in_inner_order, bool in_is_real, DofType in_dof_type, std::vector<DofData> &);
};




#endif //WAVEGUIDEPROBLEM_HSIESURFACE_H
