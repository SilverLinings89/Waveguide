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
#include <deal.II/fe/fe_values.h>
#include "HSIEPolynomial.h"
#include "DofData.h"
#include <deal.II/fe/fe_nedelec_sz.h>



struct DofCount {
    unsigned int owned = 0;
    unsigned int non_owned = 0;
    unsigned int hsie = 0;
    unsigned int non_hsie = 0;
    unsigned int owned_hsie = 0;
    unsigned int total = 0;
};

static void  print_dof_count(DofCount in_dofs) {
    std::cout << "Owned: " << in_dofs.owned << std::endl;
    std::cout << "Non-Owned: " << in_dofs.non_owned << std::endl;
    std::cout << "HSIE: " << in_dofs.hsie << std::endl;
    std::cout << "NON-HSIE: " << in_dofs.non_hsie << std::endl;
    std::cout << "Owned-HSIE: " << in_dofs.owned_hsie << std::endl;
    std::cout << "Total: " << in_dofs.total << std::endl;
}

template<unsigned int ORDER>
class HSIESurface {
    std::map<dealii::Triangulation<2,3>::cell_iterator, dealii::Triangulation<3,3>::face_iterator > association;
    const unsigned int b_id;
    unsigned int dof_counter;
    dealii::DoFHandler<2> dof_h_nedelec;
    dealii::DoFHandler<2> dof_h_q;

    const unsigned int Inner_Element_Order;
    dealii::FE_NedelecSZ<2> fe_nedelec;
    dealii::FE_Q<2> fe_q;
    std::complex<double> k0;
    unsigned int level;
    DofCount n_edge_dofs, n_face_dofs, n_vertex_dofs;
    dealii::Triangulation<2> surface_triangulation;

public:
    std::vector<DofData> face_dof_data, edge_dof_data, vertex_dof_data;
    HSIESurface( dealii::Triangulation<2,2> & in_surf_tria, unsigned int in_boundary_id, unsigned int in_level, unsigned int in_inner_order, std::complex<double> k0, std::map<dealii::Triangulation<2,3>::cell_iterator, dealii::Triangulation<3,3>::face_iterator > in_assoc );
    std::vector<HSIEPolynomial> build_curl_term(DofData, const dealii::FEValuesViews::Vector<2,2>&, unsigned int q_index, HSIEPolynomial, unsigned int);
    std::vector<HSIEPolynomial> build_non_curl_term(DofData, const dealii::FEValuesViews::Vector<2,2>&, unsigned int q_index, HSIEPolynomial , unsigned int);
    std::vector<HSIEPolynomial> build_curl_term(DofData, const dealii::FEValuesViews::Scalar<2,2>&, unsigned int q_index, HSIEPolynomial, unsigned int);
    std::vector<HSIEPolynomial> build_non_curl_term(DofData, const dealii::FEValuesViews::Scalar<2,2>&, unsigned int q_index, HSIEPolynomial , unsigned int);

    std::vector<DofData> get_dof_data_for_cell(dealii::Triangulation<2,2>::cell_iterator);
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
    void register_new_edge_dofs(dealii::DoFHandler<2>::active_cell_iterator cell, dealii::DoFHandler<2>::active_cell_iterator cell_2, unsigned  int edge);
    void register_new_surface_dofs(dealii::DoFHandler<2>::active_cell_iterator cell, dealii::DoFHandler<2>::active_cell_iterator cell2);
    unsigned int register_dof();
    void register_single_dof(std::string in_id, int in_hsie_order, int in_inner_order, bool in_is_real, DofType in_dof_type, std::vector<DofData> &, unsigned int);
    void register_single_dof(unsigned int in_id, int in_hsie_order, int in_inner_order, bool in_is_real, DofType in_dof_type, std::vector<DofData> &, unsigned int);
    static std::complex<double> evaluate_a(std::vector<HSIEPolynomial> &u, std::vector<HSIEPolynomial> &v, unsigned int gauss_order);
    void transform_coordinates_in_place(std::vector<HSIEPolynomial> *);
};

#endif //WAVEGUIDEPROBLEM_HSIESURFACE_H
