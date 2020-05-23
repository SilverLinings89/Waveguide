//
// Created by kraft on 16.08.19.
//

#ifndef WAVEGUIDEPROBLEM_HSIESURFACE_H
#define WAVEGUIDEPROBLEM_HSIESURFACE_H

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/tria.h>
#include "DofData.h"
#include "HSIEPolynomial.h"
#include "../Helpers/Parameters.h"
#include "../Helpers/Structs.h"


class HSIESurface {
  const unsigned int order;
  const unsigned int b_id;
  dealii::DoFHandler<2> dof_h_nedelec;
  dealii::DoFHandler<2> dof_h_q;

  const unsigned int Inner_Element_Order;
  dealii::FE_NedelecSZ<2> fe_nedelec;
  dealii::FE_Q<2> fe_q;
  std::complex<double> k0;
  dealii::Triangulation<2> surface_triangulation;
  bool ** edge_ownership_by_level_and_id;
  std::vector<unsigned int> corner_cell_ids;
  const double additional_coordinate;
  std::vector<std::pair<unsigned int, dealii::Point<3, double>>> surface_dofs;
  bool surface_dof_sorting_done;

 public:
  unsigned int dof_counter;
  std::vector<DofData> face_dof_data;
  std::vector<DofData> edge_dof_data;
  std::vector<DofData> vertex_dof_data;
  unsigned int n_edge_dofs;
  unsigned int n_face_dofs;
  unsigned int n_vertex_dofs;
  HSIESurface(unsigned int in_order,
      const dealii::Triangulation<2, 2> &in_surf_tria,
      unsigned int in_boundary_id,
              unsigned int in_inner_order, std::complex<double> k0,
      double in_additional_coordinate);
  void identify_corner_cells();
  void compute_edge_ownership_object(Parameters params);
  std::vector<HSIEPolynomial> build_curl_term_q(unsigned int,
                                                const dealii::Tensor<1, 2>);
  std::vector<HSIEPolynomial> build_curl_term_nedelec(
      unsigned int, const dealii::Tensor<1, 2>, const dealii::Tensor<1, 2>,
      const double, const double);
  std::vector<HSIEPolynomial> build_non_curl_term_q(unsigned int, const double);
  std::vector<HSIEPolynomial> build_non_curl_term_nedelec(unsigned int,
                                                          const double,
                                                          const double);

  void fill_sparsity_pattern(dealii::DynamicSparsityPattern *pattern);
  std::vector<DofData> get_dof_data_for_cell(
      dealii::DoFHandler<2>::active_cell_iterator,
      dealii::DoFHandler<2>::active_cell_iterator);
  void fill_matrix(dealii::SparseMatrix<double> *, dealii::IndexSet);
  void fill_matrix(dealii::SparseMatrix<double>*, unsigned int);
  void fill_matrix(dealii::TrilinosWrappers::SparseMatrix*, dealii::IndexSet);
  void fill_matrix(dealii::TrilinosWrappers::SparseMatrix*, unsigned int);
  void fill_sparsity_pattern(dealii::SparsityPattern *in_dsp,
      unsigned int shift);
  void fill_sparsity_pattern(dealii::SparsityPattern *in_dsp,
      dealii::IndexSet shift);
  void make_hanging_node_constraints(dealii::AffineConstraints<double>*,
      dealii::IndexSet);
  DofCount compute_n_edge_dofs();
  DofCount compute_n_vertex_dofs();
  DofCount compute_n_face_dofs();
  unsigned int compute_dofs_per_edge(bool only_hsie_dofs);
  unsigned int compute_dofs_per_face(bool only_hsie_dofs);
  unsigned int compute_dofs_per_vertex();
  void initialize();
  void initialize_dof_handlers_and_fe();
  void update_dof_counts_for_edge(
      dealii::DoFHandler<2>::active_cell_iterator cell, unsigned int edge,
      DofCount&);
  void update_dof_counts_for_face(
      dealii::DoFHandler<2>::active_cell_iterator cell, DofCount&);
  void update_dof_counts_for_vertex(
      dealii::DoFHandler<2>::active_cell_iterator cell, unsigned int edge,
      unsigned int vertex, DofCount&);
  void register_new_vertex_dofs(
      dealii::DoFHandler<2>::active_cell_iterator cell, unsigned int edge,
      unsigned int vertex);
  void register_new_edge_dofs(
      dealii::DoFHandler<2>::active_cell_iterator cell,
      dealii::DoFHandler<2>::active_cell_iterator cell_2, unsigned int edge);
  void register_new_surface_dofs(
      dealii::DoFHandler<2>::active_cell_iterator cell,
      dealii::DoFHandler<2>::active_cell_iterator cell2);
  unsigned int register_dof();
  void register_single_dof(std::string in_id, int in_hsie_order,
                           int in_inner_order, bool in_is_real,
                           DofType in_dof_type, std::vector<DofData> &,
                           unsigned int);
  void register_single_dof(unsigned int in_id, int in_hsie_order,
                           int in_inner_order, bool in_is_real,
                           DofType in_dof_type, std::vector<DofData> &,
                           unsigned int);
  static std::complex<double> evaluate_a(std::vector<HSIEPolynomial> &u,
                                         std::vector<HSIEPolynomial> &v);
  void transform_coordinates_in_place(std::vector<HSIEPolynomial> *);
  bool check_dof_assignment_integrity();
  bool check_number_of_dofs_for_cell_integrity();
  void set_mesh_boundary_ids();
  std::vector<unsigned int> get_boundary_ids();
  std::vector<DofData> get_dof_data_for_base_dof_nedelec(
      unsigned int base_dof_index);
  std::vector<DofData> get_dof_data_for_base_dof_q(unsigned int base_dof_index);
  unsigned int get_dof_count_by_boundary_id(unsigned int in_boundary_id);
  std::vector<unsigned int> get_dof_association();
  dealii::Point<3> undo_transform(dealii::Point<2>);
  void add_surface_relevant_dof(unsigned int in_global_index,
      dealii::Point<3, double> point);
  std::vector<unsigned int> get_dof_association_by_boundary_id(
      unsigned int in_boundary_id);
};


#endif  // WAVEGUIDEPROBLEM_HSIESURFACE_H
