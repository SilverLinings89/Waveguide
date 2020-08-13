#pragma once

#include "../Core/Types.h"
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

class HSIESurface {
  const HSIEElementOrder order;
  const BoundaryId b_id;
  DofHandler2D dof_h_nedelec;
  DofHandler2D dof_h_q;
  bool is_metal = false;
  const unsigned int Inner_Element_Order;
  dealii::FE_NedelecSZ<2> fe_nedelec;
  dealii::FE_Q<2> fe_q;
  ComplexNumber k0;
  dealii::Triangulation<2> surface_triangulation;
  std::array<std::array<bool,6>,4> edge_ownership_by_level_and_id;
  std::vector<unsigned int> corner_cell_ids;
  const double additional_coordinate;
  std::vector<DofIndexAndOrientationAndPosition> surface_dofs;
  bool surface_dof_sorting_done;
  dealii::Tensor<2,3,double> C;
  dealii::Tensor<2,3,double> G;
  Position V0;

public:
  DofCount dof_counter;
  DofDataVector face_dof_data;
  DofDataVector edge_dof_data;
  DofDataVector vertex_dof_data;
  DofCount n_edge_dofs;
  DofCount n_face_dofs;
  DofCount n_vertex_dofs;
  HSIESurface(HSIEElementOrder in_order,
      const dealii::Triangulation<2, 2> &in_surf_tria,
      BoundaryId in_boundary_id,
      NedelecElementOrder in_inner_order, ComplexNumber k0,
      double in_additional_coordinate,
      bool in_is_metal = false);
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
  void set_V0(Position);
  void fill_sparsity_pattern(dealii::DynamicSparsityPattern *pattern);
  auto get_dof_data_for_cell(CellIterator2D, CellIterator2D) -> DofDataVector;
  void fill_matrix(dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, dealii::IndexSet, const Position &V0, dealii::AffineConstraints<ComplexNumber> *constraints);
  void fill_matrix(dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, DofNumber shift,  const Position &V0, dealii::AffineConstraints<ComplexNumber> *constraints);
  void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix*, NumericVectorDistributed* rhs, dealii::IndexSet, const Position &V0, dealii::AffineConstraints<ComplexNumber> *constraints);
  void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix*, NumericVectorDistributed* rhs, DofNumber shift,  const Position &V0, dealii::AffineConstraints<ComplexNumber> *constraints);
  void fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp,
      DofNumber shift, dealii::AffineConstraints<ComplexNumber> *constraints);
  void make_hanging_node_constraints(dealii::AffineConstraints<ComplexNumber>*, DofNumber shift);
  auto compute_n_edge_dofs() -> DofCountsStruct;
  auto compute_n_vertex_dofs() -> DofCountsStruct;
  auto compute_n_face_dofs() -> DofCountsStruct;
  auto compute_dofs_per_edge(bool only_hsie_dofs) -> DofCount;
  auto compute_dofs_per_face(bool only_hsie_dofs) -> DofCount;
  auto compute_dofs_per_vertex() -> DofCount;
  void initialize();
  void initialize_dof_handlers_and_fe();
  void update_dof_counts_for_edge(CellIterator2D cell, unsigned int edge, DofCountsStruct&);
  void update_dof_counts_for_face(CellIterator2D cell, DofCountsStruct&);
  void update_dof_counts_for_vertex(CellIterator2D cell, unsigned int edge, unsigned int vertex, DofCountsStruct&);
  void register_new_vertex_dofs(CellIterator2D cell, unsigned int edge, unsigned int vertex);
  void register_new_edge_dofs(CellIterator2D cell, CellIterator2D cell_2, unsigned int edge);
  void register_new_surface_dofs(CellIterator2D cell, CellIterator2D cell2);
  auto register_dof() -> DofNumber;
  void register_single_dof(std::string in_id, int in_hsie_order, int in_inner_order, DofType in_dof_type, DofDataVector &, unsigned int);
  void register_single_dof(unsigned int in_id, int in_hsie_order, int in_inner_order, DofType in_dof_type, DofDataVector &, unsigned int);
  static ComplexNumber evaluate_a(std::vector<HSIEPolynomial> &u, std::vector<HSIEPolynomial> &v, dealii::Tensor<2,3,double> G);
  void transform_coordinates_in_place(std::vector<HSIEPolynomial> *);
  bool check_dof_assignment_integrity();
  bool check_number_of_dofs_for_cell_integrity();
  void set_mesh_boundary_ids();
  auto get_boundary_ids() -> std::vector<BoundaryId>;
  auto get_dof_data_for_base_dof_nedelec(DofNumber base_dof_index) -> DofDataVector;
  auto get_dof_data_for_base_dof_q(DofNumber base_dof_index) -> DofDataVector;
  auto get_dof_count_by_boundary_id(BoundaryId in_boundary_id) -> DofCount;
  auto get_dof_association() -> std::vector<DofIndexAndOrientationAndPosition>;
  auto undo_transform(dealii::Point<2>) -> Position;
  void add_surface_relevant_dof(
      DofIndexAndOrientationAndPosition in_gindex_and_orientation);
  auto get_dof_association_by_boundary_id(BoundaryId in_boundary_id) ->
      std::vector<DofIndexAndOrientationAndPosition>;
  void clear_user_flags();
  void set_b_id_uses_hsie(unsigned int, bool);
  bool get_orientation(const Position2D &vertex_1, const Position2D &vertex_2);
};



