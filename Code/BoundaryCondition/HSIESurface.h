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
#include "./BoundaryCondition.h"

class HSIESurface : public BoundaryCondition {
  const HSIEElementOrder order;
  DofHandler2D dof_h_nedelec;
  DofHandler2D dof_h_q;
  const unsigned int Inner_Element_Order;
  dealii::FE_NedelecSZ<2> fe_nedelec;
  dealii::FE_Q<2> fe_q;
  ComplexNumber k0;
  const double kappa;
  std::array<std::array<bool,6>,4> edge_ownership_by_level_and_id;
  dealii::Tensor<2,3,double> C;
  dealii::Tensor<2,3,double> G;
  Position V0;

public:
  DofDataVector face_dof_data;
  DofDataVector edge_dof_data;
  DofDataVector vertex_dof_data;
  DofCount n_edge_dofs;
  DofCount n_face_dofs;
  DofCount n_vertex_dofs;
  
  HSIESurface(HSIEElementOrder in_order, const dealii::Triangulation<2, 2> &in_surf_tria, BoundaryId in_boundary_id, NedelecElementOrder in_inner_order, ComplexNumber k0, double in_additional_coordinate);
  ~HSIESurface();  
  void identify_corner_cells() override;
  std::vector<HSIEPolynomial> build_curl_term_q(unsigned int, const dealii::Tensor<1, 2>);
  std::vector<HSIEPolynomial> build_curl_term_nedelec(unsigned int, const dealii::Tensor<1, 2>, const dealii::Tensor<1, 2>, const double, const double);
  std::vector<HSIEPolynomial> build_non_curl_term_q(unsigned int, const double);
  std::vector<HSIEPolynomial> build_non_curl_term_nedelec(unsigned int, const double, const double);
  void set_V0(Position);
  auto get_dof_data_for_cell(CellIterator2D, CellIterator2D) -> DofDataVector;
  void fill_matrix(dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, dealii::IndexSet,  std::array<bool, 6> surfaces_hsie,  dealii::AffineConstraints<ComplexNumber> *constraints) override;
  void fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, dealii::IndexSet, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *constraints) override;
  void fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, DofNumber shift, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *constraints) override;
  void fill_matrix(dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, DofNumber shift, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *constraints) override;
  void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix*, NumericVectorDistributed* rhs, dealii::IndexSet, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *constraints) override;
  void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix*, NumericVectorDistributed* rhs, DofNumber shift, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *constraints) override;
  void fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, DofNumber shift, dealii::AffineConstraints<ComplexNumber> *constraints) override;
  bool is_point_at_boundary(Position2D in_p, BoundaryId in_bid) override;
  auto get_vertices_for_boundary_id(BoundaryId) -> std::vector<unsigned int>;
  auto get_n_vertices_for_boundary_id(BoundaryId) -> unsigned int;
  auto get_lines_for_boundary_id(BoundaryId) -> std::vector<unsigned int>;
  auto get_n_lines_for_boundary_id(BoundaryId) -> unsigned int;
  auto compute_n_edge_dofs() -> DofCountsStruct;
  auto compute_n_vertex_dofs() -> DofCountsStruct;
  auto compute_n_face_dofs() -> DofCountsStruct;
  auto compute_dofs_per_edge(bool only_hsie_dofs) -> DofCount;
  auto compute_dofs_per_face(bool only_hsie_dofs) -> DofCount;
  auto compute_dofs_per_vertex() -> DofCount;
  void initialize() override;
  void initialize_dof_handlers_and_fe();
  void update_dof_counts_for_edge(CellIterator2D cell, unsigned int edge, DofCountsStruct&);
  void update_dof_counts_for_face(CellIterator2D cell, DofCountsStruct&);
  void update_dof_counts_for_vertex(CellIterator2D cell, unsigned int edge, unsigned int vertex, DofCountsStruct&);
  void register_new_vertex_dofs(CellIterator2D cell, unsigned int edge, unsigned int vertex);
  void register_new_edge_dofs(CellIterator2D cell, CellIterator2D cell_2, unsigned int edge);
  void register_new_surface_dofs(CellIterator2D cell, CellIterator2D cell2);
  auto register_dof() -> DofNumber;
  void register_single_dof(std::string in_id, int in_hsie_order, int in_inner_order, DofType in_dof_type, DofDataVector &, unsigned int);
  void register_single_dof(unsigned int in_id, int in_hsie_order, int in_inner_order, DofType in_dof_type, DofDataVector &, unsigned int, bool orientation = true);
  ComplexNumber evaluate_a(std::vector<HSIEPolynomial> &u, std::vector<HSIEPolynomial> &v, dealii::Tensor<2,3,double> G);
  void transform_coordinates_in_place(std::vector<HSIEPolynomial> *);
  bool check_dof_assignment_integrity();
  bool check_number_of_dofs_for_cell_integrity();
  auto get_dof_data_for_base_dof_nedelec(DofNumber base_dof_index) -> DofDataVector;
  auto get_dof_data_for_base_dof_q(DofNumber base_dof_index) -> DofDataVector;
  auto get_dof_count_by_boundary_id(BoundaryId in_boundary_id) -> DofCount override;
  auto get_dof_association() -> std::vector<InterfaceDofData> override;
  auto undo_transform(dealii::Point<2>) -> Position;
  auto undo_transform_for_shape_function(dealii::Point<2>) -> Position;
  void add_surface_relevant_dof(InterfaceDofData in_gindex_and_orientation);
  auto get_dof_association_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<InterfaceDofData> override;
  void clear_user_flags();
  void set_b_id_uses_hsie(unsigned int, bool);
  auto build_fad_for_cell(CellIterator2D cell) -> FaceAngelingData;
  void compute_extreme_vertex_coordinates();
  auto vertex_positions_for_ids(std::vector<unsigned int> ids) -> std::vector<Position>;
  auto line_positions_for_ids(std::vector<unsigned int> ids) -> std::vector<Position>;
  void setup_neighbor_couplings(std::array<bool, 6> is_b_id_truncated) override;
  void reset_neighbor_couplings(std::array<bool, 6> is_b_id_truncated) override;
  void output_results(const dealii::Vector<ComplexNumber> & , std::string) override;
  void fill_sparsity_pattern_for_neighbor(const BoundaryId in_bid, const unsigned int own_first_dof, const unsigned int partner_index, dealii::AffineConstraints<ComplexNumber> * constraints, dealii::DynamicSparsityPattern * dsp) override;
  void fill_sparsity_pattern_for_boundary_id(const BoundaryId in_bid, const unsigned int own_first_dof_index, dealii::AffineConstraints<ComplexNumber> * constraints, dealii::DynamicSparsityPattern * dsp) override;
  SurfaceCellData get_surface_cell_data_for_cell_index(const int in_index, const BoundaryId in_bid);
};



