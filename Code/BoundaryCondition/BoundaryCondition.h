#pragma once

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <vector>
#include "../Core/Types.h"
#include "./HSIEPolynomial.h"

class BoundaryCondition {
public:
  const BoundaryId b_id;
  const unsigned int level;
  const double additional_coordinate;
  const DofNumber first_own_dof;
  std::vector<unsigned int> corner_cell_ids;
  std::vector<InterfaceDofData> surface_dofs;
  bool surface_dof_sorting_done;
  bool boundary_coordinates_computed = false;
  std::array<double, 6> boundary_vertex_coordinates;
  DofCount dof_counter;

  BoundaryCondition(unsigned int in_bid, unsigned int in_level, double in_additional_coordinate, DofNumber first_own_index);

  virtual void identify_corner_cells() = 0;
  virtual void fill_matrix(dealii::PETScWrappers::SparseMatrix*,      NumericVectorDistributed* rhs, dealii::AffineConstraints<ComplexNumber> *constraints) = 0;
  virtual void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix*, NumericVectorDistributed* rhs, dealii::AffineConstraints<ComplexNumber> *constraints) = 0;
  virtual void fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, dealii::AffineConstraints<ComplexNumber> *constraints) = 0;
  virtual void fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, dealii::AffineConstraints<ComplexNumber> *constraints);
  virtual void fill_sparsity_pattern_for_inner_surface(dealii::DynamicSparsityPattern *in_dsp, dealii::AffineConstraints<ComplexNumber> *constraints);
  virtual bool is_point_at_boundary(Position2D in_p, BoundaryId in_bid) = 0;
  virtual void initialize() = 0;
  void set_mesh_boundary_ids();
  auto get_boundary_ids() -> std::vector<BoundaryId>;
  virtual auto get_dof_count_by_boundary_id(BoundaryId in_boundary_id) -> DofCount = 0;
  virtual auto get_dof_association() -> std::vector<InterfaceDofData> = 0;
  virtual auto get_dof_association_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<InterfaceDofData> = 0;
  void compute_extreme_vertex_coordinates();
  virtual void output_results(const dealii::Vector<ComplexNumber> & , std::string) = 0;
  virtual void fill_sparsity_pattern_for_boundary_id(const BoundaryId in_bid, dealii::AffineConstraints<ComplexNumber> * constraints, dealii::DynamicSparsityPattern * dsp) = 0;
  virtual void make_surface_constraints(dealii::AffineConstraints<ComplexNumber> * constraints) = 0;
  virtual void make_edge_constraints(dealii::AffineConstraints<ComplexNumber> * constraints, BoundaryId other_boundary) = 0;
  virtual auto get_surface_cell_data(BoundaryId in_bid) -> std::vector<SurfaceCellData> = 0;
  virtual auto get_inner_surface_cell_data() -> std::vector<SurfaceCellData> = 0;
  std::vector<unsigned int> dof_indices_from_surface_cell_data(std::vector<SurfaceCellData> in_data);
  void fill_sparsity_pattern_with_surface_data_vectors(std::vector<SurfaceCellData> first_data_vector, std::vector<SurfaceCellData> second_data_vector, dealii::DynamicSparsityPattern *in_dsp, dealii::AffineConstraints<ComplexNumber> *constraints);
};