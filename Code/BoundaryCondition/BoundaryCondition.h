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

  virtual void initialize() = 0;
  virtual std::string output_results(const dealii::Vector<ComplexNumber> & , std::string) = 0;
  
  // Geometry functionality and Boundary Ids.
  virtual void identify_corner_cells() = 0;
  virtual bool is_point_at_boundary(Position2D in_p, BoundaryId in_bid) = 0;
  void set_mesh_boundary_ids();
  auto get_boundary_ids() -> std::vector<BoundaryId>;
  void compute_extreme_vertex_coordinates();
  
  // Make constraints
  virtual auto get_dof_count_by_boundary_id(BoundaryId in_boundary_id) -> DofCount = 0;
  virtual auto get_dof_association() -> std::vector<InterfaceDofData> = 0;
  virtual auto get_dof_association_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<InterfaceDofData> = 0;
  virtual void make_surface_constraints(Constraints * constraints, bool make_inhomogeneities) = 0;
  virtual void make_edge_constraints(Constraints * constraints, BoundaryId other_boundary) = 0;
  
  // Generate the sparsity pattern
  virtual auto get_surface_cell_data(BoundaryId in_bid) -> std::vector<SurfaceCellData> = 0;
  virtual auto get_inner_surface_cell_data() -> std::vector<SurfaceCellData> = 0;
  virtual auto get_corner_surface_cell_data(BoundaryId main_boundary, BoundaryId secondary_boundary) -> std::vector<SurfaceCellData> = 0;
  virtual void fill_internal_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, Constraints * in_constriants) = 0;
  virtual void fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, Constraints *constraints);
  void fill_sparsity_pattern_for_inner_surface(dealii::DynamicSparsityPattern *in_dsp, Constraints *constraints);
  void fill_sparsity_pattern_with_surface_data_vectors(std::vector<SurfaceCellData> first_data_vector, std::vector<SurfaceCellData> second_data_vector, dealii::DynamicSparsityPattern *in_dsp, Constraints *constraints);
  void make_edge_sparsity_pattern(const BoundaryId in_bid, Constraints * constraints, dealii::DynamicSparsityPattern * dsp);
  auto dof_indices_from_surface_cell_data(std::vector<SurfaceCellData> in_data) -> std::vector<unsigned int>;

  
  // Fill the system matrix
  virtual void fill_matrix(dealii::PETScWrappers::SparseMatrix*,      NumericVectorDistributed* rhs, Constraints *constraints) = 0;
  virtual void fill_matrix(dealii::SparseMatrix<ComplexNumber> *, Constraints *constraints) = 0;
  virtual void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix*, NumericVectorDistributed* rhs, Constraints *constraints) = 0;
  virtual void fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, Constraints *constraints) = 0;
};