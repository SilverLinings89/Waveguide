#pragma once

#include "../Core/Types.h"
#include "./HSIEPolynomial.h"

const std::vector<std::vector<unsigned int>>  edge_to_boundary_id = {
    {4,5,2,3}, {5,4,2,3}, {0,1,4,5}, {0,1,5,4}, {1,0,2,3}, {0,1,2,3}
};

class BoundaryCondition {
public:
  const BoundaryId b_id;
  const double additional_coordinate;
  dealii::Triangulation<2> surface_triangulation;
  std::vector<unsigned int> corner_cell_ids;
  std::vector<InterfaceDofData> surface_dofs;
  bool surface_dof_sorting_done;
  bool boundary_coordinates_computed = false;
  std::array<double, 6> boundary_vertex_coordinates;
  DofCount dof_counter;
  
  BoundaryCondition(unsigned int in_bid, double in_additional_coordinate, const dealii::Triangulation<2> & in_surf_tria);

  virtual void identify_corner_cells() = 0;
  virtual void fill_matrix(dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, dealii::IndexSet,  std::array<bool, 6> surfaces_hsie,  dealii::AffineConstraints<ComplexNumber> *constraints) = 0;
  virtual void fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, dealii::IndexSet, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *constraints) = 0;
  virtual void fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, DofNumber shift, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *constraints) = 0;
  virtual void fill_matrix(dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, DofNumber shift, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *constraints) = 0;
  virtual void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix*, NumericVectorDistributed* rhs, dealii::IndexSet, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *constraints) = 0;
  virtual void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix*, NumericVectorDistributed* rhs, DofNumber shift, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *constraints) = 0;
  virtual void fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, DofNumber shift, dealii::AffineConstraints<ComplexNumber> *constraints) = 0;
  virtual bool is_point_at_boundary(Position2D in_p, BoundaryId in_bid) = 0;
  virtual void initialize() = 0;
  void set_mesh_boundary_ids();
  auto get_boundary_ids() -> std::vector<BoundaryId>;
  virtual auto get_dof_count_by_boundary_id(BoundaryId in_boundary_id) -> DofCount = 0;
  virtual auto get_dof_association() -> std::vector<InterfaceDofData> = 0;
  virtual auto get_dof_association_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<InterfaceDofData> = 0;
  void compute_extreme_vertex_coordinates();
  virtual void setup_neighbor_couplings(std::array<bool, 6> is_b_id_truncated) = 0;
  virtual void reset_neighbor_couplings(std::array<bool, 6> is_b_id_truncated) = 0;
};