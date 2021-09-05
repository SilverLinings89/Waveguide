#pragma once

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <vector>
#include "../Core/Types.h"
#include "./HSIEPolynomial.h"
#include "../Core/FEDomain.h"

class BoundaryCondition: public FEDomain {
public:
  std::array<std::array<bool, 6>, 6> is_surface_owned;
  const BoundaryId b_id;
  const unsigned int level;
  const double additional_coordinate;
  std::vector<unsigned int> corner_cell_ids;
  std::vector<InterfaceDofData> surface_dofs;
  bool surface_dof_sorting_done;
  bool boundary_coordinates_computed = false;
  std::array<double, 6> boundary_vertex_coordinates;
  DofCount dof_counter;

  BoundaryCondition(unsigned int in_bid, unsigned int in_level, double in_additional_coordinate);

  virtual void initialize() = 0;
  virtual std::string output_results(const dealii::Vector<ComplexNumber> & , std::string) = 0;
  
  // Geometry functionality and Boundary Ids.
  virtual void identify_corner_cells() = 0;
  virtual bool is_point_at_boundary(Position2D in_p, BoundaryId in_bid) = 0;
  void set_mesh_boundary_ids();
  auto get_boundary_ids() -> std::vector<BoundaryId>;
  
  virtual auto get_dof_count_by_boundary_id(BoundaryId in_boundary_id) -> DofCount = 0;
  virtual auto get_dof_association() -> std::vector<InterfaceDofData> = 0;
  virtual auto get_dof_association_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<InterfaceDofData> = 0;
  virtual auto get_global_dof_indices_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<DofNumber>;

  // Generate the sparsity pattern
  virtual void fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, Constraints *constraints) = 0;
  
  // Fill the system matrix
  virtual void fill_matrix(dealii::PETScWrappers::SparseMatrix*,      NumericVectorDistributed* rhs, Constraints *constraints) = 0;
  virtual void fill_matrix(dealii::SparseMatrix<ComplexNumber> *, Constraints *constraints) = 0;
  virtual void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix*, NumericVectorDistributed* rhs, Constraints *constraints) = 0;
  virtual void fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, Constraints *constraints) = 0;

  virtual void send_up_inner_dofs();
  virtual void receive_from_below_dofs();
  virtual void finish_dof_index_initialization();
  virtual std::vector<DofNumber> receive_boundary_dofs(unsigned int other_bid);
};