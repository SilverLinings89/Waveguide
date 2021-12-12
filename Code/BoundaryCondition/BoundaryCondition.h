#pragma once

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <vector>
#include "../Core/Types.h"
#include "./HSIEPolynomial.h"
#include "../Core/FEDomain.h"

/**
 * \class BoundaryCondition 
 * @brief This is the base type for boundary coniditions. Some implementations are done on this level, some below.
 * 
 * There are several deriveed classes for this type: Dirichlet, Empty, Hardy, PML and Neighbor. Details about them can be found in the derived classes. To the rest of the code, the most relevant functions are:
 * - Handling the dofs (number of dofs and association to boundaries)
 * - Assembly (of sparsity pattern and matrices)
 * - Building constraints
 * 
 * For the numbering, I always use the scheme 0 = -x, 1 = +x, 2 = -y, 3 = +y, 4 = -z and 5 = +z for all domain types. All domains are cuboid, so there are always 6 surfaces in the coordinate orthogonal directions.
 * 
 * Boundary conditions in this code have three types of surfaces (best visualized with a pml domain, i.e. a FE-domain):
 * - The surface shared with the inner domain,
 * - The sufaces shared with other boundary conditions,
 * - An outward surface, where dofs only couple with the interior of this boundary condition. 
 * 
 * Similar to all objects in this code, these objects have an initialize function that is implemented in the derived classes.
 * 
 */

class BoundaryCondition: public FEDomain {
public:
  const BoundaryId b_id; // This value is the boundary id as seen from the inner domain (for pml domains for example, there is a difference there. From the standpoint of the interior domain, the pml domain is boundary 0 but the interface between the boundary condition and the interior domain would be boundary 1 from the perspective of the pml domain).
  const unsigned int level; // This value is the sweeping level. The value is required to retrieve relevant data from the global objects.
  const double additional_coordinate; // Since all surfaces are axis-parallel and the boundary id determines the normal axis, there is only one value required to completely determine the interface plane between the inner domain and the surface domain. For example, if b_id = 0 (-x surface of the inner domain) and additional_coordinate = 1.0, then the boundary surface are all points with p[0] == 1.0.
  std::vector<InterfaceDofData> surface_dofs; // This is a cache so I don't have to perform expensive operations over and over again during assembly.
  bool surface_dof_sorting_done; // internal
  bool boundary_coordinates_computed = false;
  std::array<double, 6> boundary_vertex_coordinates;
  DofCount dof_counter; // This value stores the number of owned dofs of this boundary condition. It is 0 for boundary types like neighbor, empty or dirichlet, but non-zero for pml and hsie, where the boundary condition introduces additional dofs.
  unsigned int global_partner_mpi_rank; // This value is only relevant for neighbors but stored on the base type for convenience. It determines the MPI rank of the neighbor associated with this surface on the level communicator.
  const std::vector<BoundaryId> adjacent_boundaries; // For more performant implementation, I have stored this value. It is the set of boundary indices excluding the own boundary id and the boundary that is opposed to this one from the inner domain, since there is no coupling between the domain and itself and no domain intersection with the opposing side.
  std::array<bool, 6> are_edge_dofs_owned; // If the i-th value of this array is set to true, the surface dofs on that boundary surface are locally owned.
  DofHandler3D dof_handler;

  BoundaryCondition(unsigned int in_bid, unsigned int in_level, double in_additional_coordinate);

  virtual void initialize() = 0;
  virtual std::string output_results(const dealii::Vector<ComplexNumber> & , std::string) = 0;
  
  // Geometry functionality and Boundary Ids.
  virtual bool is_point_at_boundary(Position2D in_p, BoundaryId in_bid) = 0;
  void set_mesh_boundary_ids();
  auto get_boundary_ids() -> std::vector<BoundaryId>;
  
  virtual auto get_dof_association() -> std::vector<InterfaceDofData> = 0;
  virtual auto get_dof_association_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<InterfaceDofData> = 0;
  virtual auto get_global_dof_indices_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<DofNumber>;

  // Generate the sparsity pattern
  virtual void fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, Constraints *constraints) = 0; // Implementation happens in the derived classes and fills the sparsity pattern. Sometime these sparsity patterns depend on constraints as well, therefore constraints get passed aswell. The dealii documentation has more details on this topic.
  
  // Fill the system matrix
  virtual void fill_matrix(dealii::PETScWrappers::SparseMatrix*,      NumericVectorDistributed* rhs, Constraints *constraints) = 0; // For a given constraints object, this function writes all contributions from this boundary condition into the matrix (if any). Additionally, it assembles the right-hand-side vector associated with this boundary condition.
  virtual void fill_matrix(dealii::SparseMatrix<ComplexNumber> *, Constraints *constraints) = 0; // Similar to the function above, but it only assembles the matrix contributions.
  virtual void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix*, NumericVectorDistributed* rhs, Constraints *constraints) = 0; // Same as the first version of this function but for a distributed matrix type.
  virtual void fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, Constraints *constraints) = 0; // Same as the second version of this function but for a distributed matrix type.

  virtual void send_up_inner_dofs(); // This function is only required on neighbor surfaces but made available on this interface for code convenience. Find details in the neighbor surface documentation.
  virtual void receive_from_below_dofs(); // This function is only required on neighbor surfaces but made available on this interface for code convenience. Find details in the neighbor surface documentation.
  virtual void finish_dof_index_initialization(); // In cases where not all locally active dofs are locally owned (for example for two pml domains, the dofs on the shared surface are only owned by one of two processes) this function handles the numbering of the dofs once the non-owned dofs have been communicated.
  virtual std::vector<DofNumber> receive_boundary_dofs(unsigned int other_bid);

  virtual auto make_constraints() -> Constraints; // This is one of the core functions. Most boundary conditions come with some constraints on dofs (except the neighbor surface). For example in the derived DirichletSurface type, this function adds the dirichlet constraints to the constraints object passed in as an argument.

  double boundary_norm(NumericVectorDistributed *); // Computes the L2-norm of the solution passed in on the shared interface with the interior domain.
  double boundary_surface_norm(NumericVectorDistributed *, BoundaryId); // Computes the L2-norm of the solution passed in as an argument on the solution passed in as the second argument.

  virtual unsigned int cells_for_boundary_id(unsigned int boundary_id); // Counts the number of cells associated with the boundary passed in as an argument.

  void print_dof_validation();
};