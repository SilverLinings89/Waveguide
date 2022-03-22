#pragma once
/**
 * @file BoundaryCondition.h
 * @author Pascal Kraft (kraft.pascal@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <vector>
#include "../Core/Types.h"
#include "./HSIEPolynomial.h"
#include "../Core/FEDomain.h"

/**
 * \class BoundaryCondition 
 * @brief This is the base type for boundary coniditions. Some implementations are done on this level, some in the derived types.
 * 
 * There are several deriveed classes for this type: Dirichlet, Empty, Hardy, PML and Neighbor. Details about them can be found in the derived classes. To the rest of the code, the most relevant functions are:
 * - Handling the dofs (number of dofs and association to boundaries)
 * - Assembly (of sparsity pattern and matrices)
 * - Building constraints
 * 
 * For the boundary numbering, I always use the scheme 0 = -x, 1 = +x, 2 = -y, 3 = +y, 4 = -z and 5 = +z for all domain types. All domains are cuboid, so there are always 6 surfaces in the coordinate orthogonal directions, so the code always considers one interior domain and 6 surfaces, which each need a boundary condition associated with them.
 * 
 * Boundary conditions in this code have three types of surfaces (best visualized with a pml domain, i.e. a FE-domain):
 * - The surface shared with the inner domain, This is always one.
 * - The sufaces shared with other boundary conditions, There are always four neighbors since there are always six boundary methods for a domain and the boundary conditions handle the outer sides of this domain like the sides of a cube.
 * - An outward surface, where dofs only couple with the interior of this boundary condition domain (if that exists). 
 * 
 * Similar to all objects in this code, these objects have an initialize function that is implemented in the derived classes. It is important to note, that boundary conditions can introduce their own degrees of freedom to the system assemble and are therefore derived from the abstract base class FEDomain, which basically means they have owned and locally active dofs and these may need to be added to sets of degrees of freedom or handled otherwise.
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
  int global_partner_mpi_rank; // This value is only relevant for neighbors but stored on the base type for convenience. It determines the MPI rank of the neighbor associated with this surface on the level communicator.
  int local_partner_mpi_rank;
  const std::vector<BoundaryId> adjacent_boundaries; // For more performant implementation, I have stored this value. It is the set of boundary indices excluding the own boundary id and the boundary that is opposed to this one from the inner domain, since there is no coupling between the domain and itself and no domain intersection with the opposing side.
  std::array<bool, 6> are_edge_dofs_owned; // If the i-th value of this array is set to true, the surface dofs on that boundary surface are locally owned.
  DofHandler3D dof_handler;

  BoundaryCondition(unsigned int in_bid, unsigned int in_level, double in_additional_coordinate);

  /**
     *  @brief Not all data for objects of this type will be available at time of construction. This function exists on many objects in this code and handles initialization once all data is configured.
     *
     *  @details Typically, this function will perform actions like initializing matrices and vectors and enumerating dofs. It is part of the typical pattern Construct -> Initialize -> Run -> Output -> Delete. 
     * However, since this is an abstract base class, this function cannot be implemented on this level. No data needs to be passed as an argument and no value is returned. Make sure you understand this function before calling or adapting it on a derived class.
     * 
     *  @see   This function is also often implemented in deal.II examples and derives its name from there.    
     */
  virtual void initialize() = 0;

  /**
     *  @brief Writes output for a provided solution to a file with the provided name.
     *
     *  @details In some cases (currently only the PMLSurface) the boundary condition can have its own mesh and can thus also have data to visualize. As an example of the distinction: For a surface of Dirichlet data (DirichletSurface) all the boundary does is set the degrees of freedom on the surface of the inner domain to the values they should have.
     * As a consequence, the object has no interior mesh and the it can be checked in the output of the inner domain if the boundary method has done its job correctly so no output is required. For a PML domain, however, there is an interior mesh in which the solution is damped. Visual output of the solution in the PML domain can be helpful to understand problems with reflections etc. As a consequence, this function will usually be called on all boundary conditions but most won't perform any tasks.
     * 
     *  @see   PMLSurface::output_results()
     * 
     *  @param in_solution This parameter provides the values of the local dofs. In the case of the PMLSurface, these values are the computed E-field on the degrees of freedom that are active in the PMLDomain, i.e. have support in the PML domain. 
     *  @param filename The output will typically be written to a paraview-compatible format like .vtk and .vtu. This string does not contain the file endings. So if you want to write to a file solution.vtk you would only provide "solution".
     *  @return  This function returns the complete filename to which it has written the data. This can be used by the caller to generate meta-files for paraview which load for example the solution on the interior and all adjacent pml domains together.
     */
  virtual std::string output_results(const dealii::Vector<ComplexNumber> & in_solution, std::string filename) = 0;
  
  /**
     *  @brief Checks if a 2D coordinate is on the a surface of the boundary methods domain.
     *
     *  @details This function is currently only being used for HSIE. It checks if a point on the interface shared between the inner domain and the boundary method is also at a surface of that boundary, i.e. if this point is also relevant for another boundary method.
     * 
     *  @see   HSIESurface::HSIESurface::get_vertices_for_boundary_id()
     * 
     *  @param in_p The point in the 2D parametrization of the surface.
     *  @param in_bid The boundary id of the other boundary condition, for which it should be checked if this point is on it.
     *  @return     Returns true if this is on such an edge and false if it isn't.
     */
  virtual bool is_point_at_boundary(Position2D in_p, BoundaryId in_bid) = 0;

  /**
     *  @brief If the boundary condition has its own mesh, this function iterates over the mesh and sets boundary ids on the mesh
     *
     *  @details Consider, as an example, a PML domain. For such a domain we have one surface facing the inner domain, 4 surfaces facing other boundary conditions and the remainder of the boundary condition faces outward. All of these surfaces have to be dealt with individually. On the boundary facing the interior we need to identify the dofs with their equivalent dofs on the interior domain. On durfaces shared with other boundary conditions we have to decide on ownership and set them properly (if the other boundary condition is a Dirichlet Boundary, for example, we need to enforce a PML-damped dirichlet data. If it is a neighbor surface, we need to perform communication with the neighbor. etc.)
     * For the outward surface on the other hand we need to set metallic boundary conditions. To make these actions more efficient, we set boundary ids on the cells, so after that we can simply derive the operation required on a cell by asking for its boundary id and we can also simply get all dofs that require a certain action simply by their boundary id.
     * 
     *  @see  PMLSurface::set_mesh_boundary_ids()
     */
  void set_mesh_boundary_ids();

  /**
     *  @brief Returns a vector of all boundary ids associated with dofs in this domain.
     *  @return  The returned vector contains all boundary IDs that are relevant on this domain.
     */
  auto get_boundary_ids() -> std::vector<BoundaryId>;
  
  /**
     *  @brief Returns a vector of all degrees of freedom shared with the inner domain.
     *
     *  @details For those boundary conditions that generate their own dofs (HSIE, PML and Neighbor) we need to figure out dpf sets that need to be coupled. For example: The PML domain has dofs on the surface shared with the interior domain. These should have the same index as their counterpart in the interior domain. To this goal, we exchange a vector of all dofs on the surface we have previously sorted. That way, we only need to call this function on the interior domain and the boundary method and identify the dofs in the two returned vectors that have the same index.
     * 
     *  @see   InnerDomain::get_surface_dof_vector_for_boundary_id()
     * 
     *  @return InterfaceDofData always contains a reference points and index for every index found on the surface. The reference points are used for sorting, the index is the actual data used by the caller.
     */
  virtual auto get_dof_association() -> std::vector<InterfaceDofData> = 0;

  /**
     *  @brief More general version of the function above that can also handle interfaces with other boundary ids.
     *
     *  @details This function typically holds the actual implementation of the function above as well as implementations for the boundaries shared with other boudnary conditions. It differs in all the derived types.
     * 
     *  @see  PMLSurface::get_dof_association_by_boundary_id()
     * 
     *  @param boundary_id This is the boundary id as seen from this domain.
     *  @return InterfaceDofData always contains a reference points and index for every index found on the surface. The reference points are used for sorting, the index is the actual data used by the caller.
     */
  virtual auto get_dof_association_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<InterfaceDofData> = 0;

  /**
     *  @brief Specific version of the function above that provides the indices in the returned vector by their globally unique id instead of local numbering. Lets say a Boundary Condition has 1000 own degrees of freedom then the method above will return dof ids in the range [0,1000] whereas this function will return the index ids in the numbering relevant to the current sweep of local problem which is globally unique to that problem.
     *
     *  @details This function performs the same task as the one above but returns the global indices of the dofs instead of the local ones.
     * 
     *  @see  get_dof_association()
     * 
     *  @param boundary_id This is the boundary id as seen from this domain.
     *  @return At this point, the base_points are no longer required since this function gets called later in the preparation stage. For that reason, this function does not return the base points of the dofs anymore and instead only returns the dof indices. The indices, however, are still in the same order.
     */
  virtual auto get_global_dof_indices_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<DofNumber>;

  /**
     *  @brief If this object owns degrees of freedom, this function fills a sparsity pattern for their global indices.
     *
     *  @details The classes local and non-local problem manage matrices to solve either directly or iteratively. Matrices in a HPC setting that are generated from a fem system are usually sparse. A sparsity pattern is an object, that describes in which positions of a matrix there are non-zero entries that require storing. 
     * This function updates a given sparsity pattern with the entries related to this object. An important sidemark: In deal.II there are constraint object which store hanging node constraints as well as inhomogenous constraints like Dirichlet data. When filling a matrix, there can sometimes be ways of making use of such constraints and reducing the required memory this way.
     * 
     *  @see deal.II description of sparsity patterns and constraints
     * 
     *  @param in_dsp The sparsity pattern to be updated
     *  @param constraints The constraint object that is used to perform this action effectively
     */
  
  virtual void fill_sparsity_pattern(dealii::DynamicSparsityPattern * in_dsp, Constraints * constraints) = 0;
  
  /**
     *  @brief Fills a provided matrix and right-hand side vector with the data related to the current fem system under consideration and related to this boundary condition.
     *
     *  @details Most of a fem code is preparation to assemble a matrix. This function is the last step in that process. Once dofs have been enumerated and materials and geometries setup, this function performs the task of filling a system matrix with the contributions to the set of linear equations. Called after the previous function, this function writes the actual values into the system matrix that were marked as non-zero in the previous function. The same function exists on the InnerDomain object and these objects together build the entire system matrix.
     * 
     *  @see InnerDomain::fill_matrix()
     * 
     *  @param matrix The matrix to fill with the entries related to this object.
     *  @param rhs If dofs in this system are inhomogenously constraint (as in the case of Dirichlet data or jump coupling) the system has a non-zero right hand side (in the sense of a linear system A*x = b). It makes sense to assemble the matrix and the right-hand side together. This is the vector that will store the vector b.
     *  @param constraints The constraint object is used to determine values that have a fixed value and to use that information to reduce the memory consumption of the matrix as well as assembling the right-hand side vector.
     */

  virtual void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix* matrix, NumericVectorDistributed* rhs, Constraints * constraints) = 0; 

  /**
     *  @brief Handles the communication of non-locally owned dofs and thus finishes the setup of the object.
     *
     *  @details In cases where not all locally active dofs are locally owned (for example for two pml domains, the dofs on the shared surface are only owned by one of two processes) this function handles the numbering of the dofs once the non-owned dofs have been communicated.
     * 
     */
  virtual void finish_dof_index_initialization();  
  
  /**
     *  @brief Builds a constraint object that represents fixed values of degrees of freedom associated with this object.
     *
     *  @details For a Dirichlet-data surface, this writes the dirichlet data into the AffineConstraints object. In a PML Surface this writes the zero constraints of the outward surface to the constraint object. Constraint objects can be merged. Therefore this object builds a new one, containing only the constraints related to this boundary contidion. It can then be merged into another one.
     * 
     *  @return Returns a new constraint object relating only to the current boundary condition to be merged into one for the entire local computation-
     */
  virtual auto make_constraints() -> Constraints;

  /**
   *  @brief Computes the L2-norm of the solution passed in on the shared interface with the interior domain.
   *
   *  @details This function evaluates the provided dof values as a solution on the surface connected to the interior domain. That function is then integrated across the surface as an L2 integral.
   * 
   *  @param solution The provided values of the degrees of freedom related to this boundary condition.
   *  @return The function returns the L2 norm of the function computed along the surface connecting the boundary condition with the interior domain.
   */
  double boundary_norm(NumericVectorDistributed * solution); 

  /**
     *  @brief Computes the L2-norm of the solution passed in as an argument on the solution passed in as the second argument.
     *
     *  @details Thisi function performs the same action as the previous function but does so an an arbitrary surface of the boundary condition instead of only working for the surface facing the interior domain.
     * 
     *  @param solution The values of the degrees of freedom to be used for this computation. These dof values represent an electircal field that can be integrated over the somain surface.
     *  @param b_id The boundary id of the surface the function is supposed to integrate across.
     *  @return The function returns the L2 norm of the field provided in the solution argument across the surface b_id.
     */
  double boundary_surface_norm(NumericVectorDistributed * solution, BoundaryId b_id);

/**
     *  @brief Counts the number of cells associated with the boundary passed in as an argument.
     *
     *  @details  It can be useful for testing purposes to count the number of cells forming a certain surface. Imagine if you will a domain discretized by 3 cells in x-direction, 4 in y and 5 in z-direction. The suraces for any combination of 2 directions then have a known number of cells. We can use this knowledge to test if our mesh-coloring algoithms work or not.
     * 
     *  @param  boundary_id The boundary we are counting the cells for.
     *  @return The number of cells the method found that connect directly with the boundary boundary_id 
     */
  virtual unsigned int cells_for_boundary_id(unsigned int boundary_id);

/**
     *  @brief In some cases we have more then one option to validate how many dofs a domain should have. This is one way of computng that value for comparison with numbers that arise from the compuataion directly.
     *
     *  @details This is an internal function and should be used with caution. The function only warns the user. It does not abort the execution.
     * 
     */
  void print_dof_validation();

/**
     *  @brief Triggers the internal validation routine. Prints an error message if invalid.
     *
     *  @details This is for internal use. It validates if all dofs have a value that is valid in the current scope. Since this is mainly a core implementation concern there is only an error message printed to the console - errors in this code should no longer be occuring.
     * 
     */
  void force_validation();

  /**
     *  @brief Counts the number of cells used in the object.
     *
     *  @details For msot derived types, this is the number of 2D surface cells of the inner domain. For PML, however the value is the number of 3D cellx. It is always the number of steps a dof_handler iterates to handle the matrix filling operation.
     * 
     *  @return The number of cells.
     */
  virtual unsigned int n_cells();
};