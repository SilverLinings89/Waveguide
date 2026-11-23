#pragma once

/**
 * @file GeometryManager.h
 * @author Pascal Kraft
 * @brief Contains the GeometryManager header, which handles the distribution of the computational domain onto processes and most of the initialization.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <array>
#include <vector>
#include <deal.II/base/index_set.h>
#include "../Core/Types.h"
#include "../BoundaryCondition/BoundaryCondition.h"
#include <memory>
#include <utility>
#include "../Core/Enums.h"

class InnerDomain;

/**
 * \class GeometryManager
 * \brief One object of this type is globally available to handle the geometry of the computation (what is the global computational domain, what is computed locally).
 * 
 * This object is one of the first to be initialized. It contains the coordinate ranges locally and globally. It also has several LevelGeometry objects in a vector. This is the core data behind the sweeping hierarchy. These level objects contain:
 * - the surface types for all boundaries on this level
 * - pointers to the boundary condition objects
 * - dof counting data (how many dofs exist on the level, how many dofs does this process own on this level) and also which dofs are stored where in the dof_distribution member.
 * 
 * This object can also determine if a coordinate is inside or outside of the waveguide and computes kappa squared required for the assembly of Maxwell's equations.
 * 
 */

/**
 * @brief Geometry and dof data of one level of the sweeping hierarchy as seen by this process.
 *
 * GeometryManager::levels holds one object per level. Level 0 is the local problem of this process. Arrays of
 * length six are indexed by boundary id: 0 = -x, 1 = +x, 2 = -y, 3 = +y, 4 = -z, 5 = +z.
 */
struct LevelGeometry {
  /** @brief Type of each of the six surfaces of the local domain on this level, see GeometryManager::get_surface_type(). */
  std::array<SurfaceType, 6> surface_type;
  /** @brief True for surfaces of type ABC_SURFACE, i.e. surfaces that carry a PML or Hardy space infinite elements. The constructor of LocalProblem sets all six entries of level 0 to true. */
  CubeSurfaceTruncationState is_surface_truncated;
  /** @brief Boundary condition object of each surface. The derived type matches surface_type. Created in GeometryManager::initialize_surfaces_on_level(). */
  std::array<std::shared_ptr<BoundaryCondition> , 6> surfaces;
  /** @brief Locally owned dofs of every process of the level communicator, indexed by the rank in GlobalMPI.communicators_by_level[level]. The indices are global dof numbers of this level and each process owns one contiguous range. */
  std::vector<dealii::IndexSet> dof_distribution;
  /** @brief Number of dofs this process owns on this level (inner domain and boundary methods). */
  DofNumber n_local_dofs;
  /** @brief Number of dofs of the whole problem this process takes part in on this level, summed over all processes of the level communicator. */
  DofNumber n_total_level_dofs;
  /** @brief Mesh, dof handler and assembly of the interior of the local domain on this level. Created in GeometryManager::initialize_inner_domain(). */
  InnerDomain * inner_domain;
};

class GeometryManager {
 public:
  /** @brief Legacy member. It is never assigned or read. */
  double input_connector_length;
  /** @brief Legacy member. It is never assigned or read. */
  double output_connector_length;
  /** @brief Legacy member. It is never assigned or read. */
  double shape_sector_length;
  /** @brief Legacy member. It is never assigned or read. */
  unsigned int shape_sector_count;
  /** @brief Legacy member. It is never assigned or read. */
  unsigned int local_inner_dofs;
  /** @brief True once surface_meshes has been filled. This happens in the first call of initialize_inner_domain(). */
  bool are_surface_meshes_initialized;
  /** @brief Edge length in x-direction of the cells of the local inner domain (length of local_x_range divided by the parameter Cells_in_x). */
  double h_x;
  /** @brief Edge length in y-direction of the cells of the local inner domain. */
  double h_y;
  /** @brief Edge length in z-direction of the cells of the local inner domain. */
  double h_z;

  /** @brief Legacy member. It is never assigned or read. */
  std::array<unsigned int, 6> dofs_at_surface;
  /** @brief Two-dimensional meshes of the six surfaces of the local inner domain, indexed by boundary id. Each one is extracted from a copy of the inner domain mesh that has been transformed such that the surface takes the place of boundary 5 (+z), and is then flattened to two dimensions. Used by the Hardy space infinite elements. */
  std::array<dealii::Triangulation<2, 2>, 6> surface_meshes;
  /** @brief Position of each surface of the local inner domain along its normal axis, indexed by boundary id: lower and upper end of local_x_range, local_y_range and local_z_range. */
  std::array<double, 6> surface_extremal_coordinate;
  /** @brief Lower and upper x-coordinate of the inner domain of this process, see compute_x_range(). */
  std::pair<double, double> local_x_range;
  /** @brief Lower and upper y-coordinate of the inner domain of this process, see compute_y_range(). */
  std::pair<double, double> local_y_range;
  /** @brief Lower and upper z-coordinate of the inner domain of this process, see compute_z_range(). */
  std::pair<double, double> local_z_range;

  /** @brief Lower and upper x-coordinate of the global computational domain without boundary layers. The domain is centered at x = 0. */
  std::pair<double, double> global_x_range;
  /** @brief Lower and upper y-coordinate of the global computational domain without boundary layers. The domain is centered at y = 0. */
  std::pair<double, double> global_y_range;
  /** @brief Lower and upper z-coordinate of the global computational domain without boundary layers. It starts at the parameter global_z_shift and has the length Geometry_Size_Z. */
  std::pair<double, double> global_z_range;

  /** @brief Data of the levels of the sweeping hierarchy, indexed by level. Only level 0 and the levels of the directions with more than one block are initialized. */
  std::array<LevelGeometry,4> levels;

  /** @brief Marks the surface meshes as not built. All other setup happens in initialize(). */
  GeometryManager();

  /** @brief Does nothing. The inner domain objects of the levels are not deleted. */
  virtual ~GeometryManager();

  /**
   * @brief Parent of the entire initialization loop
   * This initializes all levels of the computation.
   */
  void initialize();

  /**
   * @brief On the level in_level this builds the InnerDomain object
   * 
   * @param in_level The level to perform the action on.
   */
  void initialize_inner_domain(unsigned int in_level);

  /**
   * @brief This function computes the term epsilon_r * omega^2 at a given location. 
   * This is required for the assembly of the Maxwell system.
   * 
   * @return double \f$\epsilon_r \omega^2\f$
   */
  double eps_kappa_2(Position);

  /**
   * @brief Like the function above but without epsilon_r.
   * Since this value is independent of the position, this function has no arguments.
   * @return double \f$\omega^2\f$
   */
  double kappa_2();

  /**
   * @brief Computes the range of the coordinate x this process is responsible for.
   * Since the local domains are always of the form \f$[min_x, max_x]\times[min_y, max_y]\times[min_z, max_z]\f$, these ranges can be used to describe the local problem.
   * @return std::pair<double, double> first is the lower bound of the range, second is the upper bound.
   */
  std::pair<double, double> compute_x_range();

  /**
   * @brief Same as above but for y.
   * 
   * @return std::pair<double, double> see above. 
   */
  std::pair<double, double> compute_y_range();

  /**
   * @brief Same as above but for z.
   * 
   * @return std::pair<double, double> see above.
   */
  std::pair<double, double> compute_z_range();

  /**
   * @brief Fixes the x-range this process is working on for its inner domain.
   * Boundary conditions can extend beyond this value however. The idea is to use the return value of compute_x_range().
   * @param inp_x the x_range to use locally.
   */
  void set_x_range(std::pair<double, double> inp_x);
  
  /**
   * @brief Fixes the y-range this process is working on for its inner domain.
   * Boundary conditions can extend beyond this value however. The idea is to use the return value of compute_y_range().
   * @param inp_y the y_range to use locally.
   */
  void set_y_range(std::pair<double, double> inp_y);

  /**
   * @brief Fixes the z-range this process is working on for its inner domain.
   * Boundary conditions can extend beyond this value however. The idea is to use the return value of compute_z_range().
   * @param inp_z the z_range to use locally.
   */
  void set_z_range(std::pair<double, double> inp_z);

  /**
   * @brief For a given direction, this function computes if there is a neighbor of this process in that direction and, if so, that process's rank.
   * 
   * @param dir The direction to go to
   * @return std::pair<bool, unsigned int> first: is there a process there? second: whats its rank.
   */
  std::pair<bool, unsigned int> get_global_neighbor_for_interface(Direction dir);

  /**
   * @brief Similar to the function above but gets the rank of the neighbor in a level communicator for the level in_level
   * 
   * @param dir Direction to check in
   * @param level The level we are operating on.
   * @return std::pair<bool, unsigned int> Same as above but second returns the rank in the level communicator.
   */
  std::pair<bool, unsigned int> get_level_neighbor_for_interface(Direction dir, unsigned int level);

  /**
   * @brief Checks if the coordinate is in the waveguide core or not.
   * The shape of the core depends on GlobalParams.cross_section: a rectangle, a circle, or, for a cross-section
   * from a mode file, the region where the permittivity is in the upper half of its range.
   *
   * @return true Location in mathematical coordinates corresponds with the interior of the waveguide.
   * @return false it does not.
   */
  bool math_coordinate_in_waveguide(Position) const;

  /**
   * @brief Returns a diagonalized material tensor that does not use transformation optics. Artifact.
   * 
   * @return dealii::Tensor<2,3> 
   */
  dealii::Tensor<2,3> get_epsilon_tensor(const Position &);

  /**
   * @brief Computes scalar \f$\epsilon_r\f$ for the given location in mathematical coordinates.
   * For the rectangular and the circular cross-section this is the permittivity of the core or of the cladding.
   * For a cross-section from a mode file it is the permittivity stored in the file (ExactSolution::permittivity_at()).
   *
   * @return double \f$\epsilon_r\f$ of material at given location.
   */
  double get_epsilon_for_point(const Position &);
  /**
   * @brief Translates a direction into the id of the boundary whose outward normal points in that direction.
   *
   * @return BoundaryId 0 = -x, 1 = +x, 2 = -y, 3 = +y, 4 = -z, 5 = +z.
   */
  auto get_boundary_for_direction(Direction) -> BoundaryId;
  /**
   * @brief Inverse of get_boundary_for_direction().
   *
   * @return Direction The outward normal direction of the boundary. An id above 5 prints a message and returns Direction::MinusX.
   */
  auto get_direction_for_boundary_id(BoundaryId) -> Direction;
  /**
   * @brief Checks that every dof of the inner domain and of the six surfaces has received a global index on the level.
   * Dofs without a global index are reported on the console. Nothing is changed.
   *
   * @param in_level The level to check.
   */
  void validate_global_dof_indices(unsigned int in_level);
  /**
   * @brief Determines the type of one surface of the local domain on a level.
   *
   * Surfaces on the boundary of the global domain are of type ABC_SURFACE (PML or Hardy space infinite elements).
   * There are two exceptions: the input side (boundary 4 of the first block in z) is a DIRICHLET_SURFACE if the
   * signal is coupled in by Dirichlet values, and the side walls are of type OPEN_SURFACE if the parameter
   * pec_side_walls is set. Interior interfaces depend on the level: they are NEIGHBOR_SURFACE if the neighboring
   * process belongs to the same problem of the level, and OPEN_SURFACE or ABC_SURFACE otherwise.
   *
   * @param b_id Boundary id of the surface.
   * @param level The level.
   * @return SurfaceType The type of the surface.
   */
  SurfaceType get_surface_type(BoundaryId b_id, unsigned int level);
  /**
   * @brief Numbers the dofs of the level globally.
   *
   * The inner domain and the six surfaces count their active and owned dofs. From the owned counts of all processes
   * of the level communicator an ascending partitioning is built (LevelGeometry::dof_distribution). Every process
   * then numbers its inner dofs first, followed by the dofs of its surfaces that are not neighbor surfaces. On levels
   * above 0 the indices of the dofs on interfaces between processes are exchanged in perform_mpi_dof_exchange().
   *
   * @param level The level.
   */
  void distribute_dofs_on_level(unsigned int level);
  /**
   * @brief Fills LevelGeometry::surface_type and LevelGeometry::is_surface_truncated for all six surfaces of the level.
   *
   * @param level The level.
   */
  void set_surface_types_and_properties(unsigned int level);
  /**
   * @brief Creates and initializes the boundary condition objects of the level according to the surface types.
   * An ABC_SURFACE becomes an HSIESurface or a PMLSurface depending on the parameter BoundaryCondition.
   *
   * @param level The level.
   */
  void initialize_surfaces_on_level(unsigned int level);
  /**
   * @brief Performs the complete setup of one level: inner domain, surface types, boundary condition objects and dof numbering.
   *
   * @param level The level.
   */
  void initialize_level(unsigned int level);
  /**
   * @brief Prints the total number of dofs of the level, split into dofs of the inner domains and dofs of the boundary methods.
   *
   * @param level The level.
   */
  void print_level_dof_counts(unsigned int level);
  /**
   * @brief Exchanges the global dof indices on the interfaces between neighboring processes of a level.
   *
   * The processes take turns in the order of their global MPI rank. The process whose turn it is completes the
   * index data of its neighbor surfaces 1, 3 and 5 (+x, +y, +z). All other processes complete the surfaces
   * 0, 2 and 4 that face this process.
   *
   * @param level The level.
   */
  void perform_mpi_dof_exchange(unsigned int level);
};
