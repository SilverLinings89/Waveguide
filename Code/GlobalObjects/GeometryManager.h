#pragma once

/**
 * @file GeometryManager.h
 * @author your name (you@domain.com)
 * @brief Contains the GeometryManager header, which handles the distribution of the computational domain onto processes and most of the initialization.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

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

struct LevelGeometry {
  std::array<SurfaceType, 6> surface_type;
  CubeSurfaceTruncationState is_surface_truncated;
  std::array<std::shared_ptr<BoundaryCondition> , 6> surfaces;
  std::vector<dealii::IndexSet> dof_distribution;
  DofNumber n_local_dofs;
  DofNumber n_total_level_dofs;
  InnerDomain * inner_domain;
};

class GeometryManager {
 public:
  double input_connector_length;
  double output_connector_length;
  double shape_sector_length;
  unsigned int shape_sector_count;
  unsigned int local_inner_dofs;
  bool are_surface_meshes_initialized;
  double h_x;
  double h_y;
  double h_z;

  std::array<unsigned int, 6> dofs_at_surface;
  std::array<dealii::Triangulation<2, 2>, 6> surface_meshes;
  std::array<double, 6> surface_extremal_coordinate;
  std::pair<double, double> local_x_range;
  std::pair<double, double> local_y_range;
  std::pair<double, double> local_z_range;

  std::pair<double, double> global_x_range;
  std::pair<double, double> global_y_range;
  std::pair<double, double> global_z_range;

  std::array<LevelGeometry,4> levels;

  GeometryManager();

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
   * @return double \epsilon_r * \omega^2
   */
  double eps_kappa_2(Position);

  /**
   * @brief Like the function above but without epsilon_r.
   * Since this value is independent of the position, this function has no arguments.
   * @return double \omega^2
   */
  double kappa_2();

  /**
   * @brief Computes the range of the coordinate x this process is responsible for.
   * Since the local domains are always of the form [min_x, max_x]\times[min_y, max_y]\times[min_z, max_z], these ranges can be used to describe the local problem.
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
   * @brief Computes scalar \epsilon_r for the given location.
   * 
   * @return double \epsilon_r of material at given location.
   */
  double get_epsilon_for_point(const Position &);
  auto get_boundary_for_direction(Direction) -> BoundaryId;
  auto get_direction_for_boundary_id(BoundaryId) -> Direction;
  void validate_global_dof_indices(unsigned int in_level);
  SurfaceType get_surface_type(BoundaryId b_id, unsigned int level);
  void distribute_dofs_on_level(unsigned int level);
  void set_surface_types_and_properties(unsigned int level);
  void initialize_surfaces_on_level(unsigned int level);
  void initialize_level(unsigned int level);
  void print_level_dof_counts(unsigned int level);
  void perform_mpi_dof_exchange(unsigned int level);
};
