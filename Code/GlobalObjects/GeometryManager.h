#pragma once

#include <deal.II/base/index_set.h>
#include "../Core/Types.h"
#include "../BoundaryCondition/BoundaryCondition.h"
#include <memory>
#include <utility>
#include "../Core/Enums.h"

class InnerDomain;

struct LevelGeometry {
  std::array<SurfaceType, 6> surface_type;
  CubeSurfaceTruncationState is_surface_truncated;
  std::array<std::shared_ptr<BoundaryCondition> , 6> surfaces;
  unsigned int inner_first_dof;
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

  void initialize();
  void initialize_inner_domain(unsigned int in_level);
  void initialize_surfaces();
  void perform_initialization(unsigned int level);
  double eps_kappa_2(Position);

  std::pair<double, double> compute_x_range();
  std::pair<double, double> compute_y_range();
  std::pair<double, double> compute_z_range();

  void set_x_range(std::pair<double, double>);
  void set_y_range(std::pair<double, double>);
  void set_z_range(std::pair<double, double>);

  std::pair<bool, unsigned int> get_global_neighbor_for_interface(Direction);
  std::pair<bool, unsigned int> get_level_neighbor_for_interface(Direction, unsigned int);
  bool math_coordinate_in_waveguide(Position) const;
  dealii::Tensor<2,3> get_epsilon_tensor(const Position &);
  double get_epsilon_for_point(const Position &);
  Position get_global_center();
  Position get_local_center();
  auto get_boundary_for_direction(Direction) -> BoundaryId;
  auto get_direction_for_boundary_id(BoundaryId) -> Direction;
  void validate_global_dof_indices(unsigned int in_level);
  SurfaceType get_surface_type(BoundaryId b_id, unsigned int level);
  void distribute_dofs_on_level(unsigned int level);
  void set_surface_types_and_properties(unsigned int level);
  void initialize_surfaces_on_level(unsigned int level);
  void initialize_level(unsigned int level);
  bool is_surface_isolated(BoundaryId b_id, unsigned int in_level);
  void print_level_dof_counts(unsigned int level);
};
