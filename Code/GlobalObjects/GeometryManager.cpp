#include <deal.II/base/mpi.h>
#include <deal.II/grid/tria.h>
#include "GlobalObjects.h"
#include "../Core/Enums.h"
#include "GeometryManager.h"
#include "../Core/InnerDomain.h"
#include "../BoundaryCondition/EmptySurface.h"
#include "../BoundaryCondition/DirichletSurface.h"
#include "../BoundaryCondition/HSIESurface.h"
#include "../BoundaryCondition/PMLSurface.h"
#include "../BoundaryCondition/NeighborSurface.h"
#include "../Helpers/staticfunctions.h"

unsigned int count_occurences_of_max_uint(std::vector<unsigned int> vector) {
  unsigned int ret = 0;
  for(unsigned int i = 0; i < vector.size(); i++) {
    if(vector[i] == UINT_MAX) {
      ret++;
    }
  }
  return ret;
}

unsigned int count_occurences_of_zero(std::vector<unsigned int> vector) {
  unsigned int ret = 0;
  for(unsigned int i = 0; i < vector.size(); i++) {
    if(vector[i] == 0) {
      ret++;
    }
  }
  return ret;
}

GeometryManager::GeometryManager() {
  are_surface_meshes_initialized = false;
}

GeometryManager::~GeometryManager() {}

void GeometryManager::initialize() {
  print_info("GeometryManager::initialize", "Start");
  set_x_range(compute_x_range());
  set_y_range(compute_y_range());
  set_z_range(compute_z_range());
  h_x = (local_x_range.second - local_x_range.first) / GlobalParams.Cells_in_x;
  h_y = (local_y_range.second - local_y_range.first) / GlobalParams.Cells_in_y;
  h_z = (local_z_range.second - local_z_range.first) / GlobalParams.Cells_in_z;
  surface_extremal_coordinate[0] = local_x_range.first;
  surface_extremal_coordinate[1] = local_x_range.second;
  surface_extremal_coordinate[2] = local_y_range.first;
  surface_extremal_coordinate[3] = local_y_range.second;
  surface_extremal_coordinate[4] = local_z_range.first;
  surface_extremal_coordinate[5] = local_z_range.second;
  
  initialize_level(0);
  validate_global_dof_indices(0);
  if(GlobalParams.Blocks_in_z_direction != 1) {
    initialize_level(1);
    validate_global_dof_indices(1);
  }
  if(GlobalParams.Blocks_in_y_direction != 1) {
    initialize_level(2);
    validate_global_dof_indices(2);
  }
  if(GlobalParams.Blocks_in_x_direction != 1) {
    initialize_level(3);
    validate_global_dof_indices(3);
  }
  print_info("GeometryManager::initialize", "End");
}

void GeometryManager::initialize_inner_domain(unsigned int in_level) {
  levels[in_level].inner_domain = new InnerDomain(in_level);
  levels[in_level].inner_domain->make_grid();
  if(!are_surface_meshes_initialized) {
    for (unsigned int side = 0; side < 6; side++) {
      dealii::Triangulation<2, 3> temp_triangulation;
      dealii::Triangulation<2> surf_tria;
      Mesh tria;
      tria.copy_triangulation(levels[in_level].inner_domain->triangulation);
      std::set<unsigned int> b_ids;
      b_ids.insert(side);
      switch (side) {
        case 0:
          dealii::GridTools::transform(Transform_0_to_5, tria);
          break;
        case 1:
          dealii::GridTools::transform(Transform_1_to_5, tria);
          break;
        case 2:
          dealii::GridTools::transform(Transform_2_to_5, tria);
          break;
        case 3:
          dealii::GridTools::transform(Transform_3_to_5, tria);
          break;
        case 4:
          dealii::GridTools::transform(Transform_4_to_5, tria);
          break;
        default:
          break;
      }
      dealii::GridGenerator::extract_boundary_mesh(tria, temp_triangulation, b_ids);
      dealii::GridGenerator::flatten_triangulation(temp_triangulation, surface_meshes[side]);
    }
    are_surface_meshes_initialized = true;
  }
}

void GeometryManager::validate_global_dof_indices(unsigned int in_level) {
  unsigned int occurences = count_occurences_of_max_uint(levels[in_level].inner_domain->global_index_mapping);
  if(occurences != 0) {
    std::cout << "On level " << in_level << " on " << GlobalParams.MPI_Rank << " in the interior there were " << occurences <<std::endl;
  }
  for(unsigned int i = 0; i < 6; i++) {
    occurences = count_occurences_of_max_uint(levels[in_level].surfaces[i]->global_index_mapping);
    if(occurences != 0) {
      levels[in_level].surfaces[i]->print_dof_validation();
    }
  }
}

void GeometryManager::distribute_dofs_on_level(unsigned int in_level) {
  unsigned int n_owned_dofs = 0;
  levels[in_level].inner_domain->initialize_dof_counts(levels[in_level].inner_domain->compute_n_locally_active_dofs(), levels[in_level].inner_domain->compute_n_locally_owned_dofs());
  n_owned_dofs += levels[in_level].inner_domain->n_locally_owned_dofs;
  for(unsigned int surf = 0; surf < 6; surf++) {
    unsigned int surf_active = levels[in_level].surfaces[surf]->compute_n_locally_active_dofs();
    unsigned int surf_owned = levels[in_level].surfaces[surf]->compute_n_locally_owned_dofs();
    levels[in_level].surfaces[surf]->initialize_dof_counts(surf_active, surf_owned);
    n_owned_dofs += levels[in_level].surfaces[surf]->n_locally_owned_dofs;
  }
  levels[in_level].dof_distribution = dealii::Utilities::MPI::create_ascending_partitioning(GlobalMPI.communicators_by_level[in_level], n_owned_dofs);
  unsigned int first_dof = levels[in_level].dof_distribution[GlobalMPI.rank_on_level[in_level]].nth_index_in_set(0);
  levels[in_level].n_local_dofs = levels[in_level].dof_distribution[GlobalMPI.rank_on_level[in_level]].n_elements();
  levels[in_level].n_total_level_dofs = levels[in_level].dof_distribution[0].size();
  levels[in_level].inner_domain->determine_non_owned_dofs();
  for(unsigned int i = 0; i < 6; i++) {
    levels[in_level].surfaces[i]->determine_non_owned_dofs();
  }
  levels[in_level].inner_domain->freeze_ownership();
  for(unsigned int i = 0; i < 6; i++) {
    levels[in_level].surfaces[i]->freeze_ownership();
  }
  levels[in_level].inner_domain->finish_initialization(first_dof);
  
  
  first_dof += levels[in_level].inner_domain->n_locally_owned_dofs;

  for(unsigned int i = 0; i < 6; i++) {
    if(Geometry.levels[in_level].surface_type[i] != SurfaceType::NEIGHBOR_SURFACE) {
      levels[in_level].surfaces[i]->finish_initialization(first_dof);
      first_dof += levels[in_level].surfaces[i]->n_locally_owned_dofs;
    }
  }  

  MPI_Barrier(MPI_COMM_WORLD);
  for(unsigned int i = 0; i < 6; i++) {
    if(Geometry.levels[in_level].surface_type[i] != SurfaceType::NEIGHBOR_SURFACE) {
      Geometry.levels[in_level].surfaces[i]->finish_dof_index_initialization();
    }
  }
  
  if(Geometry.levels[in_level].surface_type[0] == SurfaceType::NEIGHBOR_SURFACE) {
    Geometry.levels[in_level].surfaces[0]->finish_dof_index_initialization();
  }  
  if(Geometry.levels[in_level].surface_type[1] == SurfaceType::NEIGHBOR_SURFACE) {
    Geometry.levels[in_level].surfaces[1]->finish_dof_index_initialization();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(Geometry.levels[in_level].surface_type[2] == SurfaceType::NEIGHBOR_SURFACE) {
    Geometry.levels[in_level].surfaces[2]->finish_dof_index_initialization();
  }
  if(Geometry.levels[in_level].surface_type[3] == SurfaceType::NEIGHBOR_SURFACE) {
    Geometry.levels[in_level].surfaces[3]->finish_dof_index_initialization();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(Geometry.levels[in_level].surface_type[4] == SurfaceType::NEIGHBOR_SURFACE) {
    Geometry.levels[in_level].surfaces[4]->finish_dof_index_initialization();
  }
  if(Geometry.levels[in_level].surface_type[5] == SurfaceType::NEIGHBOR_SURFACE) {
    Geometry.levels[in_level].surfaces[5]->finish_dof_index_initialization();
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  if(Geometry.levels[in_level].surface_type[0] == SurfaceType::NEIGHBOR_SURFACE) {
    Geometry.levels[in_level].surfaces[0]->finish_dof_index_initialization();
  }  
  if(Geometry.levels[in_level].surface_type[1] == SurfaceType::NEIGHBOR_SURFACE) {
    Geometry.levels[in_level].surfaces[1]->finish_dof_index_initialization();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(Geometry.levels[in_level].surface_type[2] == SurfaceType::NEIGHBOR_SURFACE) {
    Geometry.levels[in_level].surfaces[2]->finish_dof_index_initialization();
  }
  if(Geometry.levels[in_level].surface_type[3] == SurfaceType::NEIGHBOR_SURFACE) {
    Geometry.levels[in_level].surfaces[3]->finish_dof_index_initialization();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(Geometry.levels[in_level].surface_type[4] == SurfaceType::NEIGHBOR_SURFACE) {
    Geometry.levels[in_level].surfaces[4]->finish_dof_index_initialization();
  }
  if(Geometry.levels[in_level].surface_type[5] == SurfaceType::NEIGHBOR_SURFACE) {
    Geometry.levels[in_level].surfaces[5]->finish_dof_index_initialization();
  }
}

dealii::Tensor<2,3> GeometryManager::get_epsilon_tensor(const Position & in_p) {
  dealii::Tensor<2,3> ret;
  const double local_epsilon = get_epsilon_for_point(in_p);
  for(unsigned int i = 0; i < 3; i++) {
    for(unsigned int j = 0; j < 3; j++) {
      if(i == j) {
        ret[i][j] = local_epsilon;
      } else {
        ret[i][j] = 0;
      }
    }
  }
  return ret;
}

double GeometryManager::get_epsilon_for_point(const Position & in_p) {
  if(math_coordinate_in_waveguide(in_p)) {
    return GlobalParams.Epsilon_R_in_waveguide;
  } else {
    return GlobalParams.Epsilon_R_outside_waveguide;
  }
}

double GeometryManager::eps_kappa_2(Position in_p) {
  return (math_coordinate_in_waveguide(in_p)? GlobalParams.Epsilon_R_in_waveguide : GlobalParams.Epsilon_R_outside_waveguide) * GlobalParams.Omega * GlobalParams.Omega;
}

double GeometryManager::kappa_2() {
  return GlobalParams.Omega * GlobalParams.Omega;
}

void GeometryManager::set_x_range(std::pair<double, double> in_range) {
  this->local_x_range = in_range;
  global_x_range = std::pair<double, double>(-GlobalParams.Geometry_Size_X / 2.0, GlobalParams.Geometry_Size_X / 2.0);
}

void GeometryManager::set_y_range(std::pair<double, double> in_range) {
  this->local_y_range = in_range;
  global_y_range = std::pair<double, double> (-GlobalParams.Geometry_Size_Y / 2.0, GlobalParams.Geometry_Size_Y / 2.0);
}

void GeometryManager::set_z_range(std::pair<double, double> in_range) {
  this->local_z_range = in_range;
  global_z_range = std::pair<double, double>(0.0, GlobalParams.Geometry_Size_Z);
}

std::pair<double, double> GeometryManager::compute_x_range() {
  if (GlobalParams.Blocks_in_x_direction == 1) {
    return std::pair<double, double>(-GlobalParams.Geometry_Size_X / 2.0, GlobalParams.Geometry_Size_X / 2.0);
  } else {
    double length = GlobalParams.Geometry_Size_X / ((double) GlobalParams.Blocks_in_x_direction);
    int block_index = GlobalParams.MPI_Rank % GlobalParams.Blocks_in_x_direction;
    double min = -GlobalParams.Geometry_Size_X / 2.0 + block_index * length;
    return std::pair<double, double>(min, min + length);
  }
}

std::pair<double, double> GeometryManager::compute_y_range() {
  if (GlobalParams.Blocks_in_y_direction == 1) {
    return std::pair<double, double>(-GlobalParams.Geometry_Size_Y / 2.0, GlobalParams.Geometry_Size_Y / 2.0);
  } else {
    double length = GlobalParams.Geometry_Size_Y / ((double) GlobalParams.Blocks_in_y_direction);
    int block_processor_count = GlobalParams.Blocks_in_x_direction;
    int block_index = (GlobalParams.MPI_Rank % (GlobalParams.Blocks_in_x_direction * GlobalParams.Blocks_in_y_direction)) / block_processor_count;
    double min = -GlobalParams.Geometry_Size_Y / 2.0 + block_index * length;
    return std::pair<double, double>(min, min + length);
  }
}

std::pair<double, double> GeometryManager::compute_z_range() {
  if (GlobalParams.Blocks_in_z_direction == 1) {
    return std::pair<double, double>(0, GlobalParams.Geometry_Size_Z);
  } else {
    double length = GlobalParams.Geometry_Size_Z / ((double) GlobalParams.Blocks_in_z_direction);
    int block_processor_count = GlobalParams.Blocks_in_x_direction * GlobalParams.Blocks_in_y_direction;
    int block_index = GlobalParams.MPI_Rank / block_processor_count;
    double min = block_index * length;
    return std::pair<double, double>(min, min + length);
  }
}

std::pair<bool, unsigned int> GeometryManager::get_global_neighbor_for_interface(Direction in_direction) {
  std::pair<bool, unsigned int> ret(true, 0);
  switch (in_direction) {
    case Direction::MinusX:
      if (GlobalParams.Index_in_x_direction == 0) {
        ret.first = false;
      } else {
        ret.second = GlobalParams.MPI_Rank - 1;
      }
      break;
    case Direction::PlusX:
      if (GlobalParams.Index_in_x_direction == GlobalParams.Blocks_in_x_direction - 1) {
        ret.first = false;
      } else {
        ret.second = GlobalParams.MPI_Rank + 1;
      }
      break;
    case Direction::MinusY:
      if (GlobalParams.Index_in_y_direction == 0) {
        ret.first = false;
      } else {
        ret.second = GlobalParams.MPI_Rank - GlobalParams.Blocks_in_x_direction;
      }
      break;
    case Direction::PlusY:
      if (GlobalParams.Index_in_y_direction == GlobalParams.Blocks_in_y_direction - 1) {
        ret.first = false;
      } else {
        ret.second = GlobalParams.MPI_Rank + GlobalParams.Blocks_in_x_direction;
      }
      break;
    case Direction::MinusZ:
      if (GlobalParams.Index_in_z_direction == 0) {
        ret.first = false;
      } else {
        ret.second = GlobalParams.MPI_Rank - (GlobalParams.Blocks_in_x_direction * GlobalParams.Blocks_in_y_direction);
      }
      break;
    case Direction::PlusZ:
      if (GlobalParams.Index_in_z_direction == GlobalParams.Blocks_in_z_direction - 1) {
        ret.first = false;
      } else {
        ret.second = GlobalParams.MPI_Rank + (GlobalParams.Blocks_in_x_direction * GlobalParams.Blocks_in_y_direction);
      }
      break;
  }
  return ret;
}

std::pair<bool, unsigned int> GeometryManager::get_level_neighbor_for_interface(Direction in_direction, unsigned int level) {
  std::pair<bool, unsigned int> ret(true, 0);
  if(level == 0) {
    return get_global_neighbor_for_interface(in_direction);
  }
  if(level == 1) {
    switch (in_direction) {
      case Direction::MinusX:
        if (GlobalParams.Index_in_x_direction == 0) {
          ret.first = false;
        } else {
          ret.second = (GlobalParams.MPI_Rank - 1) % (GlobalParams.Blocks_in_x_direction * GlobalParams.Blocks_in_y_direction);
        }
        break;
      case Direction::PlusX:
        if (GlobalParams.Index_in_x_direction == GlobalParams.Blocks_in_x_direction - 1) {
          ret.first = false;
        } else {
          ret.second = (GlobalParams.MPI_Rank + 1) % (GlobalParams.Blocks_in_x_direction * GlobalParams.Blocks_in_y_direction);
        }
        break;
      case Direction::MinusY:
        if (GlobalParams.Index_in_y_direction == 0) {
          ret.first = false;
        } else {
          ret.second = (GlobalParams.MPI_Rank - GlobalParams.Blocks_in_y_direction) % (GlobalParams.Blocks_in_x_direction * GlobalParams.Blocks_in_y_direction);
        }
        break;
      case Direction::PlusY:
        if (GlobalParams.Index_in_y_direction == GlobalParams.Blocks_in_y_direction - 1) {
          ret.first = false;
        } else {
          ret.second = (GlobalParams.MPI_Rank + GlobalParams.Blocks_in_y_direction) % (GlobalParams.Blocks_in_x_direction * GlobalParams.Blocks_in_y_direction);
        }
        break;
      case Direction::MinusZ:
        ret.first = false;
        break;
      case Direction::PlusZ:
        ret.first = false;
        break;
    }
  }
  if(level == 2) {
    switch (in_direction) {
      case Direction::MinusX:
        if (GlobalParams.Index_in_x_direction == 0) {
          ret.first = false;
        } else {
          ret.second = (GlobalParams.MPI_Rank - 1) % GlobalParams.Blocks_in_x_direction;
        }
        break;
      case Direction::PlusX:
        if (GlobalParams.Index_in_x_direction == GlobalParams.Blocks_in_x_direction - 1) {
          ret.first = false;
        } else {
          ret.second = (GlobalParams.MPI_Rank + 1) % GlobalParams.Blocks_in_x_direction;
        }
        break;
      case Direction::MinusY:
        ret.first = false;
        break;
      case Direction::PlusY:
        ret.first = false;
        break;
      case Direction::MinusZ:
        ret.first = false;
        break;
      case Direction::PlusZ:
        ret.first = false;
        break;
    }
  }  
  return ret;
}

bool GeometryManager::math_coordinate_in_waveguide(Position in_position) const {
  return (std::abs(in_position[0]) < (GlobalParams.Width_of_waveguide  / 2.0)) && (std::abs(in_position[1]) < (GlobalParams.Height_of_waveguide / 2.0));
}

BoundaryId GeometryManager::get_boundary_for_direction(Direction in_direction) {
  switch (in_direction) {
    case Direction::MinusX:
      return 0;
      break;
    case Direction::PlusX:
      return 1;
      break;
    case Direction::MinusY:
      return 2;
      break;
    case Direction::PlusY:
      return 3;
      break;
    case Direction::MinusZ:
      return 4;
      break;
    case Direction::PlusZ:
      return 5;
      break;
    default:
      std::cout << "Weird call in get boundary id for direction function" << std::endl;
      return 6;
  }
}

Direction GeometryManager::get_direction_for_boundary_id(BoundaryId in_bid) {
  switch (in_bid) {
    case 0:
      return Direction::MinusX;
      break;
    case 1:
      return Direction::PlusX;
      break;
    case 2:
      return Direction::MinusY;
      break;
    case 3:
      return Direction::PlusY;
      break;
    case 4:
      return Direction::MinusZ;
      break;
    case 5:
      return Direction::PlusZ;
      break;
    default:
      std::cout << "Weird call in get direction for boundary id function" << std::endl;
      return Direction::MinusX;
  }
}

SurfaceType get_surf_type_for_level_0(BoundaryId in_b_id) {
  // The lower interfaces for all sweeping directions are Open. Additionally there is dirichlet data. All others are ABC.
  if(in_b_id == 4) {
    return SurfaceType::OPEN_SURFACE;
  }
  if(GlobalParams.Sweeping_Level > 1 && in_b_id == 2 ) {
    return SurfaceType::OPEN_SURFACE;
  }
  if(GlobalParams.Sweeping_Level == 3 && in_b_id == 0 ) {
    return SurfaceType::OPEN_SURFACE;
  }
  return SurfaceType::ABC_SURFACE;
}

SurfaceType get_surf_type_for_level_1(BoundaryId b_id) {
  if(GlobalParams.Sweeping_Level == 1) {
    if(b_id == 4 || b_id == 5) {
      return SurfaceType::NEIGHBOR_SURFACE;
    }
    return SurfaceType::ABC_SURFACE;
  }

  if(GlobalParams.Sweeping_Level == 2) {
    if(b_id == 2 || b_id == 3) {
      return SurfaceType::NEIGHBOR_SURFACE;
    }
    if(b_id == 4) {
      return SurfaceType::OPEN_SURFACE;
    }
    return SurfaceType::ABC_SURFACE;
  }
  
  if(GlobalParams.Sweeping_Level == 3) {
    if(b_id == 0 || b_id == 1) {
      return SurfaceType::NEIGHBOR_SURFACE;
    }
    if(b_id == 4 || b_id == 2) {
      return SurfaceType::OPEN_SURFACE;
    }
    return SurfaceType::ABC_SURFACE;
  }
  return SurfaceType::ABC_SURFACE;
}

SurfaceType get_surf_type_for_level_2(BoundaryId b_id) {
  if(GlobalParams.Sweeping_Level == 2) {
    if(b_id == 2 || b_id == 3 || b_id == 4 || b_id == 5) {
      return SurfaceType::NEIGHBOR_SURFACE;
    }
    return SurfaceType::ABC_SURFACE;
  }
  
  if(GlobalParams.Sweeping_Level == 3) {
    if(b_id == 0 || b_id == 1 || b_id == 2 || b_id == 3) {
      return SurfaceType::NEIGHBOR_SURFACE;
    }
    if(b_id == 4) {
      return SurfaceType::OPEN_SURFACE;
    }
    return SurfaceType::ABC_SURFACE;
  }
  return SurfaceType::ABC_SURFACE;
}

SurfaceType get_surf_type_for_level_3(BoundaryId ) {
  return SurfaceType::NEIGHBOR_SURFACE;
}

std::pair<bool, SurfaceType> handle_exterior_boundaries(BoundaryId b_id) {
  // The fist value of the return pair describes if the boundary has been handled.
  std::pair<bool, SurfaceType> ret;
  ret.first = true;
  ret.second = SurfaceType::ABC_SURFACE;
  if(b_id == 0 && GlobalParams.Index_in_x_direction == 0) {
    return ret;
  }
  if(b_id == 1 && GlobalParams.Index_in_x_direction == GlobalParams.Blocks_in_x_direction - 1) {
    return ret;
  }
  if(b_id == 2 && GlobalParams.Index_in_y_direction == 0) {
    return ret;
  }
  if(b_id == 3 && GlobalParams.Index_in_y_direction == GlobalParams.Blocks_in_y_direction - 1) {
    return ret;
  }
  if(b_id == 4 && GlobalParams.Index_in_z_direction == 0 ) {
    if(GlobalParams.Signal_coupling_method == SignalCouplingMethod::Dirichlet) {
      ret.second = SurfaceType::DIRICHLET_SURFACE;
      return ret;
    } else {
      ret.second = SurfaceType::ABC_SURFACE;
      return ret;
    }
  }
  if(b_id == 5 && GlobalParams.Index_in_z_direction == GlobalParams.Blocks_in_z_direction - 1) {
    return ret;
  }
  ret.first = false;
  return ret;
}

SurfaceType GeometryManager::get_surface_type(BoundaryId b_id, unsigned int in_level) {
  std::pair<bool, SurfaceType> exterior_boundary = handle_exterior_boundaries(b_id);
  if(exterior_boundary.first) return exterior_boundary.second;
  switch(in_level) {
    case 0:
      return get_surf_type_for_level_0(b_id);
      break;
    case 1:
      return get_surf_type_for_level_1(b_id);
      break;
    case 2:
      return get_surf_type_for_level_2(b_id);
      break;
    case 3:
      return get_surf_type_for_level_3(b_id);
      break;
  }
  return SurfaceType::ABC_SURFACE;
}

void GeometryManager::set_surface_types_and_properties(unsigned int in_level) {
  for(unsigned int i = 0; i < 6; i++) {
    levels[in_level].surface_type[i] = get_surface_type(i,in_level);
    levels[in_level].is_surface_truncated[i] = (levels[in_level].surface_type[i] == SurfaceType::ABC_SURFACE);
  }
}

void GeometryManager::initialize_surfaces_on_level(unsigned int in_level) {
  for(unsigned int surf = 0; surf < 6; surf++) {
    switch (levels[in_level].surface_type[surf])
    {
      case SurfaceType::ABC_SURFACE:
        if(GlobalParams.BoundaryCondition == BoundaryConditionType::HSIE) {
          levels[in_level].surfaces[surf] = std::shared_ptr<BoundaryCondition>(new HSIESurface(surf, in_level));
        } else {
          levels[in_level].surfaces[surf] = std::shared_ptr<BoundaryCondition>(new PMLSurface(surf, in_level));
        }
        break;
      case SurfaceType::DIRICHLET_SURFACE:
        levels[in_level].surfaces[surf] = std::shared_ptr<BoundaryCondition>(new DirichletSurface(surf, in_level));
        break;
      case SurfaceType::NEIGHBOR_SURFACE:
        levels[in_level].surfaces[surf] = std::shared_ptr<BoundaryCondition>(new NeighborSurface(surf, in_level));
        break;
      case SurfaceType::OPEN_SURFACE:
        levels[in_level].surfaces[surf] = std::shared_ptr<BoundaryCondition>(new EmptySurface(surf,in_level));
        break;
      default:
        break;
    }
  }
  std::string output_details = "On level " + std::to_string(in_level) + " for process " + std::to_string(GlobalParams.MPI_Rank) + ": ";
  for(unsigned int surf = 0; surf < 6; surf++) {
    levels[in_level].surfaces[surf]->initialize();
    output_details += "Boundary " + std::to_string(surf) + " type ";
    switch (levels[in_level].surface_type[surf])
    {
      case SurfaceType::ABC_SURFACE:
        output_details += "PML(" + std::to_string(levels[in_level].surfaces[surf]->dof_counter) +")";
        break;
      case SurfaceType::DIRICHLET_SURFACE:
        output_details += "Dirichlet";
        break;
      case SurfaceType::NEIGHBOR_SURFACE:
        output_details += "Neighbor";
        break;
      case SurfaceType::OPEN_SURFACE:
        output_details += "Open";
        break;
      default:
        break;
    }
    output_details += ". ";
  }
  // std::cout << output_details <<std::endl;
}

void GeometryManager::initialize_level(unsigned int in_level) {
  print_info("GeometryManager::initialize_level", "Start level " + std::to_string(in_level));
  initialize_inner_domain(in_level);
  set_surface_types_and_properties(in_level);
  initialize_surfaces_on_level(in_level);
  distribute_dofs_on_level(in_level);
  print_level_dof_counts(in_level);
  print_info("GeometryManager::initialize_level", "End level " + std::to_string(in_level));
}

void GeometryManager::print_level_dof_counts(unsigned int level) {
  unsigned int n_total_inner_dofs =  dealii::Utilities::MPI::sum(Geometry.levels[level].inner_domain->n_locally_owned_dofs, GlobalMPI.communicators_by_level[level]);
  print_info("GeometryManager", "On level " + std::to_string(level) + " there are " + std::to_string(levels[level].n_total_level_dofs) + " total dofs. "+ std::to_string(n_total_inner_dofs) + " inner ones, " + std::to_string(levels[level].n_total_level_dofs-n_total_inner_dofs) + " boundary dofs.");
}