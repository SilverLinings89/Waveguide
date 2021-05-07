#include <deal.II/base/mpi.h>
#include <deal.II/grid/tria.h>
#include "../Core/GlobalObjects.h"
#include "Enums.h"
#include "GeometryManager.h"
#include "../Core/NumericProblem.h"
#include "../BoundaryCondition/HSIESurface.h"
#include "../BoundaryCondition/PMLSurface.h"
#include "../BoundaryCondition/NeighborSurface.h"
#include "staticfunctions.h"

GeometryManager::GeometryManager() {
  inner_domain = new NumericProblem();
}

GeometryManager::~GeometryManager() {
  delete inner_domain;
}

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
  inner_domain->make_grid();
  initialize_inner_domain();
  initialize_local_level();
  initialize_surfaces();
  print_info("GeometryManager::initialize", "End");
}

unsigned int GeometryManager::compute_n_dofs_on_level(const unsigned int level) {
  unsigned int ret = local_inner_dofs;
  for(unsigned int i = 0; i < 6; i++) {
    if(levels[level].is_surface_truncated[i]) {
      ret += levels[0].surfaces[i]->dof_counter;
    }
  }
  return ret;
}

Position GeometryManager::get_local_center() {
  Position ret;
  ret[0] = (local_x_range.first + local_x_range.second) / 2.0;
  ret[1] = (local_y_range.first + local_y_range.second) / 2.0;
  ret[2] = (local_z_range.first + local_z_range.second) / 2.0;
  return ret;
}

Position GeometryManager::get_global_center() {
  Position ret;
  ret[0] = (global_x_range.first + global_x_range.second) / 2.0;
  ret[1] = (global_y_range.first + global_y_range.second) / 2.0;
  ret[2] = (global_z_range.first + global_z_range.second) / 2.0;
  return ret;
}

void GeometryManager::initialize_inner_domain() {
  for (unsigned int side = 0; side < 6; side++) {
    dealii::Triangulation<2, 3> temp_triangulation;
    dealii::Triangulation<2> surf_tria;
    Mesh tria;
    tria.copy_triangulation(inner_domain->triangulation);
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
    set_mesh_boundary_ids(side);
  }
}

void GeometryManager::initialize_surfaces() {
  local_inner_dofs = inner_domain->n_dofs;
  initialize_local_level();
  if(GlobalParams.Blocks_in_z_direction != 1) {
    initialize_level_1();
  }
  if(GlobalParams.Blocks_in_y_direction != 1) {
    initialize_level_2();
  }
  if(GlobalParams.Blocks_in_x_direction != 1) {
    initialize_level_3();
  }
  validate_surface_first_dof();
}

void GeometryManager::validate_surface_first_dof() {
  for(unsigned int i = 0; i < 6; i++) {
    if(levels[0].surface_first_dof[i] == 0) std::cout << "Level 0, Surface "<< i << " Error. Surface first dof has wrong value." << std::endl;
  }
  if(GlobalParams.Blocks_in_z_direction != 1) {
    for(unsigned int i = 0; i < 6; i++) {
      if(levels[1].surface_first_dof[i] == 0) std::cout << "Level 1, Surface "<< i << " Error. Surface first dof has wrong value." << std::endl;
    }
  }
  if(GlobalParams.Blocks_in_y_direction != 1) {
    for(unsigned int i = 0; i < 6; i++) {
      if(levels[2].surface_first_dof[i] == 0) std::cout << "Level 2, Surface "<< i << " Error. Surface first dof has wrong value." << std::endl;
    }
  }
  if(GlobalParams.Blocks_in_x_direction != 1) {
    for(unsigned int i = 0; i < 6; i++) {
      if(levels[3].surface_first_dof[i] == 0) std::cout << "Level 3, Surface "<< i << " Error. Surface first dof has wrong value." << std::endl;
    }
  }
}

void GeometryManager::initialize_local_level() {
  for(unsigned int i = 0; i < 6; i++) {
    levels[0].is_surface_truncated[i] = true;
  }
  levels[0].inner_first_dof = 0;
  unsigned int counter = local_inner_dofs;
  for(unsigned int i = 0; i < 6; i++) {
    levels[0].surface_first_dof[i] = counter;
    if(GlobalParams.BoundaryCondition == BoundaryConditionType::HSIE) {
      levels[0].surfaces[i] = std::shared_ptr<BoundaryCondition>(new HSIESurface(i, 0, counter));
    } else {
      levels[0].surfaces[i] = std::shared_ptr<BoundaryCondition>(new PMLSurface(i, 0, counter));
    }
    levels[0].surfaces[i]->initialize();
    counter += levels[0].surfaces[i]->dof_counter;
  }
  levels[0].n_local_dofs = counter;
  levels[0].n_total_level_dofs = counter;
}

void GeometryManager::initialize_level_1() {
  for(unsigned int i = 0; i < 6; i++) {
    levels[1].is_surface_truncated[i] = true;
  }
  levels[1].is_surface_truncated[4] = GlobalParams.Index_in_z_direction == 0;
  levels[1].is_surface_truncated[5] = GlobalParams.Index_in_z_direction == (GlobalParams.Blocks_in_z_direction - 1);
  perform_initialization(1);
  
}

void GeometryManager::initialize_level_2() {
  for(unsigned int i = 0; i < 6; i++) {
    levels[2].is_surface_truncated[i] = true;
  }
  levels[2].is_surface_truncated[2] = GlobalParams.Index_in_y_direction == 0;
  levels[2].is_surface_truncated[3] = GlobalParams.Index_in_y_direction == (GlobalParams.Blocks_in_y_direction - 1);
  levels[2].is_surface_truncated[4] = GlobalParams.Index_in_z_direction == 0;
  levels[2].is_surface_truncated[5] = GlobalParams.Index_in_z_direction == (GlobalParams.Blocks_in_z_direction - 1);
  perform_initialization(2);
}

void GeometryManager::initialize_level_3() {
  levels[3].is_surface_truncated[0] = GlobalParams.Index_in_x_direction == 0;
  levels[3].is_surface_truncated[1] = GlobalParams.Index_in_x_direction == (GlobalParams.Blocks_in_x_direction - 1);
  levels[3].is_surface_truncated[2] = GlobalParams.Index_in_y_direction == 0;
  levels[3].is_surface_truncated[3] = GlobalParams.Index_in_y_direction == (GlobalParams.Blocks_in_y_direction - 1);
  levels[3].is_surface_truncated[4] = GlobalParams.Index_in_z_direction == 0;
  levels[3].is_surface_truncated[5] = GlobalParams.Index_in_z_direction == (GlobalParams.Blocks_in_z_direction - 1);
  perform_initialization(3);
}

void GeometryManager::perform_initialization(unsigned int in_level) {
  print_info("GeometryManager::perform_initialization", "Start level " + std::to_string(in_level));
  unsigned int count = compute_n_dofs_on_level(in_level);
  std::cout << "Count: " << count << std::endl;
  std::cout << "Rank on level " << in_level <<": " <<  GlobalMPI.rank_on_level[in_level] << std::endl;
  levels[in_level].dof_distribution = dealii::Utilities::MPI::create_ascending_partitioning(GlobalMPI.communicators_by_level[in_level], count);
  levels[in_level].inner_first_dof = levels[in_level].dof_distribution[GlobalMPI.rank_on_level[in_level]].nth_index_in_set(0);
  levels[in_level].n_total_level_dofs = levels[in_level].dof_distribution[0].size();
  
  unsigned int first_dof = levels[in_level].inner_first_dof + local_inner_dofs;

  for(unsigned int surf = 0; surf < 6; surf++) {
    levels[in_level].surface_first_dof[surf] = first_dof;
    if(levels[in_level].is_surface_truncated[surf]) {
      if(GlobalParams.BoundaryCondition == BoundaryConditionType::HSIE) {
        levels[in_level].surfaces[surf] = std::make_shared<HSIESurface>(surf, in_level, first_dof);
      } else {
        levels[in_level].surfaces[surf] = std::make_shared<PMLSurface>(surf, in_level, first_dof);
      }
    } else {
      levels[in_level].surfaces[surf] = std::make_shared<NeighborSurface>(surf, in_level, first_dof);
    }
    levels[in_level].surfaces[surf]->initialize();
    first_dof += levels[in_level].surfaces[surf]->dof_counter;
  }
  levels[in_level].n_local_dofs = compute_n_dofs_on_level(in_level);
  if(GlobalParams.MPI_Rank == 3) {
    for(unsigned int i = 0; i < 6; i++) {
      std::cout << "Surface " << i << ": " << levels[in_level].surface_first_dof[i] << std::endl;
    } 
  }
  print_info("GeometryManager::perform_initialization", "End level " + std::to_string(in_level));
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

void GeometryManager::set_x_range(std::pair<double, double> in_range) {
  this->local_x_range = in_range;
  std::pair<double, double> global_range(-GlobalParams.Geometry_Size_X / 2.0, GlobalParams.Geometry_Size_X / 2.0);
  this->global_x_range = global_range;
  return;
}

void GeometryManager::set_y_range(std::pair<double, double> in_range) {
  this->local_y_range = in_range;
  std::pair<double, double> global_range(-GlobalParams.Geometry_Size_Y / 2.0, GlobalParams.Geometry_Size_Y / 2.0);
  this->global_y_range = global_range;
  return;
}

void GeometryManager::set_z_range(std::pair<double, double> in_range) {
  this->local_z_range = in_range;
  std::pair<double, double> global_range(0.0, GlobalParams.Geometry_Size_Z);
  this->global_z_range = global_range;
  return;
}

std::pair<double, double> GeometryManager::compute_x_range() {
  if (GlobalParams.Blocks_in_x_direction == 1) {
    return std::pair<double, double>(-GlobalParams.Geometry_Size_X / 2.0,
        GlobalParams.Geometry_Size_X / 2.0);
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
      if (GlobalParams.Index_in_x_direction ==
          GlobalParams.Blocks_in_x_direction - 1) {
        ret.first = false;
      } else {
        ret.second = GlobalParams.MPI_Rank + 1;
      }
      break;
    case Direction::MinusY:
      if (GlobalParams.Index_in_y_direction == 0) {
        ret.first = false;
      } else {
        ret.second = GlobalParams.MPI_Rank - GlobalParams.Blocks_in_y_direction;
      }
      break;
    case Direction::PlusY:
      if (GlobalParams.Index_in_y_direction == GlobalParams.Blocks_in_y_direction - 1) {
        ret.first = false;
      } else {
        ret.second = GlobalParams.MPI_Rank + GlobalParams.Blocks_in_y_direction;
      }
      break;
    case Direction::MinusZ:
      if (GlobalParams.Index_in_z_direction == 0) {
        ret.first = false;
      } else {
        ret.second =
            GlobalParams.MPI_Rank - (GlobalParams.Blocks_in_x_direction * GlobalParams.Blocks_in_y_direction);
      }
      break;
    case Direction::PlusZ:
      if (GlobalParams.Index_in_z_direction == GlobalParams.Blocks_in_z_direction - 1) {
        ret.first = false;
      } else {
        ret.second =
            GlobalParams.MPI_Rank + (GlobalParams.Blocks_in_x_direction * GlobalParams.Blocks_in_y_direction);
      }
      break;
  }
  return ret;
}

std::pair<bool, unsigned int> GeometryManager::get_level_neighbor_for_interface(Direction in_direction, unsigned int level) {
  std::pair<bool, unsigned int> ret(true, 0);
  const unsigned int difference = GlobalParams.HSIE_SWEEPING_LEVEL - level;
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

void GeometryManager::set_mesh_boundary_ids(unsigned int surface) {
    auto it = surface_meshes[surface].begin_active();
    std::vector<double> x;
    std::vector<double> y;
    while(it != this->surface_meshes[surface].end()){
      if(it->at_boundary()) {
        for (unsigned int face = 0; face < GeometryInfo<2>::faces_per_cell; ++face) {
          if (it->face(face)->at_boundary()) {
            dealii::Point<2, double> c;
            c = it->face(face)->center();
            x.push_back(c[0]);
            y.push_back(c[1]);
          }
        }
      }
      ++it;
    }
    double x_max = *max_element(x.begin(), x.end());
    double y_max = *max_element(y.begin(), y.end());
    double x_min = *min_element(x.begin(), x.end());
    double y_min = *min_element(y.begin(), y.end());
    it = this->surface_meshes[surface].begin_active();
    while(it != this->surface_meshes[surface].end()){
    if (it->at_boundary()) {
      for (unsigned int face = 0; face < dealii::GeometryInfo<2>::faces_per_cell; ++face) {
        Point<2, double> center;
        center = it->face(face)->center();
        if (std::abs(center[0] - x_min) < FLOATING_PRECISION) {
          it->face(face)->set_all_boundary_ids(edge_to_boundary_id[surface][0]);
        }
        if (std::abs(center[0] - x_max) < FLOATING_PRECISION) {
          it->face(face)->set_all_boundary_ids(edge_to_boundary_id[surface][1]);
        }
        if (std::abs(center[1] - y_min) < FLOATING_PRECISION) {
          it->face(face)->set_all_boundary_ids(edge_to_boundary_id[surface][2]);
        }
        if (std::abs(center[1] - y_max) < FLOATING_PRECISION) {
          it->face(face)->set_all_boundary_ids(edge_to_boundary_id[surface][3]);
        }
        }
    }
    ++it;
  }
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
