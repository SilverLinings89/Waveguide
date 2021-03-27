#include "../Core/GlobalObjects.h"
#include "GeometryManager.h"
#include "../Core/NumericProblem.h"

GeometryManager::GeometryManager() {}

GeometryManager::~GeometryManager() {
}

void GeometryManager::initialize() {
  set_x_range(compute_x_range());
  set_y_range(compute_y_range());
  set_z_range(compute_z_range());
  h_x = (local_x_range.second - local_x_range.first) / GlobalParams.Cells_in_x;
  h_y = (local_y_range.second - local_y_range.first) / GlobalParams.Cells_in_y;
  h_z = (local_z_range.second - local_z_range.first) / GlobalParams.Cells_in_z;
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
    double length =
        GlobalParams.Geometry_Size_X
        / ((double) GlobalParams.Blocks_in_x_direction);
    int block_index = GlobalParams.MPI_Rank
        % GlobalParams.Blocks_in_x_direction;
    double min = -GlobalParams.Geometry_Size_X / 2.0 + block_index * length;
    return std::pair<double, double>(min, min + length);
  }
}

std::pair<double, double> GeometryManager::compute_y_range() {
  if (GlobalParams.Blocks_in_y_direction == 1) {
    return std::pair<double, double>(-GlobalParams.Geometry_Size_Y / 2.0,
        GlobalParams.Geometry_Size_Y / 2.0);
  } else {
    double length =
        GlobalParams.Geometry_Size_Y
        / ((double) GlobalParams.Blocks_in_y_direction);
    int block_processor_count = GlobalParams.Blocks_in_x_direction;
    int block_index = (GlobalParams.MPI_Rank
        % (GlobalParams.Blocks_in_x_direction
            * GlobalParams.Blocks_in_y_direction)) /
                      block_processor_count;
    double min = -GlobalParams.Geometry_Size_Y / 2.0 + block_index * length;
    return std::pair<double, double>(min, min + length);
  }
}

std::pair<double, double> GeometryManager::compute_z_range() {
  if (GlobalParams.Blocks_in_z_direction == 1) {
    return std::pair<double, double>(0,
        GlobalParams.Geometry_Size_Z);
  } else {
    double length = GlobalParams.Geometry_Size_Z / ((double) GlobalParams.Blocks_in_z_direction);
    int block_processor_count =
        GlobalParams.Blocks_in_x_direction
        * GlobalParams.Blocks_in_y_direction;
    int block_index = GlobalParams.MPI_Rank / block_processor_count;
    double min = block_index * length;
    return std::pair<double, double>(min, min + length);
  }
}

std::pair<bool, unsigned int> GeometryManager::get_neighbor_for_interface(Direction in_direction) {
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
      if (GlobalParams.Index_in_y_direction ==
          GlobalParams.Blocks_in_y_direction - 1) {
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
            GlobalParams.MPI_Rank - (GlobalParams.Blocks_in_x_direction *
                                     GlobalParams.Blocks_in_y_direction);
      }
      break;
    case Direction::PlusZ:
      if (GlobalParams.Index_in_z_direction ==
          GlobalParams.Blocks_in_z_direction - 1) {
        ret.first = false;
      } else {
        ret.second =
            GlobalParams.MPI_Rank + (GlobalParams.Blocks_in_x_direction *
                                     GlobalParams.Blocks_in_y_direction);
      }
      break;
  }
  return ret;
}

bool GeometryManager::math_coordinate_in_waveguide(Position in_position) const {
  return (std::abs(in_position[0]) < (GlobalParams.Width_of_waveguide  / 2.0)) && (std::abs(in_position[1]) < (GlobalParams.Height_of_waveguide / 2.0));
}
