/*
 * GemoetryManager.cpp
 *
 *  Created on: Jun 19, 2019
 *      Author: kraft
 */

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
}

void GeometryManager::set_x_range(std::pair<double, double> in_range) {
  this->x_range = in_range;
  return;
}

void GeometryManager::set_y_range(std::pair<double, double> in_range) {
  this->y_range = in_range;
  return;
}

void GeometryManager::set_z_range(std::pair<double, double> in_range) {
  this->z_range = in_range;
  return;
}

std::pair<double, double> GeometryManager::compute_x_range() {
  if (GlobalParams.Blocks_in_x_direction == 1) {
    return std::pair<double, double>(-GlobalParams.M_R_XLength / 2.0,
        GlobalParams.M_R_XLength / 2.0);
  } else {
    double length =
        GlobalParams.M_R_XLength
        / ((double) GlobalParams.Blocks_in_x_direction);
    int block_index = GlobalParams.MPI_Rank
        % GlobalParams.Blocks_in_x_direction;
    double min = -GlobalParams.M_R_XLength / 2.0 + block_index * length;
    return std::pair<double, double>(min, min + length);
  }
}

std::pair<double, double> GeometryManager::compute_y_range() {
  if (GlobalParams.Blocks_in_y_direction == 1) {
    return std::pair<double, double>(-GlobalParams.M_R_YLength / 2.0,
        GlobalParams.M_R_YLength / 2.0);
  } else {
    double length =
        GlobalParams.M_R_YLength
        / ((double) GlobalParams.Blocks_in_y_direction);
    int block_processor_count = GlobalParams.Blocks_in_x_direction;
    int block_index = (GlobalParams.MPI_Rank
        % (GlobalParams.Blocks_in_x_direction
            * GlobalParams.Blocks_in_y_direction)) /
                      block_processor_count;
    double min = -GlobalParams.M_R_YLength / 2.0 + block_index * length;
    return std::pair<double, double>(min, min + length);
  }
}

std::pair<double, double> GeometryManager::compute_z_range() {
  if (GlobalParams.Blocks_in_z_direction == 1) {
    return std::pair<double, double>(-GlobalParams.M_R_ZLength / 2.0,
        GlobalParams.M_R_ZLength / 2.0);
  } else {
    double length =
        GlobalParams.M_R_ZLength
        / ((double) GlobalParams.Blocks_in_z_direction);
    int block_processor_count =
        GlobalParams.Blocks_in_x_direction
        * GlobalParams.Blocks_in_y_direction;
    int block_index = GlobalParams.MPI_Rank / block_processor_count;
    double min = -GlobalParams.M_R_ZLength / 2.0 + block_index * length;
    return std::pair<double, double>(min, min + length);
  }
}

std::pair<bool, unsigned int> GeometryManager::get_neighbor_for_interface(
    Direction in_direction) {
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

bool GeometryManager::math_coordinate_in_waveguide(
    dealii::Point<3, double> in_position) const {
  return std::abs(in_position[0]) <
             (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out) / 2.0 &&
         std::abs(in_position[1]) <
             (GlobalParams.M_C_Dim2In + GlobalParams.M_C_Dim2Out) / 2.0;
}
