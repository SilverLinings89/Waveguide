/*
 * GemoetryManager.cpp
 *
 *  Created on: Jun 19, 2019
 *      Author: kraft
 */

#include "Geometry.h"

Geometry::Geometry() {}

Geometry::~Geometry() {
  // TODO Auto-generated destructor stub
}

void Geometry::initialize(Parameters in_params) {
  set_x_range(compute_x_range(in_params));
  set_y_range(compute_y_range(in_params));
  set_z_range(compute_z_range(in_params));
}

void Geometry::set_x_range(std::pair<double, double> in_range) {
  // TODO: Testing
  this->x_range = in_range;
  return;
}

void Geometry::set_y_range(std::pair<double, double> in_range) {
  // TODO: Testing
  this->y_range = in_range;
  return;
}

void Geometry::set_z_range(std::pair<double, double> in_range) {
  // TODO: Testing
  this->z_range = in_range;
  return;
}

std::pair<double, double> Geometry::compute_x_range(Parameters in_params) {
  if (in_params.Blocks_in_x_direction == 1) {
    return std::pair(-in_params.M_R_XLength / 2.0, in_params.M_R_XLength / 2.0);
  } else {
    double length =
        in_params.M_R_XLength / ((double)in_params.Blocks_in_x_direction);
    int block_index = in_params.MPI_Rank % in_params.Blocks_in_x_direction;
    double min = -in_params.M_R_XLength / 2.0 + block_index * length;
    return std::pair(min, min + length);
  }
}

std::pair<double, double> Geometry::compute_y_range(Parameters in_params) {
  if (in_params.Blocks_in_y_direction == 1) {
    return std::pair(-in_params.M_R_YLength / 2.0, in_params.M_R_YLength / 2.0);
  } else {
    double length =
        in_params.M_R_YLength / ((double)in_params.Blocks_in_y_direction);
    int block_processor_count = in_params.Blocks_in_x_direction;
    int block_index = (in_params.MPI_Rank % (in_params.Blocks_in_x_direction *
                                             in_params.Blocks_in_y_direction)) /
                      block_processor_count;
    double min = -in_params.M_R_YLength / 2.0 + block_index * length;
    return std::pair(min, min + length);
  }
}

std::pair<double, double> Geometry::compute_z_range(Parameters in_params) {
  if (in_params.Blocks_in_z_direction == 1) {
    return std::pair(-in_params.M_R_ZLength / 2.0, in_params.M_R_ZLength / 2.0);
  } else {
    double length =
        in_params.M_R_ZLength / ((double)in_params.Blocks_in_z_direction);
    int block_processor_count =
        in_params.Blocks_in_x_direction * in_params.Blocks_in_y_direction;
    int block_index = in_params.MPI_Rank / block_processor_count;
    double min = -in_params.M_R_ZLength / 2.0 + block_index * length;
    return std::pair(min, min + length);
  }
}
