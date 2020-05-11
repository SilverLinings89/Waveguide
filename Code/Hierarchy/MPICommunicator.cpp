/*
 * MPICommunicator.cpp
 *
 *  Created on: Apr 27, 2020
 *      Author: kraft
 */

#include <mpi.h>
#include "MPICommunicator.h"
#include "../Core/NumericProblem.h"

inline unsigned int get_index_for_direction_index(int in_direction) {
  if (in_direction == 1) {
    return GlobalParams.Index_in_z_direction;
  }
  if (in_direction == 2) {
    return GlobalParams.Index_in_y_direction;
  }
  if (in_direction == 3) {
    return GlobalParams.Index_in_x_direction;
  }
  return 0;
}

MPICommunicator::MPICommunicator() {
  // start with MPI Comm world and work the way down.
  int local_level = 1;
  MPI_Comm local = MPI_COMM_WORLD;
  communicators_by_level.push_back(MPI_COMM_WORLD);
  while (local_level <= GlobalParams.HSIE_SWEEPING_LEVEL) {
    MPI_Comm new_com;
    MPI_Comm_split(local, get_index_for_direction_index(local_level),
        GlobalParams.MPI_Rank, &new_com);
    local = new_com;
    local_level++;
    communicators_by_level.push_back(new_com);
  }
}

std::pair<bool, unsigned int> MPICommunicator::get_neighbor_for_interface(
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
    if (GlobalParams.Index_in_x_direction
        == GlobalParams.Blocks_in_x_direction - 1) {
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
    if (GlobalParams.Index_in_y_direction
        == GlobalParams.Blocks_in_y_direction - 1) {
      ret.first = false;
    } else {
      ret.second = GlobalParams.MPI_Rank + GlobalParams.Blocks_in_y_direction;
    }
    break;
  case Direction::MinusZ:
    if (GlobalParams.Index_in_z_direction == 0) {
      ret.first = false;
    } else {
      ret.second = GlobalParams.MPI_Rank
          - (GlobalParams.Blocks_in_x_direction
              * GlobalParams.Blocks_in_y_direction);
    }
    break;
  case Direction::PlusZ:
    if (GlobalParams.Index_in_z_direction
        == GlobalParams.Blocks_in_z_direction - 1) {
      ret.first = false;
    } else {
      ret.second = GlobalParams.MPI_Rank
          + (GlobalParams.Blocks_in_x_direction
              * GlobalParams.Blocks_in_y_direction);
    }
    break;
  }
  return ret;
}

MPICommunicator::~MPICommunicator() {
  // TODO Auto-generated destructor stub
}

