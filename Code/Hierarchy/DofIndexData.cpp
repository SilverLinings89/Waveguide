/*
 * DofIndexData.cpp
 *
 *  Created on: Apr 24, 2020
 *      Author: kraft
 */

#include "DofIndexData.h"
#include "../Core/GlobalObjects.h"
#include "LevelDofIndexData.h"

DofIndexData::DofIndexData() {
  for (unsigned int i = 0; i < GlobalParams.HSIE_SWEEPING_LEVEL; i++) {
    indexCountsByLevel.push_back(*(new LevelDofIndexData()));
  }
  isSurfaceNeighbor = new bool[6];
  isSurfaceNeighbor[0] = (GlobalParams.Index_in_x_direction > 0);
  isSurfaceNeighbor[1] = (GlobalParams.Index_in_x_direction
      < GlobalParams.Blocks_in_x_direction - 1);
  isSurfaceNeighbor[2] = (GlobalParams.Index_in_y_direction > 0);
  isSurfaceNeighbor[3] = (GlobalParams.Index_in_y_direction
      < GlobalParams.Blocks_in_y_direction - 1);
  isSurfaceNeighbor[4] = (GlobalParams.Index_in_z_direction > 0);
  isSurfaceNeighbor[5] = (GlobalParams.Index_in_z_direction
      < GlobalParams.Blocks_in_z_direction - 1);
}

DofIndexData::~DofIndexData() {
  delete[] isSurfaceNeighbor;
}

void DofIndexData::communicateSurfaceDofs() {
}

void DofIndexData::initialize() {

}

void DofIndexData::initialize_level(unsigned int level) {

}
