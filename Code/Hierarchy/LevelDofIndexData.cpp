/*
 * LevelDofIndexData.cpp
 *
 *  Created on: Apr 29, 2020
 *      Author: kraft
 */

#include "LevelDofIndexData.h"

const unsigned int sides_of_cube = 6;

LevelDofIndexData::LevelDofIndexData() {
  boundary_dofs = new unsigned int*[sides_of_cube];
  has_hsie_part = new bool*[sides_of_cube];
  for (unsigned int i = 0; i < sides_of_cube; i++) {
    boundary_dofs[i] = new unsigned int[sides_of_cube];
    has_hsie_part[i] = new bool[sides_of_cube];
  }
  for (unsigned int i = 0; i < sides_of_cube; i++) {
    for (unsigned int j = 0; j < sides_of_cube; j++) {
      boundary_dofs[i][j] = 0;
      has_hsie_part[i][j] = false;
    }
  }
}

LevelDofIndexData::~LevelDofIndexData() {
  for (unsigned int i = 0; i < sides_of_cube; i++) {
    delete[] boundary_dofs[i];
    delete[] has_hsie_part[i];
  }
  delete[] boundary_dofs;
  delete[] has_hsie_part;
}

