/*
 * JacobianForCell.cpp
 *
 *  Created on: Jun 16, 2020
 *      Author: kraft
 */

#include "JacobianForCell.h"

JacobianForCell::JacobianForCell() {
  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      if (i == j) {
        J[i][j] = 1;
      } else {
        J[i][j] = 0;
      }
    }
  }
  is_x_tilted = false;
  is_y_tilted = false;
  x0 = 0;
  y0 = 0;
  Lx = 0;
  Ly = 0;
}

JacobianForCell::~JacobianForCell() {
  // TODO Auto-generated destructor stub
}

void JacobianForCell::reset() {
  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      if (i == j) {
        J[i][j] = 1;
      } else {
        J[i][j] = 0;
      }
    }
  }
  is_x_tilted = false;
  is_y_tilted = false;
  x0 = 0;
  y0 = 0;
  Lx = 0;
  Ly = 0;
}

void JacobianForCell::reinit_for_cell(
    dealii::DoFHandler<2>::active_cell_iterator cell,
    std::vector<bool> b_ids_have_hsie) {
  if (!cell->at_boundary()) {
    reset();
    return;
  }
}

double JacobianForCell::compute_L_x_for_cell(
    dealii::DoFHandler<2>::active_cell_iterator cell) {

}

double JacobianForCell::compute_L_y_for_cell(
    dealii::DoFHandler<2>::active_cell_iterator cell) {

}

bool JacobianForCell::is_cell_x_tilted(
    dealii::DoFHandler<2>::active_cell_iterator cell) {

}

bool JacobianForCell::is_cell_y_tilted(
    dealii::DoFHandler<2>::active_cell_iterator cell) {

}

bool JacobianForCell::is_cell_at_x_interface(
    dealii::DoFHandler<2>::active_cell_iterator cell) {

}

bool JacobianForCell::is_cell_at_y_interface(
    dealii::DoFHandler<2>::active_cell_iterator cell) {

}
