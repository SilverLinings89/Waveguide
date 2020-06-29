/*
 * JacobianForCell.cpp
 *
 *  Created on: Jun 16, 2020
 *      Author: kraft
 */

#include "../Core/Types.h"
#include "JacobianForCell.h"
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <vector>

JacobianForCell::JacobianForCell() {
  reset();
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

void JacobianForCell::reinit_for_cell(CellIterator2D cell,
    std::vector<bool> in_b_ids_have_hsie) {
  if (!cell->at_boundary()) {
    reset();
    return;
  }
  this->b_ids_have_hsie = in_b_ids_have_hsie;
  Lx = compute_L_x_for_cell(cell);
  Ly = compute_L_y_for_cell(cell);
}

double JacobianForCell::compute_L_x_for_cell(CellIterator2D cell) {
  return 2.0 * std::abs(cell->center()[0] - (cell->vertex(0))[0]);
}

double JacobianForCell::compute_L_y_for_cell(CellIterator2D cell) {
  return 2.0 * std::abs(cell->center()[1] - (cell->vertex(0))[1]);
}

bool JacobianForCell::is_cell_x_tilted(
    CellIterator2D cell) {
  if(!is_cell_at_x_interface()) {
    return false;
  }
  for(unsigned int edge = 0; edge < 4; edge++) {
    if(cell->line(edge)->at_boundary() && is_line_in_x_direction(cell->line(edge))) {
      if(b_ids_have_hsie[cell->line(edge)->boundary_id()]) {
        return true;
      }
    }
  }
  return false;
}

bool JacobianForCell::is_line_in_x_direction(dealii::internal::DoFHandlerImplementation::Iterators<class DoFHandler<2, 2>, false>::line_iterator line) {
    dealii::Point<2> vertex_1 = line->vertex(0);
    dealii::Point<2> vertex_2 = line->vertex(1);
    return (std::abs(vertex_1[0]-vertex_2[0]) > std::abs(vertex_1[1]-vertex_2[1]));
}

bool JacobianForCell::is_line_in_y_direction(dealii::internal::DoFHandlerImplementation::Iterators<class DoFHandler<2, 2>, false>::line_iterator line) {
    dealii::Point<2> vertex_1 = line->vertex(0);
    dealii::Point<2> vertex_2 = line->vertex(1);
    return (std::abs(vertex_1[0]-vertex_2[0]) < std::abs(vertex_1[1]-vertex_2[1]));
}

bool JacobianForCell::is_cell_y_tilted(
    CellIterator2D cell) {
  if(!is_cell_at_y_interface()) {
    return false;
  }
  for(unsigned int edge = 0; edge < 4; edge++) {
    if(cell->line(edge)->at_boundary() && is_line_in_y_direction(cell->line(edge))) {
      if(b_ids_have_hsie[cell->line(edge)->boundary_id()]) {
        return true;
      }
    }
  }
  return false;
}

bool JacobianForCell::is_cell_at_x_interface(
    CellIterator2D cell) {
  
}

bool JacobianForCell::is_cell_at_y_interface(
    CellIterator2D cell) {

}
