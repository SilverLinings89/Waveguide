//
// Created by pascal on 03.02.20.
//

#include "NonLocalProblem.h"
#include "../Helpers/GeometryManager.h"
#include "LocalProblem.h"
#include "../Core/NumericProblem.h"

NonLocalProblem::NonLocalProblem(unsigned int local_level) :
    HierarchicalProblem(local_level) {
    if(local_level > 1) {
    child = new NonLocalProblem(local_level - 1);
    } else {
    child = new LocalProblem();
    }
  switch (GlobalParams.HSIE_SWEEPING_LEVEL) {
  case 1:
    this->sweeping_direction = SweepingDirection::Z;
    break;
  case 2:
    if (local_level == 1) {
      this->sweeping_direction = SweepingDirection::Y;
    } else {
      this->sweeping_direction = SweepingDirection::Z;
    }
    break;
  case 3:
    if (local_level == 1) {
      this->sweeping_direction = SweepingDirection::X;
    } else {
      if (local_level == 2) {
        this->sweeping_direction = SweepingDirection::Y;
      } else {
        this->sweeping_direction = SweepingDirection::Z;
      }
    }
    break;
  default:
    this->sweeping_direction = SweepingDirection::Z;
    break;
  }
  is_hsie_surface = new bool[6];
  for (unsigned int i = 0; i < 6; i++) {
    is_hsie_surface[i] = false;
  }
  if (GlobalParams.Index_in_x_direction == 0) {
    is_hsie_surface[0] = true;
  }
  if (GlobalParams.Index_in_x_direction
      == GlobalParams.Blocks_in_x_direction - 1) {
    is_hsie_surface[1] = true;
  }
  if (GlobalParams.Index_in_y_direction == 0) {
    is_hsie_surface[2] = true;
  }
  if (GlobalParams.Index_in_y_direction
      == GlobalParams.Blocks_in_y_direction - 1) {
    is_hsie_surface[3] = true;
  }
  if (GlobalParams.Index_in_z_direction == 0) {
    is_hsie_surface[4] = true;
  }
  if (GlobalParams.Index_in_z_direction
      == GlobalParams.Blocks_in_z_direction - 1) {
    is_hsie_surface[5] = true;
  }
  if (GlobalParams.HSIE_SWEEPING_LEVEL == local_level + 1) {
    is_hsie_surface[4] = true;
    is_hsie_surface[5] = true;
  }
  if (GlobalParams.HSIE_SWEEPING_LEVEL == local_level + 2) {
    is_hsie_surface[2] = true;
    is_hsie_surface[3] = true;
    is_hsie_surface[4] = true;
    is_hsie_surface[5] = true;
  }
  n_own_dofs = 0;
}

NonLocalProblem::~NonLocalProblem() {

}

void NonLocalProblem::assemble() {

}

void NonLocalProblem::solve() {

}

void NonLocalProblem::initialize() {
  initialize_own_dofs();
  initialize_index_sets();
}

void NonLocalProblem::generate_sparsity_pattern() {

}

void NonLocalProblem::initialize_index_sets() {
  local_indices = dealii::Utilities::MPI::create_ascending_partitioning(
      GlobalMPI.communicators_by_level[local_level], n_own_dofs);
}

LocalProblem* NonLocalProblem::get_local_problem() {
  return child->get_local_problem();
}

void NonLocalProblem::initialize_own_dofs() {
  n_own_dofs = compute_own_dofs();
}

unsigned int NonLocalProblem::compute_own_dofs() {
  unsigned int ret =
      this->get_local_problem()->base_problem.dof_handler.n_dofs();
  for (unsigned int i = 0; i < 6; i++) {
    if (is_hsie_surface[i]) {
      ret += this->get_local_problem()->surfaces[i]->dof_counter;
    }
  }
  return ret;
}

unsigned int NonLocalProblem::compute_lower_interface_dof_count() {
  // TODO implement this. Use the HSIE-suface in that direction and take the number of non-hsie dofs.
  return 0;
}

unsigned int NonLocalProblem::compute_upper_interface_dof_count() {
  // TODO implement this. Use the HSIE-suface in that direction and take the number of non-hsie dofs.
  return 0;
}

void NonLocalProblem::apply_sweep(
    dealii::LinearAlgebra::distributed::Vector<double>) {

}

