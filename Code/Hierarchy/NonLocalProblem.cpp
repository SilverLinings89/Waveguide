#include "NonLocalProblem.h"
#include "../Helpers/GeometryManager.h"
#include "LocalProblem.h"
#include "../Core/NumericProblem.h"
#include <deal.II/lac/solver_gmres.h>

NonLocalProblem::NonLocalProblem(unsigned int local_level) :
  HierarchicalProblem(local_level),
  sc(GlobalParams.So_TotalSteps, GlobalParams.So_Precision, true, true), 
  solver(sc, dealii::SolverGMRES<NumericVectorDistributed>::AdditionalData(GlobalParams.So_RestartSteps))
{
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
  delete system_matrix;
  delete system_rhs;
}

void NonLocalProblem::make_constraints() {

}

void NonLocalProblem::assemble() {

}

dealii::Vector<std::complex<double>> NonLocalProblem::get_local_vector_from_global() {
  dealii::Vector<std::complex<double>> ret(n_own_dofs);
  return ret;
}

void NonLocalProblem::solve(NumericVectorDistributed src,
    NumericVectorDistributed &dst) {
  dealii::Vector<std::complex<double>> inputb(n_own_dofs);
  for (unsigned int i = 0; i < n_own_dofs; i++) {
    inputb[i] = src(i);
  }

  // solver.solve(system_matrix, dst, system_rhs, this->child);

  for (unsigned int i = 0; i < n_own_dofs; i++) {
    dst[i] = inputb[i];
  }
}

void NonLocalProblem::run() {

}

void NonLocalProblem::reinit() {
  // TODO reimplement
}

void NonLocalProblem::initialize() {
  child->initialize();
  initialize_own_dofs();
  initialize_index_sets();
}

void NonLocalProblem::generate_sparsity_pattern() {

}

void NonLocalProblem::initialize_index_sets() {
  std::vector<dealii::IndexSet> index_sets_per_process;
  n_procs_in_sweep = dealii::Utilities::MPI::n_mpi_processes(
      GlobalMPI.communicators_by_level[local_level]);
  rank = dealii::Utilities::MPI::this_mpi_process(
      GlobalMPI.communicators_by_level[local_level]);
  index_sets_per_process =
      dealii::Utilities::MPI::create_ascending_partitioning(
      GlobalMPI.communicators_by_level[local_level], n_own_dofs);
  local_indices = index_sets_per_process[rank];
  unsigned int ret =
      this->get_local_problem()->base_problem.dof_handler.n_dofs()
          + local_indices.nth_index_in_set(0);
  surface_first_dofs.push_back(ret);
  for (unsigned int i = 0; i < 6; i++) {
    if (is_hsie_surface[i]) {
      ret += this->get_local_problem()->surfaces[i]->dof_counter;
    }
    if (i != 5) {
      surface_first_dofs.push_back(ret);
    }
  }

  if (rank > 0) {
    dofs_process_below = index_sets_per_process[rank - 1].n_elements();
  }
  if (rank + 1 < n_procs_in_sweep) {
    dofs_process_above = index_sets_per_process[rank + 1].n_elements();
  }
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
    dealii::LinearAlgebra::distributed::Vector<std::complex<double>>) {

}

