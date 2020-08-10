#include "NonLocalProblem.h"
#include "../Helpers/GeometryManager.h"
#include "LocalProblem.h"
#include "../Core/NumericProblem.h"
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_pattern.h>

NonLocalProblem::NonLocalProblem(unsigned int local_level) :
  HierarchicalProblem(local_level),
  sc(GlobalParams.GMRES_max_steps, GlobalParams.Solver_Precision, true, true),
  solver(sc, dealii::SolverGMRES<NumericVectorDistributed>::AdditionalData(GlobalParams.GMRES_Steps_before_restart))
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
  delete matrix;
  delete system_rhs;
  delete[] is_hsie_surface;
}

void NonLocalProblem::make_constraints() {

}

void NonLocalProblem::assemble() {

}

dealii::Vector<ComplexNumber> NonLocalProblem::get_local_vector_from_global() {
  dealii::Vector<ComplexNumber> ret(n_own_dofs);
  return ret;
}

void NonLocalProblem::solve(NumericVectorDistributed src,
    NumericVectorDistributed &dst) {
  dealii::Vector<ComplexNumber> inputb(n_own_dofs);
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
  DynamicSparsityPattern dsp = { total_number_of_dofs_on_level,
      total_number_of_dofs_on_level, local_indices };
  DofNumber first_index = local_indices.nth_index_in_set(0);
  get_local_problem()->base_problem.make_sparsity_pattern(&dsp, first_index);
  first_index += get_local_problem()->base_problem.n_dofs;
  for (unsigned int i = 0; i < 6; i++) {
    if (is_hsie_surface[i]) {
      get_local_problem()->surfaces[i]->fill_sparsity_pattern(&dsp,
          first_index);
      first_index += get_local_problem()->surfaces[i]->dof_counter;
    }
  }
  sp.copy_from(dsp);
  // matrix->reinit(sp); // TODO: implement this.
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
  total_number_of_dofs_on_level = 0;
  for (unsigned int i = 0; i < n_procs_in_sweep; i++) {
    total_number_of_dofs_on_level += index_sets_per_process[i].n_elements();
  }
  DofCount n_inner_dofs =
      this->get_local_problem()->base_problem.dof_handler.n_dofs()
          + local_indices.nth_index_in_set(0);
  surface_first_dofs.push_back(n_inner_dofs);
  for (unsigned int i = 0; i < 6; i++) {
    if (is_hsie_surface[i]) {
      n_inner_dofs += this->get_local_problem()->surfaces[i]->dof_counter;
    }
    if (i != 5) {
      surface_first_dofs.push_back(n_inner_dofs);
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

auto NonLocalProblem::compute_lower_interface_id() -> BoundaryId {
  if (this->sweeping_direction == SweepingDirection::X) {
    return 0;
  }
  if (this->sweeping_direction == SweepingDirection::Y) {
    return 2;
  }
  if (this->sweeping_direction == SweepingDirection::Z) {
    return 4;
  }
  return 0;
}

auto NonLocalProblem::compute_upper_interface_id() -> BoundaryId {
  if (this->sweeping_direction == SweepingDirection::X) {
    return 1;
  }
  if (this->sweeping_direction == SweepingDirection::Y) {
    return 3;
  }
  if (this->sweeping_direction == SweepingDirection::Z) {
    return 5;
  }
  return 0;
}

auto NonLocalProblem::compute_lower_interface_dof_count() -> DofCount {
  return get_local_problem()->surfaces[compute_lower_interface_id()]->get_dof_association().size();
}

auto NonLocalProblem::compute_upper_interface_dof_count() -> DofCount {
  return get_local_problem()->surfaces[compute_upper_interface_id()]->get_dof_association().size();
}

void NonLocalProblem::apply_sweep(NumericVectorDistributed) {

}

auto NonLocalProblem::get_center() -> Position const {
  Position local_contribution = (this->get_local_problem())->get_center();
  double x = dealii::Utilities::MPI::min_max_avg(local_contribution[0],
      GlobalMPI.communicators_by_level[this->local_level]).avg;
  double y = dealii::Utilities::MPI::min_max_avg(local_contribution[1],
      GlobalMPI.communicators_by_level[this->local_level]).avg;
  double z = dealii::Utilities::MPI::min_max_avg(local_contribution[2],
      GlobalMPI.communicators_by_level[this->local_level]).avg;
  return {x,y,z};
}
