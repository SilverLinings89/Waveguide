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
  if (GlobalParams.Index_in_y_direction == 0) {
    is_hsie_surface[2] = true;
  }
  if (GlobalParams.Index_in_z_direction == 0) {
    is_hsie_surface[4] = true;
  }
  if (GlobalParams.Index_in_x_direction
      == GlobalParams.Blocks_in_x_direction - 1) {
    is_hsie_surface[1] = true;
  }
  if (GlobalParams.Index_in_y_direction
      == GlobalParams.Blocks_in_y_direction - 1) {
    is_hsie_surface[3] = true;
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
  communicate_sweeping_direction(sweeping_direction);
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
  receive_local_upper_dofs();
  H_inverse(src, dst);
  send_local_lower_dofs();

  H_inverse(dst, dst);

  receive_local_lower_dofs();
  H_inverse(src, dst);
  send_local_upper_dofs();
}

void NonLocalProblem::run() {

}

void NonLocalProblem::reinit() {
  DynamicSparsityPattern dsp = { total_number_of_dofs_on_level,
      total_number_of_dofs_on_level, local_indices };
  DofNumber first_index = local_indices.nth_index_in_set(0);
  get_local_problem()->base_problem.make_sparsity_pattern(&dsp, first_index , &constraints);
  first_index += get_local_problem()->base_problem.n_dofs;
  for (unsigned int i = 0; i < 6; i++) {
    if (is_hsie_surface[i]) {
      get_local_problem()->surfaces[i]->fill_sparsity_pattern(&dsp,
          first_index, &constraints);
      first_index += get_local_problem()->surfaces[i]->dof_counter;
    }
  }
  sp.copy_from(dsp);
  // matrix->reinit(sp); // TODO: implement this.
}

void NonLocalProblem::initialize() {
  child->initialize();
  initialize_own_dofs();
  dofs_process_above = compute_upper_interface_dof_count();
  dofs_process_below = compute_lower_interface_dof_count();
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

DofCount NonLocalProblem::compute_interface_dofs(BoundaryId interface_id, BoundaryId opposing_interface_id) {
  DofCount ret = 0;
  for(unsigned int i = 0; i < 6; i++) {
    if( i == interface_id) {
      ret += get_local_problem()->base_problem.get_surface_dof_vector_for_boundary_id(interface_id).size();
    } else {
      if(i != opposing_interface_id) {
        if(is_hsie_surface[i]) {
          ret += get_local_problem()->surfaces[i]->get_dof_association_by_boundary_id(i).size();
        }
      }
    }
  }
  return ret;
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
  return compute_interace_dofs(compute_lower_interface_id(), compute_upper_interface_id());
}

auto NonLocalProblem::compute_upper_interface_dof_count() -> DofCount {
  return compute_interace_dofs(compute_upper_interface_id(), compute_lower_interface_id());
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

auto NonLocalProblem::communicate_sweeping_direction(SweepingDirection) -> void {
  child->communicate_sweeping_direction(sweeping_direction);
};

void NonLocalProblem::H_inverse(NumericVectorDistributed &, NumericVectorDistributed &) {

};

NumericVectorLocal NonLocalProblem::extract_local_upper_dofs() {

};

NumericVectorLocal NonLocalProblem::extract_local_upper_lower() {

};

bool NonLocalProblem::is_lowest_in_sweeping_direction() {
  if(sweeping_direction == SweepingDirection::X) {
    if(GlobalParams.Index_in_x_direction == 0) {
      return true;
    }
  }
  if(sweeping_direction == SweepingDirection::Y) {
    if(GlobalParams.Index_in_y_direction == 0) {
      return true;
    }
  }
  if(sweeping_direction == SweepingDirection::Z) {
    if(GlobalParams.Index_in_z_direction == 0) {
      return true;
    }
  }
  return false;
};

bool NonLocalProblem::is_highest_in_sweeping_direction() {
  if(sweeping_direction == SweepingDirection::X) {
    if(GlobalParams.Index_in_x_direction == GlobalParams.Blocks_in_x_direction-1) {
      return true;
    }
  }
  if(sweeping_direction == SweepingDirection::Y) {
    if(GlobalParams.Index_in_y_direction == GlobalParams.Blocks_in_y_direction-1) {
      return true;
    }
  }
  if(sweeping_direction == SweepingDirection::Z) {
    if(GlobalParams.Index_in_z_direction == GlobalParams.Blocks_in_z_direction-1) {
      return true;
    }
  }
  return false;
};

Direction get_lower_boundary_id_for_sweeping_direction(SweepingDirection in_direction) {
  if(in_direction == SweepingDirection::X) {
    return Direction::MinusX;
  }
  if(in_direction == SweepingDirection::Y) {
    return Direction::MinusY;
  }
  return Direction::MinusZ;
}

Direction get_upper_boundary_id_for_sweeping_direction(SweepingDirection in_direction) {
  if(in_direction == SweepingDirection::X) {
    return Direction::PlusX;
  }
  if(in_direction == SweepingDirection::Y) {
    return Direction::PlusY;
  }
  return Direction::PlusZ;
}

void NonLocalProblem::send_local_lower_dofs() {
  if(is_lowest_in_sweeping_direction()) {
    return;
  }
  Direction communication_direction = get_lower_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data =GlobalMPI.get_neighbor_for_interface(communication_direction);
  ComplexNumber * data = new ComplexNumber(dofs_process_below);
  MPI_Send(&data[0], dofs_process_below, MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD);
}

void NonLocalProblem::receive_local_lower_dofs() {
  if(is_lowest_in_sweeping_direction()) {
    return;
  }
  Direction communication_direction = get_lower_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data =GlobalMPI.get_neighbor_for_interface(communication_direction);
  ComplexNumber * data = new ComplexNumber(dofs_process_below);
  MPI_Recv(&data[0], dofs_process_below, MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
};

void NonLocalProblem::send_local_upper_dofs() {
  if(is_highest_in_sweeping_direction()) {
    return;
  }
  Direction communication_direction = get_upper_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data =GlobalMPI.get_neighbor_for_interface(communication_direction);
  ComplexNumber * data = new ComplexNumber(dofs_process_above);
  MPI_Send(&data[0], dofs_process_above, MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD);
};

void NonLocalProblem::receive_local_upper_dofs() {
  if(is_highest_in_sweeping_direction()) {
    return;
  }
  Direction communication_direction = get_upper_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data =GlobalMPI.get_neighbor_for_interface(communication_direction);
  ComplexNumber * data = new ComplexNumber(dofs_process_above);
  MPI_Recv(&data[0], dofs_process_above, MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
};