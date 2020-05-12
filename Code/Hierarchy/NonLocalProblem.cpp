//
// Created by pascal on 03.02.20.
//

#include "NonLocalProblem.h"
#include "../Helpers/GeometryManager.h"
#include "LocalProblem.h"
#include "../Core/NumericProblem.h"

NonLocalProblem::NonLocalProblem(unsigned int local_level,
    unsigned int global_level) :
    HierarchicalProblem(local_level, global_level) {
    if(local_level > 1) {
    child = new NonLocalProblem(local_level - 1, global_level);
    } else {
    child = new LocalProblem(local_level - 1, global_level);
    }
}

NonLocalProblem::~NonLocalProblem() {

}

void NonLocalProblem::initialize_MPI_communicator_for_level() {

}

void NonLocalProblem::assemble() {

}

void NonLocalProblem::solve() {

}

void NonLocalProblem::initialize() {

}

void NonLocalProblem::generate_sparsity_pattern() {

}

void NonLocalProblem::initialize_index_sets() {

}

LocalProblem* NonLocalProblem::get_local_problem() {
  return child->get_local_problem();
}

unsigned int NonLocalProblem::compute_own_dofs() {
  // TODO implement compuatation of the number of local degrees of freedom. It is always at least the number of dofs from the inner dof handler + HSIE dofs for all sides where HSIE are applied.
  return 0;
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

