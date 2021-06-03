#include "NonLocalProblem.h"
#include "../GlobalObjects/GeometryManager.h"
#include "../Helpers/staticfunctions.h"
#include "HierarchicalProblem.h"
#include "LocalProblem.h"
#include "../Core/InnerDomain.h"
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/petsc_precondition.h>
#include <mpi.h>
#include <petscksp.h>
#include <petscsystypes.h>

#include <algorithm>
#include <iterator>
#include <ostream>
#include <ratio>
#include <string>
#include <vector>

double l2_norm(DofFieldTrace in_trace) {
  double ret;
  for(unsigned int i = 0; i < in_trace.size(); i++) {
    ret += std::abs(in_trace[i])*std::abs(in_trace[i]);
  }
  return std::sqrt(ret);
}

static double last_residual = -1;

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

static PetscErrorCode MonitorError(KSP , PetscInt its, PetscReal rnorm, void *)
{
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    std::cout << "Residual in step " << std::to_string(its) << "  : " << std::to_string(rnorm);
    if(last_residual != -1 ) {
      std::cout << " Factor: " << std::to_string(rnorm / last_residual);
    }
    std::cout << std::endl;
    last_residual = rnorm;
  }
  return(0);
}

double l2_norm_of_vector(std::vector<ComplexNumber> input) {
  double norm = 0;
  for(unsigned int i = 0; i< input.size(); i++) {
    norm += std::abs(input[i]);
  }
  return std::sqrt(norm);
}

int convergence_test(KSP , const PetscInt iteration, const PetscReal residual_norm, KSPConvergedReason *reason, void * solver_control_x)
{
  SolverControl &solver_control =
    *reinterpret_cast<SolverControl *>(solver_control_x);

  const SolverControl::State state = solver_control.check(iteration, residual_norm);

  switch (state)
    {
      case ::SolverControl::iterate:
        *reason = KSP_CONVERGED_ITERATING;
        break;

      case ::SolverControl::success:
        *reason = static_cast<KSPConvergedReason>(1);
        break;

      case ::SolverControl::failure:
        if (solver_control.last_step() > solver_control.max_steps())
          *reason = KSP_DIVERGED_ITS;
        else
          *reason = KSP_DIVERGED_DTOL;
        break;

      default:
        Assert(false, ExcNotImplemented());
    }

  // return without failure
  return 0;
}

void get_petsc_index_array_from_index_set(PetscInt* in_array, dealii::IndexSet in_set) {
  for(unsigned int i = 0; i < in_set.n_elements(); i++) {
    in_array[i] = in_set.nth_index_in_set(i);
  }
}

PetscErrorCode pc_apply(PC in_pc, Vec x_in, Vec x_out) {
  SampleShellPC  *shell;
  PCShellGetContext(in_pc,(void**)&shell);
  shell->parent->apply_sweep(x_in, x_out);
  return 0;
}

PetscErrorCode pc_create(SampleShellPC *shell, NonLocalProblem * parent)
{
  SampleShellPC  *newctx;
  PetscNew(&newctx);
  newctx->parent = parent;
  *shell = *newctx;
  return 0;
}

NonLocalProblem::NonLocalProblem(unsigned int level) :
  HierarchicalProblem(level, static_cast<SweepingDirection> (2 + GlobalParams.HSIE_SWEEPING_LEVEL - level)),
  sc(GlobalParams.GMRES_max_steps, GlobalParams.Solver_Precision, true, true)
{
  if(level > 1) {
    child = new NonLocalProblem(level - 1);
  } else {
    child = new LocalProblem();
  }
  
  matrix = new dealii::PETScWrappers::MPI::SparseMatrix();
  is_mpi_cache_ready = false;
}

void NonLocalProblem::init_solver_and_preconditioner() {
  // dealii::PETScWrappers::PreconditionNone pc_none;
  // pc_none.initialize(*matrix);
  KSPCreate(GlobalMPI.communicators_by_level[level], &ksp);
  KSPGetPC(ksp, &pc);
  KSPSetOperators(ksp, *matrix, *matrix);
  KSPSetType(ksp, KSPGMRES);
  PCSetType(pc,PCSHELL);
  pc_create(&shell, this);
  PCShellSetApply(pc,pc_apply);
  PCShellSetContext(pc, (void*) &shell);
  KSPSetPC(ksp, pc);
}

void NonLocalProblem::reinit_rhs() {
  rhs = dealii::PETScWrappers::MPI::Vector(own_dofs, GlobalMPI.communicators_by_level[level]);
}

NonLocalProblem::~NonLocalProblem() {
  delete matrix;
  delete[] mpi_cache;
}

Direction get_direction_for_boundary_id(BoundaryId bid) {
  switch (bid)
  {
  case 0:
    return Direction::MinusX;
    break;
  case 1:
    return Direction::PlusX;
    break;
  case 2:
    return Direction::MinusY;
    break;
  case 3:
    return Direction::PlusY;
    break;
  case 4:
    return Direction::MinusZ;
    break;
  case 5:
    return Direction::PlusZ;
    break;
  
  default:
    std::cout << "Error in input for get_direction_for_boundary_id"<<std::endl;
    return Direction::MinusX;
    break;
  }
}

void remove_double_entries_from_vector(std::vector<DofNumber> * in_vector) {
  std::sort(in_vector->begin(), in_vector->end());
  std::vector<unsigned int> alternative;
  if(in_vector->size() > 0) alternative.push_back(in_vector->operator[](0));
  for(unsigned int i = 1; i < in_vector->size(); i++) {
    if(! ( in_vector->operator[](i) == in_vector->operator[](i-1))) {
      alternative.push_back(in_vector->operator[](i));
    }
  }
  in_vector->clear();
  for(unsigned int i = 0; i < alternative.size(); i++) {
    in_vector->push_back(alternative[i]);
  }
  if(in_vector->size() == 0) std::cout << "Logic error detected..." << std::endl;
}

void NonLocalProblem::print_diagnosis_data() {
  
}

void NonLocalProblem::assemble() {
  print_info("NonLocalProblem::assemble", "Begin assembly");
  GlobalTimerManager.switch_context("assemble", level);
  Geometry.inner_domain->assemble_system(Geometry.levels[level].inner_first_dof, &constraints, matrix, &rhs);
  print_info("NonLocalProblem::assemble", "Inner assembly done. Assembling boundary method contributions.");
  for(unsigned int i = 0; i< 6; i++) {
      Timer timer;
      timer.start();
      Geometry.levels[level].surfaces[i]->fill_matrix(matrix, &rhs, &constraints);
      timer.stop();
  }
  // MPI_Barrier(MPI_COMM_WORLD);
  print_info("NonLocalProblem::assemble", "Compress matrix.");
  matrix->compress(dealii::VectorOperation::add);
  // MPI_Barrier(MPI_COMM_WORLD);
  print_info("NonLocalProblem::assemble", "Assemble child.");
  child->assemble();
  print_info("NonLocalProblem::assemble", "Compress vectors.");
  rhs.compress(dealii::VectorOperation::add);
  // MPI_Barrier(MPI_COMM_WORLD);
  solution.compress(dealii::VectorOperation::add);
  // MPI_Barrier(MPI_COMM_WORLD);
  print_info("NonLocalProblem::assemble", "End assembly.");
}

dealii::Vector<ComplexNumber> NonLocalProblem::get_local_vector_from_global() {
  dealii::Vector<ComplexNumber> ret(Geometry.levels[level].n_local_dofs);
  return ret;
}

void NonLocalProblem::solve() {
  GlobalTimerManager.switch_context("solve", level);
  constraints.set_zero(solution);
  
  KSPSetConvergenceTest(ksp, &convergence_test, reinterpret_cast<void *>(&sc),nullptr);
  KSPSetTolerances(ksp, 0.000001, 1.0, 1000, GlobalParams.GMRES_max_steps);
  KSPMonitorSet(ksp, MonitorError, nullptr, nullptr);
  KSPSetUp(ksp);
  PetscErrorCode ierr = KSPSolve(ksp, rhs, solution);
  
  // constraints.distribute(solution);
  
  if(ierr != 0) {
    std::cout << "Error code from Petsc: " << std::to_string(ierr) << std::endl;
    throw new ExcPETScError(ierr);
  }
}

void NonLocalProblem::propagate_up(){
  const unsigned int count = Geometry.inner_domain->n_dofs;
  // Propagate the inner values. HSIE-dofs can be added later.
  for(unsigned int i = 0; i < count; i++) {
    // u[i] = child->solution(Geometry.levels[level-1].inner_first_dof + i);
  }
  for(unsigned int i = 0; i < 6; i++) {
    if(Geometry.levels[level].is_surface_truncated[i]) {
      for(unsigned int j = 0; j < Geometry.levels[level].surfaces[i]->dof_counter; j++) {
      //  u[Geometry.levels[level].surface_first_dof[i] - Geometry.levels[level].inner_first_dof + j] = child->solution(Geometry.levels[level-1].surface_first_dof[i] + j);
      }
    }
  }
}

void NonLocalProblem::apply_sweep(Vec x_in, Vec x_out) {
  u = u_from_x_in(x_in);
  
  if(!is_highest_in_sweeping_direction()) {
    subtract_fields(u, vmult(trace_to_field(receive_from_above(), false)));
  }
  
  u = S_inv(u);
  
  if(!is_lowest_in_sweeping_direction()) {
    send_down(lower_trace(u));
  }
  
  if(!is_lowest_in_sweeping_direction()) {
    subtract_fields(u, S_inv(trace_to_field(receive_from_below(), true)));
    //update_u_from_lower_trace(lower_trace);
  }

  if(!is_highest_in_sweeping_direction()) {
    zero_upper_interface_dofs(&u);
    send_up(upper_trace(vmult(u)));
  }

  if(is_highest_in_sweeping_direction()) {
    send_down(lower_trace(u));
  } else {
    update_u_from_upper_trace(receive_from_above());
    if(!is_lowest_in_sweeping_direction()) {
      send_down(lower_trace(u));
    }
  }

  set_x_out_from_u(&x_out);
}

void NonLocalProblem::reinit_u_vector(NumericVectorLocal * u) {
  u->reinit(Geometry.levels[level].n_local_dofs);
}

NumericVectorLocal NonLocalProblem::u_from_x_in(Vec x_in) {
  NumericVectorLocal ret;
  reinit_u_vector(&ret);
  ComplexNumber * values = new ComplexNumber[own_dofs.n_elements()];
  VecGetValues(x_in, own_dofs.n_elements(), locally_owned_dofs_index_array, values);
  for(unsigned int i = 0; i < Geometry.levels[level].n_local_dofs; i++) {
    ret[i] = values[i]; 
  }
  delete[] values;
  return ret;
}

NumericVectorLocal NonLocalProblem::S_inv(const NumericVectorLocal & in_u) {
  NumericVectorLocal n = in_u;
  child->rhs_mismatch = child->rhs;

  // zero_upper_interface_dofs(&n);
  
  set_child_rhs_from_u(n, false);

  child->solve();

  NumericVectorLocal ret;
  
  set_u_from_child_solution(&ret);
  
  // zero_upper_interface_dofs(&ret);
  
  child->rhs = child->rhs_mismatch;
  child->rhs.compress(VectorOperation::insert);

  return ret;
}

DofFieldTrace NonLocalProblem::lower_trace(const NumericVectorLocal & u) {
  DofFieldTrace ret;
  for(unsigned int i = 0; i < lower_interface_dofs.n_elements(); i++) {
    const unsigned int index = lower_interface_dofs.nth_index_in_set(i) - Geometry.levels[level].inner_first_dof;
    ret.push_back(u[index]);
  }
  return ret;
}

DofFieldTrace NonLocalProblem::upper_trace(const NumericVectorLocal & u) {
  DofFieldTrace ret;
  for(unsigned int i = 0; i < upper_interface_dofs.n_elements(); i++) {
    const unsigned int index = upper_interface_dofs.nth_index_in_set(i) - Geometry.levels[level].inner_first_dof;
    ret.push_back(u[index]);
  }
  return ret;
}

void NonLocalProblem::send_down(DofFieldTrace trace_values) {
  reinit_mpi_cache(trace_values.size());
  std::cout << "Sending down: " << std::to_string(l2_norm(trace_values)) << std::endl;
  Direction communication_direction = get_lower_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data = GlobalMPI.get_neighbor_for_interface(communication_direction);
  for(unsigned int i = 0; i < trace_values.size(); i++) {
    mpi_cache[i] = trace_values[i];
  }
  MPI_Send(&mpi_cache[0], trace_values.size(), MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD);
}

void NonLocalProblem::send_up(DofFieldTrace trace_values) {
  reinit_mpi_cache(trace_values.size());
  std::cout << "Sending up: " << std::to_string(l2_norm(trace_values)) << std::endl;
  Direction communication_direction = get_upper_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data = GlobalMPI.get_neighbor_for_interface(communication_direction);
  for(unsigned int i = 0; i < trace_values.size(); i++) {
    mpi_cache[i] = trace_values[i];
  }
  MPI_Send(&mpi_cache[0], trace_values.size(), MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD);
}

DofFieldTrace NonLocalProblem::receive_from_above() {
  DofFieldTrace ret;
  const unsigned int count = upper_interface_dofs.n_elements();
  reinit_mpi_cache(count);
  Direction communication_direction = get_upper_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data = GlobalMPI.get_neighbor_for_interface(communication_direction);
  MPI_Recv(&mpi_cache[0], count, MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for(unsigned int i = 0; i < count; i++) {
    ret.push_back(mpi_cache[i]);
  }
  return ret;
}

DofFieldTrace NonLocalProblem::receive_from_below() {
  DofFieldTrace ret;
  const unsigned int count = lower_interface_dofs.n_elements();
  reinit_mpi_cache(count);
  Direction communication_direction = get_lower_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data = GlobalMPI.get_neighbor_for_interface(communication_direction);
  MPI_Recv(&mpi_cache[0], count, MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for(unsigned int i = 0; i < count; i++) {
    ret.push_back(mpi_cache[i]);
  }
  return ret;
}

NumericVectorLocal NonLocalProblem::vmult(NumericVectorLocal in_u) {
  NumericVectorLocal ret;

  set_child_solution_from_u(in_u);

  child->execute_vmult();

  set_u_from_child_solution(&ret);

  return ret;
}

NumericVectorLocal NonLocalProblem::trace_to_field(DofFieldTrace trace, bool from_lower) {
  IndexSet dofs;
  if(from_lower) {
    dofs = lower_interface_dofs;
  } else {
    dofs = upper_interface_dofs;
  }
  NumericVectorLocal ret;
  reinit_u_vector(&ret);
  for(unsigned int i = 0; i < dofs.n_elements(); i++) {
    ret[dofs.nth_index_in_set(i) - Geometry.levels[level].inner_first_dof] = trace[i];
  }
  return ret;
}

void NonLocalProblem::subtract_fields(NumericVectorLocal &a, const NumericVectorLocal &b) {
  a -= b;
}

void NonLocalProblem::set_child_solution_from_u(const NumericVectorLocal & in_u) {
  child->solution = 0;
  for(unsigned int i = 0; i < Geometry.levels[level].n_local_dofs; i++) {
    child->solution[Geometry.levels[level-1].inner_first_dof + i] = in_u[i];
  }
  
  for(unsigned int surf = 0; surf < 6; surf++) {
    if(Geometry.levels[level].is_surface_truncated[surf] && Geometry.levels[level-1].is_surface_truncated[surf]) {
      for(unsigned int i = 0; i < Geometry.levels[level].surfaces[surf]->dof_counter; i++) {
        child->solution[Geometry.levels[level-1].surface_first_dof[surf] + i] = in_u[Geometry.levels[level].surface_first_dof[surf] - Geometry.levels[level].inner_first_dof + i];
      }
    }
  }
  
  child->solution.compress(VectorOperation::insert);
}

void NonLocalProblem::set_u_from_child_solution(NumericVectorLocal * u) {
  reinit_u_vector(u);

  for(unsigned int i = 0; i < Geometry.inner_domain->n_dofs; i++) {
    (*u)[i] = child->solution[Geometry.levels[level-1].inner_first_dof + i];
  }

  for(unsigned int surf = 0; surf < 6; surf++) {
    if(Geometry.levels[level].is_surface_truncated[surf] && Geometry.levels[level-1].is_surface_truncated[surf]) {
      for(unsigned int i = 0; i < Geometry.levels[level].surfaces[surf]->dof_counter; i++) {
        (*u)[Geometry.levels[level].surface_first_dof[surf] - Geometry.levels[level].inner_first_dof + i] = child->solution[Geometry.levels[level-1].surface_first_dof[surf] + i];
      }
    }
  }
}

void NonLocalProblem::zero_upper_interface_dofs(NumericVectorLocal * in_u) {
  if(!is_highest_in_sweeping_direction()) {
    for(unsigned int i = 0; i < upper_interface_dofs.n_elements(); i++) {
      (*in_u)[upper_interface_dofs.nth_index_in_set(i) - Geometry.levels[level].inner_first_dof] = 0;
    }
  }
}

void NonLocalProblem::set_child_rhs_from_u(const NumericVectorLocal & in_u, bool add_onto_child_rhs) {
  child->rhs_mismatch = child->rhs;
  child->rhs_mismatch.compress(VectorOperation::insert);
  child->rhs = 0;
  for(unsigned int i = 0; i < Geometry.levels[level].n_local_dofs; i++) {
    child->rhs[Geometry.levels[level-1].inner_first_dof + i] = in_u[i];
  }
  
  for(unsigned int surf = 0; surf < 6; surf++) {
    if(Geometry.levels[level].is_surface_truncated[surf] && Geometry.levels[level-1].is_surface_truncated[surf]) {
      for(unsigned int i = 0; i < Geometry.levels[level].surfaces[surf]->dof_counter; i++) {
        child->rhs[Geometry.levels[level-1].surface_first_dof[surf] + i] = in_u[Geometry.levels[level].surface_first_dof[surf] - Geometry.levels[level].inner_first_dof + i];
      }
    }
  }
  
  child->rhs.compress(VectorOperation::insert);

  if(add_onto_child_rhs) {
    child->rhs += child->rhs_mismatch;
  }
}

void NonLocalProblem::update_u_from_upper_trace(DofFieldTrace trace) {
  for(unsigned int i = 0; i < upper_interface_dofs.n_elements(); i++) {
    u[upper_interface_dofs.nth_index_in_set(i) - Geometry.levels[level].inner_first_dof] = trace[i];
  }
}





void NonLocalProblem::set_x_out_from_u(Vec * x_out) {
  ComplexNumber * values = new ComplexNumber[own_dofs.n_elements()];
  for(unsigned int i = 0; i < own_dofs.n_elements(); i++) {
     values[i] = u[i];
  }
  VecSetValues(*x_out, own_dofs.n_elements(), locally_owned_dofs_index_array, values, INSERT_VALUES);
  VecAssemblyBegin(*x_out);
  VecAssemblyEnd(*x_out);
  delete[] values;
}

void NonLocalProblem::reinit() {
  print_info("Nonlocal reinit", "Reinit starting");
  child->reinit();
  
  make_constraints();
  // print_diagnosis_data();
  
  make_sparsity_pattern();

  reinit_rhs();
  
  solution.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  rhs_mismatch.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  final_rhs_mismatch.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  matrix->reinit(Geometry.levels[level].dof_distribution[rank], Geometry.levels[level].dof_distribution[rank], sp, GlobalMPI.communicators_by_level[level]);
  
  print_info("Nonlocal reinit", "Reinit done");
}

void NonLocalProblem::initialize() {
  GlobalTimerManager.switch_context("initialize", level);
  child->initialize();
  n_procs_in_sweep = dealii::Utilities::MPI::n_mpi_processes(GlobalMPI.communicators_by_level[level]);
  rank = dealii::Utilities::MPI::this_mpi_process(GlobalMPI.communicators_by_level[level]);

  dofs_process_above = compute_upper_interface_dof_count();
  dofs_process_below = compute_lower_interface_dof_count();

  initialize_index_sets();
  reinit();
  init_solver_and_preconditioner();
}
 
void NonLocalProblem::initialize_index_sets() {
  lower_sweeping_interface_id = compute_lower_interface_id();
  upper_sweeping_interface_id = compute_upper_interface_id();

  lower_interface_dofs = compute_interface_dof_set(lower_sweeping_interface_id);
  upper_interface_dofs = compute_interface_dof_set(upper_sweeping_interface_id);

  initialize_own_dofs();
  locally_owned_dofs_index_array = new PetscInt[own_dofs.n_elements()];
  get_petsc_index_array_from_index_set(locally_owned_dofs_index_array, own_dofs);
}

void NonLocalProblem::initialize_own_dofs() {
  own_dofs = IndexSet(Geometry.levels[level].n_total_level_dofs);
  unsigned int first_dof = Geometry.levels[level].inner_first_dof;
  unsigned int last_dof = Geometry.levels[level].surface_first_dof[5] + Geometry.levels[level].surfaces[5]->dof_counter;
  own_dofs.add_range(first_dof, last_dof);
}

DofCount NonLocalProblem::compute_interface_dofs(BoundaryId interface_id) {
  BoundaryId opposing_interface_id = opposing_Boundary_Id(interface_id);
  DofCount ret = 0;
  for(unsigned int i = 0; i < 6; i++) {
    if( i == interface_id) {
      ret += Geometry.inner_domain->get_surface_dof_vector_for_boundary_id(interface_id).size();
    } else {
      if(i != opposing_interface_id) {
        if(Geometry.levels[level].is_surface_truncated[i]) {
          ret += Geometry.levels[level].surfaces[i]->get_dof_association_by_boundary_id(i).size();
        }
      }
    }
  }
  return ret;
}

dealii::IndexSet NonLocalProblem::compute_interface_dof_set(BoundaryId interface_id) {
  dealii::IndexSet ret(Geometry.levels[level].n_total_level_dofs);
  std::vector<InterfaceDofData> current = Geometry.inner_domain->get_surface_dof_vector_for_boundary_id_and_level(interface_id, level);
  for(unsigned int j = 0; j < current.size(); j++) {
    ret.add_index(current[j].index);
  }
  
  for(unsigned int i = 0; i < 6; i++) {
    if( i != interface_id && !are_opposing_sites(i,interface_id)) {
      if(Geometry.levels[level].is_surface_truncated[i]) {
        std::vector<InterfaceDofData> current = Geometry.levels[level].surfaces[i]->get_dof_association_by_boundary_id(interface_id);
        for(unsigned int j = 0; j < current.size(); j++) {
          ret.add_index(current[j].index); 
        }
      }
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
  return compute_interface_dofs(compute_lower_interface_id());
}

auto NonLocalProblem::compute_upper_interface_dof_count() -> DofCount {
  return compute_interface_dofs(compute_upper_interface_id());
}

auto NonLocalProblem::get_center() -> Position const {
  Position local_contribution = Geometry.get_local_center();
  double x = dealii::Utilities::MPI::min_max_avg(local_contribution[0], GlobalMPI.communicators_by_level[level]).avg;
  double y = dealii::Utilities::MPI::min_max_avg(local_contribution[1], GlobalMPI.communicators_by_level[level]).avg;
  double z = dealii::Utilities::MPI::min_max_avg(local_contribution[2], GlobalMPI.communicators_by_level[level]).avg;
  return {x,y,z};
}

void NonLocalProblem::H_inverse() {
  child->solve();
}

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
}

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
}

void NonLocalProblem::reinit_mpi_cache(unsigned int n_elements) {
  if(is_mpi_cache_ready) {
    delete[] mpi_cache;
  }
  mpi_cache = new ComplexNumber[n_elements];
  for(unsigned int i = 0; i < n_elements; i++) {
    mpi_cache[i] = 0;
  }
  is_mpi_cache_ready = true;
}

void NonLocalProblem::send_local_lower_dofs(std::vector<ComplexNumber> values) {
  reinit_mpi_cache(values.size());
  Direction communication_direction = get_lower_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data = GlobalMPI.get_neighbor_for_interface(communication_direction);
  // std::cout << "Values size: " << values.size() << std::endl;
  for(unsigned int i = 0; i < values.size(); i++) {
    mpi_cache[i] = values[i];
  }
  MPI_Send(&mpi_cache[0], values.size(), MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD);
}

std::vector<ComplexNumber> NonLocalProblem::receive_local_lower_dofs() {
  std::vector<ComplexNumber> ret;
  const unsigned int n_elements = lower_interface_dofs.n_elements();
  reinit_mpi_cache(n_elements);
  Direction communication_direction = get_lower_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data = GlobalMPI.get_neighbor_for_interface(communication_direction);
  MPI_Recv(&mpi_cache[0], n_elements, MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for(unsigned int i = 0; i< n_elements; i++) {
    ret.push_back(mpi_cache[i]);
  }
  return ret;
}

void NonLocalProblem::apply_H_to_u(std::vector<ComplexNumber> u) {
  child->reinit_rhs();
  const unsigned int count = Geometry.inner_domain->n_dofs;
  for(unsigned int i = 0; i < count; i++) {
    const DofNumber dof = lower_interface_dofs.nth_index_in_set(i) - Geometry.levels[level].inner_first_dof + Geometry.levels[level-1].inner_first_dof;
    child->rhs[dof] = mpi_cache[i];
    u[lower_interface_dofs.nth_index_in_set(i)] = mpi_cache[i];
  }
  child->rhs.compress(VectorOperation::insert);
  child->matrix->vmult(child->temp_solution, child->rhs);
  child->rhs = child->temp_solution;
  child->temp_solution = 0;
  child->solve();
  for(unsigned int i = 0; i < count; i++) {
    if(!lower_interface_dofs.is_element(i)) {
      u[i] -= (ComplexNumber)child->solution(Geometry.levels[level-1].inner_first_dof + i);
    }
  }
  for(unsigned int i = 0; i < 6; i++) {
    if(Geometry.levels[level].is_surface_truncated[i]) {
      for(unsigned int j = 0; j < Geometry.levels[level].surfaces[i]->dof_counter; j++) {
        // u[Geometry.levels[level].surface_first_dof[i] - Geometry.levels[level].inner_first_dof + j] -= ComplexNumber(child->solution(Geometry.levels[level-1].surface_first_dof[i] + j).real(), child->solution(Geometry.levels[level-1].surface_first_dof[i] + j).imag());
      }
    }
  }
}

void NonLocalProblem::receive_local_upper_dofs() {
  reinit_mpi_cache(upper_interface_dofs.n_elements());
  Direction communication_direction = get_upper_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data = GlobalMPI.get_neighbor_for_interface(communication_direction);
  MPI_Recv(&mpi_cache[0], upper_interface_dofs.n_elements(), MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for(unsigned int i = 0; i < upper_interface_dofs.n_elements(); i++) {
    u[upper_interface_dofs.nth_index_in_set(i) - Geometry.levels[level].inner_first_dof] -= mpi_cache[i];
  }
}

void NonLocalProblem::send_local_upper_dofs(std::vector<ComplexNumber> values) {
  reinit_mpi_cache(values.size());
  Direction communication_direction = get_upper_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data = GlobalMPI.get_neighbor_for_interface(communication_direction);
  for(unsigned int i = 0; i < values.size(); i++) {
    mpi_cache[i] = values[i];
  }
  MPI_Send(&mpi_cache[0], values.size(), MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD);
}

void NonLocalProblem::update_mismatch_vector(BoundaryId, bool) {
  // rhs_mismatch.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  NumericVectorDistributed temp_solution(child->own_dofs, GlobalMPI.communicators_by_level[child->level]);
  NumericVectorDistributed temp_rhs(child->own_dofs, GlobalMPI.communicators_by_level[child->level]);
  child->rhs_mismatch = 0;
  // Copy dof values for inner problem
  for(unsigned int index = 0; index < Geometry.inner_domain->n_dofs; index++) {
    temp_solution[Geometry.levels[level-1].inner_first_dof + index] = solution[Geometry.levels[level].inner_first_dof + index];
  }
  // Copy dof values for any HSIE dofs of non-sweeping interfaces.
  for(unsigned int surface = 0; surface < 6; surface++) {
    if(surface != get_lower_boundary_id_for_sweeping_direction(sweeping_direction) && surface != get_upper_boundary_id_for_sweeping_direction(sweeping_direction)) {
      if(Geometry.levels[level].is_surface_truncated[surface]) {
        for(unsigned int index = 0; index < Geometry.levels[level].surfaces[surface]->dof_counter; index++ ){
          temp_solution[Geometry.levels[level-1].surface_first_dof[surface] + index] = solution[Geometry.levels[level].surface_first_dof[surface] + index];
        }
      }
    }
  }
  // Dof values of sweeping interfaces are left at zero, leading to a 
  // non-zero solution (since we solved this problem exacty before with 
  // the boundary elements for the sweeping interface, we now get a non-zero rhs).
  child->matrix->vmult(temp_rhs, temp_solution);
  for(unsigned int index = 0; index < Geometry.inner_domain->n_dofs; index++) {
    rhs_mismatch[Geometry.levels[level].inner_first_dof + index] = temp_rhs[Geometry.levels[level-1].inner_first_dof + index];
  }
}

NumericVectorLocal NonLocalProblem::extract_local_upper_dofs() {
  std::cout << "A" <<std::endl;
  BoundaryId bid = get_upper_boundary_id_for_sweeping_direction(sweeping_direction);
  IndexSet is = surface_index_sets[bid];
  NumericVectorLocal ret(is.n_elements());
  for(unsigned int i = 0; i < is.n_elements(); i++) {
    std::cout << "i: " << i <<std::endl;
    ret[i] = rhs_mismatch[Geometry.levels[level].inner_first_dof + is.nth_index_in_set(i)];
  }
  std::cout << "b" <<std::endl;
  return ret;
}

std::vector<ComplexNumber> NonLocalProblem::extract_local_lower_dofs() {
  BoundaryId bid = get_lower_boundary_id_for_sweeping_direction(sweeping_direction);
  IndexSet is = surface_index_sets[bid];
  std::vector<ComplexNumber> ret;
  for(unsigned int i = 0; i < is.n_elements(); i++) {
    ret.push_back(u[is.nth_index_in_set(i) - Geometry.levels[level].inner_first_dof]);
  }
  return ret;
}

void NonLocalProblem::compute_solver_factorization() {
  child->compute_solver_factorization();
  // child->output_results();
}

std::vector<ComplexNumber> NonLocalProblem::UpperBlockProductAfterH() {
  setChildRhsComponentsFromU();
  child->solve();
  // H_{-1} complete

  child->update_mismatch_vector(lower_sweeping_interface_id, true);
  // rhs_mismatch is updated;

  std::vector<ComplexNumber> ret(lower_interface_dofs.n_elements());
  for(unsigned int i = 0; i < lower_interface_dofs.n_elements(); i++) {
    ret[i] = child->rhs_mismatch[lower_interface_dofs.nth_index_in_set(i) - Geometry.levels[level].inner_first_dof];
  }
  
  return ret;
}

std::vector<ComplexNumber> NonLocalProblem::LowerBlockProduct() {
  /** 
  setChildSolutionComponentsFromU();
  child->update_mismatch_vector(compute_upper_interface_id(), false);
  std::vector<ComplexNumber> ret(upper_interface_dofs.n_elements());
  for(unsigned int i = 0; i < upper_interface_dofs.n_elements(); i++) {
    ret[i] = child->rhs_mismatch[upper_interface_dofs.nth_index_in_set(i) - Geometry.levels[level].inner_first_dof + Geometry.levels[level-1].inner_first_dof];
  }
  **/

  std::vector<ComplexNumber> ret(upper_interface_dofs.n_elements());

  return ret;
}

void NonLocalProblem::setSolutionFromVector(Vec x_in) {
  ComplexNumber * values = new ComplexNumber[own_dofs.n_elements()];
  VecGetValues(x_in, own_dofs.n_elements(), locally_owned_dofs_index_array, values);
  for(unsigned int i = 0; i < Geometry.levels[level].n_local_dofs; i++) {
    u[i] = values[i]; 
  }
  delete[] values;
}

void NonLocalProblem::setChildSolutionComponentsFromU() {
  
  for(unsigned int i = 0; i < Geometry.inner_domain->n_dofs; i++) {
    child->solution[Geometry.levels[level-1].inner_first_dof + i] = u[i];
  }
  for(unsigned int surface = 0; surface < 6; surface++) {
    if(Geometry.levels[level].is_surface_truncated[surface]) {
      for(unsigned int i = 0; i < Geometry.levels[level].surfaces[surface]->dof_counter; i++) {
        child->solution[Geometry.levels[level-1].surface_first_dof[surface] + i] = u[Geometry.levels[level].surface_first_dof[surface] - Geometry.levels[level].inner_first_dof + i];
      }
    } else {
      if(Geometry.levels[level-1].is_surface_truncated[surface]) {
        std::vector<InterfaceDofData> vec = Geometry.levels[level-1].surfaces[surface]->get_dof_association();
        std::sort(vec.begin(), vec.end(), compareDofDataByGlobalIndex);
        unsigned int index = 0;
        for(unsigned int i = 0; i < Geometry.levels[level].surfaces[surface]->dof_counter; i++) {
          while(vec[index].index < i){
            index++;
          }
          if(i != vec[index].index){
            child->solution[Geometry.levels[level-1].surface_first_dof[surface] + i] = 0;
          } 
        }
      }
    }
  }
  child->solution.compress(VectorOperation::insert);
  
}

void NonLocalProblem::setChildRhsComponentsFromU() {
  child->rhs = 0;
  // This is the v-argument of eq. 3.9
  for(unsigned int i = 0; i < Geometry.inner_domain->n_dofs; i++) {
    child->rhs[Geometry.levels[level-1].inner_first_dof + i] = u[i];
  }

  for(unsigned int surf = 0; surf < 6; surf++) {
    if(Geometry.levels[level].is_surface_truncated[surf] && Geometry.levels[level-1].is_surface_truncated[surf]) {
      for(unsigned int i = 0; i < Geometry.levels[level-1].surfaces[surf]->dof_counter; i++) {
        child->rhs[Geometry.levels[level-1].surface_first_dof[surf] + i] = u[Geometry.levels[level].surface_first_dof[surf] + i - Geometry.levels[level].inner_first_dof];
      }
    }
  }

  // This is the upper 0 in eq. 3.9.
  if(!is_lowest_in_sweeping_direction()) {
    std::vector<InterfaceDofData> dofs = Geometry.inner_domain->get_surface_dof_vector_for_boundary_id(lower_sweeping_interface_id);
    for(unsigned int i = 0; i < dofs.size(); i++) {
      child->rhs[Geometry.levels[level-1].inner_first_dof + dofs[i].index] = 0;
    }
  }

  // Recheck if this works or not.

  child->rhs.compress(VectorOperation::insert);
  
}

DofOwner NonLocalProblem::get_dof_owner(unsigned int dof) {
  DofOwner ret;
  if(dof < Geometry.levels[level].inner_first_dof || dof > Geometry.levels[level].inner_first_dof + Geometry.levels[level].n_local_dofs) {
    std::cout << "get_dof_data was called for a dof that is not locally owned" << std::endl;
    return ret;
  }
  ret.owner = rank;
  ret.is_boundary_dof = dof > Geometry.levels[level].surface_first_dof[0];
  if(ret.is_boundary_dof) {
    for(unsigned int i = 0; i < 5; i++) {
      if(dof < Geometry.levels[level].surface_first_dof[i+1]) {
        ret.surface_id = i;
        return ret;
      }
    }
  }
  ret.surface_id = 5;
  return ret;
}

void NonLocalProblem::print_dof_details(unsigned int dof) {
  if(is_dof_locally_owned(dof)) {
    DofOwner data = get_dof_owner(dof);
    if(!data.is_boundary_dof) {
      std::cout << "Dof " + std::to_string(dof) + " is a inner dof on process " + std::to_string(rank) << std::endl;
    } else {
      std::cout << "Dof " + std::to_string(dof) + " is a boundary dof on process " + std::to_string(rank) + " for boundary " + std::to_string(data.surface_id)<< std::endl;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

bool NonLocalProblem::is_dof_locally_owned(unsigned int dof) {
  return (dof >= Geometry.levels[level].inner_first_dof && dof < Geometry.levels[level].inner_first_dof + Geometry.levels[level].n_local_dofs);
}
