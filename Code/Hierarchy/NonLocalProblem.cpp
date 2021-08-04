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
  double ret = 0;
  for(unsigned int i = 0; i < in_trace.size(); i++) {
    ret += in_trace[i].real()*in_trace[i].real() + in_trace[i].imag() * in_trace[i].imag();
  }
  return std::sqrt(ret);
}

double l2_norm(NumericVectorLocal in_vector) {
  double ret = 0;
  for(unsigned int i = 0; i < in_vector.size(); i++) {
    ret += in_vector[i].real()*in_vector[i].real() + in_vector[i].imag() * in_vector[i].imag();
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
  if(sweeping_direction == SweepingDirection::X) {
    n_blocks_in_sweep = GlobalParams.Blocks_in_x_direction;
    index_in_sweep = GlobalParams.Index_in_x_direction;
  }
  if(sweeping_direction == SweepingDirection::Y) {
    n_blocks_in_sweep = GlobalParams.Blocks_in_y_direction;
    index_in_sweep = GlobalParams.Index_in_y_direction;
  }
  if(sweeping_direction == SweepingDirection::Z) {
    n_blocks_in_sweep = GlobalParams.Blocks_in_z_direction;
    index_in_sweep = GlobalParams.Index_in_z_direction;
  }
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
  rhs.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
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
  MPI_Barrier(MPI_COMM_WORLD);
  print_info("NonLocalProblem::assemble", "Compress matrix.");
  matrix->compress(dealii::VectorOperation::add);
  MPI_Barrier(MPI_COMM_WORLD);
  print_info("NonLocalProblem::assemble", "Assemble child.");
  child->assemble();
  print_info("NonLocalProblem::assemble", "Assemble rhs.");
  compute_rhs_representation_of_incoming_wave();
  MPI_Barrier(MPI_COMM_WORLD);
  print_info("NonLocalProblem::assemble", "Compress vectors.");
  solution.compress(dealii::VectorOperation::add);
  MPI_Barrier(MPI_COMM_WORLD);
  print_info("NonLocalProblem::assemble", "Assemble local system for vmult.");
  assemble_local_system();
  MPI_Barrier(MPI_COMM_WORLD);
  print_info("NonLocalProblem::assemble", "End assembly.");
}

void NonLocalProblem::assemble_local_system() {
  local.n_dofs = Geometry.levels[level].n_local_dofs;
  local.local_dofs = dealii::IndexSet(local.n_dofs);
  local.local_dofs.add_range(0, local.n_dofs);
  local.constraints = Constraints(local.local_dofs);
  local.lower_sweeping_dofs.set_size(local.n_dofs);
  local.upper_sweeping_dofs.set_size(local.n_dofs);
  for(unsigned int i = 0; i < lower_interface_dofs.n_elements(); i++) {
    local.lower_sweeping_dofs.add_index(lower_interface_dofs.nth_index_in_set(i));
  }
  for(unsigned int i = 0; i < upper_interface_dofs.n_elements(); i++) {
    local.upper_sweeping_dofs.add_index(upper_interface_dofs.nth_index_in_set(i));
  }
  local.constraints.reinit(local.local_dofs);
  // fill constraints here
  for(unsigned int dof = Geometry.levels[level].inner_first_dof; dof < Geometry.levels[level].inner_first_dof + Geometry.levels[level].n_local_dofs; dof++) {
    if(constraints.is_constrained(dof)) {
      local.constraints.add_line(dof - Geometry.levels[level].inner_first_dof);
      if(constraints.is_inhomogeneously_constrained(dof)) {
        local.constraints.set_inhomogeneity(dof - Geometry.levels[level].inner_first_dof, constraints.get_inhomogeneity(dof));
      }
      for(auto it : *constraints.get_constraint_entries(dof)) {
        if(it.first >= Geometry.levels[level].inner_first_dof && it.first <  Geometry.levels[level].inner_first_dof + Geometry.levels[level].n_local_dofs) {
          for(auto it : *constraints.get_constraint_entries(dof)) {
            if(it.first >= Geometry.levels[level].inner_first_dof && it.first <  Geometry.levels[level].inner_first_dof + Geometry.levels[level].n_local_dofs) {
              local.constraints.add_entry(dof - Geometry.levels[level].inner_first_dof, it.first - Geometry.levels[level].inner_first_dof, it.second);
            }
          }
        }
      }
    }
  }
  local.constraints.close();

  dealii::DynamicSparsityPattern dsp(local.local_dofs);
  // fill dsp here
  for(unsigned int dof = Geometry.levels[level].inner_first_dof; dof < Geometry.levels[level].inner_first_dof + Geometry.levels[level].n_local_dofs; dof++) {
    for(dealii::SparsityPattern::iterator it = sp.begin(dof); it < sp.end(dof); it++) {
      if(it->column() >= Geometry.levels[level].inner_first_dof && it->column() < Geometry.levels[level].inner_first_dof + Geometry.levels[level].n_local_dofs) {
        dsp.add(it->row() - Geometry.levels[level].inner_first_dof, it->column() - Geometry.levels[level].inner_first_dof);
      }
    }
  }

  local.sp.copy_from(dsp);
  local.matrix.reinit(local.sp);

  // fill matrix here
  Geometry.inner_domain->assemble_system(&local.constraints, &local.matrix);
  for(unsigned int surf = 0; surf < 6; surf++) {
    if(Geometry.levels[level].surface_type[surf] == SurfaceType::ABC_SURFACE) {
      Geometry.levels[level].surfaces[surf]->fill_matrix(&local.matrix, &local.constraints);
    }
  }
}

dealii::Vector<ComplexNumber> NonLocalProblem::get_local_vector_from_global() {
  dealii::Vector<ComplexNumber> ret(Geometry.levels[level].n_local_dofs);
  return ret;
}

void NonLocalProblem::solve() {
  constraints.set_zero(solution);
  GlobalTimerManager.switch_context("solve", level);
  std::cout << "Norm before solving: " << rhs.l2_norm() << std::endl;
  rhs.compress(VectorOperation::add);
  if(GlobalParams.MPI_Rank == 0) {
    NumericVectorLocal u;
    reinit_u_vector(&u);
    for(unsigned int i = 0; i < Geometry.levels[level].n_local_dofs; i++) {
      u[i] = rhs[i];
    }
    set_child_rhs_from_u(u, false);
    child->solve();
    set_u_from_child_solution(&u);
    NumericVectorLocal u_truncated = NumericVectorLocal(Geometry.inner_domain->n_dofs);
    for(unsigned int i = 0; i < Geometry.inner_domain->n_dofs; i++) {
      u_truncated[i] = u[i];
    }
    std::cout << "Local solution norm on proc 0: " << l2_norm(u) << std::endl;
    Geometry.inner_domain->output_results("FirstStepFirstOutput", u_truncated);
    send_up(upper_trace(u));
  }
  if(GlobalParams.MPI_Rank == 1) {
    NumericVectorLocal u = vmult(trace_to_field( receive_from_below(), 4));
    set_child_rhs_from_u(u, false);
    child->solve();
    set_u_from_child_solution(&u);
    NumericVectorLocal u_truncated = NumericVectorLocal(Geometry.inner_domain->n_dofs);
    for(unsigned int i = 0; i < Geometry.inner_domain->n_dofs; i++) {
      u_truncated[i] = u[i];
    }
    Geometry.inner_domain->output_results("FirstStepSecondOutput", u_truncated);
  }

  if(!GlobalParams.solve_directly) {
    
    KSPSetConvergenceTest(ksp, &convergence_test, reinterpret_cast<void *>(&sc), nullptr);
    KSPSetPCSide(ksp, PCSide::PC_RIGHT);
    KSPSetTolerances(ksp, 0.000001, 1.0, 1000, GlobalParams.GMRES_max_steps);
    KSPMonitorSet(ksp, MonitorError, nullptr, nullptr);
    KSPSetUp(ksp);
    PetscErrorCode ierr = KSPSolve(ksp, rhs, solution);

  } else {
    SolverControl sc;
    dealii::PETScWrappers::SparseDirectMUMPS solver1(sc, MPI_COMM_WORLD);
    solver1.solve(*matrix, solution, rhs);
  }
  
  constraints.distribute(solution);
  
  // if(ierr != 0) {
  //   std::cout << "Error code from Petsc: " << std::to_string(ierr) << std::endl;
  //   throw new ExcPETScError(ierr);
  // }
}

void NonLocalProblem::apply_sweep(Vec b_in, Vec u_out) {
  NumericVectorLocal u = u_from_x_in(b_in);
  if(!is_highest_in_sweeping_direction()) {
    u = subtract_fields(u, vmult(trace_to_field(receive_from_above(), 5)));
  }
  if(!is_lowest_in_sweeping_direction()) {
    send_down(lower_trace(S_inv(u)));
  }
  
  u = S_inv(u);

  if(!is_lowest_in_sweeping_direction()) {
    u = subtract_fields(u, S_inv(vmult(trace_to_field(receive_from_below(), 4))));
  }
  if(!is_highest_in_sweeping_direction()) {
    send_up(upper_trace(u));
  }

  set_x_out_from_u(&u_out, u);
}

void NonLocalProblem::reinit_u_vector(NumericVectorLocal * u) {
  u->reinit(Geometry.levels[level].n_local_dofs);
}

NumericVectorLocal NonLocalProblem::u_from_x_in(Vec x_in) {
  NumericVectorLocal ret;
  reinit_u_vector(&ret);
  const unsigned int n_loc_dofs = own_dofs.n_elements();
  ComplexNumber * values = new ComplexNumber[n_loc_dofs];
  VecGetValues(x_in, n_loc_dofs, locally_owned_dofs_index_array, values);

  temp_solution.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  
  for(unsigned int i = 0; i < Geometry.levels[level].n_local_dofs; i++) {
    temp_solution[Geometry.levels[level].inner_first_dof + i] = values[i];
   }

  temp_solution.compress(dealii::VectorOperation::insert);
  
  constraints.distribute(temp_solution);
  
  for(unsigned int i = 0; i < Geometry.levels[level].n_local_dofs; i++) {
  //  ret[i] = temp_solution[Geometry.levels[level].inner_first_dof + i];
    ret[i] = values[i];
  }

  delete[] values;
  return ret;
}

void NonLocalProblem::set_x_out_from_u(Vec * x_out, NumericVectorLocal u_in) {
  const unsigned int n_loc_dofs = own_dofs.n_elements();
  ComplexNumber * values = new ComplexNumber[n_loc_dofs];
  for(unsigned int i = 0; i < n_loc_dofs; i++) {
     values[i] = u_in[i];
  }
  VecSetValues(*x_out, n_loc_dofs, locally_owned_dofs_index_array, values, INSERT_VALUES);
  VecAssemblyBegin(*x_out);
  VecAssemblyEnd(*x_out);
  delete[] values;
}

NumericVectorLocal NonLocalProblem::S_inv(NumericVectorLocal in_u) {

  set_child_rhs_from_u(in_u, false);

  child->solve();

  NumericVectorLocal ret;
  
  set_u_from_child_solution(&ret);
  
  return ret;
}

DofFieldTrace NonLocalProblem::lower_trace(NumericVectorLocal u) {
  DofFieldTrace ret;
  double max_norm = 0;
  for(unsigned int i = 0; i < lower_interface_dofs.n_elements(); i++) {
    const unsigned int index = lower_interface_dofs.nth_index_in_set(i);
    ret.push_back(u[index]);
    if(std::abs(u[index]) > max_norm) {
      max_norm = std::abs(u[index]);
    }
  }
  return ret;
}

DofFieldTrace NonLocalProblem::upper_trace(NumericVectorLocal u) {
  DofFieldTrace ret;
  double max_norm = 0;
  for(unsigned int i = 0; i < upper_interface_dofs.n_elements(); i++) {
    const unsigned int index = upper_interface_dofs.nth_index_in_set(i);
    ret.push_back(u[index]);
    if(std::abs(u[index]) > max_norm) {
      max_norm = std::abs(u[index]);
    }
  }
  return ret;
}

void NonLocalProblem::send_down(DofFieldTrace trace_values) {
  reinit_mpi_cache(trace_values.size());
  Direction communication_direction = get_lower_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data = GlobalMPI.get_neighbor_for_interface(communication_direction);
  for(unsigned int i = 0; i < trace_values.size(); i++) {
    mpi_cache[i] = trace_values[i];
  }
  MPI_Send(&mpi_cache[0], trace_values.size(), MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD);
}

void NonLocalProblem::send_up(DofFieldTrace trace_values) {
  reinit_mpi_cache(trace_values.size());
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
  reinit_u_vector(&ret);
  // local.constraints.set_zero(in_u);
  local.matrix.vmult(ret, in_u);
  // local.constraints.distribute(ret);
  return ret;
}

NumericVectorLocal NonLocalProblem::trace_to_field(DofFieldTrace trace, BoundaryId b_id) {
  std::vector<InterfaceDofData> dofs = Geometry.inner_domain->get_surface_dof_vector_for_boundary_id_and_level(b_id, level);
  NumericVectorLocal ret;
  reinit_u_vector(&ret);
  for(unsigned int i = 0; i < dofs.size(); i++) {
    ret[dofs[i].index - Geometry.levels[level].inner_first_dof] = trace[i];
  }
  return ret;
}

NumericVectorLocal NonLocalProblem::subtract_fields(NumericVectorLocal a, NumericVectorLocal b) {
  NumericVectorLocal ret;
  reinit_u_vector(&ret);
  for(unsigned int i = 0; i < ret.size(); i++) {
    ret[i] = a[i] - b[i];
  }
  return ret;
}

void NonLocalProblem::set_child_solution_from_u(NumericVectorLocal in_u) {
  child->solution = 0;
  for(unsigned int i = 0; i < Geometry.local_inner_dofs; i++) {
    child->solution[Geometry.levels[level-1].inner_first_dof + i] = in_u[i];
  }
  
  for(unsigned int surf = 0; surf < 6; surf++) {
    if(surf == upper_sweeping_interface_id && !is_highest_in_sweeping_direction()) {
      for(unsigned int i = 0; i < Geometry.levels[level-1].surfaces[surf]->dof_counter; i++) {
        child->solution[Geometry.levels[level-1].surface_first_dof[surf] + i] = 0;
      }
    }
    if(Geometry.levels[level].is_surface_truncated[surf] && Geometry.levels[level-1].is_surface_truncated[surf]) {
      for(unsigned int i = 0; i < Geometry.levels[level].surfaces[surf]->dof_counter; i++) {
        child->solution[Geometry.levels[level-1].surface_first_dof[surf] + i] = in_u[Geometry.levels[level].surface_first_dof[surf] - Geometry.levels[level].inner_first_dof + i];
      }
    }
  }
  
  child->solution.compress(VectorOperation::insert);
  child->constraints.set_zero(child->solution);
}

void NonLocalProblem::set_u_from_child_solution(NumericVectorLocal * u_in) {
  reinit_u_vector(u_in);

  for(unsigned int i = 0; i < Geometry.inner_domain->n_dofs; i++) {
    (*u_in)[i] = child->solution[Geometry.levels[level-1].inner_first_dof + i];
  }

  for(unsigned int surf = 0; surf < 6; surf++) {
    if(Geometry.levels[level].is_surface_truncated[surf] && Geometry.levels[level-1].is_surface_truncated[surf]) {
      for(unsigned int i = 0; i < Geometry.levels[level].surfaces[surf]->dof_counter; i++) {
        (*u_in)[Geometry.levels[level].surface_first_dof[surf] - Geometry.levels[level].inner_first_dof + i] = child->solution[Geometry.levels[level-1].surface_first_dof[surf] + i];
      }
    }
  }
}

double NonLocalProblem::compute_interface_norm_for_u(NumericVectorLocal u, BoundaryId in_bid) {
  double ret = 0;
  std::vector<InterfaceDofData> boundary_dofs = Geometry.inner_domain->get_surface_dof_vector_for_boundary_id_and_level(in_bid, 0);
  for(unsigned int i = 0; i < boundary_dofs.size(); i++) {
    ComplexNumber c = u[boundary_dofs[i].index];
    ret += c.real() * c.real() + c.imag() * c.imag();
  }
  return std::sqrt(ret);
}

NumericVectorLocal NonLocalProblem::zero_upper_interface_dofs(NumericVectorLocal in_u) {
  NumericVectorLocal ret;
  reinit_u_vector(&ret);
  for(unsigned int i = 0; i < in_u.size(); i++) {
    ret[i] = in_u[i];
  }
  if(!is_highest_in_sweeping_direction()) {
    for(unsigned int i = 0; i < upper_interface_dofs.n_elements(); i++) {
      ret[upper_interface_dofs.nth_index_in_set(i)] = ComplexNumber(0,0);
    }
  }
  return ret;
}

NumericVectorLocal NonLocalProblem::zero_lower_interface_dofs(NumericVectorLocal in_u) {
  NumericVectorLocal ret = in_u;
  for(unsigned int i = 0; i < in_u.size(); i++) {
    ret[i] = in_u[i];
  }
  if(!is_lowest_in_sweeping_direction()) {
    for(unsigned int i = 0; i < lower_interface_dofs.n_elements(); i++) {
      ret[lower_interface_dofs.nth_index_in_set(i)] = ComplexNumber(0,0);
    }
  }
  return ret;
}

void NonLocalProblem::set_child_rhs_from_u(NumericVectorLocal in_u, bool add_onto_child_rhs) {
  // child->rhs_mismatch = child->rhs;
  // child->rhs_mismatch.compress(VectorOperation::insert);
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
  // child->constraints.distribute(child->rhs);

  if(add_onto_child_rhs) {
    child->rhs += child->rhs_mismatch;
  }
}

void NonLocalProblem::update_u_from_trace(NumericVectorLocal * in_u, DofFieldTrace trace, bool from_lower) {
  IndexSet dofs;
  if(from_lower) {
    dofs = lower_interface_dofs;
  } else {
    dofs = upper_interface_dofs;
  }
  for(unsigned int i = 0; i < dofs.n_elements(); i++) {
    (*in_u)[dofs.nth_index_in_set(i)] = trace[i];
  }
}





void NonLocalProblem::reinit() {
  print_info("Nonlocal reinit", "Reinit starting");
  child->reinit();
  
  make_constraints();
  // print_diagnosis_data();
  
  make_sparsity_pattern();

  reinit_rhs();
  
  solution.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  temp_solution.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
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
  std::vector<InterfaceDofData> current = Geometry.inner_domain->get_surface_dof_vector_for_boundary_id_and_level(interface_id, 0);
  for(unsigned int j = 0; j < current.size(); j++) {
    ret.add_index(current[j].index);
  }
  
  for(unsigned int i = 0; i < 6; i++) {
    if( i != interface_id && !are_opposing_sites(i,interface_id)) {
      if(Geometry.levels[level].is_surface_truncated[i]) {
        std::vector<InterfaceDofData> current = Geometry.levels[level].surfaces[i]->get_dof_association_by_boundary_id(interface_id);
        for(unsigned int j = 0; j < current.size(); j++) {
          ret.add_index(current[j].index - Geometry.levels[level].inner_first_dof); 
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

void NonLocalProblem::compute_solver_factorization() {
  child->compute_solver_factorization();
  // child->output_results();
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

std::string NonLocalProblem::output_results() {
  HierarchicalProblem::output_results();
  std::vector<std::vector<std::string>> all_files = dealii::Utilities::MPI::gather(GlobalMPI.communicators_by_level[level], filenames);
  if(GlobalParams.MPI_Rank == 0) {
    std::vector<std::string> flattened_filenames;
    for(unsigned int i = 0; i < all_files.size(); i++) {
      for(unsigned int j = 0; j < all_files[i].size(); j++) {
        flattened_filenames.push_back(all_files[i][j]);
      }
    }
    std::string filename = GlobalOutputManager.get_full_filename("level_" + std::to_string(level) + "_solution.pvtu");
    std::ofstream outputvtu(filename);
    for(unsigned int i = 0; i < flattened_filenames.size(); i++) {
      flattened_filenames[i] = "../" + flattened_filenames[i];
    }
    Geometry.inner_domain->data_out.write_pvtu_record(outputvtu, flattened_filenames);
  }
  return "";
}